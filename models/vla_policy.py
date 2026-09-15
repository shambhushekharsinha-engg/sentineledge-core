"""
VLA Policy: Temporal Action Chunking Transformer
-------------------------------------------------
Architecture based on Action Chunking with Transformers (ACT) and Pi0 principles.
Features:
  - Vision encoder (CNN backbone)
  - Language encoder (maps 768-d BERT/SentenceTransformer embeddings → feature dim)
  - Action history encoder (temporal context for multi-step task memory)
  - Cross-modal Transformer (fuses vision, language, history tokens)
  - Bimanual action head (predicts 16-DOF joint targets: 8 per arm)
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvVisionEncoder(nn.Module):
    """Lightweight CNN vision backbone for extracting spatial features from RGB frames."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            # Stage 1: 480x640 → 240x320
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            # Stage 2: 240x320 → 120x160
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            # Stage 3: 120x160 → 60x80
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            # Stage 4: 60x80 → 30x40
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),   # → 256 x 4 x 4
            nn.Flatten(),                   # → 4096
            nn.Linear(4096, out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, out_dim)


class VLAPolicy(nn.Module):
    """
    Temporal Vision-Language-Action Policy for Bimanual Manipulation.

    Inputs
    ------
    pixels          : (B, 3, H, W)  — RGB camera observation
    language_emb    : (B, lang_dim) — pre-computed sentence embedding (e.g. SentenceTransformer)
    action_history  : (B, history_len * action_dim) — flattened recent actions for temporal context

    Output
    ------
    action : (B, action_dim) — predicted joint targets for both SO-101 arms
    """

    def __init__(
        self,
        action_dim: int = 16,
        history_len: int = 4,
        lang_dim: int = 768,
        feature_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.history_len = history_len
        self.feature_dim = feature_dim

        # Encoders
        self.vision_encoder  = ConvVisionEncoder(out_dim=feature_dim)
        self.language_proj   = nn.Sequential(nn.Linear(lang_dim, feature_dim), nn.GELU())
        self.history_proj    = nn.Sequential(nn.Linear(action_dim * history_len, feature_dim), nn.GELU())

        # Learnable type embeddings (tells transformer WHAT each token represents)
        self.type_embed = nn.Embedding(3, feature_dim)  # 0=vision, 1=language, 2=history

        # Cross-modal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead,
            dim_feedforward=feature_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bimanual action head (separate heads for left/right arms for better specialization)
        self.left_arm_head  = nn.Linear(feature_dim, action_dim // 2)
        self.right_arm_head = nn.Linear(feature_dim, action_dim // 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        pixels: torch.Tensor,
        language_emb: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = pixels.size(0)
        device = pixels.device

        if action_history is None:
            action_history = torch.zeros(B, self.history_len * self.action_dim, device=device)

        # Encode each modality → (B, 1, feature_dim)
        v = self.vision_encoder(pixels).unsqueeze(1)  + self.type_embed(torch.tensor([0], device=device))
        l = self.language_proj(language_emb).unsqueeze(1) + self.type_embed(torch.tensor([1], device=device))
        h = self.history_proj(action_history).unsqueeze(1)  + self.type_embed(torch.tensor([2], device=device))

        # Fuse via cross-modal Transformer: (B, 3, feature_dim)
        seq = torch.cat([v, l, h], dim=1)
        out = self.transformer(seq)  # (B, 3, feature_dim)

        # Decode from vision token (richest spatial signal)
        vision_out = out[:, 0, :]

        # Bimanual specialised heads
        left_action  = self.left_arm_head(vision_out)   # (B, 8)
        right_action = self.right_arm_head(vision_out)  # (B, 8)

        return torch.cat([left_action, right_action], dim=-1)  # (B, 16)


def get_policy(model_path: Optional[str] = None) -> VLAPolicy:
    """Factory function. Loads weights if a checkpoint path is provided."""
    policy = VLAPolicy()
    if model_path:
        state = torch.load(model_path, map_location="cpu")
        policy.load_state_dict(state)
        print(f"[Policy] Loaded weights from {model_path}")
    policy.eval()
    return policy


if __name__ == "__main__":
    # Quick sanity check
    model = get_policy()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"VLA Policy | Parameters: {total_params:.2f}M")

    dummy_img  = torch.randn(2, 3, 480, 640)
    dummy_lang = torch.randn(2, 768)
    out = model(dummy_img, dummy_lang)
    print(f"Action output shape: {out.shape}")   # Expected: torch.Size([2, 16])
