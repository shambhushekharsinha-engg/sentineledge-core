import torch
import torch.nn as nn

class DummyVLAPolicy(nn.Module):
    """
    A Vision-Language-Action (VLA) policy based on ACT/Pi0 architectures.
    Features:
    - Multi-modal Fusion (Vision + Language)
    - Multi-step Task Context (Temporal Action Chunking)
    """
    def __init__(self, action_dim=16, history_len=4):
        super().__init__()
        self.history_len = history_len
        
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 239 * 319, 256)
        )
        self.language_encoder = nn.Linear(768, 256)
        
        # Temporal encoder to maintain multi-step task context (Historical Actions)
        self.history_encoder = nn.Linear(action_dim * history_len, 256)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=2
        )
        self.action_head = nn.Linear(256, action_dim)

    def forward(self, pixels, language_embedding, action_history=None):
        """
        pixels: (B, 3, 480, 640)
        language_embedding: (B, 768)
        action_history: (B, history_len * action_dim) past context to maintain task state
        """
        v_feat = self.vision_encoder(pixels).unsqueeze(1) # (B, 1, 256)
        l_feat = self.language_encoder(language_embedding).unsqueeze(1) # (B, 1, 256)
        
        if action_history is None:
            action_history = torch.zeros(pixels.size(0), self.history_len * 16, device=pixels.device)
            
        h_feat = self.history_encoder(action_history).unsqueeze(1) # (B, 1, 256)
        
        # Concat features as sequence to maintain multi-step context
        seq = torch.cat([v_feat, l_feat, h_feat], dim=1) # (B, 3, 256)
        out = self.transformer(seq)
        
        # Predict action from the output
        action = self.action_head(out[:, 0, :])
        return action

def get_policy(model_path=None):
    policy = DummyVLAPolicy()
    if model_path:
        policy.load_state_dict(torch.load(model_path))
    policy.eval()
    return policy
