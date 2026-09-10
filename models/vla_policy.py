import torch
import torch.nn as nn

class DummyVLAPolicy(nn.Module):
    """
    A dummy Vision-Language-Action (VLA) policy based on ACT/Pi0.
    In a real scenario, this would use `lerobot.models.act.ACTPolicy` or similar,
    loading pretrained weights and tokenizing language instructions.
    """
    def __init__(self, action_dim=16):
        super().__init__()
        # Simulating a small vision backbone (e.g., ResNet18) + Transformer
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 239 * 319, 256)  # Assumes 480x640 input
        )
        self.language_encoder = nn.Linear(768, 256) # Assuming BERT embeddings
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
            num_layers=2
        )
        self.action_head = nn.Linear(256, action_dim)

    def forward(self, pixels, language_embedding):
        """
        pixels: (B, 3, 480, 640) float tensor
        language_embedding: (B, 768) float tensor
        """
        v_feat = self.vision_encoder(pixels).unsqueeze(1) # (B, 1, 256)
        l_feat = self.language_encoder(language_embedding).unsqueeze(1) # (B, 1, 256)
        
        # Concat features as sequence
        seq = torch.cat([v_feat, l_feat], dim=1) # (B, 2, 256)
        out = self.transformer(seq)
        
        # Predict action from first token
        action = self.action_head(out[:, 0, :])
        return action

def get_policy(model_path=None):
    """
    Factory function to return the VLA policy.
    """
    policy = DummyVLAPolicy()
    if model_path:
        policy.load_state_dict(torch.load(model_path))
    policy.eval()
    return policy

if __name__ == "__main__":
    # Test the policy
    model = get_policy()
    dummy_img = torch.randn(1, 3, 480, 640)
    dummy_lang = torch.randn(1, 768)
    out = model(dummy_img, dummy_lang)
    print(f"Action output shape: {out.shape}")
