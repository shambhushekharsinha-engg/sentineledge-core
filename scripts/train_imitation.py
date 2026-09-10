import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vla_policy import get_policy

class DummyVLADataset(Dataset):
    """
    Dummy Dataset for Imitation Learning / Behavioral Cloning.
    In a real scenario, this would load HDF5/Zarr trajectories containing:
    - Camera RGB observations
    - Natural Language instructions (or precomputed embeddings)
    - Expert Joint Actions (labels)
    """
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # image: (3, 480, 640)
        img = torch.randn(3, 480, 640, dtype=torch.float32)
        # instruction embedding: (768,)
        lang = torch.randn(768, dtype=torch.float32)
        # target action (dual SO-101 arm joint positions): (16,)
        action = torch.randn(16, dtype=torch.float32)
        return img, lang, action


def train_policy(epochs=5, batch_size=16, lr=1e-4, save_dir="models/weights"):
    """
    Trains the VLA policy using standard behavioral cloning (MSE Loss).
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Initializing Model and Dataset...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_policy().to(device)
    model.train()
    
    dataset = DummyVLADataset(num_samples=500)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # Simple BC loss
    
    print(f"Starting Training on {device} for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (imgs, langs, target_actions) in enumerate(dataloader):
            imgs = imgs.to(device)
            langs = langs.to(device)
            target_actions = target_actions.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predicted_actions = model(imgs, langs)
            
            # Compute loss
            loss = criterion(predicted_actions, target_actions)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(dataloader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---\n")
        
    # Save final model weights
    save_path = os.path.join(save_dir, "vla_policy_final.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Training Complete! Model saved to {save_path}")

if __name__ == "__main__":
    train_policy()
