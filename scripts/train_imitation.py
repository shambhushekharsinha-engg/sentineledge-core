"""
train_imitation.py — Behavioral Cloning Training Loop
------------------------------------------------------
Trains the VLA Policy from expert demonstrations using imitation learning.
Supports:
  - HDF5 dataset loading (output of record_teleop.py / generate_dataset.py)
  - Dummy synthetic dataset fallback (for CI / first-run verification)
  - Mixed-precision training (torch.amp) for faster GPU training
  - LR scheduling (CosineAnnealing)
  - Checkpoint saving with best-val tracking
  - WandB / TensorBoard logging stubs
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vla_policy import get_policy


# ─── Dataset ──────────────────────────────────────────────────────────────────

class HDF5VLADataset(Dataset):
    """
    Loads expert trajectories from an HDF5 file produced by record_teleop.py
    or generate_dataset.py.

    Each episode group contains:
      observations/pixels : (T, H, W, 3)  uint8
      observations/state  : (T, 16)        float32
      actions             : (T, 16)        float32
    """
    def __init__(self, h5_path: str):
        import h5py
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.episode_keys = list(f.keys())
            # Pre-count total transitions
            self.lengths = [f[k]["actions"].shape[0] for k in self.episode_keys]
        self.cumlen = np.cumsum([0] + self.lengths)

    def __len__(self):
        return self.cumlen[-1]

    def __getitem__(self, idx):
        import h5py
        # Find which episode
        ep_idx = np.searchsorted(self.cumlen[1:], idx, side="right")
        step   = idx - self.cumlen[ep_idx]
        with h5py.File(self.h5_path, "r") as f:
            ep = f[self.episode_keys[ep_idx]]
            frame  = ep["observations/pixels"][step].astype(np.float32) / 255.0  # (H, W, 3)
            action = ep["actions"][step]                                           # (16,)
        frame = np.transpose(frame, (2, 0, 1))  # → (3, H, W)
        lang  = np.random.randn(768).astype(np.float32)  # Replace with real embeddings
        return (
            torch.from_numpy(frame),
            torch.from_numpy(lang),
            torch.from_numpy(action)
        )


class SyntheticVLADataset(Dataset):
    """Synthetic fallback dataset — use only for debugging / CI smoke tests."""
    def __init__(self, num_samples: int = 500):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img    = torch.randn(3, 480, 640)
        lang   = torch.randn(768)
        action = torch.randn(16)
        return img, lang, action


# ─── Training ─────────────────────────────────────────────────────────────────

def train_policy(
    dataset_path: str = None,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    val_split: float = 0.1,
    save_dir: str = "models/weights",
    amp: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Dataset
    if dataset_path and os.path.exists(dataset_path):
        print(f"[Train] Loading HDF5 dataset: {dataset_path}")
        full_ds = HDF5VLADataset(dataset_path)
    else:
        print("[Train] No HDF5 dataset found — using synthetic data for demonstration.")
        full_ds = SyntheticVLADataset(num_samples=500)

    val_len   = max(1, int(len(full_ds) * val_split))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model     = get_policy().to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler    = GradScaler(enabled=(amp and device.type == "cuda"))

    best_val_loss  = float("inf")
    best_ckpt_path = os.path.join(save_dir, "vla_policy_best.pth")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, langs, targets in train_loader:
            imgs, langs, targets = imgs.to(device), langs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast(enabled=(amp and device.type == "cuda")):
                preds = model(imgs, langs)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, langs, targets in val_loader:
                imgs, langs, targets = imgs.to(device), langs.to(device), targets.to(device)
                preds    = model(imgs, langs)
                val_loss += criterion(preds, targets).item()

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch:>2}/{epochs}]  train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ↳ New best saved → {best_ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(save_dir, "vla_policy_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\n[Train] Complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Final  checkpoint: {final_path}")
    print(f"  Best   checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLA Policy via Behavioral Cloning")
    parser.add_argument("--dataset",    type=str,   default=None,        help="Path to HDF5 dataset (defaults to synthetic)")
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch-size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save-dir",   type=str,   default="models/weights")
    parser.add_argument("--no-amp",     action="store_true", help="Disable mixed-precision training")
    args = parser.parse_args()

    train_policy(
        dataset_path=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        amp=not args.no_amp,
    )
