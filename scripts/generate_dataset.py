"""
generate_dataset.py -- Synthetic expert dataset generator for Behavioral Cloning.

Runs N episodes of random actions in BimanualDinnerEnv and saves the
trajectories to an HDF5 file (data/bc_dataset.h5) suitable for BC training.

Usage
-----
python scripts/generate_dataset.py
python scripts/generate_dataset.py --episodes 50 --steps 200 --output data/bc_dataset.h5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import h5py
except ImportError:
    print("[ERROR] h5py is not installed. Run: pip install h5py")
    sys.exit(1)

try:
    from simulation.env import BimanualDinnerEnv
    _ENV_AVAILABLE = True
except ImportError:
    print("[WARN] simulation.env not importable -- using DummyEnv for demonstration.")
    _ENV_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback dummy environment
# ---------------------------------------------------------------------------

class DummyEnv:
    """Stand-in environment that mimics the BimanualDinnerEnv API."""

    OBS_PIXELS_SHAPE = (480, 640, 3)
    OBS_STATE_DIM = 48
    ACTION_DIM = 16

    def __init__(self):
        self.action_space_shape = (self.ACTION_DIM,)

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        obs = {
            "pixels": rng.integers(0, 256, self.OBS_PIXELS_SHAPE, dtype=np.uint8),
            "state": rng.random(self.OBS_STATE_DIM).astype(np.float32),
        }
        return obs, {}

    def step(self, action):
        obs = {
            "pixels": np.random.randint(0, 256, self.OBS_PIXELS_SHAPE, dtype=np.uint8),
            "state": np.random.random(self.OBS_STATE_DIM).astype(np.float32),
        }
        return obs, float(np.random.uniform(-0.1, 1.0)), False, False, {}

    def close(self):
        pass


TASK_INSTRUCTIONS = [
    "Place the plate and cup on the table in their correct positions.",
    "Open the drawer and retrieve the spoon, then place it beside the plate.",
    "Pick up the cup with arm A, pass it to arm B, then place it on the table.",
    "Place the fork on the right side of the plate while avoiding the moving obstacle.",
]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(num_episodes, steps_per_episode, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    EnvClass = BimanualDinnerEnv if _ENV_AVAILABLE else DummyEnv
    env = EnvClass()

    print(f"\n{'='*60}")
    print(f"  SentinelEdge BC Dataset Generator")
    print(f"  Env       : {EnvClass.__name__}")
    print(f"  Episodes  : {num_episodes}")
    print(f"  Steps/ep  : {steps_per_episode}")
    print(f"  Output    : {output_path}")
    print(f"{'='*60}\n")

    t_start = time.time()

    with h5py.File(output_path, "w") as hf:
        hf.attrs["generator"] = "SentinelEdge generate_dataset.py"
        hf.attrs["num_episodes"] = num_episodes
        hf.attrs["steps_per_episode"] = steps_per_episode
        hf.attrs["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        for ep_idx in range(num_episodes):
            seed = ep_idx * 137 + 42
            obs, _ = env.reset(seed=seed)

            if isinstance(obs, dict):
                pixels_buf = np.zeros(
                    (steps_per_episode, *obs["pixels"].shape), dtype=np.uint8
                )
                state_buf = np.zeros(
                    (steps_per_episode, obs["state"].shape[0]), dtype=np.float32
                )
            else:
                pixels_buf = np.zeros((steps_per_episode, 480, 640, 3), dtype=np.uint8)
                state_buf = np.zeros((steps_per_episode, 48), dtype=np.float32)

            action_dim = env.action_space_shape[0] if hasattr(env, "action_space_shape") else 16
            action_buf = np.zeros((steps_per_episode, action_dim), dtype=np.float32)

            for step_idx in range(steps_per_episode):
                action = np.random.uniform(-1.0, 1.0, action_dim).astype(np.float32)
                if isinstance(obs, dict):
                    pixels_buf[step_idx] = obs["pixels"]
                    state_buf[step_idx] = obs["state"]
                action_buf[step_idx] = action
                obs, _reward, terminated, truncated, _info = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset(seed=seed + step_idx)

            grp = hf.create_group(f"episode_{ep_idx:04d}")
            obs_grp = grp.create_group("observations")
            obs_grp.create_dataset("pixels", data=pixels_buf, compression="gzip", compression_opts=4)
            obs_grp.create_dataset("state", data=state_buf, compression="gzip", compression_opts=4)
            grp.create_dataset("actions", data=action_buf, compression="gzip", compression_opts=4)
            instruction = TASK_INSTRUCTIONS[ep_idx % len(TASK_INSTRUCTIONS)]
            grp.attrs["instruction"] = instruction
            grp.attrs["seed"] = seed
            grp.attrs["steps"] = steps_per_episode

            elapsed = time.time() - t_start
            eps_done = ep_idx + 1
            rate = eps_done / elapsed
            eta = (num_episodes - eps_done) / rate if rate > 0 else 0
            short_instr = instruction[:50]
            print(f"  Episode {eps_done:>4}/{num_episodes}  seed={seed:<6}  instruction=\"{short_instr}...\"  ETA {eta:.1f}s")

    env.close()
    total = time.time() - t_start
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\nDataset saved -> {output_path}  ({size_mb:.1f} MB)  [{total:.1f}s total]\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic BC dataset for SentinelEdge.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes (default: 20)")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode (default: 100)")
    parser.add_argument("--output", type=str, default="data/bc_dataset.h5", help="Output HDF5 path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        output_path=ROOT / args.output,
    )
