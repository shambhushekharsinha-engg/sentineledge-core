"""
Teleoperation & Data Collection Script
--------------------------------------
This script allows human operators to manually control the simulated dual SO-101 
arms to collect 'Expert Demonstrations' for Behavioral Cloning (train_imitation.py).

Keyboard controls (mock):
- WASD: Move Left Arm
- IJKL: Move Right Arm
- SPACE: Record Frame
"""

import os
import h5py
import numpy as np
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulation.env import BimanualDinnerEnv

def collect_demonstrations(num_episodes=5, save_dir="data/teleop"):
    os.makedirs(save_dir, exist_ok=True)
    env = BimanualDinnerEnv(render_mode="human")
    
    print("=== Teleoperation Mode Started ===")
    print("In a full setup, use a SpaceMouse or VR controllers to move the arms.")
    print("Recording expert trajectories...")

    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        frames = []
        actions = []
        
        print(f"Episode {ep+1} - Instruction: {obs['instruction']}")
        
        # Simulating 50 steps of human expert control
        for step in range(50):
            # MOCK TELEOP: Normally we would read from joy/keyboard here
            expert_action = np.random.uniform(-0.1, 0.1, size=(16,)).astype(np.float32)
            
            obs, _, _, _, _ = env.step(expert_action)
            
            frames.append(obs["pixels"])
            actions.append(expert_action)
            time.sleep(0.05) # simulate human reaction time
            
        # Save trajectory to HDF5 (standard format for Imitation Learning datasets)
        file_path = os.path.join(save_dir, f"expert_demo_{ep}.h5")
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('observations/pixels', data=np.array(frames))
            f.create_dataset('actions', data=np.array(actions))
            f.attrs['instruction'] = obs['instruction']
            
        print(f"Saved Expert Trajectory to {file_path}")

    env.close()
    print("Data Collection Complete. Ready for `train_imitation.py`.")

if __name__ == "__main__":
    collect_demonstrations()
