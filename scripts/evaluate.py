import os
import argparse
import numpy as np
import imageio
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulation.env import BimanualDinnerEnv
import openvino as ov

def load_policy(ir_path="inference/ir_model/vla_policy_int8.xml", device="AUTO"):
    if not os.path.exists(ir_path):
        print(f"Warning: OpenVINO INT8 model not found at {ir_path}. Falling back to random policy.")
        return None
    core = ov.Core()
    model = core.read_model(ir_path)
    print(f"Loaded INT8 Quantized Model on device: {device}")
    return core.compile_model(model, device)

def run_evaluation(num_seeds=10, max_steps=100, record_video=True, custom_instruction=None):
    """
    Runs the evaluation loop. 
    Returns the path to the last recorded video (useful for UI integration).
    """
    env = BimanualDinnerEnv(render_mode="rgb_array")
    policy = load_policy()
    
    success_count = 0
    os.makedirs("data/videos", exist_ok=True)
    
    print(f"Starting evaluation over {num_seeds} randomized seeds...")
    last_video_path = None
    
    for seed in range(num_seeds):
        options = {"instruction": custom_instruction} if custom_instruction else None
        obs, _ = env.reset(seed=seed, options=options)
        frames = []
        
        instruction = obs['instruction']
        print(f"--- Running Seed {seed} ---")
        print(f"Instruction: {instruction}")
        
        for step in range(max_steps):
            if record_video and step % 2 == 0: 
                frame = obs["pixels"].copy()
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # HUD overlay
                cv2.putText(frame_bgr, f"Seed: {seed} | Step: {step}/{max_steps}", (15, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cmd_text = f"Cmd: {instruction}"
                if len(cmd_text) > 60: cmd_text = cmd_text[:57] + "..."
                cv2.putText(frame_bgr, cmd_text, (15, 65), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
            lang_emb = np.random.randn(1, 768).astype(np.float32)
            img_input = np.transpose(obs["pixels"], (2, 0, 1))
            img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
            
            if policy is not None:
                action = policy([img_input, lang_emb])[0][0]
            else:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or step == max_steps - 1:
                if info.get("is_success", False):
                    success_count += 1 
                break
                
        if record_video:
            video_path = f"data/videos/demo_seed_{seed}.mp4"
            imageio.mimsave(video_path, frames, fps=15, macro_block_size=None)
            print(f"Saved video for seed {seed} at {video_path}")
            last_video_path = video_path
            
    success_rate = (success_count / num_seeds) * 100
    print(f"\nEvaluation Complete! Success Rate: {success_rate:.1f}%")
    env.close()
    
    return last_video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="Number of evaluation seeds")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--instruction", type=str, default=None, help="Custom language instruction")
    args = parser.parse_args()
    
    run_evaluation(num_seeds=args.seeds, record_video=not args.no_video, custom_instruction=args.instruction)
