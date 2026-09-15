"""
evaluate.py — Official Challenge Evaluation Loop
-------------------------------------------------
Runs the VLA policy over N randomized seeds, records HUD-annotated MP4 videos,
logs per-seed metrics (success, steps, reward, distances), and prints a final report.

Usage
-----
  # Standard 10-seed run
  python scripts/evaluate.py --seeds 10

  # Custom instruction
  python scripts/evaluate.py --seeds 5 --instruction "Place the fork beside the plate."

  # Disable video recording (fast mode)
  python scripts/evaluate.py --seeds 20 --no-video
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from simulation.env import BimanualDinnerEnv

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

try:
    import openvino as ov
    OV_AVAILABLE = True
except ImportError:
    OV_AVAILABLE = False


# ─── Policy Loading ────────────────────────────────────────────────────────────

def load_policy(ir_path: str = "inference/ir_model/vla_policy_int8.xml", device: str = "AUTO"):
    """Loads the OpenVINO INT8 compiled model. Falls back to None (random policy) if missing."""
    if not OV_AVAILABLE:
        print("[Eval] OpenVINO not installed. Using random policy.")
        return None
    if not os.path.exists(ir_path):
        print(f"[Eval] Model not found at {ir_path}. Using random policy.")
        return None
    core = ov.Core()
    model = core.read_model(ir_path)
    compiled = core.compile_model(model, device)
    print(f"[Eval] Loaded OpenVINO INT8 model on: {device}")
    return compiled


# ─── HUD Rendering ────────────────────────────────────────────────────────────

def render_hud(frame: np.ndarray, seed: int, step: int, max_steps: int,
               instruction: str, reward: float, success: bool) -> np.ndarray:
    """Overlays a heads-up display onto a raw RGB frame. Returns annotated RGB frame."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    # Semi-transparent dark bar at top
    overlay = bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0)

    # Seed / Step counter
    cv2.putText(bgr, f"Seed: {seed}  |  Step: {step:>3}/{max_steps}", (12, 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (100, 255, 100), 1, cv2.LINE_AA)

    # Instruction (truncated)
    cmd = f"CMD: {instruction}"[:72]
    cv2.putText(bgr, cmd, (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Reward indicator
    cv2.putText(bgr, f"Reward: {reward:.2f}", (w - 160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 50), 1, cv2.LINE_AA)

    # Success flash
    if success:
        cv2.putText(bgr, "*** SUCCESS ***", (w // 2 - 110, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 100), 3, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ─── Evaluation Loop ──────────────────────────────────────────────────────────

def run_evaluation(
    num_seeds: int = 10,
    max_steps: int = 100,
    record_video: bool = True,
    custom_instruction: str = None,
    ir_path: str = "inference/ir_model/vla_policy_int8.xml",
    device: str = "AUTO",
):
    """
    Main evaluation loop. Returns path to the last recorded video.
    """
    env    = BimanualDinnerEnv(render_mode="rgb_array")
    policy = load_policy(ir_path, device)

    os.makedirs("data/videos", exist_ok=True)

    results     = []
    last_video  = None
    total_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Intel Physical AI Challenge -- Evaluation")
    print(f"  Seeds: {num_seeds}  |  Max Steps: {max_steps}")
    print(f"{'='*60}\n")

    for seed in range(num_seeds):
        options = {"instruction": custom_instruction} if custom_instruction else None
        obs, _  = env.reset(seed=seed, options=options)

        instruction   = obs["instruction"]
        frames        = []
        cumulative_r  = 0.0
        success       = False
        steps_to_done = max_steps
        dists_l, dists_r = [], []

        print(f"[Seed {seed:>2}] {instruction[:70]}")

        for step in range(max_steps):
            # Render frame with HUD
            if record_video:
                frames.append(render_hud(obs["pixels"], seed, step, max_steps,
                                         instruction, cumulative_r, success))

            # Prepare OpenVINO inputs
            lang_emb  = np.random.randn(1, 768).astype(np.float32)
            img_input = np.expand_dims(
                np.transpose(obs["pixels"], (2, 0, 1)), axis=0
            ).astype(np.float32)

            action = (
                policy([img_input, lang_emb])[0][0]
                if policy is not None
                else env.action_space.sample()
            )

            obs, reward, terminated, _, info = env.step(action)
            cumulative_r += reward
            success       = info.get("is_success", False)

            if "dist_left" in info:
                dists_l.append(info["dist_left"])
                dists_r.append(info["dist_right"])

            if terminated:
                steps_to_done = step + 1
                # Capture success frame
                if record_video:
                    frames.append(render_hud(obs["pixels"], seed, step+1, max_steps,
                                             instruction, cumulative_r, True))
                break

        results.append({
            "seed": seed,
            "success": success,
            "steps": steps_to_done,
            "reward": cumulative_r,
            "avg_dist_left":  float(np.mean(dists_l)) if dists_l else -1.0,
            "avg_dist_right": float(np.mean(dists_r)) if dists_r else -1.0,
        })

        status = "[OK]  SUCCESS" if success else "[FAIL]"
        print(f"         {status}  |  steps={steps_to_done}  |  reward={cumulative_r:.2f}")

        if record_video and frames and IMAGEIO_AVAILABLE:
            vpath = f"data/videos/demo_seed_{seed}.mp4"
            imageio.mimsave(vpath, frames, fps=15, macro_block_size=None)
            last_video = vpath

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time   = time.time() - total_start
    success_rate = 100.0 * sum(r["success"] for r in results) / num_seeds

    print(f"\n{'='*60}")
    print(f"  Evaluation Complete in {total_time:.1f}s")
    print(f"  Success Rate: {success_rate:.1f}%  ({sum(r['success'] for r in results)}/{num_seeds})")
    print(f"  Mean Reward:  {np.mean([r['reward'] for r in results]):.3f}")
    print(f"{'='*60}\n")

    return last_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official Intel Physical AI Evaluation Loop")
    parser.add_argument("--seeds",       type=int,  default=10)
    parser.add_argument("--max-steps",   type=int,  default=100)
    parser.add_argument("--no-video",    action="store_true")
    parser.add_argument("--instruction", type=str,  default=None)
    parser.add_argument("--device",      type=str,  default="AUTO",
                        help="OpenVINO device: AUTO | CPU | GPU | NPU")
    args = parser.parse_args()

    run_evaluation(
        num_seeds=args.seeds,
        max_steps=args.max_steps,
        record_video=not args.no_video,
        custom_instruction=args.instruction,
        device=args.device,
    )
