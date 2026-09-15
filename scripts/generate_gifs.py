"""
generate_gifs.py — Convert per-seed demo MP4s to optimized GIFs for GitHub README
-----------------------------------------------------------------------------------
Usage:
  python scripts/generate_gifs.py
  python scripts/generate_gifs.py --input data/videos --output data/gifs --fps 10 --scale 320
"""

import os
import sys
import argparse
import subprocess
import shutil
import numpy as np

try:
    import imageio
except ImportError:
    print("Run: pip install imageio imageio[ffmpeg]")
    sys.exit(1)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def resize_frame(frame, width=320):
    """Resize frame maintaining aspect ratio."""
    if not CV2_AVAILABLE:
        return frame
    h, w = frame.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_AREA)


def mp4_to_gif(mp4_path: str, gif_path: str, fps: int = 10, scale: int = 320,
               every_n: int = 2, max_frames: int = 60):
    """
    Converts an MP4 to an optimized GIF.
    - Skips every_n frames to reduce file size
    - Resizes to scale width
    - Caps at max_frames total frames
    """
    reader = imageio.get_reader(mp4_path)
    frames = []
    for i, frame in enumerate(reader):
        if i % every_n != 0:
            continue
        resized = resize_frame(frame, width=scale)
        frames.append(resized)
        if len(frames) >= max_frames:
            break
    reader.close()

    if not frames:
        print(f"  [WARN] No frames read from {mp4_path}")
        return

    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    size_kb = os.path.getsize(gif_path) / 1024
    print(f"  Saved {os.path.basename(gif_path)} — {len(frames)} frames, {size_kb:.0f} KB")


def generate_all_gifs(input_dir: str, output_dir: str, fps: int = 10,
                       scale: int = 320, pattern: str = "demo_seed_{}.mp4"):
    os.makedirs(output_dir, exist_ok=True)

    seed = 0
    generated = []
    while True:
        mp4 = os.path.join(input_dir, pattern.format(seed))
        if not os.path.exists(mp4):
            break
        gif = os.path.join(output_dir, f"seed_{seed}.gif")
        print(f"Converting seed {seed} ...", end=" ", flush=True)
        mp4_to_gif(mp4, gif, fps=fps, scale=scale)
        generated.append((seed, gif))
        seed += 1

    if not generated:
        print(f"[ERROR] No MP4 files found in '{input_dir}'.")
        sys.exit(1)

    print(f"\nGenerated {len(generated)} GIFs in '{output_dir}'")
    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert seed demo MP4s to GIFs")
    parser.add_argument("--input",  type=str, default="data/videos")
    parser.add_argument("--output", type=str, default="data/gifs")
    parser.add_argument("--fps",    type=int, default=10)
    parser.add_argument("--scale",  type=int, default=320, help="Output GIF width in pixels")
    args = parser.parse_args()

    generate_all_gifs(args.input, args.output, fps=args.fps, scale=args.scale)
