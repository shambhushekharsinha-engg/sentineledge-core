"""
stitch_demo.py — Combine 10 per-seed demo videos into one submission reel
--------------------------------------------------------------------------
Usage:
  python scripts/stitch_demo.py                         # default: data/videos/
  python scripts/stitch_demo.py --input data/videos --output demo_final.mp4
  python scripts/stitch_demo.py --add-title-cards       # adds "Seed N" intro frames
"""

import os
import sys
import argparse
import numpy as np

try:
    import imageio
except ImportError:
    print("imageio is required. Run: pip install imageio imageio[ffmpeg]")
    sys.exit(1)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def make_title_card(seed: int, fps: int = 15, duration_sec: float = 1.5,
                    width: int = 640, height: int = 480) -> list:
    """Generates a dark title card (e.g. 'Seed 3') as a list of RGB frames."""
    n_frames = int(fps * duration_sec)
    frames = []
    for _ in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if CV2_AVAILABLE:
            text = f"Seed {seed}"
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = 2.0
            thickness = 3
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = (width - tw) // 2
            y = (height + th) // 2
            cv2.putText(frame, text, (x, y), font, scale, (100, 200, 255), thickness, cv2.LINE_AA)
            # Subtitle
            sub = "Intel Physical AI Challenge"
            (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
            cv2.putText(frame, sub, ((width - sw) // 2, y + 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1, cv2.LINE_AA)
        frames.append(frame)
    return frames


def stitch_videos(input_dir: str, output_path: str, fps: int = 15,
                  add_title_cards: bool = True, pattern: str = "demo_seed_{}.mp4"):
    """
    Reads demo_seed_0.mp4 … demo_seed_N.mp4 from input_dir and concatenates
    them (optionally with title cards) into a single output MP4.
    """
    # Discover available seed videos in order
    seed = 0
    video_paths = []
    while True:
        p = os.path.join(input_dir, pattern.format(seed))
        if os.path.exists(p):
            video_paths.append((seed, p))
            seed += 1
        else:
            break

    if not video_paths:
        print(f"[ERROR] No demo videos found in '{input_dir}'. "
              f"Run: python scripts/evaluate.py --seeds 10")
        sys.exit(1)

    print(f"Found {len(video_paths)} demo video(s). Stitching...")

    all_frames = []

    for seed_idx, vpath in video_paths:
        print(f"  [{seed_idx:>2}] Reading {vpath} ...", end=" ", flush=True)

        # Title card
        if add_title_cards and CV2_AVAILABLE:
            all_frames.extend(make_title_card(seed_idx, fps=fps))

        reader = imageio.get_reader(vpath)
        frames = [f for f in reader]
        reader.close()
        all_frames.extend(frames)
        print(f"{len(frames)} frames")

    print(f"\nWriting combined video → {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    imageio.mimsave(output_path, all_frames, fps=fps, macro_block_size=None)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    total_sec = len(all_frames) / fps
    print(f"\n✅ Done!")
    print(f"   Total frames : {len(all_frames)}")
    print(f"   Duration     : {total_sec:.1f}s  ({total_sec/60:.1f} min)")
    print(f"   File size    : {size_mb:.1f} MB")
    print(f"   Output       : {output_path}")
    print(f"\n📤 Upload this file to YouTube / HuggingFace, then add the link to README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stitch per-seed demo videos into one submission reel")
    parser.add_argument("--input",           type=str, default="data/videos",
                        help="Directory containing demo_seed_X.mp4 files")
    parser.add_argument("--output",          type=str, default="data/demo_final.mp4",
                        help="Path for the stitched output video")
    parser.add_argument("--fps",             type=int, default=15)
    parser.add_argument("--add-title-cards", action="store_true", default=True,
                        help="Insert a 'Seed N' title card between clips (default: on)")
    parser.add_argument("--no-title-cards",  action="store_true",
                        help="Disable title cards")
    args = parser.parse_args()

    stitch_videos(
        input_dir=args.input,
        output_path=args.output,
        fps=args.fps,
        add_title_cards=(args.add_title_cards and not args.no_title_cards),
    )
