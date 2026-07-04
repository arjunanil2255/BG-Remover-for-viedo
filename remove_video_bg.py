"""
====================================================
  Powerful Video Background Removal Tool
  Uses: rembg (U2-Net / BiRefNet deep learning) + OpenCV
====================================================

INSTALL DEPENDENCIES:
    pip install rembg[gpu] opencv-python-headless pillow tqdm onnxruntime-gpu
    # CPU-only (no GPU):
    pip install rembg opencv-python-headless pillow tqdm onnxruntime

USAGE EXAMPLES:
    # Remove background -> transparent output (green background by default)
    python remove_video_bg.py input.mp4 output.mp4

    # Replace with green screen
    python remove_video_bg.py input.mp4 output.mp4 --bg green

    # Replace with custom background image
    python remove_video_bg.py input.mp4 output.mp4 --bg background.jpg

    # Replace with custom background video
    python remove_video_bg.py input.mp4 output.mp4 --bg background_video.mp4

    # Use the most accurate model (BiRefNet - best for humans)
    python remove_video_bg.py input.mp4 output.mp4 --model birefnet-general

    # Transparent output (WEBM/MOV with alpha channel)
    python remove_video_bg.py input.mp4 output.webm --transparent

    # Process every other frame (2x faster, good for most cases)
    python remove_video_bg.py input.mp4 output.mp4 --skip-frames 1

    # Fine-tune mask (erode edges to remove fringe, feather for smoothness)
    python remove_video_bg.py input.mp4 output.mp4 --erode 2 --feather 3
"""

import cv2
import numpy as np
from PIL import Image
from rembg import new_session, remove
import argparse
import sys
import os
from tqdm import tqdm


# ──────────────────────────────────────────────
#  Available models (accuracy vs speed tradeoff)
# ──────────────────────────────────────────────
MODELS = {
    "u2net":              "General purpose - fast, good accuracy",
    "u2net_human_seg":    "Optimized for humans/portraits - recommended for people",
    "birefnet-general":   "BiRefNet - MOST ACCURATE, slower, great for all subjects",
    "birefnet-portrait":  "BiRefNet portrait - best for face/bust shots",
    "isnet-general-use":  "IS-Net - very accurate, good balance",
    "silueta":            "Silhouette model - fast, good for people",
}


def get_background(bg_arg, frame_size, frame_idx, bg_video_cap=None):
    """Return a background frame (H, W, 3) BGR."""
    W, H = frame_size

    if bg_arg is None:
        return np.zeros((H, W, 3), dtype=np.uint8)  # black

    if bg_arg == "green":
        bg = np.zeros((H, W, 3), dtype=np.uint8)
        bg[:, :] = (0, 177, 64)  # BGR green screen
        return bg

    if bg_arg == "white":
        return np.full((H, W, 3), 255, dtype=np.uint8)

    if bg_arg == "black":
        return np.zeros((H, W, 3), dtype=np.uint8)

    # Background video
    if bg_video_cap is not None:
        ret, bg_frame = bg_video_cap.read()
        if not ret:
            bg_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, bg_frame = bg_video_cap.read()
        if ret:
            return cv2.resize(bg_frame, (W, H))

    # Background image (loaded once, stored as numpy)
    if isinstance(bg_arg, np.ndarray):
        return cv2.resize(bg_arg, (W, H))

    return np.zeros((H, W, 3), dtype=np.uint8)


def refine_mask(alpha, erode_px=0, feather_px=0):
    """
    Post-process the alpha mask:
      - erode_px  : remove edge fringe (positive = shrink mask)
      - feather_px: soften edges (Gaussian blur radius)
    """
    if erode_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)
        )
        alpha = cv2.erode(alpha, kernel, iterations=1)

    if feather_px > 0:
        ksize = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), 0)

    return alpha


def composite(frame_bgr, alpha, background_bgr):
    """Blend foreground over background using alpha mask."""
    fg = frame_bgr.astype(np.float32)
    bg = background_bgr.astype(np.float32)
    a = (alpha / 255.0)[..., np.newaxis]  # (H, W, 1)
    out = fg * a + bg * (1.0 - a)
    return out.astype(np.uint8)


def remove_video_background(
    input_path,
    output_path,
    model_name="u2net_human_seg",
    bg_arg=None,
    skip_frames=0,
    erode_px=0,
    feather_px=2,
    transparent=False,
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=10,
    post_process_mask=True,
):
    print(f"\n🎬 Video Background Removal")
    print(f"   Model     : {model_name}")
    print(f"   Input     : {input_path}")
    print(f"   Output    : {output_path}")
    print(f"   Background: {bg_arg if bg_arg else 'transparent'}")
    print(
        f"   Skip frames: {skip_frames} (process 1 in every {skip_frames+1})")
    print()

    # ── Load rembg session ──────────────────────────────────────────────
    print(
        f"⏳ Loading model '{model_name}'... (first run downloads weights ~170MB)")
    session = new_session(model_name)
    print("✅ Model loaded.\n")

    # ── Open input video ────────────────────────────────────────────────
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        sys.exit(f"❌ Cannot open input video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 {W}x{H} @ {fps:.2f} fps — {total_frames} frames")

    # ── Background setup ────────────────────────────────────────────────
    bg_video_cap = None
    bg_image = None

    if bg_arg and os.path.isfile(str(bg_arg)):
        ext = os.path.splitext(bg_arg)[1].lower()
        if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
            bg_video_cap = cv2.VideoCapture(bg_arg)
        else:
            bg_image = cv2.imread(bg_arg)
            if bg_image is None:
                sys.exit(f"❌ Cannot load background image: {bg_arg}")
            bg_arg = bg_image  # store as ndarray

    # ── Output video writer ─────────────────────────────────────────────
    ext_out = os.path.splitext(output_path)[1].lower()

    if transparent and ext_out in (".webm",):
        # VP9 with alpha (WebM)
        fourcc = cv2.VideoWriter_fourcc(*"VP90")
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (W, H), isColor=True)
        print("⚠️  True alpha channel in video requires ffmpeg post-processing.")
        print("   Writing BGRA... will composite to green for preview.")
        transparent = False  # cv2 can't write real alpha; fall back
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    if not writer.isOpened():
        sys.exit(f"❌ Cannot open output video writer: {output_path}")

    # ── Process frames ──────────────────────────────────────────────────
    prev_alpha = None
    frame_idx = 0

    with tqdm(total=total_frames, unit="frame", ncols=80) as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Interpolate skipped frames using previous mask
            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                if prev_alpha is not None:
                    bg = get_background(
                        bg_arg, (W, H), frame_idx, bg_video_cap)
                    out = composite(frame_bgr, prev_alpha, bg)
                    writer.write(out)
                else:
                    writer.write(frame_bgr)
                frame_idx += 1
                pbar.update(1)
                continue

            # Convert BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            # ── Run background removal ──────────────────────────────────
            result_pil = remove(
                pil_img,
                session=session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
                post_process_mask=post_process_mask,
            )

            # Extract alpha channel
            result_rgba = np.array(result_pil)        # (H, W, 4)
            alpha = result_rgba[:, :, 3]         # (H, W)

            # ── Refine mask ─────────────────────────────────────────────
            alpha = refine_mask(alpha, erode_px=erode_px,
                                feather_px=feather_px)
            prev_alpha = alpha

            # ── Composite ───────────────────────────────────────────────
            bg = get_background(bg_arg, (W, H), frame_idx, bg_video_cap)
            out = composite(frame_bgr, alpha, bg)

            writer.write(out)
            frame_idx += 1
            pbar.update(1)

    cap.release()
    writer.release()
    if bg_video_cap:
        bg_video_cap.release()

    print(f"\n✅ Done! Output saved to: {output_path}")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   File size: {size_mb:.1f} MB")


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Powerful video background removal using deep learning (rembg + U2-Net/BiRefNet)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("input",  help="Input video file path")
    parser.add_argument("output", help="Output video file path")

    parser.add_argument(
        "--model", "-m",
        default="u2net_human_seg",
        choices=list(MODELS.keys()),
        help="Model to use. Default: u2net_human_seg (best for people). "
             "Use birefnet-general for maximum accuracy on any subject.",
    )

    parser.add_argument(
        "--bg",
        default=None,
        help="Background: 'green' | 'white' | 'black' | path/to/image.jpg | path/to/video.mp4. "
             "Default: black.",
    )

    parser.add_argument(
        "--skip-frames", "-s",
        type=int, default=0,
        help="Skip N frames between processed frames (0=process all). "
             "E.g. --skip-frames 1 processes every 2nd frame (2x faster). "
             "Skipped frames reuse the previous mask.",
    )

    parser.add_argument(
        "--erode", type=int, default=0,
        help="Erode mask edges by N pixels to remove green/color fringe. Default: 0",
    )

    parser.add_argument(
        "--feather", type=int, default=2,
        help="Feather/blur mask edges by N pixels for smooth blending. Default: 2",
    )

    parser.add_argument(
        "--transparent", action="store_true",
        help="Output transparent video (use .webm extension for alpha support).",
    )

    parser.add_argument(
        "--no-alpha-matting", action="store_true",
        help="Disable alpha matting (faster but less accurate edges).",
    )

    parser.add_argument(
        "--fg-threshold", type=int, default=240,
        help="Alpha matting foreground threshold (0-255). Default: 240",
    )

    parser.add_argument(
        "--bg-threshold", type=int, default=10,
        help="Alpha matting background threshold (0-255). Default: 10",
    )

    parser.add_argument(
        "--list-models", action="store_true",
        help="List all available models and exit.",
    )

    args = parser.parse_args()

    if args.list_models:
        print("\nAvailable models:\n")
        for name, desc in MODELS.items():
            print(f"  {name:<28} {desc}")
        print()
        sys.exit(0)

    if not os.path.isfile(args.input):
        sys.exit(f"❌ Input file not found: {args.input}")

    remove_video_background(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        bg_arg=args.bg,
        skip_frames=args.skip_frames,
        erode_px=args.erode,
        feather_px=args.feather,
        transparent=args.transparent,
        alpha_matting=not args.no_alpha_matting,
        alpha_matting_foreground_threshold=args.fg_threshold,
        alpha_matting_background_threshold=args.bg_threshold,
    )


if __name__ == "__main__":
    main()
