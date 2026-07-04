<div align="center">

# BG_Remover 🎬

**AI-Powered Video Background Removal Tool**

Remove backgrounds from videos using deep learning — with support for custom backgrounds, alpha matting, and mask refinement.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![rembg](https://img.shields.io/badge/built%20with-rembg-green)](https://github.com/danielgatis/rembg)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red)](https://opencv.org/)

</div>

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Background Options](#background-options)
  - [Model Selection](#model-selection)
  - [Mask Refinement](#mask-refinement)
  - [Performance Tuning](#performance-tuning)
- [Model Comparison](#model-comparison)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Deep Learning Segmentation** — Uses `rembg` with U²-Net, BiRefNet, and IS-Net models
- **Multiple Backgrounds** — Solid colors, images, or videos as replacement backgrounds
- **Alpha Matting** — Accurate edge detection for hair and fine details
- **Mask Refinement** — Erode and feather controls for clean edges
- **Skip Frames** — Process every Nth frame and interpolate for faster output
- **Transparent Output** — Export with alpha channel (WebM)

---

## Requirements

- Python 3.9 or higher
- FFmpeg (optional, for transparent WebM output)

---

## Installation

### Basic (CPU)

```bash
pip install rembg opencv-python-headless pillow tqdm onnxruntime
```

### GPU (CUDA)

```bash
pip install rembg[gpu] opencv-python-headless pillow tqdm onnxruntime-gpu
```

> **Note:** The first run downloads the selected model weights (~170 MB). An internet connection is required.

---

## Usage

```
python remove_video_bg.py <input_video> <output_video> [options]
```

### Basic Usage

Remove background and replace with black (default):

```bash
python remove_video_bg.py input.mp4 output.mp4
```

### Background Options

```bash
# Green screen background
python remove_video_bg.py input.mp4 output.mp4 --bg green

# Solid colors
python remove_video_bg.py input.mp4 output.mp4 --bg white
python remove_video_bg.py input.mp4 output.mp4 --bg black

# Custom background image
python remove_video_bg.py input.mp4 output.mp4 --bg background.jpg

# Custom background video
python remove_video_bg.py input.mp4 output.mp4 --bg bg.mp4

# Transparent output (WebM with alpha channel)
python remove_video_bg.py input.mp4 output.webm --transparent
```

### Model Selection

```bash
# Best accuracy for any subject (BiRefNet)
python remove_video_bg.py input.mp4 output.mp4 --model birefnet-general --feather 2

# Best for portraits/people (U²-Net human segmentation)
python remove_video_bg.py input.mp4 output.mp4 --model u2net_human_seg

# Best balance of speed and accuracy
python remove_video_bg.py input.mp4 output.mp4 --model isnet-general-use

# List all available models
python remove_video_bg.py input.mp4 output.mp4 --list-models
```

### Mask Refinement

```bash
# Remove fringe artifacts by eroding mask edges
python remove_video_bg.py input.mp4 output.mp4 --erode 2

# Smooth edges with feathering
python remove_video_bg.py input.mp4 output.mp4 --feather 3

# Combine both for clean, natural edges
python remove_video_bg.py input.mp4 output.mp4 --erode 1 --feather 2
```

### Performance Tuning

```bash
# Fastest processing (skip frames + no alpha matting)
python remove_video_bg.py input.mp4 output.mp4 --model u2net_human_seg --skip-frames 1 --no-alpha-matting

# Process every 3rd frame (3x faster)
python remove_video_bg.py input.mp4 output.mp4 --skip-frames 2

# Adjust alpha matting thresholds for difficult edges
python remove_video_bg.py input.mp4 output.mp4 --fg-threshold 250 --bg-threshold 5
```

---

## Model Comparison

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| `birefnet-general` | Slow | ★★★★★ | Maximum accuracy on any subject |
| `birefnet-portrait` | Slow | ★★★★★ | Face and bust shots |
| `isnet-general-use` | Medium | ★★★★☆ | Balanced speed/accuracy |
| `u2net_human_seg` | Fast | ★★★★☆ | People and portraits (default) |
| `u2net` | Fast | ★★★☆☆ | General purpose |
| `silueta` | Fastest | ★★★☆☆ | People silhouettes |

---

## CLI Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `input` | Input video file path | _(required)_ |
| `output` | Output video file path | _(required)_ |
| `--model`, `-m` | Model to use | `u2net_human_seg` |
| `--bg` | Background: `green`, `white`, `black`, or path to image/video | `black` |
| `--skip-frames`, `-s` | Skip N frames between processed frames | `0` |
| `--erode` | Erode mask edges by N pixels | `0` |
| `--feather` | Feather/blur mask edges by N pixels | `2` |
| `--transparent` | Output transparent video (WebM) | `False` |
| `--no-alpha-matting` | Disable alpha matting (faster, less accurate) | `False` |
| `--fg-threshold` | Alpha matting foreground threshold (0–255) | `240` |
| `--bg-threshold` | Alpha matting background threshold (0–255) | `10` |
| `--list-models` | List available models and exit | `False` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Model download fails** | Check internet connection. Manual download: place weights in `~/.u2net/` or `~/.rembg/` |
| **CUDA out of memory** | Use a smaller model (`u2net_human_seg`), reduce video resolution, or add `--skip-frames` |
| **Choppy output with `--skip-frames`** | Lower the skip value. Skipped frames reuse the previous mask, so high skips cause visible interpolation |
| **Green fringe on edges** | Increase `--erode` (e.g. `--erode 2`) and `--feather` (e.g. `--feather 3`) |
| **Transparent WebM not working** | The output contains an alpha plane but requires FFmpeg for full support: `ffmpeg -i output.webm -c:v libvpx-vp9 -pix_fmt yuva420p final.webm` |
| **OpenCV errors on Windows** | Install `opencv-python` instead of `opencv-python-headless` if you need GUI features |

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.
