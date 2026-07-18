# EfficientPolySeg — Real-Time Video Inference

Real-time polyp segmentation inference on colonoscopy video, using an
ONNX-exported EfficientPolySeg model (SwiftFormer backbone with efficient
edge- and context-aware decoder modules). This repository provides a
command-line inference tool and a GUI video browser/launcher for running
the model over recorded colonoscopy footage — it does not include training
code.

**Model code / training repository:** _link coming soon_

If you use this software in your research, please cite:

> Khaled Elkarazle, Valliappan Raman, Caslon Chua, Patrick Then, M Prabhavathy,
> **EfficientPolySeg: A fast and accurate network for complex polyp segmentation**,
> *Biomedical Signal Processing and Control*, Volume 112, Part A, 2026, 108449,
> ISSN 1746-8094.

```bibtex
@article{elkarazle2026efficientpolyseg,
  title   = {EfficientPolySeg: A fast and accurate network for complex polyp segmentation},
  author  = {Elkarazle, Khaled and Raman, Valliappan and Chua, Caslon and Then, Patrick and Prabhavathy, M},
  journal = {Biomedical Signal Processing and Control},
  volume  = {112},
  number  = {Part A},
  pages   = {108449},
  year    = {2026},
  issn    = {1746-8094}
}
```

---

## Overview

EfficientPolySeg combines:

- **SwiftFormer backbone** - a lightweight 4-stage transformer encoder,
  used via Hugging Face `transformers` to extract multi-scale features.
- **EPB (Efficient Processing Block)** - multi-dilation convolutions fused
  with global context, applied to skip connections.
- **EEM (Efficient Edge Module)** - depthwise-separable convolutions with
  spatial attention to sharpen ambiguous polyp boundaries.
- **AFR (Adaptive Feature Refinement)** - channel attention with a learnable
  residual gate to refine fused features before decoding.
- **A transposed-convolution decoder** with skip connections at every stage,
  followed by a final refinement block fused with the original input
  resolution.

The model outputs raw logits for binary (polyp vs. background) semantic
segmentation; a sigmoid is applied downstream. See the linked model
repository above for architecture and training details.

This repository covers **inference only**: loading a pre-exported ONNX
checkpoint and running it over video, either from the command line or a
simple GUI browser for folders of recordings.

---

## Repository structure

```
.
├── infer.py              # real-time video inference (CLI) — ONNX Runtime + OpenCV
├── main.py        # Tkinter video browser/launcher — scans a folder tree and runs infer.py per video
├── LICENSE.md
└── README.md
```

You will also need a trained model exported to ONNX (e.g.
`efficientpolyseg_best.onnx`) — see the model repository linked above for
training and export instructions.

---

## Installation

```bash
pip install onnxruntime opencv-python numpy
```

For GPU inference, install `onnxruntime-gpu` instead of `onnxruntime`.

---

## Real-time video inference

### Command line

```bash
python infer.py --source path/to/video.mp4
python infer.py --source path/to/video.mp4 --save output.mp4
python infer.py --source path/to/video.mp4 --save output.mp4 --no-display
```

Frames are processed strictly in source order — no frames are ever dropped
or skipped, so saved output always plays back at the correct, undistorted
speed regardless of inference throughput.

**Playback controls** (while a display window is open):

| Key                | Action                                            |
|--------------------|----------------------------------------------------|
| `Q` / `Esc`        | Quit                                               |
| `Space`            | Pause / resume                                     |
| Drag seek bar      | Jump to any point in the video                     |
| `→` / `D`          | Seek forward 5s                                    |
| `←` / `A`          | Seek backward 5s                                   |
| `.` (period)       | Step one frame forward (pauses if playing)         |
| `,` (comma)        | Step one frame backward (pauses if playing)        |
| `N`                | Step one frame forward (legacy alias for `.`)      |
| `S`                | Cycle overlay: mask contour → probability heatmap → off |
| `+` / `-`          | Adjust overlay opacity                             |

Useful flags:

| Flag              | Description                                             |
|-------------------|-----------------------------------------------------------|
| `--onnx`          | Path to the exported ONNX model                          |
| `--threshold`     | Sigmoid threshold for the binary mask (default `0.5`)    |
| `--smoothing`     | Temporal EMA smoothing over the probability map (`0`–`1`) |
| `--frame-skip`    | Run inference every Nth frame (still displays every frame) |
| `--speed`         | Live-preview pacing multiplier (does not affect saved file timing) |
| `--seek-seconds`  | How far arrow-key seeking jumps, in seconds               |
| `--native`        | Use the source's native resolution instead of a fixed display size |

### GUI launcher

For browsing and running inference on a folder of colonoscopy recordings
without touching the command line:

```bash
python gui_launcher.py
python gui_launcher.py --onnx path/to/model.onnx
```

Point it at any root folder — it recursively finds every video file
(`.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`) at any depth, with no required
folder-naming convention. Double-click any entry to launch playback with
the same controls described above.

---

## Notes on intended use

This software is provided for **research purposes only** and is not a
certified medical device. Predictions should not be used for clinical
decision-making without appropriate validation and regulatory clearance.

---

## License

This project is licensed under **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**
(Attribution-NonCommercial 4.0 International) — see [LICENSE.md](LICENSE.md)
for details.

You are free to share and adapt this code for **non-commercial** purposes,
with attribution. Commercial use is not permitted under this license; if
you're interested in commercial use, please contact the authors.
