"""
Usage:
    python infer.py --source path/to/video.mp4
    python infer.py --source video.mp4 --save output.mp4
    python infer.py --source video.mp4 --save out.mp4 --no-display

Controls (while a display window is open):
    Q / ESC        — quit
    SPACE          — pause / resume
    Drag seek bar  — jump to any point in the video
    Right / D      — seek forward 5s
    Left  / A      — seek backward 5s
    .  (period)    — step one frame forward  (pauses if playing)
    ,  (comma)     — step one frame backward (pauses if playing)
    N              — step one frame forward (legacy alias for '.')
    S              — cycle overlay: mask contour -> probability heatmap -> off
    +/-            — adjust overlay opacity

If you would like to use this software, please cite:

Khaled Elkarazle, Valliappan Raman, Caslon Chua, 
Patrick Then, M Prabhavathy, 
EfficientPolySeg: A fast and accurate network for complex polyp segmentation, Biomedical Signal Processing and Control, 
Volume 112, Part A, 2026, 108449, ISSN 1746-8094

"""

import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort


ONNX_PATH    = "efficientpolyseg_best.onnx"
IMG_SIZE     = 224
THRESHOLD    = 0.5
ALPHA        = 0.45
MASK_COLOR   = (0, 255, 100)  # BGR

FRAME_SKIP   = 1              # run inference every Nth frame; still displays/writes every frame
TEMPORAL_SMOOTHING = 0.5      # EMA over probability map, 0 = disabled
PLAYBACK_SPEED = 1.0          # live-preview pacing multiplier; does not affect saved file timing
SEEK_SECONDS = 5.0            # how far Left/Right arrow jumps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

SEEK_TRACKBAR_NAME = "Seek"

# Common arrow-key codes across platforms/OpenCV builds (waitKeyEx). We check
# a handful since these are not fully standardized across backends.
LEFT_ARROW_CODES  = {2424832, 65361, 81, 63234}
RIGHT_ARROW_CODES = {2555904, 65363, 83, 63235}


def build_session(onnx_path: str) -> ort.InferenceSession:
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"ONNX loaded | provider: {sess.get_providers()[0]}")
    return sess


def center_crop_square(frame):
    """Crops the largest centered square instead of squashing to square,
    to avoid distorting 16:9 video."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    return frame[y0:y0 + side, x0:x0 + side], (x0, y0, side, side)


def preprocess(frame_bgr, img_size):
    cropped, box = center_crop_square(frame_bgr)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    normed = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return normed.transpose(2, 0, 1)[None], box


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess_prob(logits, box, out_h, out_w):
    prob_small = sigmoid(logits[0, 0])
    x0, y0, side, _ = box
    prob_crop = cv2.resize(prob_small, (side, side), interpolation=cv2.INTER_LINEAR)
    full_prob = np.zeros((out_h, out_w), dtype=np.float32)
    full_prob[y0:y0 + side, x0:x0 + side] = prob_crop
    return full_prob


def overlay_mask(frame, mask, color, alpha):
    out = frame.copy()
    coloured = np.zeros_like(frame)
    coloured[mask > 0] = color
    out = cv2.addWeighted(out, 1.0, coloured, alpha, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, 2)
    return out


def overlay_heatmap(frame, prob, alpha, colormap=cv2.COLORMAP_VIRIDIS):
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_u8, colormap)
    fade = prob[..., None].astype(np.float32)
    blended = frame.astype(np.float32) * (1 - alpha * fade) + heat.astype(np.float32) * (alpha * fade)
    return np.clip(blended, 0, 255).astype(np.uint8)


def draw_hud(frame, overlay_mode, polyp_detected, paused, cur_sec=None, total_sec=None):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (20, 20, 20), -1)
    if paused:
        status, color = "PAUSED", (0, 210, 255)
    elif polyp_detected:
        status, color = "POLYP DETECTED", (0, 60, 255)
    else:
        status, color = "clear", (80, 200, 80)
    cv2.putText(frame, status, (w // 2 - 90, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"overlay: {overlay_mode}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if cur_sec is not None and total_sec is not None and total_sec > 0:
        def fmt(s):
            m, s = divmod(int(s), 60)
            return f"{m:02d}:{s:02d}"
        time_str = f"{fmt(cur_sec)} / {fmt(total_sec)}"
        (tw, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, time_str, (w - tw - 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    if paused:
        cv2.putText(frame, "SPACE resume | ,/. step | </> or A/D seek 5s | Q quit", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 255), 1)
    return frame


def run(source, save_path, onnx_path, img_size, threshold, show,
        display_size, frame_skip=FRAME_SKIP, temporal_smoothing=TEMPORAL_SMOOTHING,
        speed=1.0, stop_event=None, window_name="Polyp Segmentation", sess=None,
        alpha=ALPHA, seek_seconds=SEEK_SECONDS):

    if sess is None:
        sess = build_session(onnx_path)
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seekable = show and total_frames > 0 and not isinstance(source, int)

    if display_size:
        disp_w, disp_h = display_size
    else:
        disp_w, disp_h = native_w, native_h

    print(f"Source: {native_w}x{native_h} @ {src_fps:.1f} fps -> output {disp_w}x{disp_h}  "
          f"skip={frame_skip}  frames={total_frames if total_frames > 0 else 'unknown'}")

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, src_fps, (disp_w, disp_h))
        print(f"Saving to: {save_path}")

    target_dt = 1.0 / max(src_fps * speed, 1e-6)
    seek_frames = max(1, int(round(seek_seconds * src_fps)))

    ema_prob = np.zeros((disp_h, disp_w), dtype=np.float32)
    last_mask = np.zeros((disp_h, disp_w), dtype=np.uint8)
    overlay_mode = "mask"  # "mask" | "heatmap" | "off"
    paused = False
    frame_idx = 0
    frame_count = 0

    seek_request = {"frame": None}
    suppress_trackbar_cb = {"flag": False}

    def on_trackbar(val):
        if suppress_trackbar_cb["flag"]:
            return
        seek_request["frame"] = val

    if seekable:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(SEEK_TRACKBAR_NAME, window_name, 0, max(total_frames - 1, 1), on_trackbar)

    def do_seek(target_frame):
        nonlocal frame_idx, ema_prob, last_mask
        target_frame = int(np.clip(target_frame, 0, max(total_frames - 1, 0) if total_frames > 0 else target_frame))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        frame_idx = target_frame
        ema_prob = np.zeros_like(ema_prob)
        last_mask = np.zeros_like(last_mask)

    print("Running… Q/ESC quit | SPACE pause | drag seek bar | </> or A/D seek 5s | ,/. step frame | S overlay | +/- opacity")

    while True:
        if stop_event is not None and stop_event.is_set():
            print("Stop requested — closing video.")
            break

        loop_t0 = time.perf_counter()
        step_forward = False
        step_backward = False

        # ── handle a pending seek-bar drag before reading the next frame ──
        if seek_request["frame"] is not None:
            do_seek(seek_request["frame"])
            seek_request["frame"] = None

        if paused:
            if show:
                key = cv2.waitKeyEx(30)
                key_low = key & 0xFF if key != -1 else -1
                if key_low in (ord("q"), 27):
                    break
                elif key_low == ord(" "):
                    paused = False
                elif key_low in (ord("n"), ord(".")):
                    step_forward = True
                elif key_low == ord(","):
                    step_backward = True
                elif key_low == ord("s"):
                    overlay_mode = {"mask": "heatmap", "heatmap": "off", "off": "mask"}[overlay_mode]
                elif key_low in (ord("+"), ord("=")):
                    alpha = min(1.0, alpha + 0.05)
                elif key_low == ord("-"):
                    alpha = max(0.0, alpha - 0.05)
                elif key in RIGHT_ARROW_CODES or key_low == ord("d"):
                    do_seek(frame_idx + seek_frames)
                elif key in LEFT_ARROW_CODES or key_low == ord("a"):
                    do_seek(frame_idx - seek_frames)
            if not (step_forward or step_backward):
                continue
            if step_backward:
                do_seek(frame_idx - 2)

        ret, frame = cap.read()
        if not ret:
            break 

        if display_size:
            frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

        if frame_idx % frame_skip == 0:
            inp, box = preprocess(frame, img_size)
            logits = sess.run(None, {input_name: inp})[0]
            frame_prob = postprocess_prob(logits, box, disp_h, disp_w)
            ema_prob = (temporal_smoothing * ema_prob
                        + (1 - temporal_smoothing) * frame_prob)
            last_mask = (ema_prob > threshold).astype(np.uint8) * 255
        frame_idx += 1

        polyp_detected = bool(last_mask.any())

        if overlay_mode == "mask":
            out = overlay_mask(frame, last_mask, MASK_COLOR, alpha)
        elif overlay_mode == "heatmap":
            out = overlay_heatmap(frame, ema_prob, alpha)
        else:
            out = frame.copy()

        cur_sec = frame_idx / src_fps if src_fps > 0 else None
        total_sec = total_frames / src_fps if (src_fps > 0 and total_frames > 0) else None
        out = draw_hud(out, overlay_mode, polyp_detected, paused=paused,
                        cur_sec=cur_sec, total_sec=total_sec)

        if writer:
            writer.write(out)  # always one write per source frame -> correct saved timing

        if show:
            cv2.imshow(window_name, out)
            if seekable:
                suppress_trackbar_cb["flag"] = True
                cv2.setTrackbarPos(SEEK_TRACKBAR_NAME, window_name,
                                    int(np.clip(frame_idx, 0, max(total_frames - 1, 0))))
                suppress_trackbar_cb["flag"] = False

            if paused:
                frame_count += 1
                continue

            elapsed = time.perf_counter() - loop_t0
            wait_ms = max(1, int((target_dt - elapsed) * 1000))
            key = cv2.waitKeyEx(wait_ms)
            key_low = key & 0xFF if key != -1 else -1
            if key_low in (ord("q"), 27):
                break
            elif key_low == ord(" "):
                paused = True
            elif key_low == ord("s"):
                overlay_mode = {"mask": "heatmap", "heatmap": "off", "off": "mask"}[overlay_mode]
            elif key_low in (ord("+"), ord("=")):
                alpha = min(1.0, alpha + 0.05)
            elif key_low == ord("-"):
                alpha = max(0.0, alpha - 0.05)
            elif key in RIGHT_ARROW_CODES or key_low == ord("d"):
                do_seek(frame_idx + seek_frames)
            elif key in LEFT_ARROW_CODES or key_low == ord("a"):
                do_seek(frame_idx - seek_frames)
            elif key_low in (ord("n"), ord(".")):
                paused = True  # single-step implicitly pauses, like VLC
            elif key_low == ord(","):
                paused = True
                do_seek(frame_idx - 2)

        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time polyp segmentation")
    parser.add_argument("--source", required=True)
    parser.add_argument("--save", default=None)
    parser.add_argument("--onnx", default=ONNX_PATH)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--display-w", type=int, default=960)
    parser.add_argument("--display-h", type=int, default=540)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--smoothing", type=float, default=TEMPORAL_SMOOTHING)
    parser.add_argument("--speed", type=float, default=1.0,
                         help="live-preview pacing multiplier; does not affect saved file timing")
    parser.add_argument("--seek-seconds", type=float, default=SEEK_SECONDS,
                         help="how far Left/Right arrow (or A/D) jumps, in seconds")
    args = parser.parse_args()

    if args.no_display and not args.save:
        parser.error("--no-display requires --save")

    display_size = None if args.native else (args.display_w, args.display_h)

    run(
        source=args.source, save_path=args.save, onnx_path=args.onnx,
        img_size=args.img_size, threshold=args.threshold, show=not args.no_display,
        display_size=display_size, frame_skip=args.frame_skip,
        temporal_smoothing=args.smoothing, speed=args.speed,
        seek_seconds=args.seek_seconds,
    )