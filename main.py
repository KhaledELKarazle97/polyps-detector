"""
Polyp Segmentation: Video Browser / Launcher
Double-click (or select + "Run Inference") on any entry in the list to launch
the infer.run() loop on that file directly — no CLI needed.
If you would like to use this software, please cite:

Khaled Elkarazle, Valliappan Raman, Caslon Chua, 
Patrick Then, M Prabhavathy, 
EfficientPolySeg: A fast and accurate network for complex polyp segmentation, Biomedical Signal Processing and Control, 
Volume 112, Part A, 2026, 108449, ISSN 1746-8094
"""

import argparse
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import infer as iv


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def find_videos(root_path: str):
    """Finds video files under root_path with no assumption about folder
    naming or structure:
      - Videos directly inside root_path are included.
      - Any subfolders (at any depth) are walked recursively too.
    In other words: just look everywhere under the given path for anything
    with a video extension."""
    found = []
    if not root_path or not os.path.isdir(root_path):
        return found

    for dirpath, dirnames, filenames in os.walk(root_path):
        for f in filenames:
            if f.lower().endswith(VIDEO_EXTS):
                found.append(os.path.join(dirpath, f))

    return sorted(found)


class LauncherApp:
    def __init__(self, root, onnx_path, img_size, threshold, smoothing, speed,
                 display_w, display_h):
        self.root = root
        self.onnx_path = onnx_path
        self.img_size = img_size
        self.threshold = threshold
        self.smoothing = smoothing
        self.speed = speed
        self.display_size = (display_w, display_h)

        self.videos = []            
        self.sess = None            
        self.worker_thread = None
        self.stop_event = None

        root.title("Polyps Detector")
        root.geometry("760x480")
        root.minsize(560, 360)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Root folder:").pack(side="left")
        self.path_var = tk.StringVar()
        entry = ttk.Entry(top, textvariable=self.path_var)
        entry.pack(side="left", fill="x", expand=True, padx=6)
        entry.bind("<Return>", lambda e: self.scan())

        ttk.Button(top, text="Browse…", command=self.browse).pack(side="left")
        ttk.Button(top, text="Scan", command=self.scan).pack(side="left", padx=(6, 0))

        # list + scrollbar
        list_frame = ttk.Frame(self.root)
        list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        self.listbox = tk.Listbox(list_frame, activestyle="dotbox")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<Double-Button-1>", lambda e: self.run_selected())

        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        bottom = ttk.Frame(self.root)
        bottom.pack(fill="x", padx=8, pady=(0, 8))

        self.run_btn = ttk.Button(bottom, text="Run Inference", command=self.run_selected)
        self.run_btn.pack(side="left")

        self.stop_btn = ttk.Button(bottom, text="Stop", command=self.stop_current, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        self.status_var = tk.StringVar(value="No folder scanned yet.")
        ttk.Label(bottom, textvariable=self.status_var).pack(side="left", padx=12)

    def browse(self):
        chosen = filedialog.askdirectory(title="Select root folder")
        if chosen:
            self.path_var.set(chosen)
            self.scan()

    def scan(self):
        root_path = self.path_var.get().strip()
        if not root_path:
            messagebox.showwarning("No path", "Enter or browse to a root folder first.")
            return
        if not os.path.isdir(root_path):
            messagebox.showerror("Invalid path", f"Not a folder:\n{root_path}")
            return

        self.videos = find_videos(root_path)
        self.listbox.delete(0, tk.END)

        if not self.videos:
            self.status_var.set("No video files found under that path.")
            return

        for path in self.videos:
            rel = os.path.relpath(path, root_path)
            self.listbox.insert(tk.END, rel)

        self.status_var.set(f"Found {len(self.videos)} video(s).")

    def run_selected(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Already running",
                                 "A video is already playing. Stop it first.")
            return

        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Select a video from the list first.")
            return

        video_path = self.videos[sel[0]]
        self._launch(video_path)

    def _launch(self, video_path: str):
        self.stop_event = threading.Event()
        self.status_var.set(f"Running: {os.path.basename(video_path)} …")
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        self.worker_thread = threading.Thread(
            target=self._run_inference_thread,
            args=(video_path, self.stop_event),
            daemon=True,
        )
        self.worker_thread.start()
        self.root.after(300, self._poll_worker)

    def _run_inference_thread(self, video_path, stop_event):
        try:
            if self.sess is None:
                self.sess = iv.build_session(self.onnx_path)

            iv.run(
                source=video_path,
                save_path=None,
                onnx_path=self.onnx_path,
                img_size=self.img_size,
                threshold=self.threshold,
                show=True,
                display_size=self.display_size,
                frame_skip=iv.FRAME_SKIP,
                temporal_smoothing=self.smoothing,
                speed=self.speed,
                stop_event=stop_event,
                window_name=os.path.basename(video_path),
                sess=self.sess,
            )
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Inference error", msg))

    def _poll_worker(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(300, self._poll_worker)
        else:
            self.run_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.status_var.set(f"Found {len(self.videos)} video(s). Idle.")

    def stop_current(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.status_var.set("Stopping…")


def main():
    parser = argparse.ArgumentParser(description="Video browser/launcher for polyp segmentation")
    parser.add_argument("--onnx", default=iv.ONNX_PATH)
    parser.add_argument("--img-size", type=int, default=iv.IMG_SIZE)
    parser.add_argument("--threshold", type=float, default=iv.THRESHOLD)
    parser.add_argument("--smoothing", type=float, default=iv.TEMPORAL_SMOOTHING)
    parser.add_argument("--speed", type=float, default=iv.PLAYBACK_SPEED)
    parser.add_argument("--display-w", type=int, default=960)
    parser.add_argument("--display-h", type=int, default=540)
    args = parser.parse_args()

    root = tk.Tk()
    LauncherApp(
        root,
        onnx_path=args.onnx,
        img_size=args.img_size,
        threshold=args.threshold,
        smoothing=args.smoothing,
        speed=args.speed,
        display_w=args.display_w,
        display_h=args.display_h,
    )
    root.mainloop()


if __name__ == "__main__":
    main()