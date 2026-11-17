#!/usr/bin/env python3
"""Real-time fall-detection demo.

    python scripts/run_realtime.py --source 0                 # webcam
    python scripts/run_realtime.py --source path/to/clip.mp4  # file
    python scripts/run_realtime.py --source 0 --no-privacy    # debug view

Privacy mode (default ON): the skeleton overlay is drawn on a BLANK
canvas — raw pixels are decoded into memory for pose inference and then
discarded; they are never displayed, buffered, or written. There is no
recording code path in this repository at all. Alerts are logged as
JSON events (timestamp, track_id, confidence) only.

Performance path for >=15 FPS on an RTX 3050 @ 640 px:
  * FP16 inference (pose.half=true), single model, no test-time aug.
  * Featurization is incremental: per-track deques of the last 30
    frames; classification runs every `stride` frames, not every frame.
  * All feature math is numpy on 30x55 arrays — microseconds.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.sequence import FEATURE_DIM, sequence_features  # noqa: E402
from src.pose.extractor import PoseExtractor, iter_video_frames   # noqa: E402
from src.pose.tracker import IoUCentroidTracker                   # noqa: E402
from src.realtime.alerts import AlertManager                      # noqa: E402

# COCO skeleton edges for drawing
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def load_classifier(ckpt_path: Path, device_str: str):
    import torch
    from src.model.nets import build_model
    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=dev)
    model = build_model(ckpt["config"]).to(dev).eval()
    model.load_state_dict(ckpt["state_dict"])
    return model, dev


def draw_scene(canvas, tracks, probs, alerts_active, fps_now, cfg, cv2):
    h, w = canvas.shape[:2]
    for trk in tracks:
        kp = trk.keypoints
        prob = probs.get(trk.track_id, 0.0)
        color = (0, 0, 255) if prob >= cfg["alerts"]["threshold"] else (0, 255, 0)
        for a, b in EDGES:
            if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
                cv2.line(canvas, tuple(kp[a, :2].astype(int)),
                         tuple(kp[b, :2].astype(int)), color, 2)
        for j in range(17):
            if kp[j, 2] > 0.3:
                cv2.circle(canvas, tuple(kp[j, :2].astype(int)), 3, color, -1)
        x1, y1 = trk.bbox[:2].astype(int)
        cv2.putText(canvas, f"id{trk.track_id} p={prob:.2f}",
                    (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)

    if alerts_active:
        cv2.rectangle(canvas, (0, 0), (w, 46), (0, 0, 200), -1)
        cv2.putText(canvas, "!! FALL DETECTED !!", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.putText(canvas, f"{fps_now:.1f} FPS", (w - 110, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)


def draw_sparkline(canvas, history, cfg, cv2):
    """Per-track fall-probability sparkline, bottom-left."""
    h, _ = canvas.shape[:2]
    W, H, x0 = 220, 60, 12
    y0 = h - H - 12
    cv2.rectangle(canvas, (x0, y0), (x0 + W, y0 + H), (60, 60, 60), 1)
    thr_y = int(y0 + H * (1 - cfg["alerts"]["threshold"]))
    cv2.line(canvas, (x0, thr_y), (x0 + W, thr_y), (80, 80, 160), 1)
    if len(history) >= 2:
        n = cfg["realtime"]["sparkline_len"]
        pts = history[-n:]
        xs = np.linspace(x0, x0 + W, len(pts)).astype(int)
        ys = (y0 + H * (1 - np.clip(pts, 0, 1))).astype(int)
        for i in range(1, len(pts)):
            cv2.line(canvas, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]),
                     (0, 255, 255), 1)
    cv2.putText(canvas, "p(fall)", (x0 + 4, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)


def main() -> None:
    import cv2
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0",
                    help="webcam index or video path")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best_gru.pt")
    ap.add_argument("--no-privacy", action="store_true",
                    help="debug: draw overlay on live frames (still no recording)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (smoke / CI)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    privacy = cfg["realtime"]["privacy_mode"] and not args.no_privacy
    source = int(args.source) if args.source.isdigit() else args.source

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint {ckpt_path}. Train first, e.g. "
            f"`python scripts/train.py --config configs/smoke.yaml --model gru`")

    pose = PoseExtractor(**cfg["pose"])
    tracker = IoUCentroidTracker(**cfg["tracker"])
    model, dev = load_classifier(ckpt_path, cfg["pose"]["device"])
    alerts = AlertManager(cfg["alerts"]["threshold"],
                          cfg["alerts"]["k_consecutive"],
                          cfg["alerts"]["cooldown_s"],
                          cfg["alerts"]["log_path"])

    W, S = cfg["window"]["size"], cfg["window"]["stride"]
    fps_nom = cfg["features"]["velocity_fps"]
    buffers: dict[int, deque] = {}          # track_id -> deque[(17,3)]
    probs: dict[int, float] = {}
    banner_until = 0.0
    t_prev, fps_ema = time.time(), 0.0
    frame_count = 0

    for _, frame in iter_video_frames(source, max_frames=args.max_frames):
        boxes, kpts = pose(frame)
        tracks = tracker.update(boxes, kpts)
        frame_count += 1

        # Remove per-track state only after the tracker has expired that ID.
        # A temporarily missed detection remains in the tracker and contributes
        # an all-zero pose for this frame, preserving the time axis.
        live_track_ids = set(tracker.tracks.keys())
        for stale_id in list(buffers):
            if stale_id not in live_track_ids:
                buffers.pop(stale_id, None)
                probs.pop(stale_id, None)
        alerts.retain_tracks(live_track_ids)

        for trk in tracks:
            buf = buffers.setdefault(trk.track_id, deque(maxlen=W))
            frame_keypoints = (trk.keypoints.copy() if trk.updated_this_frame
                               else np.zeros_like(trk.keypoints))
            buf.append(frame_keypoints)
            # classify every S frames once the buffer is full
            if len(buf) == W and frame_count % S == 0:
                feats = sequence_features(
                    np.stack(buf), fps=fps_nom,
                    min_conf=float(cfg["features"].get("min_kpt_conf", 0.30)),
                )
                assert feats.shape == (W, FEATURE_DIM)
                with torch.no_grad():
                    x = torch.from_numpy(feats[None]).to(dev)
                    p = float(torch.sigmoid(model(x)).item())
                probs[trk.track_id] = p
                ev = alerts.update(trk.track_id, p)
                if ev is not None:
                    banner_until = time.time() + 3.0
                    print(f"[ALERT] {ev.to_json()}")

        # ---- render: blank canvas in privacy mode (frame never shown) ---
        canvas = (np.zeros_like(frame) if privacy else frame.copy())
        now = time.time()
        inst = 1.0 / max(now - t_prev, 1e-6)
        fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
        t_prev = now
        observed_tracks = [track for track in tracks if track.updated_this_frame]
        draw_scene(canvas, observed_tracks, probs, now < banner_until,
                   fps_ema, cfg, cv2)
        if observed_tracks:
            draw_sparkline(canvas,
                           alerts.history(observed_tracks[0].track_id,
                                          cfg["realtime"]["sparkline_len"]),
                           cfg, cv2)
        # Headless smoke: skip GUI when DISPLAY-less or --max-frames set without UI
        if args.max_frames is None:
            cv2.imshow("fall-detection (privacy mode)" if privacy
                       else "fall-detection (DEBUG)", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        elif frame_count % 30 == 0:
            print(f"[smoke] frame {frame_count} tracks={len(tracks)} "
                  f"fps={fps_ema:.1f}")

        # `frame` and `canvas` go out of scope here — nothing persisted.

    alerts.close()
    if args.max_frames is None:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
