#!/usr/bin/env python3
"""Build a tiny synthetic skeleton cache for smoke-training without URFD videos.

Produces:
  data/labels.csv
  cache/skeletons/*.npz   (keypoints, frame_idx, fps — no pixels)

Usage:
    python scripts/make_synthetic_cache.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_normalize import synthetic_standing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache" / "skeletons"
LABELS = ROOT / "data" / "labels.csv"
FPS = 30.0


def _rotate_about_hip(k: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate all joints about hip center (image y down)."""
    out = k.copy()
    hip_c = out[[11, 12], :2].mean(0)  # L_HIP=11, R_HIP=12 in COCO
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    # image coords: +angle rotates standing -> lying (shoulder toward +x)
    rot = np.array([[c, -s], [s, c]], np.float32)
    xy = out[:, :2] - hip_c
    out[:, :2] = xy @ rot.T + hip_c
    return out


def make_adl(n_frames: int = 120, seed: int = 0) -> np.ndarray:
    """Upright walking-ish sway — no fall kinematics."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        cx = 320.0 + 8.0 * math.sin(2 * math.pi * t / 40)
        hip_y = 300.0 + 3.0 * math.sin(2 * math.pi * t / 20)
        k = synthetic_standing(cx=cx, hip_y=hip_y, torso=100.0)
        k[:, :2] += rng.normal(0, 0.4, size=k[:, :2].shape).astype(np.float32)
        frames.append(k)
    return np.stack(frames).astype(np.float32)


def make_fall(n_frames: int = 150, fall_start: int = 60, fall_end: int = 100,
              seed: int = 0) -> np.ndarray:
    """Stand → rapid hip drop + trunk tip → lying."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_frames):
        if t < fall_start:
            progress = 0.0
            hip_y = 280.0
        elif t <= fall_end:
            progress = (t - fall_start) / max(fall_end - fall_start, 1)
            hip_y = 280.0 + progress * 160.0
        else:
            progress = 1.0
            hip_y = 440.0
        angle = progress * (math.pi / 2)
        k = synthetic_standing(cx=320.0, hip_y=hip_y, torso=100.0)
        k = _rotate_about_hip(k, angle)
        k[:, :2] += rng.normal(0, 0.5, size=k[:, :2].shape).astype(np.float32)
        frames.append(k)
    return np.stack(frames).astype(np.float32)


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    LABELS.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    # 8 groups so default 70/15/15 split has train+val+test nonempty
    for i in range(1, 9):
        gid = f"subject{i:02d}"
        fall_id = f"fall-{i:02d}-cam0"
        adl_id = f"adl-{i:02d}-cam0"
        fall_start, fall_end = 60, 100

        fall = make_fall(seed=100 + i)
        adl = make_adl(seed=200 + i)
        for clip_id, seq in ((fall_id, fall), (adl_id, adl)):
            np.savez_compressed(
                CACHE / f"{clip_id}.npz",
                keypoints=seq,
                frame_idx=np.arange(len(seq), dtype=np.int32),
                fps=np.float32(FPS),
            )
        rows.append((fall_id, fall_start, fall_end, gid))
        rows.append((adl_id, -1, -1, gid))

    with open(LABELS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "fall_start_frame", "fall_end_frame", "group"])
        w.writerows(rows)

    print(f"[synthetic] wrote {len(rows)} clips -> {CACHE}")
    print(f"[synthetic] labels -> {LABELS}")


if __name__ == "__main__":
    main()
