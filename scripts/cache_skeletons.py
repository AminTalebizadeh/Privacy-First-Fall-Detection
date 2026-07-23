#!/usr/bin/env python3
"""Offline pose extraction: dataset videos -> .npz skeleton cache.

This step *is* the privacy pipeline demonstration: after it runs, every
downstream stage (features, training, evaluation) touches only the
cache — pixels never leave this process and are never written anywhere.

Each clip produces cache/skeletons/<clip_id>.npz with:
    keypoints : (T,17,3) float32 — primary-track [x,y,conf] per frame
    frame_idx : (T,)     int32   — original frame indices (gaps = missed
                                    detections; kept for event-time eval)
    fps       : ()       float32 — clip frame rate
No image arrays. Ever.

URFD tip: download the *camera 0* fall/adl sequences as videos (or
convert the PNG sequences with ffmpeg), name them e.g. fall-01.mp4 /
adl-01.mp4, and generate labels.csv from the official annotation file
(see README).

Usage:
    python scripts/cache_skeletons.py --videos data/videos --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pose.extractor import PoseExtractor, iter_video_frames  # noqa: E402
from src.pose.tracker import IoUCentroidTracker                  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def primary_track_sequence(video_path: Path, pose: PoseExtractor,
                           tcfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Run pose+tracking over a clip; return the longest track's sequence.

    URFD/Le2i clips contain a single actor; the longest confirmed track
    is that actor. Frames where the actor is undetected are simply
    absent (frame_idx has gaps) — the featurizer's ffill/interp handles
    short gaps.
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    tracker = IoUCentroidTracker(tcfg["iou_gate"], tcfg["centroid_gate_px"],
                                 tcfg["max_age"], tcfg["min_hits"])
    per_track: dict[int, list[tuple[int, np.ndarray]]] = {}
    for idx, frame in iter_video_frames(str(video_path)):
        boxes, kpts = pose(frame)          # frame dropped right after this
        for trk in tracker.update(boxes, kpts):
            per_track.setdefault(trk.track_id, []).append(
                (idx, trk.keypoints.copy()))

    if not per_track:
        return (np.zeros((0, 17, 3), np.float32), np.zeros(0, np.int32),
                float(fps))
    tid = max(per_track, key=lambda k: len(per_track[k]))
    frames = np.array([f for f, _ in per_track[tid]], np.int32)
    seq = np.stack([k for _, k in per_track[tid]]).astype(np.float32)
    return seq, frames, float(fps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory of dataset clips")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(cfg["data"]["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pose = PoseExtractor(**cfg["pose"])
    clips = sorted(p for p in Path(args.videos).iterdir()
                   if p.suffix.lower() in VIDEO_EXTS)
    print(f"[cache] {len(clips)} clips -> {out_dir}")

    for clip in clips:
        out = out_dir / f"{clip.stem}.npz"
        if out.exists() and not args.overwrite:
            print(f"  skip {clip.stem} (cached)")
            continue
        seq, frames, fps = primary_track_sequence(clip, pose, cfg["tracker"])
        # NOTE: keypoints + indices + fps only — no pixels in the archive.
        np.savez_compressed(out, keypoints=seq, frame_idx=frames,
                            fps=np.float32(fps))
        print(f"  {clip.stem}: {len(frames)} tracked frames @ {fps:.1f} fps")


if __name__ == "__main__":
    main()
