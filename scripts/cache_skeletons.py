#!/usr/bin/env python3
"""Build frame-aligned skeleton caches from source videos.

Missing detections are represented by all-zero keypoints (confidence=0),
not by repeating the last pose and not by deleting a time step. Therefore
cache row i continues to represent original video frame i, preserving the
timing assumptions used by velocity, acceleration, and event evaluation.
No image arrays are written to disk.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pose.tracker import IoUCentroidTracker  # noqa: E402

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


def primary_track_sequence(video_path: Path, pose,
                           tcfg: dict, frame_iterator=None
                           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a dense, frame-aligned sequence for the longest confirmed track.

    The project currently assumes one primary actor per URFD/Le2i clip.
    For every original video frame, missing/undetected poses are represented
    by zero-confidence keypoints. The returned ``frame_idx`` is contiguous
    from zero and has the same length as ``keypoints``.
    """
    import cv2

    if frame_iterator is None:
        # Lazy import keeps cache utilities testable without loading the
        # heavyweight Ultralytics runtime at module import time.
        from src.pose.extractor import iter_video_frames
        frame_iterator = iter_video_frames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    tracker = IoUCentroidTracker(
        tcfg["iou_gate"], tcfg["centroid_gate_px"],
        tcfg["max_age"], tcfg["min_hits"],
    )
    # track_id -> {original_frame_index: observed_keypoints}
    per_track: dict[int, dict[int, np.ndarray]] = {}
    last_frame_idx = -1

    for frame_idx, frame in frame_iterator(str(video_path)):
        last_frame_idx = frame_idx
        boxes, keypoints = pose(frame)  # raw frame remains transient in memory
        for track in tracker.update(boxes, keypoints):
            # Never cache the old pose of a track as a new observation.
            if not track.updated_this_frame:
                continue
            per_track.setdefault(track.track_id, {})[frame_idx] = (
                track.keypoints.copy()
            )

    if last_frame_idx < 0:
        empty_kpts = np.zeros((0, 17, 3), dtype=np.float32)
        empty_idx = np.zeros((0,), dtype=np.int32)
        return empty_kpts, empty_idx, fps

    # Keep the existing single-primary-track policy, but preserve the full
    # original frame timeline, including leading, internal, and trailing gaps.
    if per_track:
        primary_id = max(
            per_track,
            key=lambda track_id: len(per_track[track_id]),
        )
        observations = per_track[primary_id]
    else:
        observations = {}

    sequence = np.zeros((last_frame_idx + 1, 17, 3), dtype=np.float32)
    for frame_idx, observed_keypoints in observations.items():
        sequence[frame_idx] = observed_keypoints

    frame_idx = np.arange(last_frame_idx + 1, dtype=np.int32)
    return sequence, frame_idx, fps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", required=True,
                        help="directory containing dataset clips")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    out_dir = Path(cfg["data"]["cache_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = Path(args.videos)
    if not video_dir.is_dir():
        raise NotADirectoryError(f"Video directory does not exist: {video_dir}")

    from src.pose.extractor import PoseExtractor
    pose = PoseExtractor(**cfg["pose"])
    clips = sorted(
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    print(f"[cache] {len(clips)} clips -> {out_dir}")

    for clip in clips:
        output_path = out_dir / f"{clip.stem}.npz"
        if output_path.exists() and not args.overwrite:
            print(f"  skip {clip.stem} (cached; use --overwrite to rebuild)")
            continue

        keypoints, frame_idx, fps = primary_track_sequence(
            clip, pose, cfg["tracker"]
        )
        # Data archive contains skeletons and timing only; no image pixels.
        np.savez_compressed(
            output_path,
            keypoints=keypoints,
            frame_idx=frame_idx,
            fps=np.float32(fps),
        )
        observed_frames = int(np.count_nonzero(keypoints[:, :, 2].max(axis=1) > 0))
        print(
            f"  {clip.stem}: {len(frame_idx)} timeline frames, "
            f"{observed_frames} frames with observed keypoints @ {fps:.2f} fps"
        )


if __name__ == "__main__":
    main()
