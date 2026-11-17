"""Lightweight IoU + centroid multi-person tracker.

Compatibility note:
- ``update`` still returns all confirmed live tracks, including tracks that
  were not detected in the current frame. This preserves the existing API.
- Consumers must check ``track.updated_this_frame`` before treating its
  keypoints as a new observation.
- Track history is bounded to avoid unbounded memory growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


TRACK_HISTORY_SIZE = 128


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x1, y1, x2, y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / (union + 1e-9))


def _centroid(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0,
                     (box[1] + box[3]) / 2.0], dtype=np.float32)


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    keypoints: np.ndarray
    hits: int = 1
    age: int = 0
    confirmed: bool = False
    # Keep the original public list type for caller compatibility; trim it
    # explicitly so existing code can continue to use slicing/indexing.
    history: List[np.ndarray] = field(default_factory=list)
    # True only when the tracker associated a real detection to this track
    # during the most recent call to update().
    updated_this_frame: bool = True


class IoUCentroidTracker:
    def __init__(self, iou_gate: float = 0.25,
                 centroid_gate_px: float = 120.0,
                 max_age: int = 15,
                 min_hits: int = 3):
        if not 0.0 <= iou_gate <= 1.0:
            raise ValueError("iou_gate must be in [0, 1]")
        if centroid_gate_px <= 0:
            raise ValueError("centroid_gate_px must be positive")
        if max_age < 0:
            raise ValueError("max_age must be >= 0")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")

        self.iou_gate = float(iou_gate)
        self.centroid_gate_px = float(centroid_gate_px)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, boxes: np.ndarray,
               keypoints: np.ndarray) -> List[Track]:
        """Advance one frame.

        Args:
            boxes: ``(N, 4)`` xyxy detections (may be empty).
            keypoints: matching ``(N, 17, 3)`` [x, y, confidence] poses.

        Returns:
            All confirmed live tracks. A returned track is a fresh
            observation only when ``track.updated_this_frame`` is True.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 17, 3)

        if len(boxes) != len(keypoints):
            raise ValueError(
                f"boxes/keypoints count mismatch: {len(boxes)} != {len(keypoints)}"
            )
        if not np.isfinite(boxes).all() or not np.isfinite(keypoints).all():
            raise ValueError("boxes and keypoints must contain only finite values")

        # Reset freshness before association. _assign() marks matched tracks.
        for track in self.tracks.values():
            track.updated_this_frame = False

        track_ids = sorted(self.tracks)
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(track_ids)

        # Pass 1: greedy IoU matching.
        pairs: List[Tuple[float, int, int]] = []
        for track_id in track_ids:
            for det_id in range(len(boxes)):
                score = iou_xyxy(self.tracks[track_id].bbox, boxes[det_id])
                if score >= self.iou_gate:
                    pairs.append((score, track_id, det_id))

        for score, track_id, det_id in sorted(
            pairs, key=lambda item: (-item[0], item[1], item[2])
        ):
            if track_id in unmatched_trks and det_id in unmatched_dets:
                self._assign(track_id, boxes[det_id], keypoints[det_id])
                unmatched_trks.remove(track_id)
                unmatched_dets.remove(det_id)

        # Pass 2: centroid fallback for shape changes (e.g. standing -> fallen).
        pairs = []
        for track_id in sorted(unmatched_trks):
            previous_center = _centroid(self.tracks[track_id].bbox)
            for det_id in sorted(unmatched_dets):
                distance = float(np.linalg.norm(
                    previous_center - _centroid(boxes[det_id])
                ))
                if distance <= self.centroid_gate_px:
                    pairs.append((distance, track_id, det_id))

        for distance, track_id, det_id in sorted(
            pairs, key=lambda item: (item[0], item[1], item[2])
        ):
            if track_id in unmatched_trks and det_id in unmatched_dets:
                self._assign(track_id, boxes[det_id], keypoints[det_id])
                unmatched_trks.remove(track_id)
                unmatched_dets.remove(det_id)

        # Births: new detections become new tracks.
        for det_id in sorted(unmatched_dets):
            track = Track(
                track_id=self._next_id,
                bbox=boxes[det_id].copy(),
                keypoints=keypoints[det_id].copy(),
            )
            self._append_history(track, keypoints[det_id])
            track.confirmed = track.hits >= self.min_hits
            self.tracks[self._next_id] = track
            self._next_id += 1

        # Age unmatched tracks and remove expired ones.
        for track_id in list(self.tracks):
            if track_id in unmatched_trks:
                track = self.tracks[track_id]
                track.age += 1
                if track.age > self.max_age:
                    del self.tracks[track_id]

        return [track for track in self.tracks.values() if track.confirmed]

    @staticmethod
    def _append_history(track: Track, keypoints: np.ndarray) -> None:
        track.history.append(keypoints.copy())
        overflow = len(track.history) - TRACK_HISTORY_SIZE
        if overflow > 0:
            del track.history[:overflow]

    def _assign(self, track_id: int, box: np.ndarray,
                keypoints: np.ndarray) -> None:
        track = self.tracks[track_id]
        track.bbox = box.copy()
        track.keypoints = keypoints.copy()
        track.hits += 1
        track.age = 0
        track.updated_this_frame = True
        self._append_history(track, keypoints)
        if track.hits >= self.min_hits:
            track.confirmed = True
