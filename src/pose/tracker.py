"""Lightweight IoU + centroid multi-person tracker.

Deliberately minimal (no Kalman filter, no ReID, no external deps):
fall-detection needs *stable short-term identity* so that per-track
temporal windows are built from the same person — it does not need
long-term re-identification. Greedy IoU matching with a centroid
fallback is sufficient for the mostly-static single/dual-person scenes
of URFD / Le2i, and costs microseconds per frame.

Matching strategy per frame:
  1. Compute IoU between every live track's last bbox and every new
     detection; greedily match highest IoU first, gated by `iou_gate`.
  2. Unmatched pairs get a second chance via centroid distance
     (handles the "person fell -> bbox shape changed drastically ->
     IoU collapsed" case, which is exactly the moment we care about).
  3. Remaining detections spawn new tracks; tracks unseen for
     `max_age` frames are killed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two [x1,y1,x2,y2] boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (area_a + area_b - inter + 1e-9))


def _centroid(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray                       # last [x1,y1,x2,y2]
    keypoints: np.ndarray                  # last (17,3) [x,y,conf]
    hits: int = 1                          # total matched frames
    age: int = 0                           # frames since last match
    confirmed: bool = False
    history: List[np.ndarray] = field(default_factory=list)  # keypoint history


class IoUCentroidTracker:
    def __init__(self, iou_gate: float = 0.25, centroid_gate_px: float = 120.0,
                 max_age: int = 15, min_hits: int = 3):
        self.iou_gate = iou_gate
        self.centroid_gate_px = centroid_gate_px
        self.max_age = max_age
        self.min_hits = min_hits
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    # ------------------------------------------------------------------
    def update(self, boxes: np.ndarray, keypoints: np.ndarray) -> List[Track]:
        """Advance one frame.

        Args:
            boxes: (N,4) xyxy detections (may be empty).
            keypoints: (N,17,3) matching pose keypoints.
        Returns:
            list of *confirmed* live tracks after the update.
        """
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 17, 3)

        track_ids = list(self.tracks.keys())
        unmatched_dets = set(range(len(boxes)))
        unmatched_trks = set(track_ids)

        # ---- pass 1: greedy IoU ---------------------------------------
        pairs: List[Tuple[float, int, int]] = []
        for ti in track_ids:
            for di in range(len(boxes)):
                v = iou_xyxy(self.tracks[ti].bbox, boxes[di])
                if v >= self.iou_gate:
                    pairs.append((v, ti, di))
        for v, ti, di in sorted(pairs, key=lambda p: -p[0]):
            if ti in unmatched_trks and di in unmatched_dets:
                self._assign(ti, boxes[di], keypoints[di])
                unmatched_trks.discard(ti)
                unmatched_dets.discard(di)

        # ---- pass 2: centroid fallback (fall = bbox shape shock) ------
        pairs = []
        for ti in unmatched_trks:
            c_t = _centroid(self.tracks[ti].bbox)
            for di in unmatched_dets:
                d = float(np.linalg.norm(c_t - _centroid(boxes[di])))
                if d <= self.centroid_gate_px:
                    pairs.append((d, ti, di))
        for d, ti, di in sorted(pairs, key=lambda p: p[0]):
            if ti in unmatched_trks and di in unmatched_dets:
                self._assign(ti, boxes[di], keypoints[di])
                unmatched_trks.discard(ti)
                unmatched_dets.discard(di)

        # ---- births -----------------------------------------------------
        for di in unmatched_dets:
            t = Track(self._next_id, boxes[di].copy(), keypoints[di].copy())
            t.history.append(keypoints[di].copy())
            t.confirmed = t.hits >= self.min_hits   # min_hits<=1: born confirmed
            self.tracks[self._next_id] = t
            self._next_id += 1

        # ---- ageing / deaths -------------------------------------------
        for ti in list(self.tracks.keys()):
            trk = self.tracks[ti]
            if ti in unmatched_trks:
                trk.age += 1
                if trk.age > self.max_age:
                    del self.tracks[ti]

        return [t for t in self.tracks.values() if t.confirmed]

    # ------------------------------------------------------------------
    def _assign(self, ti: int, box: np.ndarray, kpts: np.ndarray) -> None:
        trk = self.tracks[ti]
        trk.bbox = box.copy()
        trk.keypoints = kpts.copy()
        trk.hits += 1
        trk.age = 0
        trk.history.append(kpts.copy())
        if trk.hits >= self.min_hits:
            trk.confirmed = True
