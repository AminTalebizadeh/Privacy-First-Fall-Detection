"""Sequence featurization: (T,17,3) skeleton track -> (T,55) features,
plus sliding-window extraction.

Per-frame feature layout (55 dims):
    [ 0:34]  normalized keypoint xy (17 * 2), torso-centered/scaled
    [34:51]  keypoint confidence mask (17)
    [51]     trunk angle vs vertical (radians; nan -> ffill -> 0)
    [52]     visible-keypoint bbox aspect ratio (w/h)
    [53]     hip-center vertical velocity   (torso-lengths / s)
    [54]     hip-center vertical acceleration (torso-lengths / s^2)
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

from .frame_features import (bbox_aspect_ratio, hip_center_y,
                             normalized_hip_dynamics, torso_normalize,
                             trunk_angle)

FEATURE_DIM = 55


def _ffill_nan(x: np.ndarray, fill0: float = 0.0) -> np.ndarray:
    """Forward-fill NaNs (occluded frames inherit the last valid value —
    temporally consistent, unlike zeroing which fakes a sudden jump)."""
    out = x.astype(np.float64).copy()
    last = fill0
    for i in range(len(out)):
        if np.isnan(out[i]):
            out[i] = last
        else:
            last = out[i]
    return out


def sequence_features(kpts_seq: np.ndarray, fps: float = 30.0,
                      min_conf: float = 0.30) -> np.ndarray:
    """Featurize one track's keypoint sequence.

    Args:
        kpts_seq: (T,17,3) pixel-space [x,y,conf] per frame.
        fps: nominal frame rate for velocity units.
    Returns:
        (T, FEATURE_DIM) float32 feature matrix.
    """
    kpts_seq = np.asarray(kpts_seq, dtype=np.float32)
    T = kpts_seq.shape[0]
    norm = np.zeros((T, 17, 2), np.float32)
    masks = np.zeros((T, 17), np.float32)
    scales = np.zeros(T, np.float64)
    angles = np.zeros(T, np.float64)
    aspects = np.zeros(T, np.float64)
    hips = np.zeros(T, np.float64)

    last_scale = None
    for t in range(T):
        # per-frame fallback_scale gives temporal stability under occlusion
        n, m, s = torso_normalize(kpts_seq[t], min_conf, fallback_scale=last_scale)
        norm[t], masks[t], scales[t] = n, m, s
        last_scale = s
        angles[t] = trunk_angle(kpts_seq[t], min_conf)
        aspects[t] = bbox_aspect_ratio(kpts_seq[t], min_conf)
        hips[t] = hip_center_y(kpts_seq[t], min_conf)

    angles = _ffill_nan(angles)
    aspects = _ffill_nan(aspects, fill0=0.5)
    vel, acc = normalized_hip_dynamics(hips, scales, fps)

    feats = np.concatenate(
        [norm.reshape(T, 34), masks,
         angles[:, None], aspects[:, None], vel[:, None], acc[:, None]],
        axis=1).astype(np.float32)
    assert feats.shape[1] == FEATURE_DIM
    return feats


def sliding_windows(feats: np.ndarray, size: int = 30, stride: int = 5
                    ) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (start_frame, (size,F) window). Drops the ragged tail —
    at 30 fps we lose <1 s of the clip end, irrelevant vs. label noise."""
    T = feats.shape[0]
    for s in range(0, T - size + 1, stride):
        yield s, feats[s:s + size]


def window_label(start: int, size: int, fall_span: Tuple[int, int] | None,
                 positive_overlap: float = 0.30) -> int:
    """Label a window from the annotated fall frame span.

    Positive iff >= `positive_overlap` of the window's frames lie inside
    [fall_start, fall_end]. Rationale: windows brushing the very edge of
    a fall contain mostly normal activity; labeling them positive
    teaches the model to fire on standing people.
    """
    if fall_span is None:
        return 0
    lo, hi = fall_span
    inter = max(0, min(start + size - 1, hi) - max(start, lo) + 1)
    return int(inter / size >= positive_overlap)
