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



def _fill_short_gaps(
    x: np.ndarray,
    max_gap: int = 2,
    default: float = 0.0,
) -> np.ndarray:
    """Interpolate only short internal NaN gaps; use default elsewhere."""
    out = np.asarray(x, dtype=np.float64).copy()
    valid = np.isfinite(out)

    i = 0
    while i < len(out):
        if valid[i]:
            i += 1
            continue

        start = i
        while i < len(out) and not valid[i]:
            i += 1
        end = i

        gap_len = end - start
        bounded = start > 0 and end < len(out)

        if bounded and gap_len <= max_gap:
            out[start:end] = np.linspace(
                out[start - 1],
                out[end],
                gap_len + 2,
                dtype=np.float64,
            )[1:-1]
            valid[start:end] = True

    out[~valid] = default
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
    scale_age = 0
    max_scale_age = 5

    for t in range(T):
        hip_visible = (masks[:, 11] > 0) | (masks[:, 12] > 0)
        shoulder_visible = (masks[:, 5] > 0) | (masks[:, 6] > 0)
        torso_valid = hip_visible & shoulder_visible

        dynamics_scales = np.where(torso_valid, scales, np.nan)

        vel, acc = normalized_hip_dynamics(
            hips,
            dynamics_scales,
            fps=fps,
            max_gap=2,
        )
        kpts = kpts_seq[t]
        conf = kpts[:, 2]
        xy = kpts[:, :2]

        visible = conf >= min_conf
        hip_ids = [11, 12]
        shoulder_ids = [5, 6]

        hips_visible = [i for i in hip_ids if visible[i]]
        shoulders_visible = [i for i in shoulder_ids if visible[i]]

        # A fresh scale requires at least one visible hip
        # and one visible shoulder, with non-degenerate geometry.
        current_torso_valid = bool(hips_visible and shoulders_visible)

        if current_torso_valid:
            hip_center = xy[hips_visible].mean(axis=0)
            shoulder_center = xy[shoulders_visible].mean(axis=0)
            current_scale = float(
                np.linalg.norm(shoulder_center - hip_center)
            )
            current_torso_valid = (
                np.isfinite(current_scale) and current_scale > 1e-6
            )

        if current_torso_valid:
            fallback = None
        elif last_scale is not None and scale_age < max_scale_age:
            fallback = last_scale
        else:
            fallback = None

        n, m, s = torso_normalize(
            kpts, min_conf, fallback_scale=fallback
        )

        norm[t], masks[t], scales[t] = n, m, s

        if current_torso_valid:
            last_scale = current_scale
            scale_age = 0
        else:
            scale_age += 1

        angles[t] = np.nan_to_num(
            trunk_angle(kpts, min_conf),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        aspects[t] = np.nan_to_num(
            bbox_aspect_ratio(kpts, min_conf),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        hips[t] = hip_center_y(kpts, min_conf)


        hip_visible = (masks[:, 11] > 0) | (masks[:, 12] > 0)
    shoulder_visible = (masks[:, 5] > 0) | (masks[:, 6] > 0)
    torso_valid = hip_visible & shoulder_visible

    dynamics_scales = np.where(torso_valid, scales, np.nan)

    vel, acc = normalized_hip_dynamics(
        hips,
        dynamics_scales,
        fps=fps,
        max_gap=2,
    )

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
