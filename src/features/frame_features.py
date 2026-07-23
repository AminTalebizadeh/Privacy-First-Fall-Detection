"""Skeleton normalization and per-frame derived features (pure numpy).

COCO-17 keypoint indices used below:
  5 L-shoulder, 6 R-shoulder, 11 L-hip, 12 R-hip.

Every transform is justified inline — see comments.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
EPS = 1e-6


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------
def torso_normalize(kpts: np.ndarray, min_conf: float = 0.30,
                    fallback_scale: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray, float]:
    """Torso-centered, torso-scaled normalization of one skeleton.

    Why: raw pixel coordinates entangle the *pose* signal with camera
    distance, image resolution and person position in the frame. A fall
    at 2 m and at 6 m must look identical to the classifier.
      - Translation: subtract the hip-center (midpoint of both hips) so
        the skeleton is centered at the pelvis — the body's kinematic
        root. This removes absolute image position.
      - Scale: divide by torso length (hip-center -> shoulder-center
        distance). Torso length is (a) rigid — it barely changes with
        pose, unlike bbox height which collapses when a person lies
        down, and (b) proportional to apparent person size, giving
        camera-distance invariance.

    Args:
        kpts: (17,3) [x, y, conf].
        min_conf: keypoints below this confidence are zeroed + masked.
        fallback_scale: scale to reuse when the torso is occluded this
            frame (caller passes the last valid scale for temporal
            stability instead of producing a garbage skeleton).

    Returns:
        norm_xy: (17,2) normalized coords (masked joints are 0).
        mask:    (17,) float 1.0 = reliable joint, 0.0 = masked. Fed to
                 the network so it can learn to discount occlusions
                 rather than treating zeros as real positions.
        scale:   torso length actually used (pixels) — caller caches it.
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    xy, conf = kpts[:, :2], kpts[:, 2]
    mask = (conf >= min_conf).astype(np.float32)

    def _mid(i, j):
        if mask[i] and mask[j]:
            return (xy[i] + xy[j]) / 2.0, True
        if mask[i]:
            return xy[i].copy(), True
        if mask[j]:
            return xy[j].copy(), True
        return np.zeros(2, np.float32), False

    hip_c, hip_ok = _mid(L_HIP, R_HIP)
    sho_c, sho_ok = _mid(L_SHOULDER, R_SHOULDER)

    if hip_ok and sho_ok:
        scale = float(np.linalg.norm(sho_c - hip_c))
    else:
        scale = 0.0
    if scale < EPS:                      # torso occluded / degenerate
        if fallback_scale and fallback_scale > EPS:
            scale = float(fallback_scale)
        else:
            # last resort: visible-keypoint bbox height, so we degrade
            # gracefully instead of dividing by ~0.
            vis = xy[mask > 0]
            scale = float(vis[:, 1].max() - vis[:, 1].min()) if len(vis) > 1 else 1.0
            scale = max(scale, 1.0)

    center = hip_c if hip_ok else (sho_c if sho_ok else xy[mask > 0].mean(0)
                                   if mask.any() else np.zeros(2, np.float32))

    norm_xy = (xy - center) / (scale + EPS)
    norm_xy *= mask[:, None]             # masked joints -> exactly 0
    return norm_xy.astype(np.float32), mask, scale


# ----------------------------------------------------------------------
# Per-frame scalar features
# ----------------------------------------------------------------------
def trunk_angle(kpts: np.ndarray, min_conf: float = 0.30) -> float:
    """Angle (radians, [0, pi]) between the trunk vector (hip-center ->
    shoulder-center) and the *upward* vertical.

    Why: it is THE single most discriminative fall feature — standing
    ~0 rad, lying ~pi/2. Velocity of this angle is what separates
    "falling" from "already lying"; the network sees both via the
    temporal window. Image y grows downward, so up = (0, -1).
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    xy, conf = kpts[:, :2], kpts[:, 2]
    ok = conf >= min_conf
    if not ((ok[L_HIP] or ok[R_HIP]) and (ok[L_SHOULDER] or ok[R_SHOULDER])):
        return float("nan")
    hip = (xy[L_HIP] * ok[L_HIP] + xy[R_HIP] * ok[R_HIP]) / max(ok[L_HIP] + ok[R_HIP], 1)
    sho = (xy[L_SHOULDER] * ok[L_SHOULDER] + xy[R_SHOULDER] * ok[R_SHOULDER]) \
        / max(ok[L_SHOULDER] + ok[R_SHOULDER], 1)
    v = sho - hip
    n = np.linalg.norm(v)
    if n < EPS:
        return float("nan")
    cos = np.clip(np.dot(v / n, np.array([0.0, -1.0])), -1.0, 1.0)
    return float(np.arccos(cos))


def bbox_aspect_ratio(kpts: np.ndarray, min_conf: float = 0.30) -> float:
    """Width/height of the visible-keypoint bounding box.

    Why: complements trunk angle — a lying body is wide-and-short
    (ratio > 1), standing is tall-and-thin (ratio < 1). Computed from
    keypoints rather than the detector box so it inherits the same
    confidence masking and is robust to loose detector boxes.
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    vis = kpts[kpts[:, 2] >= min_conf, :2]
    if len(vis) < 2:
        return float("nan")
    w = float(vis[:, 0].max() - vis[:, 0].min())
    h = float(vis[:, 1].max() - vis[:, 1].min())
    return w / (h + EPS)


def hip_center_y(kpts: np.ndarray, min_conf: float = 0.30) -> float:
    """Vertical pixel position of the hip center (nan if hips occluded)."""
    kpts = np.asarray(kpts, dtype=np.float32)
    ok = kpts[:, 2] >= min_conf
    if ok[L_HIP] and ok[R_HIP]:
        return float((kpts[L_HIP, 1] + kpts[R_HIP, 1]) / 2.0)
    if ok[L_HIP]:
        return float(kpts[L_HIP, 1])
    if ok[R_HIP]:
        return float(kpts[R_HIP, 1])
    return float("nan")


# ----------------------------------------------------------------------
# Sequence-level dynamics
# ----------------------------------------------------------------------
def finite_diff(series: np.ndarray, fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity and acceleration of a 1-D series via central differences.

    Why central differences: O(h^2) accurate and phase-aligned (forward
    differences shift the velocity peak half a frame early, which
    matters when the fall impact spans only ~5 frames). Endpoints use
    one-sided differences. NaNs (occluded frames) are linearly
    interpolated first so a single dropped detection doesn't inject an
    infinite-velocity spike.

    Args:
        series: (T,) values in pixels (or any unit).
        fps: frames per second — converts /frame to /second so the
             learned thresholds transfer between videos of different
             frame rates.
    Returns:
        vel, acc: (T,) arrays, units/s and units/s^2.
    """
    s = np.asarray(series, dtype=np.float64).copy()
    T = len(s)
    if T == 0:
        return np.zeros(0), np.zeros(0)
    nans = np.isnan(s)
    if nans.all():
        s[:] = 0.0
    elif nans.any():
        idx = np.arange(T)
        s[nans] = np.interp(idx[nans], idx[~nans], s[~nans])

    vel = np.gradient(s) * fps          # central diffs, one-sided at ends
    acc = np.gradient(vel) * fps
    return vel.astype(np.float32), acc.astype(np.float32)


def normalized_hip_dynamics(hip_y: np.ndarray, scales: np.ndarray, fps: float
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Hip vertical velocity/acceleration in *torso lengths per second*.

    Why: raw px/s velocity depends on camera distance; dividing by the
    per-frame torso scale makes "the pelvis dropped 2 torso-lengths in
    0.5 s" mean the same thing at any distance. Falls show a large
    positive (downward, image coords) velocity spike followed by a
    deceleration spike at impact — the classifier's key dynamic cue.
    """
    scales = np.maximum(np.asarray(scales, dtype=np.float64), EPS)
    vel, acc = finite_diff(np.asarray(hip_y, dtype=np.float64), fps)
    return (vel / scales).astype(np.float32), (acc / scales).astype(np.float32)
