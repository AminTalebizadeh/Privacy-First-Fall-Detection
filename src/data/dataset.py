"""Window dataset over the .npz skeleton cache.

The cache directory (one .npz per clip, produced by
`scripts/cache_skeletons.py`) IS the training set: it contains keypoint
arrays and metadata only — no pixels — which is the privacy pipeline in
action.

Split policy: by GROUP (subject/scene identifier from labels.csv), never
by clip. Splitting by clip leaks the same person/room/camera into both
train and test, inflating metrics; group-disjoint splits measure what we
actually care about — generalization to unseen people and scenes.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # torch only needed for the Dataset class, not the split utilities
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from ..features.sequence import sequence_features, sliding_windows, window_label


def load_labels(labels_csv: str | Path) -> Dict[str, dict]:
    """labels.csv columns: clip_id, fall_start_frame, fall_end_frame, group.
    Non-fall (ADL) clips use fall_start_frame = fall_end_frame = -1."""
    out: Dict[str, dict] = {}
    with open(labels_csv, newline="") as f:
        for row in csv.DictReader(f):
            lo, hi = int(row["fall_start_frame"]), int(row["fall_end_frame"])
            out[row["clip_id"]] = {
                "fall_span": None if lo < 0 else (lo, hi),
                "group": row["group"],
            }
    return out


def split_groups(groups: List[str], fractions: Dict[str, float], seed: int
                 ) -> Dict[str, str]:
    """Deterministically assign each group -> train/val/test.

    Uses a seeded shuffle of the *unique group list* so the assignment
    is reproducible and independent of clip enumeration order.
    """
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n = len(uniq)
    n_tr = max(1, round(n * fractions["train"]))
    n_va = max(1, round(n * fractions["val"]))
    assign: Dict[str, str] = {}
    for i, g in enumerate(uniq):
        assign[g] = ("train" if i < n_tr else
                     "val" if i < n_tr + n_va else "test")
    return assign


class SkeletonWindowDataset(Dataset):
    """Materializes all (window, label) pairs for one split."""

    def __init__(self, cache_dir: str | Path, labels_csv: str | Path,
                 split: str, fractions: Dict[str, float], seed: int,
                 window_size: int = 30, stride: int = 5,
                 positive_overlap: float = 0.30, fps: float = 30.0):
        cache_dir = Path(cache_dir)
        labels = load_labels(labels_csv)
        assign = split_groups([v["group"] for v in labels.values()],
                              fractions, seed)

        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []
        self.meta: List[Tuple[str, int]] = []          # (clip_id, start_frame)

        for npz_path in sorted(cache_dir.glob("*.npz")):
            clip_id = npz_path.stem
            if clip_id not in labels:
                continue
            if assign[labels[clip_id]["group"]] != split:
                continue
            with np.load(npz_path) as z:
                kpts_seq = z["keypoints"]              # (T,17,3)
                clip_fps = float(z["fps"]) if "fps" in z else fps
            feats = sequence_features(kpts_seq, fps=clip_fps)
            for start, win in sliding_windows(feats, window_size, stride):
                self.windows.append(win)
                self.labels.append(window_label(start, window_size,
                                                labels[clip_id]["fall_span"],
                                                positive_overlap))
                self.meta.append((clip_id, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        return (torch.from_numpy(self.windows[i]),
                torch.tensor(self.labels[i], dtype=torch.float32))

    @property
    def pos_fraction(self) -> float:
        return float(np.mean(self.labels)) if self.labels else 0.0
