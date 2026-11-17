"""Window dataset over frame-aligned skeleton caches.

Split assignment is group-disjoint. In URFD, the current group field is a
sequence/scene proxy, not a verified person identity; do not claim unseen-
person generalization from this split alone.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

from ..features.sequence import sequence_features, sliding_windows, window_label


def load_labels(labels_csv: str | Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with Path(labels_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lo, hi = int(row["fall_start_frame"]), int(row["fall_end_frame"])
            group = str(row.get("group", "")).strip()
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id or not group:
                raise ValueError("labels.csv requires non-empty clip_id and group")
            if clip_id in out:
                raise ValueError(f"Duplicate clip_id in labels.csv: {clip_id}")
            if lo >= 0 and hi < lo:
                raise ValueError(f"Invalid fall span for {clip_id}: {(lo, hi)}")
            out[clip_id] = {
                "fall_span": None if lo < 0 else (lo, hi),
                "group": group,
            }
    if not out:
        raise ValueError(f"No labels found in {labels_csv}")
    return out


def split_groups(groups: List[str], fractions: Dict[str, float], seed: int) -> Dict[str, str]:
    required = ("train", "val", "test")
    if any(name not in fractions for name in required):
        raise ValueError("fractions must define train, val, and test")
    values = [float(fractions[name]) for name in required]
    if any(not np.isfinite(v) or v <= 0 or v >= 1 for v in values):
        raise ValueError("each split fraction must be strictly between 0 and 1")
    if not np.isclose(sum(values), 1.0, atol=1e-6):
        raise ValueError("train/val/test fractions must sum to 1.0")
    normalized = [str(g).strip() for g in groups]
    if any(not g for g in normalized):
        raise ValueError("group IDs must be non-empty")
    uniq = sorted(set(normalized))
    if len(uniq) < 3:
        raise ValueError("at least 3 distinct groups are required for train/val/test")
    rng = np.random.default_rng(int(seed))
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = max(1, min(n - 2, round(n * values[0])))
    n_val = max(1, min(n - n_train - 1, round(n * values[1])))
    return {
        group: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
        for i, group in enumerate(uniq)
    }


class SkeletonWindowDataset(Dataset):
    """Materialize (window, label) pairs for one split."""
    def __init__(self, cache_dir: str | Path, labels_csv: str | Path,
                 split: str, fractions: Dict[str, float], seed: int,
                 window_size: int = 30, stride: int = 5,
                 positive_overlap: float = 0.30, fps: float = 30.0,
                 min_conf: float = 0.30):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if window_size < 1 or stride < 1:
            raise ValueError("window_size and stride must be positive")
        if not 0 <= min_conf <= 1:
            raise ValueError("min_conf must be in [0, 1]")

        cache_dir = Path(cache_dir)

        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Cache directory does not exist: {cache_dir}")

        labels = load_labels(labels_csv)

        # Validate the complete dataset before applying the split.
        cache_paths = {
            path.stem: path
            for path in cache_dir.glob("*.npz")
        }

        label_ids = set(labels)
        cache_ids = set(cache_paths)

        missing_caches = sorted(label_ids - cache_ids)
        extra_caches = sorted(cache_ids - label_ids)

        if missing_caches:
            raise FileNotFoundError(
                f"Missing cache files for {len(missing_caches)} labeled clips: "
                f"{missing_caches[:10]}"
            )

        if extra_caches:
            raise ValueError(
                f"Cache files without labels.csv entries: "
                f"{extra_caches[:10]}"
            )

        assignment = split_groups(
            [v["group"] for v in labels.values()],
            fractions,
            seed
        )
        self.windows: List[np.ndarray] = []
        self.labels: List[int] = []
        self.meta: List[Tuple[str, int]] = []

        for clip_id in sorted(cache_paths):
            npz_path = cache_paths[clip_id]
            if clip_id not in labels or assignment[labels[clip_id]["group"]] != split:
                continue
            with np.load(npz_path, allow_pickle=False) as archive:
                if "keypoints" not in archive:
                    raise ValueError(f"Missing keypoints in {npz_path}")
                keypoints = np.asarray(archive["keypoints"], dtype=np.float32)
                if "frame_idx" not in archive:
                    raise ValueError(
                        f"Legacy cache without frame_idx: {npz_path}. Rebuild cache with --overwrite."
                    )
                frame_idx = np.asarray(archive["frame_idx"], dtype=np.int64)
                clip_fps = float(archive["fps"]) if "fps" in archive else float(fps)

            if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
                raise ValueError(f"Invalid keypoint shape in {npz_path}: {keypoints.shape}")

            if (
                frame_idx.ndim != 1
                or len(frame_idx) != len(keypoints)
                or (len(frame_idx) > 1 and not np.all(np.diff(frame_idx) == 1))
            ):
                raise ValueError(f"Invalid or non-contiguous frame_idx in {npz_path}")

            if not np.isfinite(keypoints).all() or not np.isfinite(clip_fps) or clip_fps <= 0:
                raise ValueError(f"Non-finite keypoints or invalid FPS in {npz_path}")

            features = sequence_features(keypoints, fps=clip_fps, min_conf=min_conf)
            for start, window in sliding_windows(features, window_size, stride):
                self.windows.append(window)
                self.labels.append(window_label(
                    start, window_size, labels[clip_id]["fall_span"], positive_overlap
                ))
                self.meta.append((clip_id, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int):
        if torch is None:
            raise ImportError("PyTorch is required to index SkeletonWindowDataset")
        return (torch.from_numpy(self.windows[index]),
                torch.tensor(self.labels[index], dtype=torch.float32))

    @property
    def pos_fraction(self) -> float:
        return float(np.mean(self.labels)) if self.labels else 0.0
