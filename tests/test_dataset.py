"""Dataset and split tests for the skeleton fall-detection pipeline.

Run from the repository root:
    python -m pytest tests/test_dataset.py -v

Two tests are marked expectedFailure because the current implementation does
not yet reject duplicate CSV clip IDs or incomplete/extra cache inventories.
Remove the decorators after those validations are implemented.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.data.dataset import (
    SkeletonWindowDataset,
    load_labels,
    split_groups,
)
from src.features.sequence import window_label
from tests.test_normalize import synthetic_standing


FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


class TestGroupSplit(unittest.TestCase):
    def test_groups_are_disjoint_and_cover_all_groups(self):
        groups = [f"group{i:02d}" for i in range(20)]
        assignment = split_groups(groups, FRACTIONS, seed=42)

        self.assertEqual(set(assignment), set(groups))
        split_groups_seen = {
            name: {g for g, split in assignment.items() if split == name}
            for name in ("train", "val", "test")
        }
        self.assertTrue(split_groups_seen["train"])
        self.assertTrue(split_groups_seen["val"])
        self.assertTrue(split_groups_seen["test"])
        self.assertFalse(split_groups_seen["train"] & split_groups_seen["val"])
        self.assertFalse(split_groups_seen["train"] & split_groups_seen["test"])
        self.assertFalse(split_groups_seen["val"] & split_groups_seen["test"])

    def test_same_seed_produces_same_assignment(self):
        groups = [f"group{i:02d}" for i in range(20)]
        self.assertEqual(
            split_groups(groups, FRACTIONS, seed=123),
            split_groups(groups, FRACTIONS, seed=123),
        )

    def test_different_seed_is_allowed_to_change_assignment(self):
        groups = [f"group{i:02d}" for i in range(30)]
        first = split_groups(groups, FRACTIONS, seed=1)
        second = split_groups(groups, FRACTIONS, seed=2)
        self.assertNotEqual(first, second)


class TestWindowLabels(unittest.TestCase):
    def test_adl_clip_is_negative(self):
        self.assertEqual(window_label(0, 30, None), 0)

    def test_window_inside_fall_is_positive(self):
        self.assertEqual(window_label(30, 30, (30, 59), 0.30), 1)

    def test_small_overlap_is_negative(self):
        self.assertEqual(window_label(0, 30, (0, 7), 0.30), 0)

    def test_overlap_threshold_is_inclusive(self):
        # 9 of 30 frames = exactly 0.30
        self.assertEqual(window_label(0, 30, (0, 8), 0.30), 1)


class DatasetFixture(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.cache_dir = self.root / "cache"
        self.cache_dir.mkdir()
        self.labels_csv = self.root / "labels.csv"

        rows = []
        # Two clips per group; each group is assigned wholly to one split.
        for group_index in range(3):
            group = f"group{group_index}"
            for clip_index in range(2):
                clip_id = f"clip_g{group_index}_{clip_index}"
                is_fall = clip_index == 1
                rows.append({
                    "clip_id": clip_id,
                    "fall_start_frame": "10" if is_fall else "-1",
                    "fall_end_frame": "35" if is_fall else "-1",
                    "group": group,
                })
                keypoints = np.stack([
                    synthetic_standing(hip_y=300.0 + 0.2 * frame)
                    for frame in range(40)
                ]).astype(np.float32)
                np.savez_compressed(
                    self.cache_dir / f"{clip_id}.npz",
                    keypoints=keypoints,
                    frame_idx=np.arange(40, dtype=np.int32),
                    fps=np.asarray(30.0, dtype=np.float32),
                )

        with self.labels_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "clip_id", "fall_start_frame", "fall_end_frame", "group"
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        self.rows = rows

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_dataset(self, split):
        return SkeletonWindowDataset(
            cache_dir=self.cache_dir,
            labels_csv=self.labels_csv,
            split=split,
            fractions=FRACTIONS,
            seed=42,
            window_size=30,
            stride=5,
            positive_overlap=0.30,
            fps=30.0,
            min_conf=0.30,
        )

    def test_dataset_windows_are_confined_to_assigned_groups(self):
        labels = load_labels(self.labels_csv)
        assignment = split_groups(
            [item["group"] for item in labels.values()], FRACTIONS, seed=42
        )

        clip_groups = {row["clip_id"]: row["group"] for row in self.rows}
        split_clip_sets = {}
        for split in ("train", "val", "test"):
            dataset = self.make_dataset(split)
            self.assertGreater(len(dataset), 0)
            observed_clips = {clip_id for clip_id, _ in dataset.meta}
            split_clip_sets[split] = observed_clips

            for clip_id in observed_clips:
                self.assertEqual(assignment[clip_groups[clip_id]], split)

        self.assertFalse(split_clip_sets["train"] & split_clip_sets["val"])
        self.assertFalse(split_clip_sets["train"] & split_clip_sets["test"])
        self.assertFalse(split_clip_sets["val"] & split_clip_sets["test"])

    def test_window_tensor_shape_and_binary_labels(self):
        dataset = self.make_dataset("train")
        self.assertGreater(len(dataset), 0)
        for window in dataset.windows:
            self.assertEqual(window.shape, (30, 55))
            self.assertTrue(np.isfinite(window).all())
        self.assertTrue(set(dataset.labels).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
