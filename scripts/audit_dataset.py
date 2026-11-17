
from pathlib import Path
from collections import defaultdict
import argparse
import sys

import numpy as np
import yaml

# Add project root to Python import path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (
    SkeletonWindowDataset,
    load_labels,
    split_groups,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/urfd.yaml",
    )
    args = parser.parse_args()

    root = Path.cwd()
    config_path = root / args.config

    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    window_cfg = cfg["window"]
    feature_cfg = cfg["features"]

    cache_dir = root / data_cfg["cache_dir"]
    labels_csv = root / data_cfg["labels_csv"]

    labels = load_labels(labels_csv)
    fractions = data_cfg["split"]

    assignment = split_groups(
        [item["group"] for item in labels.values()],
        fractions,
        seed=cfg["seed"],
    )

    groups_by_split = {
        name: {g for g, s in assignment.items() if s == name}
        for name in ("train", "val", "test")
    }

    print("=== DATASET AUDIT ===")
    print(f"Config: {config_path}")
    print(f"Cache: {cache_dir}")
    print(f"Labels: {len(labels)}")
    print(f"Groups: {len(set(v['group'] for v in labels.values()))}")

    datasets = {}
    clip_groups = {
        clip_id: item["group"] for clip_id, item in labels.items()
    }

    for split in ("train", "val", "test"):
        ds = SkeletonWindowDataset(
            cache_dir=cache_dir,
            labels_csv=labels_csv,
            split=split,
            fractions=fractions,
            seed=cfg["seed"],
            window_size=window_cfg["size"],
            stride=window_cfg["stride"],
            positive_overlap=window_cfg["positive_overlap"],
            fps=feature_cfg["velocity_fps"],
            min_conf=feature_cfg["min_kpt_conf"],
        )
        datasets[split] = ds

        clip_ids = {
            clip_id for clip_id, _ in ds.meta
        }
        fall_clips = {
            cid for cid in clip_ids
            if labels[cid]["fall_span"] is not None
        }
        adl_clips = clip_ids - fall_clips

        positive = int(sum(ds.labels))
        negative = len(ds.labels) - positive

        shapes_ok = all(
            w.shape == (window_cfg["size"], cfg["model"]["input_dim"])
            for w in ds.windows
        )
        finite_ok = all(np.isfinite(w).all() for w in ds.windows)
        bad_count = 0

        feature_names = (
            [f"joint_{i}_{axis}" for i in range(17) for axis in ("x", "y")]
            + [f"mask_{i}" for i in range(17)]
            + ["trunk_angle", "bbox_aspect", "hip_velocity", "hip_acceleration"]
        )

        for i, window in enumerate(ds.windows):
            bad = np.argwhere(~np.isfinite(window))

            if len(bad):
                bad_count += 1
                clip_id, start = ds.meta[i]

                print(
                    f"NON-FINITE: split={split}, "
                    f"clip={clip_id}, start={start}, "
                    f"count={len(bad)}"
                )

                for frame_idx, feature_idx in bad[:5]:
                    print(
                        f"  frame={frame_idx}, "
                        f"feature={feature_idx} "
                        f"({feature_names[feature_idx]}), "
                        f"value={window[frame_idx, feature_idx]}"
                    )

                if bad_count >= 10:
                    break

        print(f"Bad windows found (first 10 max): {bad_count}")

        positive_by_clip = defaultdict(int)
        for (clip_id, _), label in zip(ds.meta, ds.labels):
            positive_by_clip[clip_id] += int(label)

        fall_with_positive = sum(
            positive_by_clip[cid] > 0 for cid in fall_clips
        )

        print(f"\n[{split.upper()}]")
        print(f"Groups: {len(groups_by_split[split])}")
        print(f"Clips: {len(clip_ids)} (Fall={len(fall_clips)}, ADL={len(adl_clips)})")
        print(f"Windows: {len(ds)}")
        print(f"Positive: {positive}")
        print(f"Negative: {negative}")
        print(f"Positive rate: {positive / len(ds):.4f}" if len(ds) else "Positive rate: N/A")
        print(f"Shape OK: {shapes_ok}")
        print(f"All finite: {finite_ok}")
        print(f"Fall clips with positive windows: {fall_with_positive}/{len(fall_clips)}")

        if not ds:
            raise RuntimeError(f"{split} dataset is empty")
        if not shapes_ok or not finite_ok:
            raise RuntimeError(f"Invalid feature windows in {split}")
        if positive == 0 or negative == 0:
            raise RuntimeError(f"{split} does not contain both classes")
        if fall_with_positive != len(fall_clips):
            raise RuntimeError(f"Some Fall clips have no positive windows in {split}")

    print("\n=== GROUP OVERLAP ===")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups_by_split[a] & groups_by_split[b]
        print(f"{a} vs {b}: {len(overlap)}")

        if overlap:
            raise RuntimeError(f"Group leakage between {a} and {b}: {overlap}")

    print("\nPASS: Dataset audit completed successfully.")


if __name__ == "__main__":
    main()
