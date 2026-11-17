#!/usr/bin/env python3
"""Train the temporal fall classifier with separately controlled seeds."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import SkeletonWindowDataset, load_labels, split_groups  # noqa: E402
from src.model.nets import FocalLoss, build_model  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    tp = fp = fn = tn = 0
    for x, y in loader:
        pred = torch.sigmoid(model(x.to(device))).reshape(-1).cpu().numpy() >= threshold
        truth = y.numpy().astype(bool)
        tp += int((pred & truth).sum())
        fp += int((pred & ~truth).sum())
        fn += int((~pred & truth).sum())
        tn += int((~pred & ~truth).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def make_split(cfg, split: str, split_seed: int):
    labels_csv = Path(cfg["data"]["labels_csv"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    if not cache_dir.is_dir() or not any(cache_dir.glob("*.npz")):
        raise FileNotFoundError(f"No skeleton cache found in {cache_dir}")
    return SkeletonWindowDataset(
        cache_dir, labels_csv, split, cfg["data"]["split"], split_seed,
        cfg["window"]["size"], cfg["window"]["stride"],
        cfg["window"]["positive_overlap"], cfg["features"]["velocity_fps"],
        cfg["features"].get("min_kpt_conf", 0.30),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["gru", "cnn"], default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="training seed; vary this between repeated runs")
    parser.add_argument("--split-seed", type=int, default=None,
                        help="data partition seed; keep fixed when comparing runs")
    parser.add_argument("--ckpt-dir", default=None,
                        help="separate checkpoint directory for this run")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.model:
        cfg["model"]["type"] = args.model
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    split_seed = int(cfg["data"].get("split_seed", cfg["seed"])
                     if args.split_seed is None else args.split_seed)
    cfg["data"]["split_seed"] = split_seed
    if args.ckpt_dir:
        cfg["train"]["ckpt_dir"] = args.ckpt_dir
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = make_split(cfg, "train", split_seed)
    val_set = make_split(cfg, "val", split_seed)
    if not len(train_set) or not len(val_set):
        raise RuntimeError(f"Empty split(s): train={len(train_set)} val={len(val_set)}")
    print(f"[data] train={len(train_set)} ({train_set.pos_fraction:.1%} positive) "
          f"val={len(val_set)} ({val_set.pos_fraction:.1%} positive)")

    train_cfg = cfg["train"]
    workers = 0 if sys.platform == "win32" else int(train_cfg["num_workers"])
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"],
                              shuffle=True, num_workers=workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=train_cfg["batch_size"],
                            shuffle=False, num_workers=workers)
    model = build_model(cfg).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    if parameter_count >= 1_000_000:
        raise ValueError(f"Temporal classifier has {parameter_count:,} parameters; expected < 1M")

    criterion = FocalLoss(train_cfg["focal_alpha"], train_cfg["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    checkpoint_dir = Path(train_cfg["ckpt_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(cfg["data"]["labels_csv"])
    group_assignment = split_groups([item["group"] for item in labels.values()],
                                    cfg["data"]["split"], split_seed)
    manifest = {
        "training_seed": int(cfg["seed"]),
        "split_seed": split_seed,
        "fractions": cfg["data"]["split"],
        "group_assignment": group_assignment,
        "group_id_semantics": "Dataset group field; verify identity semantics before claiming subject independence.",
    }
    with (checkpoint_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    threshold = float(cfg["alerts"]["threshold"])
    best_f1, patience = -1.0, 0
    checkpoint_path = checkpoint_dir / f"best_{cfg['model']['type']}.pt"
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(x)
        metrics = evaluate(model, val_loader, device, threshold=threshold)
        print(f"epoch {epoch:03d} loss={loss_sum / len(train_set):.4f} "
              f"val P={metrics['precision']:.3f} R={metrics['recall']:.3f} "
              f"F1@{threshold:.3f}={metrics['f1']:.3f}")
        if metrics["f1"] > best_f1:
            best_f1, patience = metrics["f1"], 0
            torch.save({
                "state_dict": model.state_dict(), "config": cfg,
                "val_metrics": metrics, "epoch": epoch,
                "training_seed": int(cfg["seed"]), "split_seed": split_seed,
                "threshold_used_for_selection": threshold,
            }, checkpoint_path)
        else:
            patience += 1
            if patience >= int(train_cfg["early_stop_patience"]):
                print(f"[early-stop] no validation F1 gain for {patience} epochs")
                break
    print(f"[done] best validation F1={best_f1:.3f}; checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
