#!/usr/bin/env python3
"""Train the temporal fall classifier (GRU or 1D-CNN baseline).

    python scripts/train.py --config configs/default.yaml --model gru
    python scripts/train.py --config configs/default.yaml --model cnn

Early stopping on validation F1; best checkpoint saved to
checkpoints/best_<model>.pt together with the config for reproducibility.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import SkeletonWindowDataset   # noqa: E402
from src.model.nets import FocalLoss, build_model    # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    tp = fp = fn = tn = 0
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device))).cpu().numpy() >= threshold
        y = y.numpy().astype(bool)
        tp += int((p & y).sum()); fp += int((p & ~y).sum())
        fn += int((~p & y).sum()); tn += int((~p & ~y).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def make_split(cfg, split):
    labels_csv = Path(cfg["data"]["labels_csv"])
    cache_dir = Path(cfg["data"]["cache_dir"])
    if not labels_csv.is_file():
        raise FileNotFoundError(
            f"Missing {labels_csv}. Copy data/labels.example.csv -> "
            f"data/labels.csv (or run: python scripts/make_synthetic_cache.py)")
    if not cache_dir.is_dir() or not any(cache_dir.glob("*.npz")):
        raise FileNotFoundError(
            f"Empty skeleton cache at {cache_dir}. Run "
            f"`python scripts/cache_skeletons.py --videos data/videos` "
            f"or `python scripts/make_synthetic_cache.py` first.")
    return SkeletonWindowDataset(
        cache_dir, labels_csv, split,
        cfg["data"]["split"], cfg["seed"],
        cfg["window"]["size"], cfg["window"]["stride"],
        cfg["window"]["positive_overlap"], cfg["features"]["velocity_fps"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["gru", "cnn"], default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.model:
        cfg["model"]["type"] = args.model
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va = make_split(cfg, "train"), make_split(cfg, "val")
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(
            f"Empty split(s): train={len(tr)} val={len(va)}. "
            "Need enough distinct `group` values in labels.csv for the "
            "configured train/val/test fractions.")
    print(f"[data] train {len(tr)} windows ({tr.pos_fraction:.1%} pos) | "
          f"val {len(va)} windows ({va.pos_fraction:.1%} pos)")

    tcfg = cfg["train"]
    # Windows spawn + torch DataLoader workers often hang/deadlock in smoke runs.
    n_workers = 0 if sys.platform == "win32" else tcfg["num_workers"]
    dl_tr = DataLoader(tr, tcfg["batch_size"], shuffle=True,
                       num_workers=n_workers, drop_last=False)
    dl_va = DataLoader(va, tcfg["batch_size"], num_workers=n_workers)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {cfg['model']['type']} — {n_params/1e3:.1f} k params")
    assert n_params < 1_000_000, "spec: temporal classifier must be < 1 M params"

    crit = FocalLoss(tcfg["focal_alpha"], tcfg["focal_gamma"])
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])

    ckpt_dir = Path(tcfg["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_f1, patience = -1.0, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        loss_sum = 0.0
        for x, y in dl_tr:
            opt.zero_grad()
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(x)
        m = evaluate(model, dl_va, device)
        print(f"epoch {epoch:03d} loss {loss_sum/len(tr):.4f} | "
              f"val P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}")

        if m["f1"] > best_f1:
            best_f1, patience = m["f1"], 0
            torch.save({"state_dict": model.state_dict(), "config": cfg,
                        "val_metrics": m, "epoch": epoch},
                       ckpt_dir / f"best_{cfg['model']['type']}.pt")
        else:
            patience += 1
            if patience >= tcfg["early_stop_patience"]:
                print(f"[early-stop] no val-F1 gain for {patience} epochs")
                break
    print(f"[done] best val F1 = {best_f1:.3f} -> "
          f"{ckpt_dir / ('best_' + cfg['model']['type'] + '.pt')}")


if __name__ == "__main__":
    main()
