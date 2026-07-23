#!/usr/bin/env python3
"""Evaluate on held-out (group-disjoint) test split.

Reports:
  1. Window-level precision / recall / F1.
  2. Event-level metrics: simulate the AlertManager over each test clip;
     a ground-truth fall counts as DETECTED if any alert fires within
     +-tolerance seconds of the annotated fall span; alerts outside all
     tolerance zones are event-level false positives.
  3. Hardest false positives: top-N negative windows by predicted
     probability, listed with clip + frame span so you can eyeball the
     "sitting down fast" / "lying down intentionally" confusions.

    python scripts/evaluate.py --ckpt checkpoints/best_gru.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import load_labels, split_groups            # noqa: E402
from src.features.sequence import (sequence_features,             # noqa: E402
                                   sliding_windows, window_label)
from src.model.nets import build_model                            # noqa: E402
from src.realtime.alerts import AlertManager                      # noqa: E402


def prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", default="checkpoints/best_gru.pt")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--top-fp", type=int, default=15)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=dev)
    model = build_model(ckpt["config"]).to(dev).eval()
    model.load_state_dict(ckpt["state_dict"])

    labels = load_labels(cfg["data"]["labels_csv"])
    assign = split_groups([v["group"] for v in labels.values()],
                          cfg["data"]["split"], cfg["seed"])
    W, S = cfg["window"]["size"], cfg["window"]["stride"]
    thr = cfg["alerts"]["threshold"]
    tol = cfg["eval"]["event_tolerance_s"]

    win_tp = win_fp = win_fn = 0
    ev_tp = ev_fn = ev_fp = 0
    hard_fps: list[tuple[float, str, int]] = []

    for npz_path in sorted(Path(cfg["data"]["cache_dir"]).glob("*.npz")):
        clip_id = npz_path.stem
        if clip_id not in labels or assign[labels[clip_id]["group"]] != args.split:
            continue
        with np.load(npz_path) as z:
            kpts, fidx = z["keypoints"], z["frame_idx"]
            fps = float(z["fps"]) if "fps" in z else 30.0
        span = labels[clip_id]["fall_span"]
        feats = sequence_features(kpts, fps=fps)

        # fresh alert manager per clip; drive it in window order with
        # synthetic timestamps derived from frame indices.
        am = AlertManager(thr, cfg["alerts"]["k_consecutive"],
                          cfg["alerts"]["cooldown_s"], log_path=None)
        alert_times: list[float] = []

        for start, win in sliding_windows(feats, W, S):
            y = window_label(start, W, span, cfg["window"]["positive_overlap"])
            with torch.no_grad():
                p = float(torch.sigmoid(
                    model(torch.from_numpy(win[None]).to(dev))).item())
            pred = p >= thr
            win_tp += pred and y; win_fp += pred and not y
            win_fn += (not pred) and y
            if pred and not y:
                hard_fps.append((p, clip_id, int(fidx[min(start, len(fidx)-1)])))
            t_end = fidx[min(start + W - 1, len(fidx) - 1)] / fps
            ev = am.update(0, p, now=t_end)
            if ev is not None:
                alert_times.append(t_end)

        # ---- event-level bookkeeping per clip ------------------------
        if span is not None:
            lo_t, hi_t = span[0] / fps - tol, span[1] / fps + tol
            hit = any(lo_t <= t <= hi_t for t in alert_times)
            ev_tp += int(hit); ev_fn += int(not hit)
            ev_fp += sum(1 for t in alert_times if not (lo_t <= t <= hi_t))
        else:
            ev_fp += len(alert_times)

    wp, wr, wf1 = prf(win_tp, win_fp, win_fn)
    ep, er, ef1 = prf(ev_tp, ev_fp, ev_fn)
    print(f"\n== window-level ({args.split}) ==")
    print(f"P {wp:.3f}  R {wr:.3f}  F1 {wf1:.3f}   "
          f"(tp {win_tp} fp {win_fp} fn {win_fn})")
    print(f"\n== event-level (+-{tol:.0f}s) ==")
    print(f"P {ep:.3f}  R {er:.3f}  F1 {ef1:.3f}   "
          f"(tp {ev_tp} fp {ev_fp} fn {ev_fn})")
    if ef1 < 0.90:
        print("  [!] below acceptance criterion F1 >= 0.90 — tune "
              "alerts.threshold / k_consecutive or retrain")

    print(f"\n== {args.top_fp} hardest false-positive windows ==")
    print("prob   clip                 start_frame")
    for p, cid, f0 in sorted(hard_fps, reverse=True)[:args.top_fp]:
        print(f"{p:.3f}  {cid:<20} {f0}")
    print("\nInspect these clips around the listed frames: expect fast "
          "sit-downs and intentional lie-downs. If they dominate, raise "
          "window.positive_overlap slightly or add such ADL clips to train.")


if __name__ == "__main__":
    main()
