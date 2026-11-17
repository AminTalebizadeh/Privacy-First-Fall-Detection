#!/usr/bin/env python3
"""Tune alert threshold and debounce on validation split only."""

from __future__ import annotations

import sys
from pathlib import Path
import argparse

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import load_labels, split_groups
from src.features.sequence import sequence_features, sliding_windows, window_label
from src.model.nets import build_model
from src.realtime.alerts import AlertManager


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    parser = argparse.ArgumentParser(
        description="Tune alert threshold and debounce on validation split only."
    )
    parser.add_argument(
        "--config",
        default="configs/urfd.yaml",
        help="Path to YAML config",
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/urfd_dense_gru_gpu/best_gru.pt",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    ckpt_path = Path(args.ckpt)

    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = build_model(checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])

    labels = load_labels(cfg["data"]["labels_csv"])
    assignment = split_groups(
        [v["group"] for v in labels.values()],
        cfg["data"]["split"],
        cfg["seed"],
    )

    window_size = cfg["window"]["size"]
    stride = cfg["window"]["stride"]
    tolerance = cfg["eval"]["event_tolerance_s"]
    cooldown = cfg["alerts"]["cooldown_s"]

    # Cache predictions in memory; inference runs only once.
    clips = []

    cache_dir = ROOT / cfg["data"]["cache_dir"]
    for npz_path in sorted(cache_dir.glob("*.npz")):
        clip_id = npz_path.stem
        if clip_id not in labels:
            continue
        if assignment[labels[clip_id]["group"]] != "val":
            continue

        with np.load(npz_path) as z:
            keypoints = z["keypoints"]
            frame_idx = z["frame_idx"]
            fps = float(z["fps"]) if "fps" in z else 30.0

        features = sequence_features(keypoints, fps=fps)
        span = labels[clip_id]["fall_span"]
        windows = []

        for start, win in sliding_windows(features, window_size, stride):
            y = int(window_label(
                start, window_size, span,
                cfg["window"]["positive_overlap"]
            ))

            with torch.no_grad():
                prob = float(torch.sigmoid(
                    model(torch.from_numpy(win[None]).to(device))
                ).item())

            t_end = float(
                frame_idx[min(start + window_size - 1, len(frame_idx) - 1)]
            ) / fps

            windows.append((prob, y, t_end))

        clips.append((clip_id, span, fps, windows))

    if not clips:
        raise RuntimeError("No validation clips found. Check split and cache.")

    thresholds = [round(x, 2) for x in np.arange(0.30, 0.801, 0.05)]
    k_values = range(1, 7)
    results = []

    for threshold in thresholds:
        for k in k_values:
            wtp = wfp = wfn = 0
            etp = efp = efn = 0

            for clip_id, span, fps, windows in clips:
                manager = AlertManager(
                    threshold=threshold,
                    k_consecutive=k,
                    cooldown_s=cooldown,
                    log_path=None,
                )
                alert_times = []

                for prob, y, t_end in windows:
                    pred = prob >= threshold
                    wtp += int(pred and y == 1)
                    wfp += int(pred and y == 0)
                    wfn += int(not pred and y == 1)

                    event = manager.update(0, prob, now=t_end)
                    if event is not None:
                        alert_times.append(t_end)

                if span is not None:
                    lo = span[0] / fps - tolerance
                    hi = span[1] / fps + tolerance
                    hit = any(lo <= t <= hi for t in alert_times)
                    etp += int(hit)
                    efn += int(not hit)
                    efp += sum(not (lo <= t <= hi) for t in alert_times)
                else:
                    efp += len(alert_times)

            wp, wr, wf = prf(wtp, wfp, wfn)
            ep, er, ef = prf(etp, efp, efn)

            results.append({
                "threshold": threshold,
                "k": k,
                "wp": wp, "wr": wr, "wf": wf,
                "ep": ep, "er": er, "ef": ef,
                "etp": etp, "efp": efp, "efn": efn,
            })

    results.sort(
        key=lambda x: (x["ef"], x["er"], x["ep"], x["wf"]),
        reverse=True,
    )

    print(f"\nValidation clips: {len(clips)}")
    print(f"Device: {device}")
    print(f"Cooldown: {cooldown}s")
    print("\nTop 12 configurations (event F1, then recall):")
    print(" threshold  k | Event P   R   F1 | Window F1 | TP FP FN")

    for r in results[:12]:
        print(
            f"   {r['threshold']:.2f}     {r['k']} |"
            f"  {r['ep']:.3f} {r['er']:.3f} {r['ef']:.3f} |"
            f"   {r['wf']:.3f}   |"
            f" {r['etp']:2d} {r['efp']:2d} {r['efn']:2d}"
        )

    best = results[0]
    print("\nRecommended validation configuration:")
    print(f"threshold       = {best['threshold']:.2f}")
    print(f"k_consecutive   = {best['k']}")
    print(f"event F1        = {best['ef']:.3f}")
    print(f"event P/R       = {best['ep']:.3f} / {best['er']:.3f}")
    print(f"window F1       = {best['wf']:.3f}")
    print("\nNo checkpoint or cache files were modified.")
    print("Selection used validation split only.")


if __name__ == "__main__":
    main()
