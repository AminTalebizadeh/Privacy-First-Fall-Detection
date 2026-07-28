#!/usr/bin/env python3
"""Download UR Fall Detection (cam0) MP4s and build data/labels.csv.

Official source: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
Uses posture CSVs (label 0=falling, 1=lying) to set fall frame spans.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data/"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
VIDEOS = ROOT / "data" / "videos"
LABELS = ROOT / "data" / "labels.csv"


def download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  skip {dest.name}")
        return
    print(f"  get  {dest.name} ...", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"  done {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def fall_span_from_csv(csv_path: Path) -> dict[str, tuple[int, int]]:
    """Map sequence stem (fall-01) -> inclusive 0-based (start, end) frames.

    Event span = first falling frame (label 0) through last lying frame
    (label 1) when present; otherwise through last falling frame.
    """
    by_seq: dict[str, list[tuple[int, int]]] = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            seq, frame, lab = row[0], int(row[1]), int(row[2])
            by_seq[seq].append((frame, lab))  # CSV frames are 1-based

    spans: dict[str, tuple[int, int]] = {}
    for seq, frames in by_seq.items():
        zeros = [fr for fr, lab in frames if lab == 0]
        ones = [fr for fr, lab in frames if lab == 1]
        if not zeros:
            continue
        start_1 = min(zeros)
        end_1 = max(ones) if ones else max(zeros)
        # convert to 0-based OpenCV indices
        spans[seq] = (start_1 - 1, end_1 - 1)
    return spans


def write_labels(fall_spans: dict[str, tuple[int, int]],
                 fall_ids: list[str], adl_ids: list[str]) -> None:
    """Group fall-N with adl-N when both exist (leakage unit ≈ scene/actor proxy)."""
    rows = [("clip_id", "fall_start_frame", "fall_end_frame", "group")]
    for seq in fall_ids:
        lo, hi = fall_spans[seq]
        n = int(seq.split("-")[1])
        rows.append((f"{seq}-cam0", lo, hi, f"group{n:02d}"))
    for seq in adl_ids:
        n = int(seq.split("-")[1])
        rows.append((f"{seq}-cam0", -1, -1, f"group{n:02d}"))
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[labels] wrote {len(rows) - 1} clips -> {LABELS}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--falls-only", action="store_true",
                    help="download only fall clips (faster smoke)")
    ap.add_argument("--max-clips", type=int, default=None)
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)

    print("[1/3] annotation CSVs")
    download(BASE + "urfall-cam0-falls.csv", RAW / "urfall-cam0-falls.csv", args.force)
    download(BASE + "urfall-cam0-adls.csv", RAW / "urfall-cam0-adls.csv", args.force)
    fall_spans = fall_span_from_csv(RAW / "urfall-cam0-falls.csv")

    fall_ids = [f"fall-{i:02d}" for i in range(1, 31)]
    adl_ids = [] if args.falls_only else [f"adl-{i:02d}" for i in range(1, 41)]
    seqs = [(s, True) for s in fall_ids] + [(s, False) for s in adl_ids]
    if args.max_clips is not None:
        seqs = seqs[: args.max_clips]

    print(f"[2/3] downloading {len(seqs)} cam0 MP4s -> {VIDEOS}")
    for seq, _ in seqs:
        name = f"{seq}-cam0.mp4"
        download(BASE + name, VIDEOS / name, args.force)

    # labels only for successfully downloaded clips that we expect
    have = {p.stem for p in VIDEOS.glob("*-cam0.mp4")}
    fall_have = [s for s in fall_ids if f"{s}-cam0" in have and s in fall_spans]
    adl_have = [s for s in adl_ids if f"{s}-cam0" in have]
    print("[3/3] building labels.csv")
    write_labels(fall_spans, fall_have, adl_have)
    print("[done] next: python scripts/cache_skeletons.py --videos data/videos")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
