# Privacy-First Fall Detection

Fall detection from 2D pose only. The idea is simple: run YOLOv8n-pose on
each frame, throw the pixels away, and classify short skeleton windows with
a tiny temporal model (GRU, plus a 1D-CNN baseline). Alerts go out as JSON.
Nothing in this repo writes raw video to disk.

I built this around the [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)
dataset (cam0 RGB). Offline, every clip becomes a `.npz` of keypoints — that
cache is what training actually sees.

```
video frame  -->  YOLOv8n-pose  -->  tracker  -->  (17,3) keypoints
                                                      |
                                                      v
                                         normalize + dynamics  -->  (T,55)
                                                      |
                                                      v
                                         window (30 @ stride 5)
                                                      |
                                                      v
                                         GRU / 1D-CNN  -->  p(fall)
                                                      |
                                                      v
                                         AlertManager (K positives + cooldown)
                                                      |
                                                      v
                                         alerts.jsonl
```

## What's in the repo

| path | what it is |
|------|------------|
| `configs/default.yaml` | main knobs (window, model, alerts, split) |
| `configs/urfd.yaml` | same as default, but `k_consecutive: 4` — better on URFD ADL false alarms |
| `configs/smoke.yaml` | short CPU-friendly run for sanity checks |
| `src/pose/` | YOLO wrapper + IoU/centroid tracker |
| `src/features/` | torso normalize, trunk angle, hip vel/acc (numpy) |
| `src/model/` | GRU, 1D-CNN, focal loss |
| `src/data/` | window dataset + group-disjoint split |
| `src/realtime/` | debounced alerts + JSON log |
| `scripts/download_urfd.py` | pulls cam0 MP4s + builds `data/labels.csv` |
| `scripts/cache_skeletons.py` | videos → `cache/skeletons/*.npz` |
| `scripts/train.py` / `evaluate.py` / `run_realtime.py` | train, eval, demo |
| `scripts/make_synthetic_cache.py` | fake skeletons if you don't want to download URFD yet |
| `scripts/audit_no_frame_writes.py` | fails if anyone adds `imwrite` / `VideoWriter` etc. |
| `tests/` | 25 unit tests (features, tracker, alert schema) |

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Torch is CPU-fine for training here; for the live demo a CUDA build helps
FPS. Pose weights (`yolov8n-pose.pt`) download on first use.

## Run on real URFD data

```bash
# 1) download 70 cam0 clips + labels (from the official URFD mirror)
python scripts/download_urfd.py

# 2) pose extraction (this is the privacy cut: after this, no pixels needed)
python scripts/cache_skeletons.py --videos data/videos --overwrite

# 3) train
python scripts/train.py --model gru
python scripts/train.py --model cnn

# 4) evaluate (use urfd.yaml — K=4)
python scripts/evaluate.py --config configs/urfd.yaml --ckpt checkpoints/best_gru.pt --split test
python scripts/evaluate.py --config configs/urfd.yaml --ckpt checkpoints/best_cnn.pt --split test

# 5) demo — privacy mode draws skeletons on a blank canvas (q to quit)
python scripts/run_realtime.py --config configs/urfd.yaml --ckpt checkpoints/best_gru.pt --source 0
python scripts/run_realtime.py --config configs/urfd.yaml --ckpt checkpoints/best_gru.pt --source data/videos/fall-01-cam0.mp4
# debug overlay on the actual frames (still not recorded):
python scripts/run_realtime.py --ckpt checkpoints/best_gru.pt --source data/videos/fall-01-cam0.mp4 --no-privacy
```

`download_urfd.py` names clips `fall-XX-cam0.mp4` / `adl-XX-cam0.mp4` and
writes `data/labels.csv`. Fall spans come from the official posture CSV
(label `0` = falling, `1` = lying). Groups are `group01`…`group40` so
`fall-N` and `adl-N` stay together — URFD doesn't ship subject IDs, so this
is a stand-in to keep train/val/test from leaking the same scene.

If you already have videos, skip the downloader and follow
`data/labels.example.csv`.

### No dataset handy?

```bash
python scripts/make_synthetic_cache.py
python scripts/train.py --config configs/smoke.yaml --model gru
```

Useful for checking imports and the train loop. Don't treat those numbers
as real performance.

## Results on URFD (group-disjoint test)

Trained locally on the full cam0 set (70 clips). Window threshold `0.60`.
Event score: alert within ±2 s of the annotated fall.

**Window-level**

| model    | P     | R     | F1    | params |
|----------|-------|-------|-------|--------|
| GRU 2×64 | 0.683 | 0.700 | 0.691 | ~48 k  |
| 1D-CNN   | 0.659 | 0.675 | 0.667 | ~142 k |

**Event-level** (`configs/urfd.yaml`, K=4, cooldown 10 s)

| model    | P     | R     | F1    |
|----------|-------|-------|-------|
| GRU 2×64 | 1.000 | 1.000 | 1.000 |
| 1D-CNN   | 0.667 | 1.000 | 0.800 |

With the default K=3 the GRU event F1 was ~0.89 — one extra ADL false
alarm. Bumping K to 4 cleared it (same advice as below for fast sit-downs).
Test set is small (a handful of held-out fall events), so take the perfect
score with a grain of salt; re-check if you change the split seed.

`evaluate.py` also dumps the hardest false-positive windows. On this run
they clustered on a few ADL clips (`adl-02`, `adl-34`, `adl-39`) — looks
like sit / lie transitions more than random noise.

What I'd tweak next if FPs come back:
- fast sit → raise `alerts.k_consecutive` first (don't jump straight to the threshold)
- intentional lie → the model is leaning on posture; add more of those ADLs to train

Target I care about in practice: event F1 ≥ 0.90 and live demo ≥ ~15 FPS
at 640 px when a GPU is available.

## Privacy (what actually gets saved)

Kept on disk:
- `cache/skeletons/*.npz` — keypoints, frame index, fps
- `checkpoints/*.pt` — weights + the config used to train them
- `alerts.jsonl` — `{event, timestamp, track_id, confidence}` only

Not kept:
- frames, crops, thumbnails, recordings — there is no save path for them

Two checks for that: the alert schema unit test, and
`scripts/audit_no_frame_writes.py` which greps `src/` + `scripts/` for
image/video writers.

Honest caveat: skeletons are still personal motion data, and the JSON log
has timestamps. How long you keep those is a deployment choice, not
something the code decides.

## Tests

```bash
python -m pytest tests/ -v
python scripts/audit_no_frame_writes.py
```

## Notes

- Seed is `42` in the yaml; splits are deterministic given the group list.
- `.gitignore` excludes videos, cache, checkpoints, and `labels.csv` — regenerate with the scripts above.
- Cite URFD if you publish with it: Kwolek & Kepski, CMPB 2014.
