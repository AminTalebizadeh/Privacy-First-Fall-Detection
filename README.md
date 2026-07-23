# Privacy-First Fall Detection (Skeleton-Only Pipeline)

Real-time fall detection from 2-D human pose: YOLOv8n-pose keypoints →
tiny temporal classifier (GRU, 1D-CNN baseline) → debounced JSON alerts.
**No raw video frame is ever persisted — the entire downstream pipeline
runs on skeleton data only.**


## Repo layout

```
configs/default.yaml        every tunable (window, model, alerts K/thr/cooldown, split)
src/pose/                   YOLOv8n-pose wrapper + lightweight IoU/centroid tracker
src/features/               normalization + derived features (pure numpy, unit-tested)
src/model/                  GRU, 1D-CNN baseline, focal loss
src/data/                   .npz window dataset, group-disjoint splitter
src/realtime/               debounced AlertManager (privacy-safe JSON log)
scripts/cache_skeletons.py  dataset videos -> skeleton cache (offline)
scripts/train.py            training (seeds, early stopping, best ckpt)
scripts/evaluate.py         window + event metrics, hardest-FP analysis
scripts/run_realtime.py     webcam/file demo (privacy-mode canvas, sparkline, banner)
scripts/audit_no_frame_writes.py   CI grep: no image-writing code allowed
tests/                      25 unit tests (numpy-only; run anywhere)
```

## Quickstart

```bash
pip install -r requirements.txt          

# 1. Get the UR Fall Detection dataset (http://fenix.ur.edu.pl/~mkepski/ds/uf.html)
#    -> put cam0 clips in data/videos/ as fall-XX-cam0.mp4 / adl-XX-cam0.mp4
#    -> build data/labels.csv (see data/labels.example.csv):
#         clip_id,fall_start_frame,fall_end_frame,group
#       ADL clips use -1,-1. URFD has no formal subject IDs, so `group`
#       is your leakage unit: assign clips of the same actor/room the
#       same group; the splitter guarantees group-disjoint train/val/test.
#       (Le2i: group by scene — Home/Coffee_room/Office/Lecture_room.)

# 2. Offline pose extraction -> skeleton cache (privacy step)
python scripts/cache_skeletons.py --videos data/videos

# 3. Train both models
python scripts/train.py --model gru
python scripts/train.py --model cnn

# 4. Evaluate on held-out groups
python scripts/evaluate.py --ckpt checkpoints/best_gru.pt --split test
python scripts/evaluate.py --ckpt checkpoints/best_cnn.pt --split test

# 5. Real-time demo (q to quit)
python scripts/run_realtime.py --source 0            # privacy mode (default)
python scripts/run_realtime.py --source clip.mp4 --no-privacy   # debug view

# Tests + privacy audit
python -m pytest tests/ -v          # or: python -m unittest discover -s tests
python scripts/audit_no_frame_writes.py
```

## Metrics (fill in after training on your machine)

Templates below — `scripts/evaluate.py` prints every number. Acceptance
criteria: **event-level F1 ≥ 0.90** on held-out groups, **≥ 15 FPS** at
640 px on the RTX 3050 (FPS shown live in the demo overlay).

Window-level (test split, threshold = 0.60):

| model      | precision | recall | F1  | params |
|------------|-----------|--------|-----|--------|
| GRU 2×64   |    —      |   —    |  —  | ~49 k  |
| 1D-CNN     |    —      |   —    |  —  | ~180 k |

Event-level (alert within ±2 s of GT fall; K=3, cooldown 10 s):

| model      | precision | recall | F1  |
|------------|-----------|--------|-----|
| GRU 2×64   |    —      |   —    |  —  |
| 1D-CNN     |    —      |   —    |  —  |

Hardest false positives: `evaluate.py` lists the top negative windows by
predicted probability with clip + start frame. Expected confusions and
mitigations:
- **Sitting down fast** — shares the downward hip-velocity spike but not
  the terminal trunk angle (~π/2) or aspect ratio > 1; if it dominates,
  raise `alerts.k_consecutive` (a sit ends upright, so positives don't
  stay consecutive) before touching the threshold.
- **Lying down intentionally** — shares the end state but not the
  velocity/deceleration spike; if it dominates, the model is over-using
  static posture — add intentional-lie ADL clips to training.

## Privacy guarantees

What **is** stored:
- `cache/skeletons/*.npz` — keypoint coordinates+confidences, frame
  indices, fps. Nothing else.
- `checkpoints/*.pt` — model weights + config.
- `alerts.jsonl` — JSON events with exactly `{event, timestamp,
  track_id, confidence}`.

What is **never** stored, by construction:
- Raw frames, crops, thumbnails, or any pixel data, in any format.
  Pixels exist only in transient memory inside the pose front-end and
  are discarded the moment keypoints are extracted.
- There is **no recording code path anywhere in the repo**, enforced
  twice: `tests/test_alerts_and_tracker.py::TestPrivacySchema` asserts
  the event-log schema is exact, image-free, and length-bounded
  (acceptance criterion), and `scripts/audit_no_frame_writes.py` fails
  CI if any image/video-writing call appears in `src/` or `scripts/`.
- Privacy display mode (default) renders skeletons on a blank canvas —
  the live video is never even shown. `--no-privacy` shows (but still
  never records) frames, for debugging only.

Residual considerations (be honest with stakeholders): skeleton
trajectories are still behavioral data about identifiable people in
their home; the JSON log timestamps activity. Retention policy for the
cache and log is deployment policy, not code.

## Reproducibility

`seed: 42` seeds python/numpy/torch (+cuda). Group split assignment is a
deterministic function of (group set, seed). Best checkpoints embed the
full config. Exact package versions: `pip freeze > requirements.lock`.
