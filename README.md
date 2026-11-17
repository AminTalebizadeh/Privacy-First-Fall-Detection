# Privacy-Preserving Fall Detection from Skeleton Dynamics

A privacy-oriented fall detection pipeline based on human pose estimation, skeletal motion features, and temporal deep learning.

The system analyzes body keypoint dynamics instead of relying on raw RGB appearance for fall classification. It combines pose tracking, temporal feature extraction, a GRU-based classifier, and a debounced alert mechanism.

> **Project status:** Research prototype. The pipeline has been trained and evaluated on the UR Fall Detection (URFD) dataset. The current results are promising, but the small number of fall events in the test split means that further evaluation is required before making real-world safety claims.

## 1. Overview

The project follows this pipeline:

1. Extract human body keypoints from video using YOLOv8 Pose.
2. Track detected people across frames.
3. Extract motion and geometric features from skeletal keypoints.
4. Convert temporal features into fixed-length sliding windows.
5. Classify each window using a GRU-based neural network.
6. Convert consecutive positive predictions into fall alerts using thresholding and temporal debouncing.

The main design objective is to reduce dependence on identifiable visual appearance by focusing on skeletal dynamics.

**Privacy note:** Skeleton-based processing is privacy-oriented, but it does not automatically guarantee anonymity or eliminate all privacy risks. The project also includes checks intended to prevent raw video-frame storage in the processing pipeline.

## 2. Main Features

- Human pose estimation using YOLOv8 Pose.
- Multi-frame person tracking.
- Skeleton-based temporal feature extraction.
- GRU-based fall classification.
- Sliding-window inference.
- Threshold-based alert generation.
- Consecutive-positive debouncing and cooldown.
- Dataset auditing and group-disjoint evaluation.
- Synthetic data support for development and testing.
- Tests for feature extraction, tracking, alerts, and dataset handling.

## 3. Project Structure

```text
Privacy_Preserving_Fall_Detection_from_Skeleton_Dynamics/
│
├── configs/
│   ├── default.yaml
│   ├── smoke.yaml
│   └── urfd.yaml
│
├── data/
│   └── labels.example.csv
│
├── scripts/
│   ├── audit_dataset.py
│   ├── audit_no_frame_writes.py
│   ├── cache_skeletons.py
│   ├── download_urfd.py
│   ├── evaluate.py
│   ├── make_synthetic_cache.py
│   ├── run_realtime.py
│   ├── train.py
│   └── tune_alerts.py
│
├── src/
│   ├── data/
│   ├── features/
│   ├── model/
│   ├── pose/
│   └── realtime/
│
├── tests/
│
├── requirements.txt
├── .gitignore
└── README.md
```

Generated files such as datasets, skeleton caches, model checkpoints, and local environments are intentionally excluded from version control.

## 4. Dataset

The experiments use the UR Fall Detection (URFD) dataset.

The current dataset setup contains:

- 70 video clips.
- 70 corresponding labels.
- 40 group identifiers.
- Group-disjoint train, validation, and test splits.

| Split | Groups | Clips |
|---|---:|---:|
| Train | 28 | 49 |
| Validation | 6 | 11 |
| Test | 6 | 10 |
| **Total** | **40** | **70** |

The `group` field is used as a sequence/scene grouping proxy to reduce leakage between related clips.

**Important limitation:** These group identifiers are not confirmed subject identities. Therefore, this evaluation must not be described as person-independent validation.

### Dataset preparation

Dataset videos and labels are not included in this repository.

Place the dataset files in the expected local data directories and prepare the label CSV using the supplied example:

```text
data/labels.example.csv
```

The active configuration expects:

```yaml
data:
  cache_dir: cache/skeletons_dense_candidate
  labels_csv: data/labels.csv
```

Check the dataset preparation script options before downloading or processing:

```powershell
.venv\Scripts\python.exe scripts\download_urfd.py --help
```

Audit the dataset after preparation:

```powershell
.venv\Scripts\python.exe scripts\audit_dataset.py --help
```

## 5. Environment Setup

The project was developed and tested on Windows with Python 3.13 and an NVIDIA GPU.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build compatible with your NVIDIA driver and CUDA environment. Verify GPU availability:

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The training and evaluation experiments reported below were run on an NVIDIA GeForce RTX 3050 Laptop GPU.

## 6. Configuration

The main URFD configuration is:

```text
configs/urfd.yaml
```

Important settings include:

| Parameter | Value |
|---|---:|
| Window size | 30 frames |
| Window stride | 5 frames |
| Model | GRU |
| Input dimension | 55 |
| Hidden dimension | 64 |
| GRU layers | 2 |
| Dropout | 0.3 |
| Alert threshold | 0.60 |
| Consecutive positive windows | 4 |
| Alert cooldown | 10 seconds |

The active skeleton cache directory is:

```text
cache/skeletons_dense_candidate
```

The `default.yaml` and `smoke.yaml` configurations are retained for general and lightweight development workflows.

## 7. Training

Train the GRU model using the URFD configuration:

```powershell
.venv\Scripts\python.exe scripts\train.py `
  --config configs\urfd.yaml `
  --model gru `
  --seed 42 `
  --split-seed 42 `
  --ckpt-dir checkpoints\urfd_dense_gru_gpu
```

The training script uses the training and validation splits. The test split is reserved for final evaluation.

The best checkpoint from the reported GPU training run is:

```text
checkpoints/urfd_dense_gru_gpu/best_gru.pt
```

The best validation window F1 during training was approximately 0.747.

## 8. Alert Parameter Tuning

Alert threshold and consecutive-positive settings can be tuned using the validation split only:

```powershell
.venv\Scripts\python.exe scripts\tune_alerts.py `
  --config configs\urfd.yaml `
  --ckpt checkpoints\urfd_dense_gru_gpu\best_gru.pt
```

The tuning experiment found several configurations with the same event-level F1.

The selected operating point retained in the final configuration is:

```yaml
alerts:
  threshold: 0.6
  k_consecutive: 4
  cooldown_s: 10.0
```

At threshold 0.60, values of 3, 4, and 5 consecutive positive windows achieved the same validation event-level metrics in the reported experiment.

The value of 4 was retained for consistency with the final test evaluation.

**Important:** Alert parameters must not be selected by repeatedly optimizing against test results.

## 9. Evaluation

Evaluate on the validation split:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py `
  --config configs\urfd.yaml `
  --ckpt checkpoints\urfd_dense_gru_gpu\best_gru.pt `
  --split val `
  --top-fp 10
```

Evaluate on the test split:

```powershell
.venv\Scripts\python.exe scripts\evaluate.py `
  --config configs\urfd.yaml `
  --ckpt checkpoints\urfd_dense_gru_gpu\best_gru.pt `
  --split test `
  --top-fp 10
```

### Reported results

The following results were obtained with the GRU checkpoint and alert threshold 0.60.

**Window-level metrics**

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.848 | 0.642 |
| Recall | 0.667 | 0.773 |
| F1-score | 0.747 | 0.701 |
| True Positives | 28 | 34 |
| False Positives | 5 | 19 |
| False Negatives | 14 | 10 |

**Event-level metrics**

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 1.000 | 0.800 |
| Recall | 0.800 | 1.000 |
| F1-score | 0.889 | 0.889 |
| True Positives | 4 | 4 |
| False Positives | 0 | 1 |
| False Negatives | 1 | 0 |

### Interpretation

The test evaluation detected all four fall events in the test split, with one additional false-positive event.

Although the event-level F1-score was 0.889, the window-level test F1-score was 0.701, indicating that false-positive windows remain an important challenge.

**Small-sample limitation:** The test split contains only four positive fall events. Consequently, event-level metrics are sensitive to individual predictions and should not be interpreted as strong evidence of generalization.

The current experiment does not establish robustness across unseen people, environments, camera viewpoints, or real-world deployment conditions.

## 10. Testing

Run the project test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q tests
```

The latest verified test result:

```text
39 passed
```

Check Python syntax:

```powershell
.venv\Scripts\python.exe -m compileall -q src tests scripts
```

Check Git whitespace errors:

```powershell
git diff --check
```

## 11. Privacy Considerations

The project is designed around skeletal keypoints and motion features rather than storing raw RGB frames as model inputs.

Privacy-related considerations include:

- Avoiding unnecessary storage of raw video frames.
- Using skeletal representations for temporal classification.
- Keeping local datasets and generated artifacts out of public version control.
- Providing an audit script for checking frame-write behavior.

The privacy audit can be inspected and run using:

```powershell
.venv\Scripts\python.exe scripts\audit_no_frame_writes.py --help
```

Skeleton data can still contain sensitive behavioral information. Therefore, access control, retention policies, and secure handling remain necessary.

## 12. Limitations and Future Work

Current limitations:

- Small number of fall events in the test split.
- Window-level false positives remain relatively high.
- Group identifiers are not verified subject identities.
- No demonstrated person-independent generalization.
- No comprehensive real-world deployment validation.
- No established clinical or safety certification.

Potential future improvements:

- Evaluation on larger and more diverse datasets.
- Subject-independent validation using verified subject identities.
- Improved false-positive suppression.
- Robustness testing across camera viewpoints and environments.
- Real-time performance benchmarking on different hardware.
- Additional privacy and data-retention audits.

## 13. Reproducibility and Artifact Policy

The following local artifacts are not distributed through Git:

```text
.venv/
cache/
checkpoints/
data/labels.csv
data/raw/
data/videos/
*.pt
*.pth
__pycache__/
.pytest_cache/
```

To reproduce the experiments, prepare the dataset, install dependencies, generate the skeleton cache, train the model, and evaluate using the commands above.

Model checkpoints and dataset artifacts must be generated or obtained separately.

## 14. Disclaimer

This repository is a research and development prototype.

It is not a certified medical device or a replacement for professional monitoring systems. It must not be relied upon as the sole mechanism for detecting falls or ensuring personal safety.

---

**Project:** Privacy-Preserving Fall Detection from Skeleton Dynamics  
**Primary model:** GRU  
**Dataset:** URFD  
**Framework:** PyTorch  
**Evaluation:** Group-disjoint train/validation/test split