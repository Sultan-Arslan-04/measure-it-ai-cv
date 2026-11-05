# Measure It — AI / Computer Vision

**Assessment:** xis.ai Technical Assessment, Version 2.0
**Objective:** Object segmentation, camera calibration and metric measurement.

## Project status

This repository is under development. No calibration result, model metric or physical accuracy result is claimed yet.

## Objective

Given a new photograph containing one UNO card and a known-size ArUco marker in the same physical plane, the system will:

1. Undistort the photograph using saved OpenCV camera parameters.
2. Segment the UNO card using a trained instance-segmentation model.
3. Detect the ArUco reference marker automatically.
4. Convert the card outline from image coordinates to millimetres using the reference geometry.
5. Save a visual overlay.
6. Report width, height and confidence.
7. Refuse unreliable measurements.

## Hardware

| Component | Specification |
|---|---|
| CPU | Intel Core i5-7300U @ 2.60 GHz |
| RAM | 8 GB |
| Local GPU | Intel HD Graphics 620; no CUDA GPU |
| OS | Windows 10 Pro 22H2 |
| Camera | HONOR X6c rear main camera |
| Camera zoom | 1x |
| Actual image resolution | Pending original-image metadata inspection |
| Training hardware | Kaggle GPU planned; actual accelerator must be recorded |

Existing photographs were captured with automatic focus. Focus was not confirmed locked. This setup deviation is disclosed in `docs/PLAN.md`.

## Physical setup

- Target object: UNO playing cards.
- Nominal card dimensions: 85 x 55 mm; actual dimensions must be measured.
- Reference: ArUco DICT_4X4_50, marker ID 23.
- Nominal outer marker dimension: 60 x 60 mm; verify actual printed dimensions.
- Calibration target: 9 x 7 internal checkerboard corners.
- Nominal checkerboard square: 20 x 20 mm.
- Manual measuring instruments currently available: 15 cm and 30 cm metric rulers, each with 1 mm graduations.

The target and reference must lie in the same physical plane. Perspective and residual lens distortion must be addressed before reporting measurements.

## Initial data inventory

| Data | Count |
|---|---:|
| Physical UNO cards | 16 |
| Original UNO photographs | 95 |
| Original calibration photographs | 25 |
| Training candidates | 67 |
| Validation candidates | 19 |
| Test candidates | 9 |

The user reports that the reference marker is present in all 95 UNO photographs. Programmatic detection and quality approval are still pending.

The split is image-level. The same physical card identity may occur in multiple splits. This limitation is disclosed and must be considered when interpreting generalization metrics.

The committed split list is:

`dataset/splits/split_manifest.csv`

Do not use test images for training, model tuning or checkpoint selection.

## Planned segmentation architecture

Initial candidate: torchvision Mask R-CNN with a ResNet-50 FPN backbone, fine-tuned on the captured and labelled UNO photographs.

Pretrained initialization is allowed by the assessment. The final model must be trained or fine-tuned using the candidate's own images.

Ultralytics YOLO models and Roboflow models are prohibited as the segmentation model.

## Repository structure

```text
Xis_ai/
├── calibration/
│   ├── images/
│   ├── results/
│   ├── calibrate.py
│   └── calibration_data.npz
├── dataset/
│   ├── pilot_originals/
│   ├── raw/
│   ├── undistorted/
│   ├── annotations/
│   │   └── masks/
│   └── splits/
│       └── split_manifest.csv
├── models/
│   ├── configs/
│   └── checkpoints/
├── inference/
├── measurement/
│   ├── results/
│   └── ground_truth.csv
├── docs/
│   ├── PLAN.md
│   ├── OBJECTIVES.md
│   └── DEFINITION_OF_DONE.md
├── README.md
└── requirements.txt
```

## Environment setup

Python 3.11 is the planned environment.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Exact installed versions must be verified and recorded. Kaggle training environment versions and accelerator details must also be recorded separately.

## Reproducibility

The implementation must record:

- Random seeds.
- Python and dependency versions.
- Camera model and actual image resolution.
- Operating system, CPU, GPU and RAM.
- Calibration parameters and reprojection error.
- Training configuration and evaluation outputs.
- Model weights and SHA-256 hash.

Original photographs must remain unchanged. Undistorted images and segmentation masks must be stored separately.

## Execution commands

Implementation commands will be added as the scripts are created and tested. Do not treat an unimplemented command as working.

## Model artifacts

**Trained weights:** Pending
**SHA-256:** Pending
**Calibration parameters:** Pending
**Measurement validation:** Pending
**Loom demonstration:** Pending
**Pull request:** Pending

## Documentation

- `docs/PLAN.md`
- `docs/OBJECTIVES.md`
- `docs/DEFINITION_OF_DONE.md`
