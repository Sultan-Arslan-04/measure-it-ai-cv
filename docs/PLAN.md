# Plan — Measure It

**Assessment:** xis.ai AI/CV Assessment v2.0
**Date:** 5 November 2026
**Deadline:** 5 November 2026, 12:00 PKT
**Work budget:** 8 active hours
**Branch:** dev-test01

## Goal and setup

Build a reproducible system that segments a physical UNO card and estimates its width and height in millimetres from a calibrated HONOR X6c photograph containing an ArUco reference.

Available data: 25 checkerboard candidates and 95 UNO photographs of 16 physical cards. The nominal checkerboard square is 20 mm; the reference is ArUco DICT_4X4_50, ID 23, nominally 60 mm. Available computer: i5-7300U, 8 GB RAM, Intel HD 620. Kaggle GPU training is planned. Ground truth currently uses 1 mm-graduation rulers.

## Execution order and active-time allocation

| Work | Hours |
|---|---:|
| Documentation and initial Git commits | 0.50 |
| Camera calibration and verification | 1.00 |
| Image audit, undistortion and dataset preparation | 1.00 |
| Segmentation annotation and quality checks | 1.25 |
| Model training and test evaluation | 1.50 |
| Image-to-millimetres inference | 1.00 |
| Physical accuracy validation | 0.75 |
| Required experiments, final docs, Loom and submission | 1.00 |
| **Total** | **8.00** |

Prioritize deliverables 1–4 before further experiments. Use at least 20 valid calibration images and at least 70 approved, self-captured, undistorted and labelled object images. The fixed image-level split is 67 train, 19 validation and 9 test. Test images must never be used for training, tuning or checkpoint selection. Physical card identities can overlap between splits; disclose this limitation.

## Approach and risks

Use OpenCV for calibration, undistortion and ArUco detection. The initial model candidate is torchvision Mask R-CNN with ResNet-50 FPN, fine-tuned on the captured data; prohibited YOLO and Roboflow models will not be used. Convert the segmented card outline using the reference geometry, accounting for perspective and requiring coplanarity.

Existing photographs used automatic rather than confirmed locked focus. Disclose this deviation and check sharpness and calibration stability. Use fixed focus, zoom and resolution for any replacement captures and their corresponding calibration. Verify actual printed dimensions. Prefer a borrowed calliper; if unavailable, disclose ruler-based validation. Existing images remain candidates until quality checks pass.

## Decision record

Chosen: OpenCV, ArUco and fine-tuned torchvision Mask R-CNN. Rejected: prohibited YOLO/Roboflow models and colour thresholding as the final segmentation method. The chosen approach costs more training and implementation time but provides learned object masks and explicit reference geometry.

Skip unrelated features and extra object categories. Record any plan changes, their reasons and consequences. Report unfinished work honestly.
