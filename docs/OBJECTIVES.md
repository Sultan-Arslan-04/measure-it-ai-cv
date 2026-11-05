# Objectives — Measure It

**Assessment:** xis.ai AI / Computer Vision Technical Assessment, Version 2.0
**Status:** Targets defined before implementation. No results have been measured yet.

## MO-1 — Physical measurement accuracy (required)

**What is measured**

The absolute error between the system-reported width and height and the corresponding physical ground-truth dimensions of at least 10 distinct physical UNO cards.

**How**

1. Measure the width and height of at least 10 physical cards by hand.
2. Prefer a calliper, as requested by the assessment. If unavailable, record the ruler's 1 mm graduation and disclose the limitation.
3. Photograph each card with the ArUco marker visible in the same plane.
4. Run the complete measurement command.
5. Save every physical measurement and system output.
6. Calculate errors separately for width and height and provide a per-instance table.

For N dimension readings:

`MAE_mm = sum(abs(predicted_mm - actual_mm)) / N`

`MAPE_percent = 100 * mean(abs(predicted_mm - actual_mm) / actual_mm)`

**Target**

- Mean absolute error (MAE) <= 2.0 mm.
- Mean absolute percentage error (MAPE) <= 3.0%.

Both targets must be reported. A missed target must be reported honestly with the observed result.

**Conditions**

- HONOR X6c rear main camera at fixed 1x zoom and fixed resolution.
- Same camera configuration used for calibration and measurement.
- Card and reference marker on the same plane.
- Reference marker actual physical dimension measured and recorded.
- At least 10 distinct physical card instances.
- No hidden, cropped or untrusted reference marker.

**Not included**

Images where the system correctly refuses because the card or reference is missing, partly hidden, badly blurred or geometrically unreliable. Refusal counts must be reported separately.

## MO-2 — Segmentation quality

**What is measured**

Mean mask Intersection over Union (IoU) on the fixed, held-out test split.

**How**

- Compare predicted binary masks against manually labelled ground-truth masks.
- Calculate IoU for every test image.
- Report the individual values and the mean.
- Use the committed test filenames only for final evaluation.
- Do not train on, tune with or select a checkpoint using test images.

`IoU = intersection(predicted_mask, ground_truth_mask) / union(predicted_mask, ground_truth_mask)`

**Target**

Mean test mask IoU >= 0.85.

**Conditions**

- Fixed test manifest with 9 images.
- Same image preprocessing and evaluation threshold for every test image.
- Ground-truth test masks remain unavailable to model training and tuning.
- The limited test-set size must be disclosed when interpreting the result.

## Other mandatory reported measurements

These are required even though they are not additional objectives:

- Camera calibration image count and reprojection error.
- Camera matrix and distortion coefficients.
- Mask IoU.
- mAP@0.5.
- mAP@0.5:0.95.
- Precision and recall.
- Per-instance physical measurement table.
- Mean, median and worst-case measurement error.
- Mean percentage error.
- Raw-image versus undistorted-image measurement errors.
- One measured speed or accuracy target.
- Error change under one selected failure condition.
- Confidence threshold and the number of refused images.

## Results

All actual results are pending implementation and execution.

| Objective | Target | Actual result | Status |
|---|---|---|---|
| MO-1 MAE | <= 2.0 mm | Not measured | Pending |
| MO-1 MAPE | <= 3.0% | Not measured | Pending |
| MO-2 mean mask IoU | >= 0.85 | Not measured | Pending |

No result may be manually invented, rounded to conceal errors or reported without raw evidence.

## Calibration experiment — initial measured result

**Status:** Calibration calculated; measurement reliability remains under review.

| Property | Measured result |
|---|---|
| Physical checkerboard squares | 10 x 8 = 80 |
| Internal corner pattern | 9 x 7 = 63 |
| Nominal square size | 20 mm |
| Calibration photographs | 25 |
| Valid photographs | 21 |
| Rejected photographs | 4 |
| Image resolution | 3056 x 4080 |
| OpenCV version | 4.10.0 |
| NumPy version | 1.26.4 |
| Model A RMS | 5.910700 px |
| Model B RMS | 5.968164 px |

Model A is provisionally retained because it has the lower reprojection residual. This does not establish that the camera calibration is sufficiently accurate for physical measurement.

The existing photographs were taken using automatic focus. This deviation is disclosed in the Plan.

Evidence:
- calibration/results/calibration_report.json
- calibration/results/per_image_errors.csv
- calibration/results/model_comparison.json
- calibration/results/detection_summary.csv

The physical measurement accuracy objective remains unmeasured.
