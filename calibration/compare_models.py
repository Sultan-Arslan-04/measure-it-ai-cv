
from pathlib import Path
import json

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
IMAGE_DIR = BASE / "images"
RESULTS = BASE / "results"

SQUARES_SIZE = (10, 8)
PATTERN_SIZE = (9, 7)
SQUARE_SIZE_MM = 20.0

FILES = sorted(IMAGE_DIR.glob("*.jpg"))

columns, rows = PATTERN_SIZE

object_template = np.zeros(
    (columns * rows, 3),
    np.float32
)

object_template[:, :2] = (
    np.mgrid[0:columns, 0:rows]
    .T.reshape(-1, 2)
    * SQUARE_SIZE_MM
)

objects = []
images = []
names = []
image_size = None

print("Physical squares:", SQUARES_SIZE)
print("Internal corners:", PATTERN_SIZE)
print("Input images:", len(FILES))

for path in FILES:

    image = cv2.imread(str(path))

    if image is None:
        continue

    h, w = image.shape[:2]

    if image_size is None:
        image_size = (w, h)

    if image_size != (w, h):
        print("SKIPPED resolution mismatch:", path.name)
        continue

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    found = False
    corners = None

    if hasattr(cv2, "findChessboardCornersSB"):

        flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )

        found, corners = cv2.findChessboardCornersSB(
            gray,
            PATTERN_SIZE,
            flags
        )

    if not found:
        continue

    objects.append(object_template.copy())
    images.append(corners.astype(np.float32))
    names.append(path.name)

print("Valid photographs:", len(names))

if len(names) < 20:
    raise RuntimeError("Fewer than 20 valid calibration images.")

def evaluate(label, flags):

    rms, matrix, distortion, rvecs, tvecs = (
        cv2.calibrateCamera(
            objects,
            images,
            image_size,
            None,
            None,
            flags=flags
        )
    )

    squared_total = 0.0
    point_total = 0
    individual = []

    for name, obj, observed, rvec, tvec in zip(
        names,
        objects,
        images,
        rvecs,
        tvecs
    ):

        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            matrix,
            distortion
        )

        delta = (
            observed.reshape(-1, 2)
            - projected.reshape(-1, 2)
        )

        squared = np.sum(delta ** 2, axis=1)

        error = float(np.sqrt(np.mean(squared)))

        squared_total += float(np.sum(squared))
        point_total += len(squared)

        individual.append({
            "filename": name,
            "rms_px": error
        })

    overall = float(
        np.sqrt(squared_total / point_total)
    )

    return {
        "model": label,
        "opencv_rms_px": float(rms),
        "overall_rms_px": overall,
        "distortion": distortion.ravel().tolist(),
        "camera_matrix": matrix.tolist(),
        "per_image_errors": individual
    }


model_a = evaluate(
    "A_all_five_coefficients",
    0
)

model_b = evaluate(
    "B_k3_fixed_zero",
    cv2.CALIB_FIX_K3
)

print("\n========== MODEL A ==========")
print("RMS:", model_a["opencv_rms_px"])
print("Distortion:", model_a["distortion"])

print("\n========== MODEL B ==========")
print("RMS:", model_b["opencv_rms_px"])
print("Distortion:", model_b["distortion"])

report = {
    "resolution": list(image_size),
    "physical_squares": list(SQUARES_SIZE),
    "internal_corners": list(PATTERN_SIZE),
    "valid_images": names,
    "model_A": model_a,
    "model_B": model_b
}

output = RESULTS / "model_comparison.json"

output.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8"
)

print("\nSaved:", output)
print("\nDiagnostic complete. No model selected automatically.")
