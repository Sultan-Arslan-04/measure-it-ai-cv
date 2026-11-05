from pathlib import Path
import csv
import json
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

SQUARES_SIZE = (10, 8)

PATTERN_SIZE = (
    SQUARES_SIZE[0] - 1,
    SQUARES_SIZE[1] - 1
)

SQUARE_SIZE_MM = 20.0

MIN_VALID_IMAGES = 20

EXTENSIONS = {".jpg", ".jpeg", ".png"}

CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)

def create_object_points():

    columns, rows = PATTERN_SIZE

    points = np.zeros(
        (columns * rows, 3),
        dtype=np.float32
    )

    points[:, :2] = np.mgrid[
        0:columns,
        0:rows
    ].T.reshape(-1, 2) * SQUARE_SIZE_MM

    return points

def detect_corners(gray):

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

        if found:
            return True, corners, "SB"

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(
        gray,
        PATTERN_SIZE,
        flags
    )

    if not found:
        return False, None, "FAILED"

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        CRITERIA
    )

    return True, corners, "CLASSIC"


def main():

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    image_files = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    )

    if not image_files:
        raise RuntimeError("No calibration images found.")

    print("=" * 60)
    print("MEASURE IT - CAMERA CALIBRATION")
    print("=" * 60)

    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("Input images:", len(image_files))
    print(
    "Physical checkerboard:",
    f"{SQUARES_SIZE[0]} columns x {SQUARES_SIZE[1]} rows",
    f"= {SQUARES_SIZE[0] * SQUARES_SIZE[1]} squares"
)

    print(
        "Internal corners:",
        f"{PATTERN_SIZE[0]} x {PATTERN_SIZE[1]}",
        f"= {PATTERN_SIZE[0] * PATTERN_SIZE[1]} corners"
    )

    print("Physical squares:", SQUARES_SIZE)
    print("Total squares:", int(np.prod(SQUARES_SIZE)))

    print("Internal corners:", PATTERN_SIZE)
    print("Total internal corners:", int(np.prod(PATTERN_SIZE)))

    print("Square size:", SQUARE_SIZE_MM, "mm")

    object_template = create_object_points()

    object_points = []
    image_points = []

    valid_names = []
    detection_records = []

    image_size = None
    preview_image = None

    for index, path in enumerate(image_files, 1):

        print(
            f"\n[{index}/{len(image_files)}] {path.name}"
        )

        image = cv2.imread(str(path))

        if image is None:

            detection_records.append({
                "filename": path.name,
                "status": "FAILED",
                "reason": "UNREADABLE",
                "detector": ""
            })

            print("  Image could not be read.")
            continue

        height, width = image.shape[:2]
        current_size = (width, height)

        if image_size is None:
            image_size = current_size

        if current_size != image_size:

            detection_records.append({
                "filename": path.name,
                "status": "FAILED",
                "reason": "RESOLUTION_MISMATCH",
                "detector": ""
            })

            print("  Resolution mismatch. Skipped.")
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        found, corners, detector = detect_corners(gray)

        if found:

            object_points.append(
                object_template.copy()
            )

            image_points.append(
                corners.astype(np.float32)
            )

            valid_names.append(path.name)

            detection_records.append({
                "filename": path.name,
                "status": "DETECTED",
                "reason": "OK",
                "detector": detector
            })

            print(
                f"  SUCCESS: {len(corners)} corners "
                f"detected using {detector}"
            )

            if preview_image is None:
                preview_image = image.copy()

        else:

            detection_records.append({
                "filename": path.name,
                "status": "FAILED",
                "reason": "CHECKERBOARD_NOT_FOUND",
                "detector": detector
            })

            print("  Checkerboard not detected.")

    summary_path = RESULTS_DIR / "detection_summary.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "status",
                "reason",
                "detector"
            ]
        )

        writer.writeheader()
        writer.writerows(detection_records)

    valid_count = len(valid_names)

    print("\n" + "=" * 60)
    print("DETECTION SUMMARY")
    print("=" * 60)

    print("Total photographs:", len(image_files))
    print("Valid photographs:", valid_count)
    print("Rejected photographs:", len(image_files) - valid_count)

    if valid_count < MIN_VALID_IMAGES:

        raise RuntimeError(
            f"Only {valid_count} valid images. "
            f"At least {MIN_VALID_IMAGES} are required."
        )

    print("\nCalculating camera parameters...")

    rms, camera_matrix, distortion, rvecs, tvecs = (
        cv2.calibrateCamera(
            object_points,
            image_points,
            image_size,
            None,
            None
        )
    )

    if not np.isfinite(rms):
        raise RuntimeError("Invalid calibration RMS.")

    error_records = []

    total_squared_error = 0.0
    total_points = 0

    for name, obj, observed, rvec, tvec in zip(
        valid_names,
        object_points,
        image_points,
        rvecs,
        tvecs
    ):

        projected, _ = cv2.projectPoints(
            obj,
            rvec,
            tvec,
            camera_matrix,
            distortion
        )

        observed = observed.reshape(-1, 2)
        projected = projected.reshape(-1, 2)

        distances = np.linalg.norm(
            observed - projected,
            axis=1
        )

        image_rms = float(
            np.sqrt(np.mean(distances ** 2))
        )

        total_squared_error += float(
            np.sum(distances ** 2)
        )

        total_points += len(distances)

        error_records.append({
            "filename": name,
            "corner_count": len(distances),
            "reprojection_rms_px": image_rms
        })

    overall_rms = float(
        np.sqrt(total_squared_error / total_points)
    )

    new_camera_matrix, roi = (
        cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            distortion,
            image_size,
            1.0,
            image_size
        )
    )

    undistorted = cv2.undistort(
        preview_image,
        camera_matrix,
        distortion,
        None,
        new_camera_matrix
    )

    cv2.imwrite(
        str(RESULTS_DIR / "undistorted_preview.jpg"),
        undistorted
    )


    preview_width = 800

    preview_height = int(
        preview_image.shape[0]
        * preview_width
        / preview_image.shape[1]
    )

    original_small = cv2.resize(
        preview_image,
        (preview_width, preview_height)
    )

    corrected_small = cv2.resize(
        undistorted,
        (preview_width, preview_height)
    )

    comparison = cv2.hconcat([
        original_small,
        corrected_small
    ])

    cv2.imwrite(
        str(RESULTS_DIR / "undistortion_comparison.jpg"),
        comparison
    )

    thumbnails = []

    for path in image_files:

        image = cv2.imread(str(path))

        if image is None or image.shape[1::-1] != image_size:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        found, corners, _ = detect_corners(gray)

        height, width = image.shape[:2]

        scale = min(
            320 / width,
            230 / height
        )

        thumbnail = cv2.resize(
            image,
            (
                int(width * scale),
                int(height * scale)
            )
        )

        if found:
            cv2.drawChessboardCorners(
                thumbnail,
                PATTERN_SIZE,
                corners * scale,
                True
            )

        canvas = np.zeros(
            (270, 340, 3),
            dtype=np.uint8
        )

        h, w = thumbnail.shape[:2]

        canvas[30:30+h, 10:10+w] = thumbnail

        cv2.putText(
            canvas,
            path.name,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        status = "DETECTED" if found else "FAILED"

        color = (
            (0, 255, 0)
            if found
            else (0, 0, 255)
        )

        cv2.putText(
            canvas,
            status,
            (10, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

        thumbnails.append(canvas)

    columns = 5
    rows = (len(thumbnails) + columns - 1) // columns

    if thumbnails:

        montage = np.zeros(
            (rows * 270, columns * 340, 3),
            dtype=np.uint8
        )

        for i, thumb in enumerate(thumbnails):

            y = (i // columns) * 270
            x = (i % columns) * 340

            montage[y:y+270, x:x+340] = thumb

        cv2.imwrite(
            str(RESULTS_DIR / "calibration_detection_montage.jpg"),
            montage
        )

    np.savez_compressed(
        RESULTS_DIR / "calibration_data.npz",

        camera_matrix=camera_matrix,

        distortion_coefficients=distortion,

        new_camera_matrix=new_camera_matrix,

        image_width=image_size[0],

        image_height=image_size[1],

        pattern_columns=PATTERN_SIZE[0],

        pattern_rows=PATTERN_SIZE[1],

        square_size_mm=SQUARE_SIZE_MM,

        rms_reprojection_error_px=rms,

        used_image_names=np.asarray(valid_names)
    )

    error_path = RESULTS_DIR / "per_image_errors.csv"

    with error_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "corner_count",
                "reprojection_rms_px"
            ]
        )

        writer.writeheader()
        writer.writerows(error_records)

    report = {

        "opencv_version": cv2.__version__,

        "numpy_version": np.__version__,

        "checkerboard_internal_corners": list(PATTERN_SIZE),

        "square_size_mm": SQUARE_SIZE_MM,

        "total_images": len(image_files),

        "valid_images": valid_count,

        "rejected_images": len(image_files) - valid_count,

        "image_resolution": {
            "width": image_size[0],
            "height": image_size[1]
        },

        "camera_matrix": camera_matrix.tolist(),

        "distortion_coefficients": distortion.ravel().tolist(),

        "opencv_rms_reprojection_error_px": float(rms),

        "overall_reprojection_rms_px": overall_rms,

        "valid_images_used": valid_names,
        "physical_squares": list(SQUARES_SIZE),
        "physical_square_count": int(np.prod(SQUARES_SIZE)),
        "internal_corners": list(PATTERN_SIZE),
        "internal_corner_count": int(np.prod(PATTERN_SIZE)),
    }

    with (
        RESULTS_DIR / "calibration_report.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(report, file, indent=4)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    print("\nCamera matrix:")
    print(camera_matrix)

    print("\nDistortion coefficients:")
    print(distortion)

    print(f"\nOpenCV RMS error: {rms:.6f} pixels")

    print(
        f"Recalculated overall RMS: "
        f"{overall_rms:.6f} pixels"
    )

    print(f"Valid images: {valid_count}/25")

    print("\nSaved evidence:")

    for path in sorted(RESULTS_DIR.iterdir()):
        print(" ", path.name)

    print("\nCalibration calculation completed.")
    print("Review the errors and visual evidence before approval.")

    return 0


if __name__ == "__main__":

    try:
        sys.exit(main())

    except Exception as error:

        print(
            f"\nERROR: {error}",
            file=sys.stderr
        )

        sys.exit(1)
