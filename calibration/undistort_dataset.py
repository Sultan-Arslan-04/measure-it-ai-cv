from pathlib import Path
import csv
import json
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_FILE = (
    ROOT / "calibration" / "results" / "calibration_data.npz"
)

MANIFEST_FILE = (
    ROOT / "dataset" / "splits" / "split_manifest.csv"
)

OUTPUT_DIR = ROOT / "dataset" / "undistorted"

RESULTS_DIR = ROOT / "measurement" / "results"

EXPECTED_IMAGES = 95
EXPECTED_SPLITS = {
    "train": 67,
    "val": 19,
    "test": 9,
}

ARUCO_ID = 23
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50

def create_aruco_detector():

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )

    parameters = cv2.aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.adaptiveThreshConstant = 5

    parameters.minMarkerPerimeterRate = 0.015
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minDistanceToBorder = 3

    parameters.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX
    )

    return cv2.aruco.ArucoDetector(
        dictionary,
        parameters
    )

def detect_reference(gray, detector):

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return False, "", None, corners, ids

    detected_ids = ids.flatten().tolist()

    if ARUCO_ID not in detected_ids:
        return (
            False,
            ";".join(map(str, detected_ids)),
            None,
            corners,
            ids
        )

    index = detected_ids.index(ARUCO_ID)

    points = corners[index].reshape(4, 2)

    sides = [
        np.linalg.norm(
            points[i] - points[(i + 1) % 4]
        )
        for i in range(4)
    ]

    mean_side_pixels = float(np.mean(sides))

    return (
        True,
        ";".join(map(str, detected_ids)),
        mean_side_pixels,
        corners,
        ids
    )

def load_inputs():

    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(CALIBRATION_FILE)

    if not MANIFEST_FILE.is_file():
        raise FileNotFoundError(MANIFEST_FILE)

    with np.load(CALIBRATION_FILE) as data:

        camera_matrix = data["camera_matrix"].copy()

        distortion = data[
            "distortion_coefficients"
        ].copy()

        new_camera_matrix = data[
            "new_camera_matrix"
        ].copy()

        resolution = (
            int(data["image_width"]),
            int(data["image_height"])
        )

        calibration_rms = float(
            data["rms_reprojection_error_px"]
        )

    with MANIFEST_FILE.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        rows = list(csv.DictReader(file))

    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_IMAGES} manifest entries; "
            f"found {len(rows)}."
        )

    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in EXPECTED_SPLITS
    }

    if counts != EXPECTED_SPLITS:
        raise ValueError(
            f"Unexpected manifest counts: {counts}"
        )

    paths = [row["relative_path"] for row in rows]

    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate manifest paths detected.")

    return (
        camera_matrix,
        distortion,
        new_camera_matrix,
        resolution,
        calibration_rms,
        rows,
        counts
    )

def main():

    (
        camera_matrix,
        distortion,
        new_camera_matrix,
        expected_resolution,
        calibration_rms,
        manifest,
        split_counts
    ) = load_inputs()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detector = create_aruco_detector()

    print("=" * 65)
    print("MEASURE IT — DATASET UNDISTORTION")
    print("=" * 65)

    print("OpenCV:", cv2.__version__)
    print("Calibration RMS:", calibration_rms)
    print("Expected resolution:", expected_resolution)
    print("Input photographs:", len(manifest))
    print("ArUco dictionary: DICT_4X4_50")
    print("Required marker ID:", ARUCO_ID)

    print("\n========== SOURCE PREFLIGHT ==========")
    for index, row in enumerate(manifest, 1):

        source = ROOT / Path(row["relative_path"])

        if not source.is_file():
            raise FileNotFoundError(source)

        image = cv2.imread(str(source))

        if image is None:
            raise ValueError(
                f"Unreadable source image: {source}"
            )

        actual_resolution = (
            image.shape[1],
            image.shape[0]
        )

        if actual_resolution != expected_resolution:
            raise ValueError(
                f"Resolution mismatch: {source.name}: "
                f"{actual_resolution} != {expected_resolution}"
            )

        if index % 20 == 0 or index == len(manifest):
            print(f"Verified {index}/{len(manifest)}")

    print("All source photographs passed resolution checks.")

    audit = []

    marker_counts = {
        "train_detected": 0,
        "train_missing": 0,
        "val_detected": 0,
        "val_missing": 0,
    }

    thumbnails = []

    for index, row in enumerate(manifest, 1):

        source = ROOT / Path(row["relative_path"])

        card_id = row["card_id"]
        split = row["split"]

        original = cv2.imread(str(source))

        undistorted = cv2.undistort(
            original,
            camera_matrix,
            distortion,
            None,
            new_camera_matrix
        )

        output_folder = OUTPUT_DIR / card_id
        output_folder.mkdir(parents=True, exist_ok=True)

        output_path = output_folder / (
            source.stem + ".png"
        )

        if not cv2.imwrite(
            str(output_path),
            undistorted,
            [cv2.IMWRITE_PNG_COMPRESSION, 3]
        ):
            raise RuntimeError(
                f"Could not save: {output_path}"
            )

        marker_status = "HELD_OUT"
        detected_ids = ""
        marker_side_pixels = None

        if split in {"train", "val"}:

            gray = cv2.cvtColor(
                undistorted,
                cv2.COLOR_BGR2GRAY
            )

            (
                found,
                ids_text,
                side_pixels,
                marker_corners,
                marker_ids
            ) = detect_reference(gray, detector)

            marker_status = (
                "DETECTED" if found else "MISSING"
            )

            detected_ids = ids_text
            marker_side_pixels = side_pixels

            key = f"{split}_{'detected' if found else 'missing'}"

            marker_counts[key] += 1

            if len(thumbnails) < 24:

                preview = undistorted.copy()

                if marker_ids is not None:

                    cv2.aruco.drawDetectedMarkers(
                        preview,
                        marker_corners,
                        marker_ids
                    )

                target_width = 300

                scale = target_width / preview.shape[1]

                thumbnail = cv2.resize(
                    preview,
                    (
                        target_width,
                        int(preview.shape[0] * scale)
                    ),
                    interpolation=cv2.INTER_AREA
                )

                canvas = np.zeros(
                    (240, 320, 3),
                    dtype=np.uint8
                )

                height, width = thumbnail.shape[:2]

                display_scale = min(
                    300 / width,
                    190 / height
                )

                thumbnail = cv2.resize(
                    thumbnail,
                    (
                        int(width * display_scale),
                        int(height * display_scale)
                    )
                )

                height, width = thumbnail.shape[:2]

                canvas[
                    35:35 + height,
                    10:10 + width
                ] = thumbnail

                color = (
                    (0, 255, 0)
                    if found
                    else (0, 0, 255)
                )

                cv2.putText(
                    canvas,
                    f"{source.name} : {marker_status}",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    color,
                    1
                )

                thumbnails.append(canvas)

        audit.append({
            "filename": source.name,
            "card_id": card_id,
            "split": split,
            "source_path": row["relative_path"],
            "undistorted_path": (
                output_path.relative_to(ROOT).as_posix()
            ),
            "width": expected_resolution[0],
            "height": expected_resolution[1],
            "marker_23_status": marker_status,
            "detected_marker_ids": detected_ids,
            "marker_mean_side_pixels": marker_side_pixels
        })

        if index % 20 == 0 or index == len(manifest):
            print(f"Undistorted {index}/{len(manifest)}")

    audit_path = (
        RESULTS_DIR / "reference_detection_audit.csv"
    )

    with audit_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(audit[0].keys())
        )

        writer.writeheader()
        writer.writerows(audit)


    if thumbnails:

        columns = 3

        rows_count = (
            len(thumbnails) + columns - 1
        ) // columns

        montage = np.zeros(
            (rows_count * 240, columns * 320, 3),
            dtype=np.uint8
        )

        for i, thumbnail in enumerate(thumbnails):

            y = (i // columns) * 240
            x = (i % columns) * 320

            montage[
                y:y + 240,
                x:x + 320
            ] = thumbnail

        cv2.imwrite(
            str(
                RESULTS_DIR
                / "reference_detection_montage.jpg"
            ),
            montage
        )


    report = {
        "project": "Measure It",
        "calibration_file": (
            CALIBRATION_FILE.relative_to(ROOT).as_posix()
        ),
        "calibration_rms_px": calibration_rms,
        "image_resolution": list(expected_resolution),
        "source_images": len(manifest),
        "undistorted_images": len(audit),
        "split_counts": split_counts,
        "output_format": "PNG",
        "reference_dictionary": "DICT_4X4_50",
        "reference_marker_id": ARUCO_ID,
        "reference_nominal_size_mm": 60.0,
        "reference_detection_counts": marker_counts,
        "test_marker_audit": "Not performed",
        "original_images_modified": False
    }

    report_path = (
        RESULTS_DIR / "undistortion_audit.json"
    )

    report_path.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("UNDISTORTION COMPLETE")
    print("=" * 65)

    print("Total images:", len(audit))
    print("Training:", split_counts["train"])
    print("Validation:", split_counts["val"])
    print("Test:", split_counts["test"])

    print("\nREFERENCE DETECTION — TRAIN/VALIDATION ONLY")

    for key, count in marker_counts.items():
        print(f"{key}: {count}")

    print("\nSaved:")
    print(audit_path)
    print(report_path)
    print(RESULTS_DIR / "reference_detection_montage.jpg")

    print("\nOriginal photographs were not modified.")
    print("Test images were not inspected for marker tuning.")

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
