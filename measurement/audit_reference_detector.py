from pathlib import Path
import csv
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "dataset" / "splits" / "split_manifest.csv"
UNDISTORTED = ROOT / "dataset" / "undistorted"
RESULTS = ROOT / "measurement" / "results"
RAW = ROOT / "dataset" / "pilot_originals"

MARKER_ID = 23
DICTIONARY_ID = cv2.aruco.DICT_4X4_50

RESULTS.mkdir(parents=True, exist_ok=True)


def detector_configs():
    """Return a small, predefined set of detector configurations."""

    configurations = []
    p = cv2.aruco.DetectorParameters()
    configurations.append(("default", p))
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 7
    p.minMarkerPerimeterRate = 0.015
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minCornerDistanceRate = 0.03
    p.minDistanceToBorder = 3
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    configurations.append(("adaptive_subpix", p))
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 5
    p.minMarkerPerimeterRate = 0.015
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minDistanceToBorder = 3
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    configurations.append(("threshold_5", p))
    p = cv2.aruco.DetectorParameters()
    p.adaptiveThreshWinSizeMin = 3
    p.adaptiveThreshWinSizeMax = 53
    p.adaptiveThreshWinSizeStep = 10
    p.adaptiveThreshConstant = 10
    p.minMarkerPerimeterRate = 0.015
    p.maxMarkerPerimeterRate = 4.0
    p.polygonalApproxAccuracyRate = 0.05
    p.minDistanceToBorder = 3
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    configurations.append(("threshold_10", p))

    return configurations


def detect(image, dictionary, parameters):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters
    )

    corners, ids, rejected = detector.detectMarkers(gray)

    found_ids = [] if ids is None else ids.flatten().tolist()

    return MARKER_ID in found_ids, found_ids, rejected


def main():

    with MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        manifest = list(csv.DictReader(f))

    # Strictly exclude test images from detector comparison.
    candidates = [
        row for row in manifest
        if row["split"] in {"train", "val"}
    ]

    dictionary = cv2.aruco.getPredefinedDictionary(
        DICTIONARY_ID
    )

    configs = detector_configs()
    results = []

    print("=" * 65)
    print("ARUCO DETECTOR COMPARISON")
    print("=" * 65)
    print("Images inspected:", len(candidates))
    print("Test images inspected: 0")
    print("Configurations:", len(configs))

    for index, row in enumerate(candidates, 1):

        filename = row["filename"]
        card_id = row["card_id"]
        split = row["split"]

        raw_path = RAW / card_id / filename
        undistorted_path = (
            UNDISTORTED / card_id / f"{Path(filename).stem}.png"
        )

        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)

        if not undistorted_path.is_file():
            raise FileNotFoundError(undistorted_path)

        raw_image = cv2.imread(str(raw_path))
        undistorted_image = cv2.imread(str(undistorted_path))

        if raw_image is None or undistorted_image is None:
            raise RuntimeError(f"Could not read {filename}")

        for image_type, image in [
            ("raw", raw_image),
            ("undistorted", undistorted_image)
        ]:

            for config_name, parameters in configs:

                found, ids, rejected = detect(
                    image,
                    dictionary,
                    parameters
                )

                results.append({
                    "filename": filename,
                    "card_id": card_id,
                    "split": split,
                    "image_type": image_type,
                    "configuration": config_name,
                    "marker_23_detected": found,
                    "detected_ids": ";".join(map(str, ids)),
                    "rejected_candidate_count": len(rejected)
                })

        if index % 20 == 0 or index == len(candidates):
            print(f"Processed {index}/{len(candidates)}")

    output = RESULTS / "aruco_detector_comparison.csv"

    with output.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== DETECTION RATES ==========")

    for image_type in ("raw", "undistorted"):

        for config_name, _ in configs:

            subset = [
                r for r in results
                if r["image_type"] == image_type
                and r["configuration"] == config_name
            ]

            found_count = sum(
                r["marker_23_detected"]
                for r in subset
            )

            print(
                f"{image_type:12} {config_name:20} "
                f"{found_count}/{len(subset)}"
            )

    print("\nSaved:", output)
    print("Test images were not inspected.")

    return 0


if __name__ == "__main__":

    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
