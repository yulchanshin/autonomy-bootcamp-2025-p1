"""
Test for DetectColours code.
"""

import pathlib
import cv2
import numpy as np
import pytest

from modules.detect_colours import DetectBlue, DetectRed

TEST_PATH = pathlib.Path(__file__).parent
GROUND_TRUTH_BLUE_PATH = [
    pathlib.Path(TEST_PATH, "ground_truth_maps", f"mask_detections_blue_{i}.jpg")
    for i in range(1, 4)
]
GROUND_TRUTH_RED_PATH = [
    pathlib.Path(TEST_PATH, "ground_truth_maps", f"mask_detections_red_{i}.jpg")
    for i in range(1, 4)
]
TEST_MAPS_PATH = [pathlib.Path(TEST_PATH, "test_maps", f"maps_{i}.jpg") for i in range(1, 4)]

IOU_THRESHOLD = 0.8


def compute_iou(written_mask: np.ndarray, expected_mask: np.ndarray) -> float:
    """
    Compute coverage of detections through Intersection over Union method.
    """
    intersection = np.logical_and(written_mask > 0, expected_mask > 0).sum()
    union = np.logical_or(written_mask > 0, expected_mask > 0).sum()
    return intersection / union if union > 0 else 0.0


@pytest.mark.parametrize("map_path, gt_path", zip(TEST_MAPS_PATH, GROUND_TRUTH_BLUE_PATH))
def test_blue_coverage(
    map_path: pathlib.Path, gt_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """
    Test coverage of blue detections.
    """
    expected_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    blue_detect = DetectBlue.create()
    blue_mask = blue_detect.run(map_path, tmp_path / "out.jpg", return_mask=True)

    iou = compute_iou(blue_mask, expected_mask)
    assert iou >= IOU_THRESHOLD, f"Blue detection is too low. IoU: {iou: .2f}"


@pytest.mark.parametrize("map_path, gt_path", zip(TEST_MAPS_PATH, GROUND_TRUTH_RED_PATH))
def test_red_coverage(
    map_path: pathlib.Path, gt_path: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """
    Test coverage of blue detections.
    """
    expected_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    red_detect = DetectRed.create()
    red_mask = red_detect.run(map_path, tmp_path / "out.jpg", return_mask=True)

    iou = compute_iou(red_mask, expected_mask)
    assert iou >= IOU_THRESHOLD, f"Red detection is too low. IoU: {iou: .2f}"
