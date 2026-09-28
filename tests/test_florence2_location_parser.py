import numpy as np
import pytest

from open_vocabulary_detector.backends.florence2 import LocationParser


def test_parse_reads_consecutive_boxes_at_bin_centers() -> None:
    boxes: np.ndarray = LocationParser.parse(
        "</s><s>cat<loc_8><loc_114><loc_497><loc_990><loc_539><loc_50><loc_998><loc_777></s>"
    )
    assert boxes.shape == (2, 4)
    assert boxes[0].tolist() == pytest.approx([0.0085, 0.1145, 0.4975, 0.9905])
    assert boxes[1].tolist() == pytest.approx([0.5395, 0.0505, 0.9985, 0.7775])


def test_parse_without_boxes_is_empty() -> None:
    assert LocationParser.parse("</s><s>cat</s>").shape == (0, 4)
    assert LocationParser.parse("</s><s>cat<loc_1><loc_2></s>").shape == (0, 4)


def test_parse_bounds_polygons() -> None:
    boxes: np.ndarray = LocationParser.parse(
        "</s><s>cat<poly><loc_10><loc_20><loc_30><loc_5><loc_20><loc_40></poly>"
        + "<poly><loc_1><loc_2><loc_3></poly><poly><loc_7></poly></s>"
    )
    assert boxes.shape == (2, 4)
    assert boxes[0].tolist() == pytest.approx([0.0105, 0.0055, 0.0305, 0.0405])
    assert boxes[1].tolist() == pytest.approx([0.0015, 0.0025, 0.0015, 0.0025])


def test_parse_reads_mixed_boxes_and_polygons_in_generation_order() -> None:
    boxes: np.ndarray = LocationParser.parse(
        "</s><s>cat<loc_100><loc_200><loc_300><loc_400>"
        + "<poly><loc_10><loc_20><loc_30><loc_5></poly><loc_500><loc_600><loc_700><loc_800></s>"
    )
    assert boxes.shape == (3, 4)
    assert boxes[0].tolist() == pytest.approx([0.1005, 0.2005, 0.3005, 0.4005])
    assert boxes[1].tolist() == pytest.approx([0.0105, 0.0055, 0.0305, 0.0205])
    assert boxes[2].tolist() == pytest.approx([0.5005, 0.6005, 0.7005, 0.8005])
