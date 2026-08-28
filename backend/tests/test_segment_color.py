from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import segment_models
import segment_service


def test_garment_color_uses_only_pixels_inside_mask():
    image = np.full((30, 30, 3), (245, 245, 245), dtype=np.uint8)
    image[8:22, 10:20] = (35, 75, 175)
    mask = np.zeros((30, 30), dtype=bool)
    mask[8:22, 10:20] = True

    assert segment_service._garment_color(image, mask) == "블루"


def test_small_bag_is_not_rejected_only_for_its_area():
    stats = {"areaRatio": 0.0005, "fillRatio": 0.05, "confidence": 0.8}

    assert segment_service._reject_reason(
        stats, segment_models.QUALITY_THRESHOLDS["가방"]
    ) is None


def test_bag_and_accessory_labels_remain_distinct():
    assert segment_models.ATR_LABEL_TO_CATEGORY["Bag"] == "가방"
    assert segment_models.ATR_LABEL_TO_CATEGORY["Belt"] == "액세서리"
