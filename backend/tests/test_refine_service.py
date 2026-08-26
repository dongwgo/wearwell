"""마스크 보수 단계 — 세그멘테이션이 남기는 구멍·조각을 어떻게 다루는지."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import refine_service


def test_labeling_counts_detached_pieces_separately():
    """신발 한 켤레처럼 정상적으로 두 조각인 마스크를 하나로 세면 안 된다."""
    mask = np.zeros((40, 60), dtype=bool)
    mask[10:30, 5:25] = True
    mask[10:30, 35:55] = True

    labels, count = refine_service.label_components(mask)

    assert count == 2
    assert labels[20, 15] != labels[20, 45]
    assert labels[0, 0] == 0  # 배경은 라벨을 받지 않는다


def test_holes_are_only_the_gaps_the_mask_encloses():
    """바깥 배경도 마스크가 아닌 픽셀이다. 반전만 하면 사진 전체가 구멍이 된다."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[15:25, 15:25] = False  # 팔에 가려진 자리

    holes = refine_service.find_holes(mask)

    assert int(holes.sum()) == 100
    assert holes[20, 20]
    assert not holes[0, 0]


def test_open_notch_is_not_a_hole():
    """경계까지 이어진 홈은 구멍이 아니다 — 메우면 옷 모양이 바뀐다."""
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:35] = True
    mask[15:25, 25:35] = False  # 오른쪽 경계까지 열린 홈

    assert not refine_service.find_holes(mask).any()


def test_repair_fills_holes_and_leaves_no_component_split():
    mask = np.zeros((80, 60), dtype=bool)
    mask[10:70, 10:50] = True
    mask[30:45, 20:35] = False

    repaired, report = refine_service.repair(mask, refine_service.DEFAULT_REPAIR)

    assert report["holesAfter"] == 0
    assert report["componentsAfter"] == 1
    assert report["pixelsAfter"] > report["pixelsBefore"]
    assert repaired[37, 27]  # 구멍이 채워졌다
    assert [step["step"] for step in report["steps"]] == ["close", "fillHoles", "dropStrays"]


def test_stray_removal_keeps_the_other_shoe_and_drops_the_speck():
    """비율 기준을 쓰는 이유. 가장 큰 조각만 남기면 신발 한 짝이 사라진다."""
    mask = np.zeros((60, 90), dtype=bool)
    mask[20:50, 5:35] = True   # 왼쪽 신발
    mask[20:50, 45:75] = True  # 오른쪽 신발 (같은 크기)
    mask[2:5, 85:88] = True    # 경계 부스러기

    _, report = refine_service.repair(mask, refine_service.DEFAULT_REPAIR)

    assert report["componentsAfter"] == 2
    assert report["steps"][-1]["delta"] < 0  # 부스러기만큼 줄었다


def test_repair_steps_can_be_turned_off_one_by_one():
    """랩에서 스위치를 끄면 그 단계는 아예 돌지 않아야 비교가 된다."""
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    mask[20:30, 20:30] = False

    _, report = refine_service.repair(
        mask, {**refine_service.DEFAULT_REPAIR, "fillHoles": False, "dropStrays": False}
    )

    assert [step["step"] for step in report["steps"]] == ["close"]
    assert report["holesAfter"] > 0


def test_stages_share_one_crop_window_so_they_can_be_compared_side_by_side():
    from PIL import Image
    import io

    mask = np.zeros((120, 100), dtype=bool)
    mask[20:100, 20:80] = True
    mask[40:60, 35:55] = False
    image = np.full((120, 100, 3), 180, dtype=np.uint8)

    stages = refine_service.build_stages(image, mask, None)

    sizes = {
        key: Image.open(io.BytesIO(stages[key])).size
        for key in ("cropPng", "defectPng", "repairedPng")
    }
    assert len(set(sizes.values())) == 1
    assert stages["diagnosis"]["holeCount"] == 1
    assert stages["repair"]["holesAfter"] == 0
    # 생성 모델 입력은 흰 배경 정사각 — 투명 영역을 그대로 넣으면 검게 들어간다.
    assert stages["image"].mode == "RGB"
    assert stages["image"].size == (refine_service.NORMALIZED_PX, refine_service.NORMALIZED_PX)
    assert stages["image"].getpixel((2, 2)) == (255, 255, 255)


def test_filled_holes_are_painted_with_the_garment_colour_not_the_arm_behind_them():
    """구멍을 알파로만 채우면 그 자리의 팔·배경 픽셀이 옷 안으로 들어간다.

    그 이미지를 생성 모델에 넣으면 살색을 옷의 일부로 읽고 그대로 그려낸다.
    """
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 10:50] = True
    mask[20:30, 20:30] = False

    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[:, :] = (20, 40, 200)      # 옷
    image[20:30, 20:30] = (230, 180, 150)  # 구멍 너머로 보이는 팔

    stages = refine_service.build_stages(image, mask, None)

    assert stages["repair"]["patchedPixels"] == 100
    assert stages["repair"]["patchColor"] == "#1428c8"
    # 정규화 이미지 어디에도 살색이 남지 않아야 한다.
    assert not (np.asarray(stages["image"]) == (230, 180, 150)).all(axis=2).any()


def test_empty_mask_does_not_crash_the_pipeline():
    """품질 필터에 걸러진 후보까지 태울 수 있으므로 빈 마스크가 들어올 수 있다."""
    mask = np.zeros((20, 20), dtype=bool)

    stages = refine_service.build_stages(np.zeros((20, 20, 3), dtype=np.uint8), mask, None)

    assert stages["diagnosis"]["pixelCount"] == 0
    assert stages["repair"]["pixelsAfter"] == 0
    assert stages["window"] == [0, 0, 20, 20]
