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


def crossed_shirt():
    """팔이 상의의 왼쪽 아래를 가로지르는, 실제로 가장 자주 나오는 결함.

    파먹힌 자리가 바깥과 이어져 있어 구멍이 아니다 — `find_holes`도, 닫기 커널도
    이걸 되돌리지 못한다. (온전한 상의, 세그멘테이션이 준 상의 마스크, 팔 마스크).
    """
    shirt = np.zeros((220, 200), dtype=bool)
    shirt[40:170, 50:150] = True
    arm = np.zeros((220, 200), dtype=bool)
    for offset in range(70):
        arm[100 + offset:118 + offset, 40 + offset // 2:70 + offset // 2] = True
    return shirt, shirt & ~arm, arm


def test_convex_hull_wraps_the_mask_without_touching_a_convex_shape():
    solid = np.zeros((40, 40), dtype=bool)
    solid[10:30, 10:30] = True
    assert (refine_service.convex_hull_mask(solid) == solid).all()  # 이미 볼록하면 그대로

    bitten = solid.copy()
    bitten[20:30, 20:30] = False  # 모서리를 베어 문 자국
    hull = refine_service.convex_hull_mask(bitten)
    assert hull[22, 22] and not bitten[22, 22]  # 베어 문 자리가 껍질 안에 들어온다
    assert not hull[28, 28]  # 껍질은 벤 자국을 비스듬히 가로지른다 — 바운딩박스가 아니다
    assert not hull[5, 5]


def test_arm_across_the_chest_is_found_even_though_it_is_not_a_hole():
    """이 파이프라인이 놓치던 결함. 구멍 진단은 0을 내지만 마스크는 18% 파먹혀 있다."""
    shirt, garment, arm = crossed_shirt()

    assert not refine_service.find_holes(garment).any()  # 구멍으로는 안 보인다

    occluded, report = refine_service.find_occluded(garment, arm, None, np.where(arm, 14, 0), {14: "Left-arm"})

    assert report["deficitRatio"] > 0.15
    assert report["occludedRatio"] > 0.12
    assert report["solidity"] < 0.85
    assert report["acceptedCount"] == 1
    assert report["by"][0]["label"] == "Left-arm"
    assert report["where"] == "lower-left"
    # 되찾은 자리는 전부 원래 옷이던 자리여야 한다. 옷이 아닌 곳을 옷으로 만들면
    # 실루엣이 부풀고, 그건 FLUX도 되돌리지 못한다.
    assert not (occluded & ~shirt).any()
    assert (occluded & shirt).sum() > (arm & shirt).sum() * 0.7


def test_filling_the_occluded_area_restores_the_original_garment_shape():
    shirt, garment, arm = crossed_shirt()
    parse = {"pred": np.where(arm, 14, 0), "names": {14: "Left-arm"}, "background": [0]}
    image = np.zeros((220, 200, 3), dtype=np.uint8)
    image[:] = (240, 240, 245)
    image[shirt] = (60, 90, 200)
    image[arm] = (225, 175, 145)

    stages = refine_service.build_stages(image, garment, None, parse)
    repaired = stages["repair"]

    assert [step["step"] for step in repaired["steps"]][2] == "fillOccluded"
    assert repaired["componentsAfter"] == 1  # 팔에 끊겼던 자락이 다시 붙었다
    assert repaired["pixelsAfter"] > repaired["pixelsBefore"] * 1.15
    # 정규화 이미지 어디에도 팔의 살색이 남으면 안 된다.
    assert not (np.asarray(stages["image"]) == (225, 175, 145)).all(axis=2).any()


def test_the_hollow_between_sleeve_and_waist_is_not_swallowed():
    """볼록껍질만 믿으면 원래 파인 자리까지 옷이 된다.

    소매와 허리 사이는 껍질이 가로지르는 오목한 자리다. 거기 팔이 지나가면
    "껍질 안 + 사람 픽셀"이 되어 버리므로, 옷이 양쪽에 남아 있는지를 함께 본다.
    """
    shirt = np.zeros((120, 120), dtype=bool)
    shirt[20:40, 20:100] = True   # 소매 — 넓다
    shirt[40:100, 45:75] = True   # 몸통 — 좁다
    arm = np.zeros((120, 120), dtype=bool)
    arm[50:70, 38:65] = True      # 몸통을 가로지르고 소매 아래 파인 자리까지 나간 팔

    occluded, report = refine_service.find_occluded(shirt & ~arm, arm)

    assert report["acceptedCount"] == 1   # 조각 자체는 "가려진 옷"으로 통과하지만
    assert occluded[60, 55]               # 몸통을 가로지른 자리만 메우고
    assert not occluded[60, 40]           # 원래 파여 있던 자리는 그대로 둔다
    assert not (occluded & ~shirt).any()


def test_an_arm_beside_the_garment_is_not_swallowed():
    """가림 판정이 '옷이 아닌 사람 픽셀'이기만 하면 팔 전체가 상의가 된다.

    옷 앞을 가로지르는 토막은 테두리가 옷에 닿지만, 옆에 나란한 팔은 배경에 닿는다.
    """
    garment = np.zeros((120, 120), dtype=bool)
    garment[20:100, 20:70] = True
    arm = np.zeros((120, 120), dtype=bool)
    arm[20:100, 72:90] = True  # 옷 오른쪽에 붙어 있을 뿐, 가리지는 않는다

    occluded, report = refine_service.find_occluded(garment, arm)

    assert not occluded.any()
    assert report["occludedPixels"] == 0


def test_an_occluder_bigger_than_the_garment_is_reported_but_not_filled():
    """통 넓은 바지에 덮인 신발처럼, 그대로 합치면 마스크가 바짓단을 삼키는 조합."""
    shoe = np.zeros((100, 100), dtype=bool)
    shoe[50:90, 20:40] = True
    shoe[50:90, 60:80] = True
    pants = np.zeros((100, 100), dtype=bool)
    pants[50:90, 40:60] = True  # 두 짝 사이를 통째로 채우는, 신발 면적의 절반짜리 조각

    occluded, report = refine_service.find_occluded(shoe, pants)

    assert not occluded.any()
    assert report["oversizedCount"] == 1
    assert report["candidatePixels"] > 0  # 후보였다는 사실은 남는다


def test_occlusion_is_skipped_without_a_label_map():
    """라벨맵이 없으면 형태만 남는다. 형태만 보고 메우면 진짜 옷 경계까지 부푼다."""
    _, garment, _ = crossed_shirt()

    occluded, report = refine_service.find_occluded(garment, None)

    assert not occluded.any()
    assert report["available"] is False
    assert report["solidity"] < 0.85  # 볼록채움도는 라벨맵 없이도 알 수 있다


def test_repair_fills_holes_and_leaves_no_component_split():
    mask = np.zeros((80, 60), dtype=bool)
    mask[10:70, 10:50] = True
    mask[30:45, 20:35] = False

    repaired, report = refine_service.repair(mask, refine_service.DEFAULT_REPAIR)

    assert report["holesAfter"] == 0
    assert report["componentsAfter"] == 1
    assert report["pixelsAfter"] > report["pixelsBefore"]
    assert repaired[37, 27]  # 구멍이 채워졌다
    # 가림 메우기는 라벨맵(occluded)을 받지 못하면 아예 돌지 않는다 — 형태만 보고
    # 바깥과 이어진 결손을 메우면 멀쩡한 옷 모양까지 바꾼다.
    assert [step["step"] for step in report["steps"]] == ["close", "fillHoles", "smooth", "dropStrays"]


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
        mask, {**refine_service.DEFAULT_REPAIR, "fillHoles": False, "dropStrays": False, "smooth": False}
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
    assert stages["repair"]["patchFill"] == "propagate"
    # 정규화 이미지 어디에도 살색이 남지 않아야 한다.
    assert not (np.asarray(stages["image"]) == (230, 180, 150)).all(axis=2).any()


def test_the_patch_follows_the_shading_instead_of_flattening_it():
    """대표색 한 가지로 칠하면 FLUX가 그 자리를 무지 패널로 읽는다.

    경계에서 색을 이어받으면 위아래 명암이 끊기지 않는다.
    """
    mask = np.zeros((120, 60), dtype=bool)
    mask[10:110, 10:50] = True
    mask[40:80, 20:40] = False

    image = np.zeros((120, 60, 3), dtype=np.uint8)
    image[:, :, 2] = 200
    image[:, :, 0] = np.linspace(0, 200, 120).astype(np.uint8)[:, None]  # 위에서 아래로 밝아지는 옷
    image[40:80, 20:40] = (230, 180, 150)

    patched, info = refine_service._patch(image, mask, mask | _box(120, 60, 40, 80, 20, 40))

    assert info["patchFill"] == "propagate"
    top = int(patched[42, 30, 0])
    bottom = int(patched[77, 30, 0])
    assert bottom - top > 30  # 메운 자리 안에서도 명암이 이어진다
    assert patched[60, 30, 1] < 60  # 살색이 아니라 옷 색이다


def _box(height, width, y0, y1, x0, x1):
    area = np.zeros((height, width), dtype=bool)
    area[y0:y1, x0:x1] = True
    return area


def test_empty_mask_does_not_crash_the_pipeline():
    """품질 필터에 걸러진 후보까지 태울 수 있으므로 빈 마스크가 들어올 수 있다."""
    mask = np.zeros((20, 20), dtype=bool)

    stages = refine_service.build_stages(np.zeros((20, 20, 3), dtype=np.uint8), mask, None)

    assert stages["diagnosis"]["pixelCount"] == 0
    assert stages["repair"]["pixelsAfter"] == 0
    assert stages["window"] == [0, 0, 20, 20]
