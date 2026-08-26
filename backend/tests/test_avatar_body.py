"""체형 레퍼런스 생성 테스트.

메시 피팅 경로는 SMPL-X 가중치가 있어야 돌아가므로 CI에서는 건너뛴다.
여기서는 가중치 없이도 항상 성립해야 하는 성질만 검증한다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from avatar_body import (  # noqa: E402
    VIEW_YAW,
    BodyReference,
    BodyTarget,
    build_avatar_prompt,
    build_body_reference,
    estimate_missing,
    measure_mesh,
    pad_for_full_body,
    render_proportional_silhouette,
    rotate_y,
)


def silhouette_widths(image) -> list[int]:
    """각 행에서 몸이 차지하는 픽셀 폭. 배경보다 어두운 픽셀을 몸으로 본다."""
    array = np.array(image.convert("L"))
    body = array < array.max() - 30
    return body.sum(axis=1).tolist()


def test_missing_circumferences_are_filled_from_height_and_weight():
    lean = estimate_missing(BodyTarget("men", 175, 60))
    heavy = estimate_missing(BodyTarget("men", 175, 95))
    assert lean["waist"] < heavy["waist"]
    assert lean["chest"] < heavy["chest"]
    # 키가 같으면 추정 어깨너비도 같아야 한다(둘레와 달리 체중에 거의 안 붙는다).
    assert lean["shoulder"] == heavy["shoulder"]


def test_supplied_measurements_are_never_overwritten():
    resolved = estimate_missing(BodyTarget("women", 165, 55, waist=61, hip=94))
    assert resolved["waist"] == 61
    assert resolved["hip"] == 94


def test_goals_only_include_fittable_measurements():
    goals = BodyTarget("men", 178, 72).goals()
    assert set(goals) >= {"height", "chest", "waist", "hip", "shoulder"}
    assert all(value > 0 for value in goals.values())


def test_taller_targets_get_a_smaller_head_relative_to_the_body():
    # 실루엣은 키와 상관없이 프레임을 꽉 채운다. 그래서 절대 신장은 픽셀 높이가
    # 아니라 두신지수(머리 대 몸 비율)로 전달된다.
    def head_fraction(height_cm: float) -> float:
        image, _ = render_proportional_silhouette(BodyTarget("men", height_cm, 65))
        widths = silhouette_widths(image)
        filled = [index for index, w in enumerate(widths) if w > 0]
        span = filled[-1] - filled[0]
        # 어깨선은 전신의 18.5% 지점이므로 위 12%는 머리만 들어 있다.
        head_only = widths[filled[0]: filled[0] + int(span * 0.12)]
        return max(head_only) / span

    assert head_fraction(190) < head_fraction(160)


def test_silhouette_is_wider_for_a_heavier_target_at_the_same_height():
    lean, _ = render_proportional_silhouette(BodyTarget("men", 175, 58))
    heavy, _ = render_proportional_silhouette(BodyTarget("men", 175, 98))
    assert max(silhouette_widths(heavy)) > max(silhouette_widths(lean))


def test_silhouette_waist_is_narrower_than_shoulders():
    image, _ = render_proportional_silhouette(BodyTarget("women", 165, 52))
    widths = silhouette_widths(image)
    filled = [index for index, w in enumerate(widths) if w > 0]
    span = filled[-1] - filled[0]
    shoulder_row = filled[0] + int(span * 0.20)
    waist_row = filled[0] + int(span * 0.375)
    assert widths[waist_row] < widths[shoulder_row]


def test_build_body_reference_falls_back_without_smplx_weights(monkeypatch):
    monkeypatch.setattr("avatar_body.SMPLX_MODEL_PATH", "")
    reference = build_body_reference(BodyTarget("men", 180, 75))
    assert reference.source == "silhouette-fallback"
    assert reference.image.size == (768, 1152)
    assert reference.target["height"] == 180


def test_errors_and_mae_report_the_gap_in_cm():
    reference = BodyReference(
        image=None,
        achieved={"height": 180.4, "waist": 79.0},
        target={"height": 180.0, "waist": 82.0},
    )
    assert reference.errors() == {"height": 0.4, "waist": -3.0}
    assert reference.mean_absolute_error() == 1.7


def test_avatar_prompt_points_at_the_reference_instead_of_naming_numbers():
    prompt = build_avatar_prompt(BodyTarget("women", 163, 52, waist=68))
    assert "Reference image 1 is a body-shape guide" in prompt
    assert "163" not in prompt and "68" not in prompt
    assert "adult Korean woman" in prompt


def test_measure_mesh_reads_a_synthetic_cylinder():
    # 반지름 0.15 m, 높이 1.7 m 원기둥: 둘레는 어느 높이에서 재도 2*pi*0.15 m.
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    rings = []
    for y in np.linspace(0, 1.7, 200):
        rings.append(np.stack([0.15 * np.cos(angles), np.full_like(angles, y), 0.15 * np.sin(angles)], axis=1))
    measured = measure_mesh(np.concatenate(rings))
    assert measured["height"] == pytest.approx(170.0, abs=1.0)
    for key in ("chest", "waist", "hip"):
        assert measured[key] == pytest.approx(2 * np.pi * 15, rel=0.02)
    assert measured["shoulder"] == pytest.approx(30.0, abs=0.5)


@pytest.mark.skipif(not os.getenv("SMPLX_MODEL_PATH"), reason="SMPL-X 가중치가 없으면 건너뛴다")
def test_smplx_fitting_lands_within_a_few_centimetres():
    reference = build_body_reference(BodyTarget("men", 178, 72, chest=98, waist=82, hip=96, shoulder=46))
    assert reference.source == "smplx-fitted"
    # 목표 5개 평균 오차 3 cm 이내면 옷 사이즈 판단에 쓸 만한 수준이다.
    assert reference.mean_absolute_error() < 3.0


def test_side_view_is_narrower_than_the_front_for_the_same_body():
    target = BodyTarget("men", 178, 72, shoulder=46, chest=98, waist=82, hip=96)
    front, _ = render_proportional_silhouette(target, view="front")
    side, _ = render_proportional_silhouette(target, view="side")
    # 옆에서 보이는 폭은 몸의 앞뒤 두께라 어깨너비보다 좁다.
    assert max(silhouette_widths(side)) < max(silhouette_widths(front))


def test_back_view_mirrors_the_front_silhouette():
    target = BodyTarget("women", 165, 55)
    front, _ = render_proportional_silhouette(target, view="front")
    back, _ = render_proportional_silhouette(target, view="back")
    # 뒤에서 봐도 실루엣 폭 분포는 같아야 한다(좌우만 뒤집힌다).
    assert silhouette_widths(back) == silhouette_widths(front)


def test_every_view_keeps_the_feet_inside_the_frame():
    # 발이 잘리던 문제. 실루엣 아래에 항상 빈 여백이 남아야 한다.
    for view in VIEW_YAW:
        image, _ = render_proportional_silhouette(BodyTarget("men", 190, 85), view=view)
        widths = silhouette_widths(image)
        bottom_margin = len(widths) - 1 - max(i for i, w in enumerate(widths) if w > 0)
        assert bottom_margin > image.height * 0.04, f"{view} 뷰의 발밑 여백이 부족하다"


def test_unknown_view_is_rejected():
    with pytest.raises(ValueError):
        build_body_reference(BodyTarget("men", 175, 70), view="top-down")


def test_rotate_y_turns_the_mesh_about_the_vertical_axis():
    points = np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]])
    turned = rotate_y(points, 90)
    assert turned[:, 1].tolist() == [0.5, 0.5]  # 높이는 그대로
    assert turned[0] == pytest.approx([0.0, 0.5, -1.0], abs=1e-6)
    assert turned[1] == pytest.approx([1.0, 0.5, 0.0], abs=1e-6)
    assert rotate_y(points, 0) is points  # 0도는 복사조차 하지 않는다


def test_side_and_back_prompts_ask_for_the_right_rotation():
    target = BodyTarget("men", 178, 72)
    side = build_avatar_prompt(target, view="side", identity_reference=True)
    back = build_avatar_prompt(target, view="back", identity_reference=True)
    assert "Exact left profile" in side and "turned 90 degrees" in side
    assert "directly behind" in back and "face turned entirely away" in back
    for prompt in (side, back):
        assert "Reference image 2 is the same person" in prompt


def test_front_prompt_has_no_identity_reference_by_default():
    prompt = build_avatar_prompt(BodyTarget("men", 178, 72))
    assert "Reference image 2" not in prompt


def test_every_view_prompt_demands_full_body_framing():
    # 신발까지 프레임 안에 들어와야 한다는 요구가 모든 시점에 들어가야 한다.
    for view in VIEW_YAW:
        prompt = build_avatar_prompt(BodyTarget("women", 165, 55), view=view)
        assert "soles of the feet inside the frame" in prompt
        assert "below the feet" in prompt


def _body_rows(image) -> tuple[int, int]:
    """이미지에서 몸이 차지하는 첫 행과 마지막 행."""
    widths = silhouette_widths(image)
    filled = [index for index, width in enumerate(widths) if width > 0]
    return filled[0], filled[-1]


def test_padding_creates_empty_space_below_a_body_that_ran_off_the_edge():
    from PIL import Image, ImageDraw

    # 종아리에서 잘린 인물 사진: 몸이 아래 모서리까지 닿아 있다.
    cropped = Image.new("RGB", (768, 1152), (154, 154, 152))
    ImageDraw.Draw(cropped).rectangle((280, 120, 490, 1151), fill=(90, 70, 60))
    assert _body_rows(cropped)[1] >= 1151

    padded = pad_for_full_body(cropped)

    assert padded.size == (768, 1152)
    top, bottom = _body_rows(padded)
    # 위아래 모두 배경이 남아야 모델이 "여기까지가 화면"이라고 읽는다.
    assert top > 0
    assert bottom < padded.height - 40


def test_padding_samples_the_background_not_the_body():
    from PIL import Image, ImageDraw

    background = (200, 30, 30)  # 눈에 띄는 색이라야 잘못 골랐을 때 바로 보인다
    image = Image.new("RGB", (768, 1152), background)
    ImageDraw.Draw(image).rectangle((300, 0, 470, 1151), fill=(20, 20, 20))

    padded = pad_for_full_body(image)

    # 맨 아랫줄 가운데는 몸이 아니라 배경색이어야 한다.
    pixel = padded.getpixel((padded.width // 2, padded.height - 4))
    assert abs(pixel[0] - background[0]) < 40 and pixel[0] > pixel[1]


def test_padding_keeps_a_body_that_already_had_margins():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (768, 1152), (240, 240, 240))
    ImageDraw.Draw(image).rectangle((300, 100, 470, 1000), fill=(40, 40, 40))
    top, bottom = _body_rows(pad_for_full_body(image))
    assert top > 0 and bottom < 1152
