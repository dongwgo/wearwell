"""착장 지시문 생성기 테스트.

이 파일이 지키는 것은 이미지 품질이 아니라 **지시문의 계약**이다.
품질 자체는 scripts/eval_tryon.py의 VLM 판정으로 따로 측정한다.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from tryon_prompt import (  # noqa: E402
    LAYER_RANK,
    build_tryon_prompt,
    build_tryon_view_prompt,
    layer_rank,
    order_garments,
    resolve_placement,
)


@dataclass
class Garment:
    category: str
    name: str = "아이템"


def test_layer_rank_puts_inner_layers_first():
    assert LAYER_RANK["upper"] < LAYER_RANK["outer"]
    assert LAYER_RANK["lower"] < LAYER_RANK["upper"]
    assert LAYER_RANK["outer"] < LAYER_RANK["bag"] < LAYER_RANK["accessory"]


def test_order_garments_sorts_outerwear_after_the_top():
    ordered = order_garments([Garment("outer"), Garment("bag"), Garment("upper"), Garment("lower")])
    assert [item.category for item in ordered] == ["lower", "upper", "outer", "bag"]


def test_order_garments_keeps_shoes_and_bag_that_used_to_be_truncated():
    # 예전에는 옷장 순서로 앞 4개만 잘라서 신발·가방이 사라졌다.
    picked = [
        Garment("accessory", "볼캡"),
        Garment("bag", "크로스백"),
        Garment("shoes", "스니커즈"),
        Garment("outer", "블레이저"),
        Garment("upper", "셔츠"),
        Garment("lower", "슬랙스"),
    ]
    ordered = order_garments(picked)
    assert [item.category for item in ordered] == [
        "lower",
        "upper",
        "outer",
        "shoes",
        "bag",
        "accessory",
    ]


def test_order_garments_drops_the_outer_layer_competing_for_one_slot():
    # 상의를 두 벌 고르면 둘이 한 벌로 합쳐진다. 몸에 가까운 쪽 하나만 남긴다.
    ordered = order_garments([Garment("upper", "티셔츠"), Garment("upper", "셔츠")])
    assert [item.name for item in ordered] == ["티셔츠"]


def test_one_piece_garment_suppresses_separate_top_and_bottom():
    ordered = order_garments([Garment("overall", "원피스"), Garment("upper"), Garment("lower")])
    assert [item.category for item in ordered] == ["overall"]


def test_multiple_accessories_are_allowed_together():
    ordered = order_garments([Garment("accessory", "볼캡"), Garment("accessory", "실버 목걸이")])
    assert len(ordered) == 2


def test_order_garments_respects_the_limit():
    assert len(order_garments([Garment("accessory", f"악세{i}") for i in range(10)], limit=6)) == 6


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("나일론 백팩", "both straps over the shoulders"),
        ("레더 크로스백", "cross-body"),
        ("캔버스 토트백", "held in one hand"),
        ("이름 없는 가방", "from one shoulder"),
    ],
)
def test_bag_placement_follows_the_bag_type(name, expected):
    assert expected in resolve_placement("bag", name)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("코튼 볼캡", "on the head"),
        ("울 머플러", "around the neck"),
        ("가죽 벨트", "around the waist"),
        ("메탈 선글라스", "squarely over both eyes"),
        ("실버 목걸이", "around the neck"),
    ],
)
def test_accessory_placement_names_the_body_part(name, expected):
    assert expected in resolve_placement("accessory", name)


def test_prompt_numbers_references_in_layering_order():
    prompt = build_tryon_prompt(order_garments([Garment("outer", "코트"), Garment("upper", "셔츠")]))
    assert prompt.index("Reference image 2 is the upper-body top") < prompt.index("Reference image 3 is the outerwear")


def test_prompt_states_the_torso_stack_when_layers_overlap():
    prompt = build_tryon_prompt(order_garments([Garment("outer", "코트"), Garment("upper", "셔츠")]))
    assert "Torso layering order from the skin outward: the top then the outerwear" in prompt
    assert "outer one overlaps and partially hides the one beneath" in prompt


def test_prompt_omits_the_stack_sentence_for_a_single_torso_layer():
    prompt = build_tryon_prompt(order_garments([Garment("upper", "셔츠"), Garment("shoes", "로퍼")]))
    assert "Torso layering order" not in prompt


def test_prompt_tells_the_model_to_keep_untouched_body_parts():
    prompt = build_tryon_prompt(order_garments([Garment("upper", "셔츠")]))
    assert "Leave the person's existing bottoms, shoes exactly as they already appear" in prompt


def test_prompt_has_no_untouched_clause_for_a_complete_outfit():
    prompt = build_tryon_prompt(
        order_garments([Garment("upper"), Garment("lower"), Garment("shoes")])
    )
    assert "Leave the person's existing" not in prompt


def test_prompt_avoids_negated_instructions():
    # guidance-distilled 모델에서는 negative가 먹지 않고 오히려 그 단어를 조건에
    # 밀어 넣는다. 제약은 전부 긍정문이어야 한다.
    prompt = build_tryon_prompt(
        order_garments([Garment("upper"), Garment("outer"), Garment("bag", "크로스백")])
    ).lower()
    for banned in (" no extra", "no invented", "no collage", "don't", "avoid ", "without "):
        assert banned not in prompt


def test_prompt_pins_identity_to_the_first_reference():
    prompt = build_tryon_prompt(order_garments([Garment("upper")]))
    assert "reference image 1 is the person" in prompt.lower()
    assert "same face" in prompt


def test_prompt_counts_the_items_it_actually_applies():
    ordered = order_garments([Garment("upper"), Garment("lower"), Garment("shoes")])
    assert "exactly these 3 referenced items" in build_tryon_prompt(ordered)


def test_prompt_demands_the_whole_body_including_the_shoes():
    # 신발을 신기면 발끝이 프레임 밖으로 밀려 잘리던 문제.
    prompt = build_tryon_prompt(order_garments([Garment("upper"), Garment("shoes", "스니커즈")]))
    assert "soles of the shoes" in prompt
    # 넣어준 사람 사진이 종아리에서 끝나도 아래로 구도를 넓히라고 지시해야 한다.
    assert "widen the composition downward" in prompt


def test_socks_go_on_before_the_shoes():
    ordered = order_garments([Garment("shoes", "화이트 스니커즈"), Garment("accessory", "크루 양말")])
    assert [item.name for item in ordered] == ["크루 양말", "화이트 스니커즈"]


def test_belt_sits_between_the_top_and_the_outerwear():
    ordered = order_garments([
        Garment("outer", "블레이저"), Garment("accessory", "가죽 벨트"),
        Garment("upper", "셔츠"), Garment("lower", "슬랙스"),
    ])
    assert [item.name for item in ordered] == ["슬랙스", "셔츠", "가죽 벨트", "블레이저"]


def test_scarf_goes_over_the_outerwear_but_hat_comes_last():
    ordered = order_garments([
        Garment("accessory", "볼캡"), Garment("accessory", "울 머플러"), Garment("outer", "코트"),
    ])
    assert [item.name for item in ordered] == ["코트", "울 머플러", "볼캡"]


def test_layer_rank_separates_accessories_by_what_they_are():
    # 양말과 모자는 둘 다 '액세서리'지만 착용 순서가 완전히 다르다.
    assert layer_rank(Garment("accessory", "크루 양말")) < layer_rank(Garment("shoes", "스니커즈"))
    assert layer_rank(Garment("accessory", "볼캡")) > layer_rank(Garment("shoes", "스니커즈"))
    # 이름을 알 수 없는 액세서리는 기본 순위로 떨어진다.
    assert layer_rank(Garment("accessory", "정체불명")) == 70


def test_prompt_spells_out_the_sock_then_shoe_order():
    prompt = build_tryon_prompt(
        order_garments([Garment("shoes", "화이트 스니커즈"), Garment("accessory", "크루 양말")])
    )
    assert "'크루 양말' goes on the bare foot first, then '화이트 스니커즈' goes on over it" in prompt


def test_sock_order_sentence_is_omitted_without_shoes():
    prompt = build_tryon_prompt(order_garments([Garment("accessory", "크루 양말"), Garment("upper")]))
    assert "goes on the bare foot first" not in prompt


def test_small_accessories_are_named_again_so_they_do_not_vanish():
    prompt = build_tryon_prompt(
        order_garments([Garment("upper", "셔츠"), Garment("accessory", "메탈 선글라스"),
                        Garment("accessory", "실버 시계")])
    )
    # 나열 순서는 착용 순서를 따른다 — 시계(손목, 44)가 선글라스(얼굴, 72)보다 앞.
    assert "'실버 시계', '메탈 선글라스' are small in the frame and must still be clearly visible" in prompt


def test_large_accessories_do_not_trigger_the_small_item_clause():
    prompt = build_tryon_prompt(order_garments([Garment("upper", "셔츠"), Garment("accessory", "볼캡")]))
    assert "small in the frame" not in prompt


def test_view_prompt_rotates_the_finished_outfit_instead_of_redressing():
    prompt = build_tryon_view_prompt("side", [Garment("upper", "셔츠"), Garment("outer", "코트")])
    assert "Reference image 2 is the finished photograph" in prompt
    assert "The outfit is '셔츠', '코트'" in prompt
    assert "exact left profile" in prompt
    # 회전 중에 레이어가 뒤집히지 않도록 순서를 다시 못 박는다.
    assert "outer layers still outside the inner ones" in prompt


def test_view_prompt_rejects_an_unknown_view():
    with pytest.raises(ValueError):
        build_tryon_view_prompt("top-down", [Garment("upper")])


# --- 프론트엔드와 공유하는 순서 픽스처 -------------------------------------
# app.js가 6벌로 자르기 전에 무엇을 남길지 스스로 정해야 해서, 같은 규칙이
# 양쪽에 존재한다. 두 구현이 갈라지지 않도록 같은 픽스처로 양쪽을 못 박는다.
# (JS 쪽은 scripts/avatar-view-test.mjs가 같은 파일을 읽어서 검사한다.)

CATEGORY_BY_KOREAN = {
    "상의": "upper", "하의": "lower", "원피스": "overall", "아우터": "outer",
    "신발": "shoes", "가방": "bag", "액세서리": "accessory",
}
ORDERING_CASES = json.loads(
    (BACKEND.parent / "eval" / "ordering-cases.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize("case", ORDERING_CASES, ids=[c["title"] for c in ORDERING_CASES])
def test_shared_ordering_fixture(case):
    items = [Garment(CATEGORY_BY_KOREAN[i["category"]], i["name"]) for i in case["items"]]
    assert [item.name for item in order_garments(items, limit=6)] == case["expected"]


def test_ordering_fixture_covers_the_rules_that_bit_us():
    titles = " ".join(case["title"] for case in ORDERING_CASES)
    for topic in ("속옷", "양말", "벨트", "원피스", "브라운"):
        assert topic in titles, f"{topic} 케이스가 픽스처에서 빠졌다"
