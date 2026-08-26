"""가상 착장(try-on) 지시문 생성기.

FLUX.2 [klein]은 guidance-distilled 모델이라 추론 시 guidance_scale=1.0으로 돌고,
negative prompt가 실질적으로 작동하지 않는다. 그래서 "~하지 마"로 쓴 문장은
오히려 그 단어를 조건에 밀어 넣는 역효과가 난다(예: "no bag" → 가방이 생김).
이 모듈은 모든 제약을 **긍정문 + 명시적 배치 규칙**으로 바꿔서 지시문을 만든다.

기존 지시문의 실패 원인 3가지를 각각 다르게 처리한다.

1. 레이어 순서 미지정
   상의와 아우터를 그냥 나열하면 모델이 둘을 하나로 합치거나(item fusion)
   아우터를 안쪽에 그린다. -> 피부에서 바깥으로 나가는 순서를 stack 문장으로
   못 박고, 인접 레이어 사이의 가림(occlusion) 관계를 한 문장 더 붙인다.

2. 가방·액세서리의 착용 지점 부재
   "fashion accessory"만으로는 모자인지 벨트인지 모델이 알 수 없어 몸 아무 데나
   붙거나 아예 무시된다. -> 아이템 이름에서 착용 부위(anchor)를 추론해
   "어디에, 어떤 방향으로, 어느 레이어 위에" 걸치는지까지 문장으로 만든다.

3. 참조 이미지 순서와 레이어 순서의 불일치
   reference image 번호가 옷장에 담긴 순서 그대로라 모델이 번호를 레이어 힌트로
   오해한다. -> 항상 안쪽 레이어부터 번호를 매긴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

# 피부에서 바깥으로 나가는 순서. 참조 이미지 번호도 이 순서로 매긴다.
# 숫자가 작을수록 몸에 가깝다.
LAYER_RANK: dict[str, int] = {
    "overall": 10,
    "lower": 20,
    "upper": 30,
    "outer": 40,
    "shoes": 50,
    "bag": 60,
    "accessory": 70,
}

# 몸의 같은 자리를 두고 경쟁하는 카테고리. 같은 슬롯에 둘 이상 들어오면
# 레이어가 낮은(= 몸에 가까운) 쪽 하나만 남긴다.
BODY_SLOT: dict[str, str] = {
    "overall": "torso+legs",
    "upper": "torso",
    "lower": "legs",
    "outer": "outer-torso",
    "shoes": "feet",
    "bag": "carried",
    "accessory": "accessory",
}

# 액세서리는 여러 개를 동시에 착용할 수 있으므로 슬롯 충돌에서 제외한다.
MULTI_ITEM_SLOTS = {"accessory"}

# 원피스를 입으면 상·하의 슬롯이 통째로 덮인다.
SLOT_COVERS: dict[str, tuple[str, ...]] = {
    "torso+legs": ("torso", "legs"),
}


@dataclass(frozen=True)
class CategorySpec:
    label: str  # reference image N: <label>
    placement: str  # 어디에 어떻게 입는가
    stack_name: str  # 레이어 stack 문장에 쓸 짧은 이름


CATEGORY_SPECS: dict[str, CategorySpec] = {
    "overall": CategorySpec(
        label="one-piece garment (dress or jumpsuit)",
        placement=(
            "worn directly on the body as the single base layer covering torso and legs, "
            "shoulder seams sitting on the shoulders and the hem falling at its own natural length"
        ),
        stack_name="the one-piece garment",
    ),
    "lower": CategorySpec(
        label="lower-body garment (pants or skirt)",
        placement=(
            "worn on the hips and legs with the waistband at the natural waistline, "
            "both legs covered symmetrically and the hem breaking naturally over the shoes"
        ),
        stack_name="the bottoms",
    ),
    "upper": CategorySpec(
        label="upper-body top",
        placement=(
            "worn directly against the torso as the inner top layer, covering chest, shoulders and arms, "
            "its collar and hem fully formed even where a later layer covers them"
        ),
        stack_name="the top",
    ),
    "outer": CategorySpec(
        label="outerwear layer (jacket, coat or cardigan)",
        placement=(
            "worn open over everything else on the torso as the outermost torso layer, "
            "its own sleeves fully covering the arms, its front edges hanging outside and in front of the top, "
            "its collar sitting on top of the top's collar"
        ),
        stack_name="the outerwear",
    ),
    "shoes": CategorySpec(
        label="pair of shoes",
        placement=(
            "worn on both feet as a matching left-right pair, "
            "flat on the ground, the trouser or skirt hem falling over the shoe opening"
        ),
        stack_name="the shoes",
    ),
    "bag": CategorySpec(
        label="bag",
        placement="carried by the person",  # anchor 로 덮어씀
        stack_name="the bag",
    ),
    "accessory": CategorySpec(
        label="worn accessory",
        placement="worn on the body",  # anchor 로 덮어씀
        stack_name="the accessory",
    ),
}

# --- 착용 지점(anchor) 추론 -------------------------------------------------
# 옷장 아이템 이름(무신사 상품명이라 한글·영문이 섞여 있다)에서 착용 부위를
# 찾는다. 첫 일치를 쓰므로 더 구체적인 키워드를 앞에 둔다.

BAG_ANCHORS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("백팩", "배낭", "backpack", "rucksack"),
        "worn on the back with both straps over the shoulders, the straps lying on top of the outermost torso layer "
        "and the body of the bag visible behind the upper arm",
    ),
    (
        ("크로스", "메신저", "cross", "messenger", "sling"),
        "worn cross-body with the strap running diagonally from one shoulder across the chest, "
        "the strap lying on top of the outermost torso layer and the bag resting at the opposite hip",
    ),
    (
        ("토트", "핸드백", "tote", "handbag", "clutch"),
        "held in one hand at the person's side, hanging by its handles below the hip, "
        "fully outside every clothing layer",
    ),
    (
        ("숄더", "shoulder", "hobo"),
        "hung from one shoulder with the strap on top of the outermost torso layer, "
        "the bag resting against the hip on the same side",
    ),
    (
        ("웨이스트", "힙색", "waist", "belt bag", "fanny"),
        "strapped around the waist over the outermost torso layer, the pouch sitting at the front hip",
    ),
)

ACCESSORY_ANCHORS: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("모자", "캡", "비니", "버킷", "cap", "hat", "beanie", "bucket"),
        "worn on the head, brim or edge level and the hair falling naturally around it",
    ),
    (
        ("목도리", "머플러", "스카프", "scarf", "muffler"),
        "wrapped around the neck on top of the outermost torso layer, its ends hanging down the front",
    ),
    (
        ("벨트", "belt"),
        "fastened around the waist through the waistband of the bottoms, the buckle centred at the front",
    ),
    (
        ("안경", "선글", "glasses", "sunglass", "eyewear"),
        "worn on the face over the eyes, the temples resting on the ears",
    ),
    (
        ("시계", "팔찌", "watch", "bracelet"),
        "worn on one wrist, outside the sleeve cuff",
    ),
    (
        ("목걸이", "펜던트", "necklace", "pendant", "chain"),
        "worn around the neck, the pendant resting on the chest over the top layer",
    ),
    (
        ("귀걸이", "이어링", "earring"),
        "worn on the ears, small and symmetric on both sides",
    ),
    (
        ("반지", "ring"),
        "worn on one finger of a visible hand",
    ),
    (
        ("양말", "삭스", "sock"),
        "worn on both feet between the ankle and the shoe opening",
    ),
    (
        ("장갑", "glove"),
        "worn on both hands, cuffs overlapping the sleeve ends",
    ),
    (
        ("타이", "넥타이", "tie"),
        "knotted at the collar of the top, hanging down the centre of the chest",
    ),
)

DEFAULT_BAG_ANCHOR = (
    "hung from one shoulder with the strap on top of the outermost torso layer, "
    "the bag resting against the hip on the same side"
)
DEFAULT_ACCESSORY_ANCHOR = (
    "worn on the body part it is designed for, sitting on top of the clothing layer beneath it"
)


class GarmentLike(Protocol):
    category: str
    name: str


def _match_anchor(name: str, table: Iterable[tuple[tuple[str, ...], str]], default: str) -> str:
    lowered = (name or "").lower()
    for keywords, anchor in table:
        if any(keyword in lowered for keyword in keywords):
            return anchor
    return default


def resolve_placement(category: str, name: str) -> str:
    """카테고리 + 아이템 이름 -> 구체적인 착용 지점 문장."""
    if category == "bag":
        return _match_anchor(name, BAG_ANCHORS, DEFAULT_BAG_ANCHOR)
    if category == "accessory":
        return _match_anchor(name, ACCESSORY_ANCHORS, DEFAULT_ACCESSORY_ANCHOR)
    return CATEGORY_SPECS[category].placement


def order_garments(garments: list[GarmentLike], limit: int = 6) -> list[GarmentLike]:
    """레이어 순서로 정렬하고 슬롯 충돌을 정리한 뒤 limit개로 자른다.

    프론트엔드가 옷장 순서대로 앞에서 4개만 잘라 보내던 탓에, 상의·하의를 고르면
    신발과 가방이 통째로 사라졌다. 여기서는 자르기 전에 정렬부터 한다.
    """
    ordered = sorted(
        enumerate(garments),
        key=lambda pair: (LAYER_RANK.get(pair[1].category, 99), pair[0]),
    )

    taken: set[str] = set()
    kept: list[GarmentLike] = []
    for _, garment in ordered:
        slot = BODY_SLOT.get(garment.category, garment.category)
        if slot not in MULTI_ITEM_SLOTS and slot in taken:
            continue  # 같은 자리에 이미 더 안쪽 레이어가 들어갔다
        covered = SLOT_COVERS.get(slot, ())
        if slot not in MULTI_ITEM_SLOTS:
            taken.add(slot)
        taken.update(covered)
        kept.append(garment)
        if len(kept) >= limit:
            break
    return kept


def _torso_stack(garments: list[GarmentLike]) -> str | None:
    """몸통 레이어가 2개 이상일 때만 stack 문장을 만든다."""
    names = [
        CATEGORY_SPECS[item.category].stack_name
        for item in garments
        if item.category in {"overall", "upper", "outer"}
    ]
    if len(names) < 2:
        return None
    return (
        "Torso layering order from the skin outward: "
        + " then ".join(names)
        + ". Each of these stays a separate garment with its own visible edge — "
        "the outer one overlaps and partially hides the one beneath it, "
        "and the inner one still shows at the collar, the front opening and the hem."
    )


def _kept_slots_clause(garments: list[GarmentLike]) -> str:
    """제공되지 않은 부위는 아바타가 원래 입고 있던 옷을 그대로 두라는 지시."""
    provided = {BODY_SLOT.get(item.category, item.category) for item in garments}
    if "torso+legs" in provided:
        provided.update(("torso", "legs"))
    untouched = [
        korean
        for slot, korean in (("torso", "top"), ("legs", "bottoms"), ("feet", "shoes"))
        if slot not in provided
    ]
    if not untouched:
        return ""
    return (
        " Leave the person's existing "
        + ", ".join(untouched)
        + " exactly as they already appear in reference image 1."
    )


def build_tryon_prompt(garments: list[GarmentLike]) -> str:
    """정렬이 끝난 착장 목록 -> FLUX.2 편집 지시문.

    호출 전에 order_garments()를 거친 목록을 넘겨야 참조 이미지 번호가
    레이어 순서와 일치한다.
    """
    roles = []
    for index, item in enumerate(garments):
        spec = CATEGORY_SPECS[item.category]
        name = (item.name or "").strip() or spec.label
        roles.append(
            f"Reference image {index + 2} is the {spec.label} '{name}', "
            f"{resolve_placement(item.category, name)}."
        )

    parts = [
        "Multi-reference virtual try-on edit. Reference image 1 is the person: keep the same face, "
        "hair, skin tone, body proportions, pose, hands, feet and background exactly as they are.",
        " ".join(roles),
    ]

    stack = _torso_stack(garments)
    if stack:
        parts.append(stack)

    parts.append(
        "Copy each referenced item as it truly is: same colour, fabric, weave, print, logo position and size, "
        "neckline, sleeve length, cut, closure and silhouette. Show each item alone on this person, "
        "cropped away from whatever model, hanger or background it came with."
    )
    parts.append(
        f"The person wears exactly these {len(garments)} referenced item"
        f"{'s' if len(garments) != 1 else ''} and nothing else is added."
        + _kept_slots_clause(garments)
    )
    parts.append(
        "Make the result physically plausible: fabric follows the body, folds and shadows fall with gravity, "
        "straps and hems rest on the surface directly beneath them, and every item shares one studio light source."
    )
    # 신발을 신기면 발끝이 프레임 밖으로 밀려나 잘리는 일이 잦다. 프레이밍을
    # 명시적으로 요구해야 모델이 아래쪽 여백을 남긴다.
    parts.append(
        "Keep the whole body inside the frame from the top of the head to the soles of the shoes, with clear "
        "empty background above the head and below the feet, and both feet fully visible on the ground. "
        "Photorealistic full-body Korean fashion e-commerce photograph, single frame, clean image with no lettering."
    )
    return " ".join(parts)
