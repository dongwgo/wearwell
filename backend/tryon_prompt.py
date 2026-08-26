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
#
# rank는 이 아이템이 레이어 스택의 어디에 끼는지다. 카테고리만으로는 부족하다 —
# 양말과 모자는 둘 다 '액세서리'지만 양말은 신발보다 **먼저** 신고 모자는 전부
# 입은 뒤에 쓴다. 하나의 rank로 묶으면 양말이 신발 위로 올라간다.


@dataclass(frozen=True)
class WornSpec:
    anchor: str  # 어디에, 어떤 방향으로, 어느 레이어 위에
    rank: int  # LAYER_RANK와 같은 축의 착용 순서
    small: bool = False  # 화면에서 작아 모델이 통째로 빠뜨리기 쉬운가
    slot: str | None = None  # 몸의 어느 자리를 차지하는가 (None이면 카테고리 기본값)
    label: str | None = None  # 참조 이미지 설명 (None이면 카테고리 기본값)


# --- 속옷 -------------------------------------------------------------------
# 속옷은 무엇보다 먼저 입는다. rank 5로 모든 옷 앞에 세운다. 이 지시가 없으면
# 바지 위에 팬티를 그리는 결과가 나온다.
#
# 속옷 판정은 카테고리가 아니라 이름으로 한다 — 브라는 '상의', 팬티는 '하의',
# 그냥 '속옷'은 '액세서리'로 분류돼 들어오기 때문이다. 대신 이름 매칭이
# 카테고리를 가로지르므로 오탐이 훨씬 위험해진다. 아래 제외어를 먼저 본다.
UNDERWEAR_EXCLUSIONS = (
    "브라운",  # brown. '브라'가 여기에 걸린다
    "슬립온", "슬립 온", "slip-on", "slipon",  # 신발
    "브레이슬릿", "bracelet",
    "트렁크 케이스", "트렁크캐리어",
)

UNDERWEAR_ANCHORS: tuple[tuple[tuple[str, ...], WornSpec], ...] = (
    (
        ("팬티", "드로즈", "드로어즈", "트렁크", "boxer", "brief", "panty", "panties", "thong"),
        WornSpec(
            "worn directly on the skin as the very first layer on the hips, completely hidden "
            "underneath the bottoms so that no part of it shows outside the trousers or skirt",
            5,
            slot="underwear-legs",
            label="pair of underwear briefs",
        ),
    ),
    (
        ("브래지어", "브라탑", "브라렛", "캐미솔", "브라 ", "brassiere", "bralette", "camisole", "bra top"),
        WornSpec(
            "worn directly on the skin as the very first layer on the torso, completely hidden "
            "underneath the top so that no part of it shows outside the clothing",
            5,
            slot="underwear-torso",
            label="bra",
        ),
    ),
    (
        ("속옷", "언더웨어", "이너웨어", "underwear", "lingerie"),
        WornSpec(
            "worn directly on the skin as the very first layer, completely hidden underneath "
            "everything else so that no part of it shows outside the clothing",
            5,
            label="undergarment",
        ),
    ),
)

# 속옷 이름 판정을 돌릴 카테고리. 가방·신발·아우터에서는 볼 필요가 없고,
# 보지 않아야 '트렁크 캐리어' 같은 오탐이 애초에 생기지 않는다.
UNDERWEAR_CATEGORIES = {"upper", "lower", "accessory"}


BAG_ANCHORS: tuple[tuple[tuple[str, ...], WornSpec], ...] = (
    (
        ("백팩", "배낭", "backpack", "rucksack"),
        WornSpec(
            "worn on the back with both straps over the shoulders, the straps lying on top of the "
            "outermost torso layer and the body of the bag visible behind the upper arm",
            60,
        ),
    ),
    (
        ("크로스", "메신저", "cross", "messenger", "sling"),
        WornSpec(
            "worn cross-body with the strap running diagonally from one shoulder across the chest, "
            "the strap lying on top of the outermost torso layer and the bag resting at the opposite hip",
            60,
        ),
    ),
    (
        ("토트", "핸드백", "tote", "handbag", "clutch"),
        WornSpec(
            "held in one hand at the person's side, hanging by its handles below the hip, "
            "fully outside every clothing layer",
            60,
        ),
    ),
    (
        ("숄더", "shoulder", "hobo"),
        WornSpec(
            "hung from one shoulder with the strap on top of the outermost torso layer, "
            "the bag resting against the hip on the same side",
            60,
        ),
    ),
    (
        ("웨이스트", "힙색", "waist", "belt bag", "fanny"),
        WornSpec(
            "strapped around the waist over the outermost torso layer, the pouch sitting at the front hip",
            60,
        ),
    ),
)

ACCESSORY_ANCHORS: tuple[tuple[tuple[str, ...], WornSpec], ...] = (
    (
        # 양말은 신발보다 먼저 신는다. rank 45로 shoes(50) 앞에 세운다.
        ("양말", "삭스", "sock"),
        WornSpec(
            "pulled onto the bare feet first, covering the ankle, so the shoes go on over them and only "
            "the sock cuff shows between the trouser hem and the shoe opening",
            45,
        ),
    ),
    (
        # 벨트는 하의 위, 아우터 아래. 상의를 넣어 입었으면 상의 위로 온다.
        ("벨트", "belt"),
        WornSpec(
            "fastened around the waist through the belt loops of the bottoms, over the hem of the top "
            "and underneath any outerwear, the buckle centred at the front",
            35,
        ),
    ),
    (
        ("타이", "넥타이", "tie"),
        WornSpec(
            "knotted at the collar of the top and hanging down the centre of the chest, "
            "lying on the shirt and underneath any outerwear",
            35,
        ),
    ),
    (
        ("목걸이", "펜던트", "necklace", "pendant", "chain"),
        WornSpec(
            "worn around the neck with the pendant resting on the chest, on top of the top layer",
            35,
            small=True,
        ),
    ),
    (
        ("시계", "팔찌", "watch", "bracelet"),
        WornSpec(
            "worn on one wrist, its face turned toward the camera and clearly readable, "
            "sitting outside the sleeve cuff",
            44,
            small=True,
        ),
    ),
    (
        ("반지", "ring"),
        WornSpec("worn on one finger of a visible hand", 44, small=True),
    ),
    (
        ("장갑", "glove"),
        WornSpec(
            "worn on both hands with the cuffs overlapping the ends of the outermost sleeves",
            65,
        ),
    ),
    (
        ("목도리", "머플러", "스카프", "scarf", "muffler"),
        WornSpec(
            "wrapped around the neck on top of the outermost torso layer, its ends hanging down the front",
            65,
        ),
    ),
    (
        # 안경류는 얼굴에서 차지하는 면적이 작아 통째로 사라지기 쉽다. 위치를
        # 구체적으로 지정하고 small=True로 강조 문장까지 붙인다.
        ("안경", "선글", "glasses", "sunglass", "eyewear", "고글"),
        WornSpec(
            "worn on the face with the lenses squarely over both eyes, the bridge resting on the nose "
            "and the temple arms running back over the ears, drawn at full size and in sharp focus",
            72,
            small=True,
        ),
    ),
    (
        ("모자", "캡", "비니", "버킷", "cap", "hat", "beanie", "bucket"),
        WornSpec(
            "worn on the head with the brim or edge level, sitting on top of the hair",
            72,
        ),
    ),
    (
        ("귀걸이", "이어링", "earring"),
        WornSpec("worn on the ears, small and symmetric on both sides", 72, small=True),
    ),
)

DEFAULT_BAG_SPEC = WornSpec(
    "hung from one shoulder with the strap on top of the outermost torso layer, "
    "the bag resting against the hip on the same side",
    60,
)
DEFAULT_ACCESSORY_SPEC = WornSpec(
    "worn on the body part it is designed for, sitting on top of the clothing layer beneath it",
    70,
)


class GarmentLike(Protocol):
    category: str
    name: str


def _match_spec(name: str, table, default: WornSpec) -> WornSpec:
    lowered = (name or "").lower()
    for keywords, spec in table:
        if any(keyword in lowered for keyword in keywords):
            return spec
    return default


def _strip_lookalikes(name: str) -> str:
    """속옷 키워드를 잘못 품고 있는 낱말을 먼저 지운다.

    통째로 거부하면 '브라운 드로즈'까지 놓친다. 문제가 되는 낱말만 걷어내고
    나머지로 판정해야 '브라운 자켓'은 거르고 '브라운 드로즈'는 잡는다.
    """
    lowered = (name or "").lower()
    for bad in UNDERWEAR_EXCLUSIONS:
        lowered = lowered.replace(bad, " ")
    return lowered


def is_underwear(category: str, name: str) -> bool:
    if category not in UNDERWEAR_CATEGORIES:
        return False
    cleaned = _strip_lookalikes(name)
    return any(word in cleaned for words, _ in UNDERWEAR_ANCHORS for word in words)


def resolve_spec(category: str, name: str) -> WornSpec | None:
    """이 아이템을 어디에, 몇 번째로 입는가. 해당 없으면 None.

    속옷을 가장 먼저 본다 — 브라는 '상의', 팬티는 '하의'로 들어오기 때문에
    카테고리로 갈라놓으면 잡을 수 없다.
    """
    if is_underwear(category, name):
        spec = _match_spec(_strip_lookalikes(name), UNDERWEAR_ANCHORS, DEFAULT_ACCESSORY_SPEC)
        if spec.slot:
            return spec
        # '속옷'처럼 부위를 알 수 없는 이름은 카테고리에서 자리를 가져온다.
        base = BODY_SLOT.get(category, category)
        return WornSpec(spec.anchor, spec.rank, spec.small, f"underwear-{base}", spec.label)
    if category == "bag":
        return _match_spec(name, BAG_ANCHORS, DEFAULT_BAG_SPEC)
    if category == "accessory":
        return _match_spec(name, ACCESSORY_ANCHORS, DEFAULT_ACCESSORY_SPEC)
    return None


def body_slot(item: GarmentLike) -> str:
    """이 아이템이 몸의 어느 자리를 차지하는가.

    속옷은 겉옷과 같은 자리를 다투지 않는다 — 팬티와 바지는 함께 입는 것이지
    둘 중 하나를 고르는 게 아니다. 그래서 별도 슬롯을 준다.
    """
    spec = resolve_spec(item.category, getattr(item, "name", ""))
    if spec and spec.slot:
        return spec.slot
    return BODY_SLOT.get(item.category, item.category)


def resolve_placement(category: str, name: str) -> str:
    """카테고리 + 아이템 이름 -> 구체적인 착용 지점 문장."""
    spec = resolve_spec(category, name)
    return spec.anchor if spec else CATEGORY_SPECS[category].placement


def layer_rank(item: GarmentLike) -> int:
    """이 아이템이 레이어 스택의 어디에 끼는가.

    카테고리만 보면 양말과 모자가 같은 '액세서리'라 같은 순위를 받고, 양말이
    신발 위로 올라간다. 이름에서 종류를 알아낸 경우에는 그 종류의 순위를 쓴다.
    """
    spec = resolve_spec(item.category, getattr(item, "name", ""))
    return spec.rank if spec else LAYER_RANK.get(item.category, 99)


def order_garments(garments: list[GarmentLike], limit: int = 6) -> list[GarmentLike]:
    """레이어 순서로 정렬하고 슬롯 충돌을 정리한 뒤 limit개로 자른다.

    프론트엔드가 옷장 순서대로 앞에서 4개만 잘라 보내던 탓에, 상의·하의를 고르면
    신발과 가방이 통째로 사라졌다. 여기서는 자르기 전에 정렬부터 한다.
    """
    ordered = sorted(enumerate(garments), key=lambda pair: (layer_rank(pair[1]), pair[0]))

    taken: set[str] = set()
    kept: list[GarmentLike] = []
    for _, garment in ordered:
        slot = body_slot(garment)
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


def _base_layer_clause(garments: list[GarmentLike]) -> str | None:
    """속옷이 포함됐을 때 "맨 아래, 그리고 안 보이게"를 못 박는다.

    이 문장이 없으면 바지 위에 팬티가 그려진다. 참조 이미지에서 속옷은 단독
    컷이라 모델이 겉옷으로 오해하기 쉽다.
    """
    names = [
        (item.name or "").strip()
        for item in garments
        if is_underwear(item.category, getattr(item, "name", ""))
    ]
    names = [name for name in names if name]
    if not names:
        return None
    listed = ", ".join(f"'{name}'" for name in names)
    verb = "is" if len(names) == 1 else "are"
    return (
        f"{listed} {verb} underwear: it goes on the bare skin before every other item and every later "
        "layer covers it completely, so the finished photograph shows the outer clothing and the "
        "underwear stays out of sight beneath it."
    )


def _feet_stack(garments: list[GarmentLike]) -> str | None:
    """양말과 신발이 함께 있을 때만 신는 순서를 못 박는다.

    둘 다 발에 가는 아이템이라 지시가 없으면 모델이 양말을 신발 위에 그리거나
    양말을 통째로 빼먹는다.
    """
    names = {}
    for item in garments:
        spec = resolve_spec(item.category, getattr(item, "name", ""))
        if spec and spec.rank == 45:
            names["socks"] = (item.name or "socks").strip()
        elif item.category == "shoes":
            names["shoes"] = (item.name or "shoes").strip()
    if len(names) < 2:
        return None
    return (
        f"On the feet the order is: '{names['socks']}' goes on the bare foot first, then "
        f"'{names['shoes']}' goes on over it. The shoe covers the foot and only the sock cuff "
        "stays visible above the shoe opening."
    )


def _small_item_clause(garments: list[GarmentLike]) -> str | None:
    """작은 아이템은 이름을 한 번 더 불러 준다.

    안경·시계·귀걸이는 프레임에서 차지하는 면적이 워낙 작아, 참조 이미지로
    줘도 모델이 통째로 빠뜨린다. 목록 마지막에 이름을 다시 세워 두면 눈에
    띄게 살아난다.
    """
    small = [
        (item.name or "").strip()
        for item in garments
        if (spec := resolve_spec(item.category, getattr(item, "name", ""))) and spec.small
    ]
    small = [name for name in small if name]
    if not small:
        return None
    listed = ", ".join(f"'{name}'" for name in small)
    verb = "is" if len(small) == 1 else "are"
    return (
        f"{listed} {verb} small in the frame and must still be clearly visible on the person, "
        "drawn sharply at natural size in the right place."
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
        name = (item.name or "").strip()
        worn = resolve_spec(item.category, name)
        label = (worn.label if worn and worn.label else CATEGORY_SPECS[item.category].label)
        roles.append(
            f"Reference image {index + 2} is the {label} '{name or label}', "
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

    base = _base_layer_clause(garments)
    if base:
        parts.append(base)

    feet = _feet_stack(garments)
    if feet:
        parts.append(feet)

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
    small = _small_item_clause(garments)
    if small:
        parts.append(small)

    parts.append(
        "Make the result physically plausible: fabric follows the body, folds and shadows fall with gravity, "
        "straps and hems rest on the surface directly beneath them, and every item shares one studio light source."
    )
    # 신발을 신기면 발끝이 프레임 밖으로 밀려나 잘리는 일이 잦다. 프레이밍을
    # 명시적으로 요구해야 모델이 아래쪽 여백을 남긴다.
    parts.append(
        "Frame the shot as a complete full-length photograph: the top of the head and the soles of the shoes "
        "are both inside the picture, with clear empty background above the head and below the feet. "
        "Where reference image 1 stops short of the feet, widen the composition downward and draw the rest of "
        "the legs and the shoes standing on the ground. "
        "Photorealistic full-body Korean fashion e-commerce photograph, single frame, clean image with no lettering."
    )
    return " ".join(parts)


# --- 다른 시점에서 본 착장 ---------------------------------------------------

ROTATION_DIRECTION = {
    "side": (
        "an exact left profile, the body turned 90 degrees so only one side faces the camera, "
        "arms hanging relaxed beside the torso"
    ),
    "back": (
        "seen from directly behind, the back of the head and the shoulder blades toward the camera, "
        "the face turned entirely away"
    ),
}


def build_tryon_view_prompt(view: str, garments: list[GarmentLike]) -> str:
    """완성된 정면 착장을 다른 시점으로 돌리는 지시문.

    참조 순서가 중요하다. 예전에는 체형 가이드(회색 마네킹)를 참조 1번에 두고
    완성된 정면을 2번에 뒀는데, FLUX.2는 1번 참조를 가장 강하게 따라가서
    "옷 없는 회색 인체"가 착장을 밀어냈다 — 후면에서 패딩이 통째로 사라지거나
    토트백이 백팩으로 바뀌는 결과가 여기서 나왔다.

    그래서 이제 **완성된 정면이 참조 1번**이고, 옷 사진들을 다시 뒤에 붙인다.
    옷 참조를 다시 넣는 게 낭비 같지만, 회전 중에 아이템이 다른 물건으로
    바뀌는 것을 막는 유일한 근거다. 마네킹 가이드는 아예 빼 버렸다 — 체형
    정보는 이미 정면 사진 안에 다 들어 있다.
    """
    if view not in ROTATION_DIRECTION:
        raise ValueError(f"unknown view: {view}")

    roles = []
    for index, item in enumerate(garments):
        name = (item.name or "").strip()
        worn = resolve_spec(item.category, name)
        label = (worn.label if worn and worn.label else CATEGORY_SPECS[item.category].label)
        roles.append(
            f"Reference image {index + 2} is the same {label} '{name or label}' that the person is "
            f"already wearing, {resolve_placement(item.category, name)}."
        )

    parts = [
        "Rotate the camera around a dressed person. Reference image 1 is the finished photograph of "
        "this person wearing the complete outfit, seen from the front. Keep that exact person and that "
        "exact outfit: same face structure, same hair, same body, same garments, same studio lighting "
        "and background.",
    ]
    if roles:
        parts.append(" ".join(roles))
        parts.append(
            "Each of those items stays the same object it is in its own reference image — same shape, "
            "colour, fabric, print and hardware, worn the same way — while the camera moves around it."
        )
    parts.append(
        f"Draw that identical person in that identical outfit {ROTATION_DIRECTION[view]}. "
        "The layering order is unchanged, with the outer layers still outside the inner ones, and every "
        "item that is visible from the front is still on the body from this angle."
    )
    parts.append(
        "Frame the shot as a complete full-length photograph: the top of the head and the soles of the "
        "shoes are both inside the picture, with clear empty background above the head and below the feet. "
        "Photorealistic full-body Korean fashion e-commerce photograph, single frame, clean image with "
        "no lettering."
    )
    return " ".join(parts)
