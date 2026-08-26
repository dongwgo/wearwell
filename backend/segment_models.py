"""세그멘테이션 모델 레지스트리 — 옷 분리 성능을 모델별로 바꿔가며 비교하기 위한 정의.

모델마다 라벨 체계가 다르다(ATR 18 클래스 vs Fashionpedia 46 클래스). 그래서
"라벨 -> 옷장 카테고리" 매핑을 전역 상수가 아니라 각 모델 스펙에 붙여 둔다.
torch/transformers를 import하지 않는 순수 파이썬 모듈이라 app.py가 모델 없이도
목록을 응답할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 앱의 옷장 카테고리(app.js의 `categories`)와 같은 이름을 쓴다.
CATEGORIES = ["아우터", "상의", "하의", "원피스", "신발", "가방", "액세서리"]

# 오버레이 색. 모델이 달라도 같은 카테고리는 같은 색으로 칠해야 눈으로 비교가 된다.
CATEGORY_COLORS = {
    "상의": (255, 113, 91),
    "아우터": (255, 176, 59),
    "하의": (74, 144, 226),
    "원피스": (183, 110, 224),
    "신발": (72, 187, 120),
    "가방": (240, 201, 72),
    "액세서리": (236, 98, 160),
}

# 완성도 낮은 검출을 옷장에 넣지 않기 위한 3중 필터.
# 하나라도 못 넘으면 그 아이템은 버린다(사진에 옷이 없다는 뜻이 아니라
# 이번 검출의 잘라낸 결과물을 믿을 수 없다는 뜻).
#
# fill ratio(바운딩박스 안에서 마스크가 실제로 채우는 비율)는 카테고리별로 기준이
# 다르다 — 상의/하의/원피스는 몸통을 덩어리로 덮어서 박스를 잘 채우지만, 신발은
# 대각선으로 놓이고 가방/액세서리는 끈처럼 가늘어서 정상 검출도 박스를 듬성듬성
# 채운다. 하나의 기준을 쓰면 후자 세 카테고리가 통째로 걸러진다(실측: 0.28 고정
# 기준에서 신발 56%->9%, 가방 36%->8%, 액세서리 26%->2%로 붕괴).
QUALITY_THRESHOLDS = {
    "상의": {"minArea": 0.012, "minFill": 0.25, "minConfidence": 0.55},
    "아우터": {"minArea": 0.012, "minFill": 0.25, "minConfidence": 0.55},
    "하의": {"minArea": 0.012, "minFill": 0.25, "minConfidence": 0.55},
    "원피스": {"minArea": 0.02, "minFill": 0.25, "minConfidence": 0.55},
    "신발": {"minArea": 0.004, "minFill": 0.08, "minConfidence": 0.5},
    "가방": {"minArea": 0.004, "minFill": 0.08, "minConfidence": 0.5},
    "액세서리": {"minArea": 0.003, "minFill": 0.08, "minConfidence": 0.5},
}

# ATR(18 클래스) 라벨 -> 옷장 카테고리.
# ATR에는 아우터 클래스가 없다. 코트/재킷도 전부 Upper-clothes로 나오므로 ATR
# 계열 모델은 구조적으로 "아우터"를 만들어낼 수 없다 — 비교할 때 이 점이 핵심이다.
ATR_LABEL_TO_CATEGORY = {
    "Upper-clothes": "상의",
    "Dress": "원피스",
    "Skirt": "하의",
    "Pants": "하의",
    "Belt": "액세서리",
    "Left-shoe": "신발",
    "Right-shoe": "신발",
    "Bag": "가방",
    "Scarf": "액세서리",
}

# Fashionpedia(46 클래스) 라벨 -> 옷장 카테고리. 상의/아우터를 실제로 구분한다.
FASHIONPEDIA_LABEL_TO_CATEGORY = {
    "shirt, blouse": "상의",
    "top, t-shirt, sweatshirt": "상의",
    "sweater": "상의",
    "vest": "상의",
    "jacket": "아우터",
    "coat": "아우터",
    "cardigan": "아우터",
    "cape": "아우터",
    "pants": "하의",
    "shorts": "하의",
    "skirt": "하의",
    "dress": "원피스",
    "jumpsuit": "원피스",
    "shoe": "신발",
    "bag, wallet": "가방",
    "belt": "액세서리",
    "glasses": "액세서리",
    "hat": "액세서리",
    "headband, head covering, hair accessory": "액세서리",
    "scarf": "액세서리",
    "tie": "액세서리",
    "glove": "액세서리",
    "watch": "액세서리",
    "sock": "액세서리",
    "tights, stockings": "액세서리",
    "leg warmer": "액세서리",
    "umbrella": "액세서리",
}

# Fashionpedia는 옷의 "부속"까지 별도 클래스로 예측한다. argmax는 픽셀당 라벨을
# 하나만 주므로, 소매가 sleeve로 잡히면 그 픽셀은 상의 마스크에서 빠진다 —
# 즉 상의 마스크에 팔 모양 구멍이 뚫린다. 임의로 본체에 합치면(소매를 상의로
# 흡수) 아우터를 입은 사진에서 틀리므로 매핑하지 않고, 대신 응답의 rawLabels에
# 그대로 실어 보내 얼마나 파먹혔는지 눈으로 확인하게 한다.
FASHIONPEDIA_PART_LABELS = {
    "sleeve", "collar", "neckline", "lapel", "pocket", "hood", "epaulette",
    "zipper", "buckle", "rivet", "sequin", "bead", "bow", "ribbon", "ruffle",
    "fringe", "tassel", "applique", "flower",
}


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    title: str
    taxonomy: str
    weights_mb: int
    label_to_category: dict[str, str]
    summary: str
    watch_for: str
    part_labels: frozenset[str] = field(default_factory=frozenset)

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "modelId": self.model_id,
            "title": self.title,
            "taxonomy": self.taxonomy,
            "weightsMb": self.weights_mb,
            "summary": self.summary,
            "watchFor": self.watch_for,
            "categories": sorted(set(self.label_to_category.values()), key=CATEGORIES.index),
            "labelCount": len(self.label_to_category),
        }


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="b2_clothes",
            model_id="mattmdjaga/segformer_b2_clothes",
            title="SegFormer-B2 Clothes",
            taxonomy="ATR 18",
            weights_mb=110,
            label_to_category=ATR_LABEL_TO_CATEGORY,
            summary="현재 프로덕션 기본값. 셋 중 가장 가볍고 빠르다.",
            watch_for="아우터 클래스가 없어 코트도 상의로 나온다. 경계가 뭉툭한 편.",
        ),
        ModelSpec(
            key="b3_clothes",
            model_id="sayeed99/segformer_b3_clothes",
            title="SegFormer-B3 Clothes",
            taxonomy="ATR 18",
            weights_mb=189,
            label_to_category=ATR_LABEL_TO_CATEGORY,
            summary="B2와 라벨 체계가 같고 인코더만 키운 버전. 순수 용량 비교용.",
            watch_for="B2와 카테고리가 같으므로 차이는 경계 품질과 IoU로만 드러난다.",
        ),
        ModelSpec(
            key="b3_fashion",
            model_id="sayeed99/segformer-b3-fashion",
            title="SegFormer-B3 Fashion",
            taxonomy="Fashionpedia 46",
            weights_mb=189,
            label_to_category=FASHIONPEDIA_LABEL_TO_CATEGORY,
            part_labels=frozenset(FASHIONPEDIA_PART_LABELS),
            summary="옷 종류를 46개로 세분한다. 유일하게 상의와 아우터를 구분한다.",
            watch_for="소매·카라·주머니를 별도 클래스로 떼어내서 본체 마스크에 구멍이 생긴다.",
        ),
        ModelSpec(
            key="b5_human_parsing",
            model_id="matei-dorian/segformer-b5-finetuned-human-parsing",
            title="SegFormer-B5 Human Parsing",
            taxonomy="ATR 18",
            weights_mb=339,
            label_to_category=ATR_LABEL_TO_CATEGORY,
            summary="가장 큰 ATR 모델. 느린 대신 경계가 가장 깨끗한지 확인용.",
            watch_for="CPU에서는 눈에 띄게 느리다. 정확도 대비 비용을 볼 때 쓴다.",
        ),
    ]
}

PRODUCTION_MODEL = "b2_clothes"
DEFAULT_COMPARE = ["b2_clothes", "b3_clothes", "b3_fashion"]


def resolve(key: str | None) -> ModelSpec:
    spec = MODELS.get(key or PRODUCTION_MODEL)
    if spec is None:
        raise KeyError(f"알 수 없는 세그멘테이션 모델 '{key}'. 사용 가능: {', '.join(MODELS)}")
    return spec
