"""기존 Pinterest 상품컷의 카테고리와 색상을 정리한다.

세그멘테이션 라벨을 조건으로 생성한 상품컷 중 Bag/Belt 오검출은 사람이 접촉표로
검수했다. 색은 원본 전신 사진의 대표색이 아니라 흰 배경을 제외한 최종 상품컷에서
다시 계산한다.
"""

from __future__ import annotations

import colorsys
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "assets" / "lookbook-data.js"
SOURCE_FILE = ROOT / "assets" / "lookbook-sources.json"
INFLUENCER_FILE = ROOT / "assets" / "influencer-data.js"

# 2026-08-27 상품컷 접촉표 검수. 나열되지 않은 가방/액세서리는 현재 분류가 맞다.
CATEGORY_OVERRIDES = {
    **{number: "상의" for number in (
        188, 195, 201, 202, 204, 206, 207, 208, 211, 212, 213, 214, 215,
        216, 217, 218, 222, 226, 228, 230, 231, 235, 239, 264, 267, 276, 301,
    )},
    259: "하의",
    278: "상의",
    293: "하의",
    295: "상의",
    **{number: "아우터" for number in (
        4, 16, 23, 30, 40, 59, 67, 72, 75, 82, 85, 88, 99, 103, 108,
        118, 121, 123, 135, 153, 159, 167, 170, 177, 179, 184,
    )},
}


def load_catalog() -> list[dict]:
    text = DATA_FILE.read_text(encoding="utf-8")
    payload = re.sub(r"^window\.MUSINSA_RANKING\s*=\s*|;\s*$", "", text)
    return json.loads(payload)


def color_name(path: Path) -> str:
    pixels = np.asarray(Image.open(path).convert("RGB").resize((160, 160))).reshape(-1, 3)
    rgb = pixels / 255.0
    maximum, minimum = rgb.max(axis=1), rgb.min(axis=1)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    # 생성 상품컷의 흰 배경과 옅은 그림자를 제외한다.
    foreground = pixels[~((maximum > 0.88) & (saturation < 0.10))]
    if len(foreground) < 20:
        foreground = pixels
    buckets = (foreground // 24).astype(np.int16)
    keys = buckets[:, 0] * 121 + buckets[:, 1] * 11 + buckets[:, 2]
    key = int(np.bincount(keys).argmax())
    red, green, blue = np.median(foreground[keys == key], axis=0) / 255.0
    hue, sat, value = colorsys.rgb_to_hsv(float(red), float(green), float(blue))
    hue *= 360
    if sat < 0.12:
        return "블랙" if value < 0.20 else "화이트" if value > 0.84 else "라이트 그레이" if value > 0.62 else "그레이"
    if value < 0.22:
        return "네이비" if 190 <= hue < 260 else "블랙"
    if hue < 15 or hue >= 345:
        return "브라운" if value < 0.62 or sat < 0.45 else "레드"
    if hue < 45:
        return "브라운" if value < 0.62 else "베이지"
    if hue < 75:
        return "베이지"
    if hue < 165:
        return "그린"
    if hue < 195:
        return "민트"
    if hue < 255:
        return "네이비" if value < 0.48 else "블루"
    if hue < 300:
        return "퍼플"
    return "핑크"


def main() -> None:
    catalog = load_catalog()
    source = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    # source 메타데이터는 분리 PNG, 앱 카탈로그는 FLUX 정제 JPG를 가리키므로 image
    # 경로가 다르다. 두 파일에서 안정적으로 같은 항목을 뜻하는 rank로 연결한다.
    source_by_rank = {item["rank"]: item for item in source["images"]}
    changed_categories = 0
    for item in catalog:
        look_number = int(re.search(r"look-(\d+)", item["image"]).group(1))
        category = CATEGORY_OVERRIDES.get(look_number, item["category"])
        if category != item["category"]:
            changed_categories += 1
            item["category"] = category
            item["subcategory"] = {"상의": "상의", "하의": "하의", "아우터": "재킷·가디건"}.get(category, category)
            item["segmentation"]["reviewedCategory"] = category
        item["color"] = color_name(ROOT / item["image"])
        item.pop("brand", None)
        kind = {"상의": "상의", "하의": "하의", "가방": "가방", "액세서리": "액세서리"}.get(category, category)
        item["name"] = f"{item['color']} {kind} {item['rank']:03d}"
        if item["rank"] in source_by_rank:
            source_by_rank[item["rank"]].pop("brand", None)
            source_by_rank[item["rank"]].update({
                "category": item["category"], "subcategory": item["subcategory"],
                "color": item["color"], "name": item["name"],
            })
            source_by_rank[item["rank"]]["segmentation"] = item["segmentation"]

    DATA_FILE.write_text(
        "window.MUSINSA_RANKING = " + json.dumps(catalog, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )
    SOURCE_FILE.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Trends 카드에도 같은 문구가 creator로 노출된다. 출처 링크·credit은 유지하고
    # 화면용 이름만 중립적인 표현으로 바꾼다.
    influencer = INFLUENCER_FILE.read_text(encoding="utf-8").replace(
        '"creator": "Pinterest 남성패션"', '"creator": "스타일 레퍼런스"'
    )
    INFLUENCER_FILE.write_text(influencer, encoding="utf-8")
    print(f"카테고리 {changed_categories}개 교정, 색상 {len(catalog)}개 재계산")


if __name__ == "__main__":
    main()
