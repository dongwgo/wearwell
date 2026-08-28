"""Pinterest 스테이징 이미지를 프로젝트의 SegFormer 로직으로 선별한다."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

import segment_service  # noqa: E402

STAGING = ROOT / "assets" / "pinterest-next"
OUTPUT = STAGING / "lookbook"
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCES = json.loads((STAGING / "sources.json").read_text(encoding="utf-8"))
BY_FILE = {item["file"]: item for item in SOURCES["images"]}


def analyze(path: Path) -> dict:
    return segment_service.analyze(path.read_bytes(), "b3_clothes")


garments: list[dict] = []
closet_reports: list[dict] = []
closet_files = sorted((STAGING / "closet-sources").iterdir())
for index, source_path in enumerate(closet_files, 1):
    result = analyze(source_path)
    source = BY_FILE[source_path.name]
    accepted = [item for item in result["items"] if item["accepted"]]
    closet_reports.append({
        "sourceFile": source_path.name, "pinId": source["id"], "pinUrl": source["sourceUrl"],
        "acceptedCount": len(accepted), "detectedCount": len(result["items"]),
        "inferenceSeconds": result["inferenceSeconds"],
        "items": [{key: item[key] for key in ("category", "label", "accepted", "rejectReason", "areaRatio", "fillRatio", "confidence")} for item in result["items"]],
    })
    for item in accepted:
        garments.append({"source": source, "sourceFile": source_path.name, "item": item})
    print(f"\r옷장 분리 {index:3d}/100 · 통과 의류 {len(garments):3d}", end="", flush=True)

# 100장 전체를 검사한 뒤 품질 기준을 통과한 의류를 모두 넣는다.
garments.sort(key=lambda row: (row["item"]["areaRatio"] * row["item"]["confidence"]), reverse=True)
selected = garments
records = []
for index, row in enumerate(selected, 1):
    item = row["item"]
    filename = f"look-{index:03d}.png"
    (OUTPUT / filename).write_bytes(item["png_bytes"])
    source = row["source"]
    records.append({
        "id": f"pinterest-{source['id']}-{item['category']}",
        "image": f"assets/lookbook/{filename}", "gender": "men", "rankingGender": "men",
        "category": item["category"], "subcategory": item["label"],
        "name": f"{item['category']} {index:03d}",
        "color": item["color"], "rank": index, "sourceRank": index, "price": 0,
        "sourceUrl": source["sourceUrl"], "sourceImageUrl": source["imageUrl"],
        "worn": 0, "userAdded": False,
        "segmentation": {"model": "sayeed99/segformer_b3_clothes", "label": item["label"], "confidence": item["confidence"], "areaRatio": item["areaRatio"], "fillRatio": item["fillRatio"]},
    })

if "--closet-only" in sys.argv:
    metadata = {
        "query": SOURCES["query"], "searchUrl": SOURCES["searchUrl"], "retrievedAt": SOURCES["retrievedAt"],
        "processedAt": datetime.now(timezone.utc).isoformat(), "model": "sayeed99/segformer_b3_clothes",
        "sourcePhotoCount": len(closet_files), "acceptedGarmentCount": len(garments), "selectedGarmentCount": len(records),
        "images": records, "reports": closet_reports,
    }
    (STAGING / "lookbook-sources.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (STAGING / "lookbook-data.js").write_text("window.MUSINSA_RANKING = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    print(f"\n완료: 100장 중 통과한 의류 {len(records)}개를 모두 저장")
    raise SystemExit(0)

trend_records = []
trend_reports = []
trend_files = sorted((STAGING / "trends").iterdir())
for index, source_path in enumerate(trend_files, 1):
    result = analyze(source_path)
    source = BY_FILE[source_path.name]
    accepted = [item for item in result["items"] if item["accepted"]]
    pieces = [{
        "category": item["category"], "label": item["label"], "colors": [item["color"]],
        "materials": ["소재 미분류"], "fits": ["핏 미분류"], "details": [item["label"]],
    } for item in accepted]
    trend_records.append({
        "id": f"pinterest-trend-{index:03d}", "gender": "men", "creator": "스타일 레퍼런스",
        "creatorHandle": "@pinterest", "creatorUrl": "https://kr.pinterest.com/", "credit": "Pinterest 공개 Pin",
        "sourceTitle": source["title"], "sourceUrl": source["sourceUrl"],
        "image": f"assets/influencers/look-{index:03d}.{source_path.suffix.lstrip('.')}",
        "published": "2026 남성패션 검색", "publicSpec": "남성 패션 룩",
        "bodyLabel": "남성", "bmiRange": [16, 35], "bodyShapes": ["보통"], "heightRange": [150, 200],
        "weather": ["맑음", "간절기"], "mood": "남성 스트리트 패션", "styles": ["남성패션", "스트리트", "캐주얼"],
        "summary": source["description"] or source["title"], "pieces": pieces,
    })
    trend_reports.append({"sourceFile": source_path.name, "pinId": source["id"], "acceptedCount": len(accepted), "pieces": pieces})
    print(f"\rTrends 분석 {index:3d}/100", end="", flush=True)

metadata = {
    "query": SOURCES["query"], "searchUrl": SOURCES["searchUrl"], "retrievedAt": SOURCES["retrievedAt"],
    "processedAt": datetime.now(timezone.utc).isoformat(), "model": "sayeed99/segformer_b3_clothes",
    "sourcePhotoCount": len(closet_files), "acceptedGarmentCount": len(garments), "selectedGarmentCount": len(records),
    "images": records, "reports": closet_reports,
}
(STAGING / "lookbook-sources.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
(STAGING / "lookbook-data.js").write_text("window.MUSINSA_RANKING = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
(STAGING / "influencer-data.js").write_text("(function () {\n  window.WEARWELL_INFLUENCER_LOOKS = " + json.dumps(trend_records, ensure_ascii=False, indent=2) + ";\n})();\n", encoding="utf-8")
(STAGING / "trend-analysis.json").write_text(json.dumps({"model": "sayeed99/segformer_b3_clothes", "reports": trend_reports}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n완료: 100장 중 {len(garments)}개 통과, 옷장 {len(records)}개 선택, Trends {len(trend_records)}개")
