#!/usr/bin/env python3
"""착장 결과를 VLM으로 자동 채점한다.

가상 착장은 정답 이미지가 없어서 SSIM/LPIPS 같은 픽셀 지표를 쓸 수 없다.
"상의가 아우터 안쪽에 있는가"는 픽셀 거리로 나오지 않기 때문이다.
그래서 Garments2Look(CVPR 2026)이 쓴 방식과 같이 **VLM 판정**으로 이진 채점한다.
심판은 이미 백엔드에 올라와 있는 Qwen3-VL-8B를 그대로 쓴다.

채점 항목 4가지 — 모두 이번 개선이 노린 실패 유형과 1:1로 대응한다.

  layering      상의/아우터 안팎이 맞는가            (레이어 순서 프롬프트)
  items         고른 옷이 전부, 그것만 입혀졌는가     (정렬 + 개수 제한 수정)
  accessories   가방·액세서리가 제 위치에 있는가      (anchor 문장)
  identity      같은 사람·같은 포즈로 남았는가        (identity 절)

사용법:

    # 1) 케이스 정의 (JSON). 아바타 1장 + 옷 여러 장의 경로 목록.
    python scripts/eval_tryon.py --cases eval/cases.json --out eval/before.json
    # 2) 개선/파인튜닝 적용 후 다시 돌리고
    python scripts/eval_tryon.py --cases eval/cases.json --out eval/after.json
    # 3) 두 결과를 비교
    python scripts/eval_tryon.py --compare eval/before.json eval/after.json

cases.json 형식:

    [
      {
        "id": "layer-01",
        "avatar": "eval/avatars/a1.png",
        "garments": [
          {"image": "eval/items/shirt.png",   "category": "상의",     "name": "화이트 옥스퍼드 셔츠"},
          {"image": "eval/items/blazer.png",  "category": "아우터",   "name": "네이비 블레이저"},
          {"image": "eval/items/bag.png",     "category": "가방",     "name": "레더 크로스백"}
        ]
      }
    ]
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import statistics
import sys
import time
from pathlib import Path

import requests

CATEGORY_TO_SLOT = {
    "상의": "upper",
    "하의": "lower",
    "원피스": "overall",
    "아우터": "outer",
    "신발": "shoes",
    "가방": "bag",
    "액세서리": "accessory",
}


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


class Backend:
    def __init__(self, base: str, token: str, timeout: float = 600.0) -> None:
        self.base = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.timeout = timeout

    def post(self, route: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.base}{route}", json=payload, headers=self.headers, timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()

    def tryon(self, avatar: str, garments: list[dict], seed: int) -> dict:
        return self.post("/api/tryon", {"avatar": avatar, "garments": garments, "seed": seed})

    def judge(self, image: str, manifest: str) -> dict:
        return self.post("/api/vlm/tryon-judge", {"image": image, "manifest": manifest})


def score_case(case: dict, backend: Backend, seed: int) -> dict:
    root = Path(case.get("root", "."))
    garments = [
        {
            "image": data_url(root / item["image"]),
            "category": CATEGORY_TO_SLOT.get(item["category"], item["category"]),
            "name": item["name"],
        }
        for item in case["garments"]
    ]
    manifest = "\n".join(f"- {item['name']} ({item['category']})" for item in case["garments"])

    started = time.time()
    result = backend.tryon(data_url(root / case["avatar"]), garments, seed)
    elapsed = round(time.time() - started, 1)

    verdict = backend.judge(result["image"], manifest)
    expected = {item["name"] for item in case["garments"]}
    present = set(verdict.get("items_present") or [])

    return {
        "id": case["id"],
        "engine": result.get("engine"),
        "appliedGarments": result.get("appliedGarments"),
        "requestedCount": result.get("requestedCount"),
        "seconds": elapsed,
        "layering_ok": bool(verdict.get("layering_ok")),
        # 고른 옷이 전부 보이고, 추가로 생긴 옷이 없고, 합쳐진 옷도 없어야 통과.
        "items_ok": expected <= present
        and not verdict.get("extra_items")
        and not verdict.get("merged_items"),
        "item_recall": round(len(expected & present) / len(expected), 3) if expected else 1.0,
        "accessories_ok": bool(verdict.get("accessories_placed_ok")),
        "identity_ok": bool(verdict.get("identity_ok")),
        "artifact_count": len(verdict.get("artifacts") or []),
        "reasons": verdict.get("reasons", ""),
        "raw": verdict,
    }


def summarise(rows: list[dict]) -> dict:
    def rate(key: str) -> float:
        return round(sum(1 for row in rows if row[key]) / len(rows), 3)

    return {
        "cases": len(rows),
        "layering_accuracy": rate("layering_ok"),
        "item_accuracy": rate("items_ok"),
        "item_recall": round(statistics.mean(row["item_recall"] for row in rows), 3),
        "accessory_accuracy": rate("accessories_ok"),
        "identity_preservation": rate("identity_ok"),
        "mean_artifacts": round(statistics.mean(row["artifact_count"] for row in rows), 2),
        "median_seconds": round(statistics.median(row["seconds"] for row in rows), 1),
    }


def compare(before: Path, after: Path) -> None:
    left = json.loads(before.read_text(encoding="utf-8"))["summary"]
    right = json.loads(after.read_text(encoding="utf-8"))["summary"]
    width = max(len(key) for key in left)
    print(f"{'metric'.ljust(width)}  {'before':>8}  {'after':>8}  {'delta':>8}")
    print("-" * (width + 30))
    for key, old in left.items():
        new = right.get(key)
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue
        print(f"{key.ljust(width)}  {old:>8}  {new:>8}  {new - old:>+8.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, help="평가 케이스 JSON")
    parser.add_argument("--out", type=Path, help="결과를 저장할 JSON")
    parser.add_argument("--api", default="http://127.0.0.1:8787", help="백엔드 주소")
    parser.add_argument("--token", default="", help="API bearer token")
    parser.add_argument("--seed", type=int, default=42, help="비교 가능하도록 고정 시드")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return 0
    if not args.cases:
        parser.error("--cases 또는 --compare 중 하나가 필요합니다")

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    backend = Backend(args.api, args.token)
    rows = []
    for index, case in enumerate(cases, 1):
        try:
            row = score_case(case, backend, args.seed)
        except Exception as error:  # 한 케이스 실패로 전체 평가를 날리지 않는다
            print(f"[{index}/{len(cases)}] {case['id']} 실패: {error}", file=sys.stderr)
            continue
        rows.append(row)
        flags = "".join(
            mark if row[key] else "." for key, mark in
            (("layering_ok", "L"), ("items_ok", "I"), ("accessories_ok", "A"), ("identity_ok", "P"))
        )
        print(f"[{index}/{len(cases)}] {row['id']:<16} {flags}  {row['seconds']}s")

    if not rows:
        print("채점된 케이스가 없습니다", file=sys.stderr)
        return 1

    summary = summarise(rows)
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
