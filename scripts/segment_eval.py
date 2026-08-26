"""여러 장을 한 모델에 돌려 카테고리별 검출률과 탈락 사유를 집계한다.

Seg Lab이 사진 한 장을 깊게 본다면, 이쪽은 얕게 많이 본다. 임계값을 바꾸거나
프로덕션 모델을 갈아탈 때 "그래서 검출률이 어떻게 변하는데"에 답하기 위한 도구다.

사용법:
    python scripts/segment_eval.py <사진폴더> [--models b2_clothes,b3_clothes] [--limit 100]
    python scripts/segment_eval.py <사진폴더> --fixed-fill 0.28   # 단일 채움 기준 실험

출력은 사람이 읽는 표와, --json 경로를 주면 원시 집계까지 남긴다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND))

import segment_models  # noqa: E402
import segment_service  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def reason_axis(reason: str | None) -> str:
    """탈락 사유를 축 이름 하나로. 어느 기준이 병목인지 세려는 것이다."""
    if reason is None:
        return "통과"
    return reason.split()[0]  # "면적" | "채움" | "확신도"


def evaluate(paths, model_key, fixed_fill=None):
    """사진들을 한 모델에 돌려 (카테고리 -> 사유 카운터)와 소요 시간을 모은다."""
    per_category: dict[str, Counter] = defaultdict(Counter)
    confidences: dict[str, list[float]] = defaultdict(list)
    seconds = []
    failures = []

    for path in paths:
        try:
            analysis = segment_service.analyze(path.read_bytes(), model_key)
        except Exception as error:
            failures.append((path.name, str(error)))
            continue
        analysis.pop("_masks", None)
        seconds.append(analysis["inferenceSeconds"])
        for item in analysis["items"]:
            category = item["category"]
            confidences[category].append(item["confidence"])
            if fixed_fill is None:
                per_category[category][reason_axis(item["rejectReason"])] += 1
                continue
            # 단일 채움 기준을 씌우면 어떻게 되는지 재판정한다.
            thresholds = item["thresholds"]
            if item["areaRatio"] < thresholds["minArea"]:
                per_category[category]["면적"] += 1
            elif item["fillRatio"] < fixed_fill:
                per_category[category]["채움"] += 1
            elif item["confidence"] < thresholds["minConfidence"]:
                per_category[category]["확신도"] += 1
            else:
                per_category[category]["통과"] += 1
    return per_category, confidences, seconds, failures


def print_table(title, per_category, confidences, total_images):
    print(f"\n=== {title} (사진 {total_images}장) ===")
    header = f"{'카테고리':<8} {'검출률':>7} {'통과':>5} {'면적':>5} {'채움':>5} {'확신도':>6} {'평균확신도':>9}"
    print(header)
    print("-" * len(header))
    for category in segment_models.CATEGORIES:
        counts = per_category.get(category)
        if not counts:
            continue
        passed = counts["통과"]
        values = confidences.get(category, [])
        mean = sum(values) / len(values) if values else 0.0
        print(
            f"{category:<8} {passed / total_images:>6.0%} {passed:>5} "
            f"{counts['면적']:>5} {counts['채움']:>5} {counts['확신도']:>6} {mean:>9.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="세그멘테이션 모델 검출률 집계")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--models", default=",".join(segment_models.DEFAULT_COMPARE))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fixed-fill", type=float, default=None, help="모든 카테고리에 같은 채움 기준을 적용해 본다")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error(f"{args.folder}에 이미지가 없습니다")

    report = {}
    for model_key in args.models.split(","):
        started = time.perf_counter()
        per_category, confidences, seconds, failures = evaluate(paths, model_key, args.fixed_fill)
        label = f"{model_key}{f' · 채움 고정 {args.fixed_fill}' if args.fixed_fill is not None else ''}"
        print_table(label, per_category, confidences, len(paths))
        mean_inference = sum(seconds) / len(seconds) if seconds else 0.0
        print(f"  추론 평균 {mean_inference:.3f}s · 전체 {time.perf_counter() - started:.1f}s"
              + (f" · 실패 {len(failures)}장" if failures else ""))
        for name, error in failures[:3]:
            print(f"    실패: {name} — {error}")
        report[label] = {
            "images": len(paths),
            "categories": {category: dict(counts) for category, counts in per_category.items()},
            "meanInferenceSeconds": round(mean_inference, 3),
        }

    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n원시 집계 -> {args.json}")


if __name__ == "__main__":
    main()
