"""세그멘테이션 마스크를 옷장에 넣을 수 있는 상품컷 직전 단계까지 다듬는다.

세그멘테이션 결과를 그대로 옷장에 넣기 어려운 이유는 argmax가 픽셀마다 라벨을
하나만 주기 때문이다.

  - 팔이 몸통을 가리면 그 픽셀은 팔로 가고 상의 마스크에는 팔 모양 구멍이 남는다
    (Fashionpedia 계열은 소매·카라까지 별도 클래스로 떼어가서 더 심하다 —
    docs/segmentation.md의 `FASHIONPEDIA_PART_LABELS` 참고)
  - 가방 끈이나 겹쳐 입은 옷에 가려지면 한 벌이 조각으로 끊어진다
  - 경계에는 점 같은 오검출 조각이 붙는다

이 모듈은 그 결함을 (1) 재고 (2) 눈에 보이게 칠하고 (3) 형태학 연산으로 보수한 뒤
(4) 생성 모델에 넣을 흰 배경 정규화 이미지를 만든다. 생성 모델을 부르지 않으므로
GPU 없이도 전부 돈다 — Refine Lab에서 보수 단계만 따로 확인할 수 있다.

보수는 "복원"이 아니라 "정리"다. 형태학 연산은 없는 픽셀을 지어내지 못하므로 팔에
가려진 몸통은 여전히 비어 있다. 그 빈 곳을 실제로 채우는 건 다음 단계(FLUX)의 몫이고,
이 모듈은 거기에 넣기 좋은 입력과 "얼마나 망가져 있었는가"라는 근거를 만든다.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageFilter

from segment_service import shrink

# 스테이지 이미지는 화면에 네 장을 나란히 놓고 보는 용도라 원본 해상도가 필요 없다.
PREVIEW_STAGE_PX = 460
# 생성 모델 입력. 너무 키우면 요청만 무거워지고 FLUX가 어차피 자체 해상도로 다시 잡는다.
NORMALIZED_PX = 768
# 정규화 캔버스에서 옷 주변에 남기는 여백 비율. 상품컷은 가장자리에 붙어 있으면 답답하다.
NORMALIZED_MARGIN = 0.06
# 닫기 연산 커널이 이보다 커지면 옷 모양 자체가 뭉개진다.
MAX_KERNEL_PX = 21

DEFAULT_REPAIR = {
    "close": True,
    "fillHoles": True,
    "dropStrays": True,
    # 커널 크기를 크롭 짧은 변의 비율로 잡는다 — 사진 해상도가 달라도 같은 세기로 닫히게.
    "closeScale": 0.012,
    # 가장 큰 조각 대비 이 비율보다 작은 조각은 버린다.
    #
    # "가장 큰 것만 남긴다"가 아니라 비율인 이유: 신발은 왼발·오른발이 정상적으로 두
    # 조각이고, 가방도 본체와 끈이 끊어져 보일 수 있다. 최대 조각만 남기면 멀쩡한
    # 한 짝이 통째로 사라진다.
    "strayRatio": 0.08,
}

HOLE_COLOR = (232, 76, 76)
STRAY_COLOR = (66, 133, 244)


def _find(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-이웃 연결 성분 라벨링. (라벨 배열, 성분 수). 배경은 0.

    scipy 없이 돌려야 해서 직접 짰다. 픽셀마다 도는 대신 행별 런(run)을 묶어
    union-find로 잇는다 — 크롭 한 장에서 런은 행당 몇 개뿐이라 충분히 빠르다.
    """
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    parent = [0]  # 0번은 배경 자리
    previous: list[tuple[int, int, int]] = []
    for y in range(height):
        padded = np.concatenate(([False], mask[y], [False]))
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        current: list[tuple[int, int, int]] = []
        for start, end in zip(edges[0::2].tolist(), edges[1::2].tolist()):
            touching = {_find(parent, label) for p_start, p_end, label in previous if p_start < end and start < p_end}
            if touching:
                label = min(touching)
                for other in touching:
                    parent[other] = label
            else:
                label = len(parent)
                parent.append(label)
            labels[y, start:end] = label
            current.append((start, end, label))
        previous = current

    if len(parent) == 1:
        return labels, 0
    roots = np.array([_find(parent, index) for index in range(len(parent))], dtype=np.int32)
    unique, compact = np.unique(roots, return_inverse=True)  # 배경(0)이 항상 첫 자리
    return compact.astype(np.int32).reshape(-1)[labels], len(unique) - 1


def find_holes(mask: np.ndarray) -> np.ndarray:
    """마스크에 둘러싸여 바깥과 이어지지 않는 빈 구멍.

    바깥 배경도 마스크가 아닌 픽셀이라 단순히 반전하면 안 된다. 반전한 뒤 테두리에
    닿는 성분을 배경으로 보고 빼야 진짜 구멍만 남는다.
    """
    labels, count = label_components(~mask)
    if not count:
        return np.zeros_like(mask)
    border = set(labels[0].tolist()) | set(labels[-1].tolist())
    border |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
    inner = [label for label in range(1, count + 1) if label not in border]
    return np.isin(labels, inner) if inner else np.zeros_like(mask)


def _kernel_size(shape: tuple[int, int], scale: float) -> int:
    size = int(round(min(shape) * max(0.0, scale)))
    return min(MAX_KERNEL_PX, max(3, size | 1))  # PIL 랭크 필터는 홀수 커널만 받는다


def _rank(mask: np.ndarray, size: int, filter_class) -> np.ndarray:
    image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    return np.array(image.filter(filter_class(size))) > 127


def close_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """팽창 후 침식. 경계의 톱니와 좁은 틈을 메우되 전체 크기는 유지한다."""
    return _rank(_rank(mask, size, ImageFilter.MaxFilter), size, ImageFilter.MinFilter)


def diagnose(mask: np.ndarray) -> dict:
    """이 마스크가 어떻게 망가져 있는지. 보수 전 상태를 숫자로 남긴다.

    `_holes`는 결함 시각화에 쓰는 numpy 배열이라 직렬화 전에 호출부가 걷어낸다.
    """
    pixel_count = int(mask.sum())
    if not pixel_count:
        return {
            "pixelCount": 0, "componentCount": 0, "strayCount": 0, "largestRatio": 0.0,
            "strayRatio": 0.0, "holeCount": 0, "holeRatio": 0.0, "fillRatio": 0.0,
            "_holes": np.zeros_like(mask),
        }
    labels, count = label_components(mask)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    sizes[0] = 0
    largest = int(sizes.max())
    holes = find_holes(mask)
    hole_pixels = int(holes.sum())
    stray = sizes[1:][sizes[1:] < largest * DEFAULT_REPAIR["strayRatio"]]
    return {
        "pixelCount": pixel_count,
        "componentCount": count,
        # 기본 기준으로 셌을 때 몇 조각이 부스러기인지. 실제로 버릴지는 요청 옵션이 정한다.
        "strayCount": int(stray.size),
        "largestRatio": round(largest / pixel_count, 4),
        "strayRatio": round(float(stray.sum()) / pixel_count, 4),
        "holeCount": int(label_components(holes)[1]),
        # 분모를 "옷이 온전했다면 차지했을 넓이"(마스크+구멍)로 잡아야 30% 파먹혔다는
        # 말이 그대로 읽힌다. 마스크 픽셀로 나누면 같은 손상이 더 커 보인다.
        "holeRatio": round(hole_pixels / (pixel_count + hole_pixels), 4),
        "fillRatio": round(pixel_count / mask.size, 4),
        "_holes": holes,
    }


def repair(mask: np.ndarray, options: dict) -> tuple[np.ndarray, dict]:
    """마스크를 정리한다. (보수된 마스크, 단계별 변화).

    `_dropped`는 버린 조각을 결함 시각화에 칠하려고 함께 돌려주는 numpy 배열이다.
    """
    kernel = _kernel_size(mask.shape, options.get("closeScale", DEFAULT_REPAIR["closeScale"]))
    before = int(mask.sum())
    steps: list[dict] = []
    working = mask

    if options.get("close", True) and before:
        closed = close_mask(working, kernel)
        steps.append({"step": "close", "detail": f"{kernel}px 커널", "delta": int(closed.sum()) - before})
        working = closed

    if options.get("fillHoles", True) and working.any():
        # 닫기가 이미 좁은 틈을 메웠으므로 남은 구멍을 다시 센다.
        holes = find_holes(working)
        steps.append({"step": "fillHoles", "detail": f"구멍 {int(holes.sum()):,}px", "delta": int(holes.sum())})
        working = working | holes

    dropped = np.zeros_like(mask)
    if options.get("dropStrays", True) and working.any():
        ratio = float(options.get("strayRatio", DEFAULT_REPAIR["strayRatio"]))
        labels, count = label_components(working)
        sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
        sizes[0] = 0
        keep = sizes >= sizes.max() * ratio
        keep[0] = False
        kept = keep[labels]
        dropped = working & ~kept
        steps.append({
            "step": "dropStrays",
            "detail": f"{count - int(keep.sum())}조각 · 최대 대비 {ratio:.0%} 미만",
            "delta": -int(dropped.sum()),
        })
        working = kept

    after = int(working.sum())
    return working, {
        "kernel": kernel,
        "steps": steps,
        "pixelsBefore": before,
        "pixelsAfter": after,
        "growth": round((after - before) / before, 4) if before else 0.0,
        "componentsAfter": label_components(working)[1],
        "holesAfter": int(find_holes(working).sum()),
        "_dropped": dropped,
    }


def crop_window(mask: np.ndarray, shape: tuple[int, int], margin_px: int) -> tuple[int, int, int, int]:
    """마스크를 감싸는 창. 보수하면서 마스크가 커지므로 여유를 두고 자른다.

    모든 스테이지 이미지가 같은 창을 쓴다 — 창이 스테이지마다 달라지면 나란히 놓고
    어디가 달라졌는지 볼 수가 없다.
    """
    height, width = shape[0], shape[1]
    ys, xs = np.where(mask)
    if not ys.size:
        return 0, 0, width, height
    return (
        max(0, int(xs.min()) - margin_px),
        max(0, int(ys.min()) - margin_px),
        min(width, int(xs.max()) + margin_px + 1),
        min(height, int(ys.max()) + margin_px + 1),
    )


def _png(image: Image.Image, max_px: int | None = PREVIEW_STAGE_PX) -> bytes:
    buffer = io.BytesIO()
    # 경계가 비교 대상이라 JPEG 링잉을 피해 PNG로 두고, 크기는 축소로 줄인다.
    shrink(image, max_px).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _cutout(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.dstack([rgb, (mask * 255).astype(np.uint8)]), mode="RGBA")


def _patch(rgb: np.ndarray, before: np.ndarray, after: np.ndarray) -> tuple[np.ndarray, int, str]:
    """보수로 새로 덮은 자리를 옷의 대표색으로 칠한다.

    구멍을 알파로만 채우면 그 자리의 원래 픽셀 — 팔, 머리카락, 뒷배경 — 이 그대로
    옷 안에 들어간다. 그 이미지를 FLUX에 넣으면 살색을 옷의 일부로 읽고 그려낸다.
    중앙값으로 칠해두면 실루엣은 온전해지고 색은 옷의 것이라, 모델이 채워야 할 일이
    "이 자리의 질감과 주름"으로 좁아진다.
    """
    added = after & ~before
    count = int(added.sum())
    if not count or not before.any():
        return rgb, 0, ""
    color = np.median(rgb[before], axis=0)
    patched = rgb.copy()
    patched[added] = color
    return patched, count, "#%02x%02x%02x" % tuple(int(value) for value in color)


def _defect_view(rgb: np.ndarray, mask: np.ndarray, holes: np.ndarray, strays: np.ndarray) -> Image.Image:
    """구멍은 빨강, 버릴 조각은 파랑. 나머지 배경은 어둡게 깔아 옷만 뜨게 한다."""
    grey = rgb.mean(axis=2, keepdims=True)
    canvas = (grey * 0.45 + 26).clip(0, 255).repeat(3, axis=2)
    canvas[mask] = rgb[mask]
    for area, color in ((holes, HOLE_COLOR), (strays, STRAY_COLOR)):
        if area.any():
            canvas[area] = rgb[area] * 0.15 + np.array(color, dtype=float) * 0.85
    return Image.fromarray(canvas.astype(np.uint8), mode="RGB")


def normalize(cutout: Image.Image, size: int = NORMALIZED_PX) -> Image.Image:
    """투명 크롭 -> 흰 배경 정사각 상품컷 규격.

    생성 모델에 넣을 입력이자, GPU가 없을 때 랩이 보여줄 최종본이다. 알파를 흰색으로
    깔아두지 않으면 투명 영역이 검게 들어가 모델이 그 검은 면을 옷의 일부로 읽는다.
    """
    inner = max(1, int(round(size * (1 - NORMALIZED_MARGIN * 2))))
    scale = inner / max(cutout.size)
    resized = cutout.resize(
        (max(1, round(cutout.width * scale)), max(1, round(cutout.height * scale))), Image.LANCZOS
    )
    canvas = Image.new("RGB", (size, size), "#ffffff")
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2), resized)
    return canvas


def build_stages(image_np: np.ndarray, mask: np.ndarray, options: dict | None = None) -> dict:
    """마스크 하나에 대한 파이프라인 2~3단계 전부.

    반환의 `image`는 FLUX에 넣을 PIL 이미지다 — 호출부가 다시 디코딩하지 않도록
    바이트가 아니라 이미지로 돌려준다.
    """
    options = {**DEFAULT_REPAIR, **(options or {})}
    margin = _kernel_size(image_np.shape[:2], options["closeScale"])
    x0, y0, x1, y1 = crop_window(mask, image_np.shape[:2], margin * 2)
    rgb = image_np[y0:y1, x0:x1]
    window_mask = mask[y0:y1, x0:x1]

    diagnosis = diagnose(window_mask)
    repaired, report = repair(window_mask, options)
    holes = diagnosis.pop("_holes")
    dropped = report.pop("_dropped")
    patched, report["patchedPixels"], report["patchColor"] = _patch(rgb, window_mask, repaired)

    normalized = normalize(_cutout(patched, repaired))
    return {
        "window": [x0, y0, x1, y1],
        "diagnosis": diagnosis,
        "repair": report,
        "cropPng": _png(_cutout(rgb, window_mask)),
        "defectPng": _png(_defect_view(rgb, window_mask, holes, dropped)),
        "repairedPng": _png(_cutout(patched, repaired)),
        "normalizedPng": _png(normalized),
        "image": normalized,
    }
