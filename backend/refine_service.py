"""세그멘테이션 마스크를 옷장에 넣을 수 있는 상품컷 직전 단계까지 다듬는다.

세그멘테이션 결과를 그대로 옷장에 넣기 어려운 이유는 argmax가 픽셀마다 라벨을
하나만 주기 때문이다.

  - 팔이 몸통을 가리면 그 픽셀은 팔로 가고 상의 마스크에는 팔 모양 구멍이 남는다
    (Fashionpedia 계열은 소매·카라까지 별도 클래스로 떼어가서 더 심하다 —
    docs/segmentation.md의 `FASHIONPEDIA_PART_LABELS` 참고)
  - 가방 끈이나 겹쳐 입은 옷에 가려지면 한 벌이 조각으로 끊어진다
  - 경계에는 점 같은 오검출 조각이 붙는다

이 모듈은 그 결함을 (1) 재고 (2) 눈에 보이게 칠하고 (3) 보수한 뒤 (4) 생성 모델에
넣을 흰 배경 정규화 이미지를 만든다. 생성 모델을 부르지 않으므로 GPU 없이도 전부
돈다 — Refine Lab에서 보수 단계만 따로 확인할 수 있다.

형태학 연산만으로는 팔에 가려진 자리를 되돌릴 수 없다. 그 결손은 바깥과 이어져
있어서 구멍이 아니고(`find_holes`가 못 본다), 닫기 커널을 팔 굵기만큼 키우면 옷
모양이 뭉개진다. 그래서 판정을 형태가 아니라 **라벨**로 한다: 세그멘테이션은 그 팔을
이미 `Left-arm`으로 찍어 놓았으므로, 옷의 볼록껍질 안에서 사람 픽셀에 덮인 자리는
배경이 아니라 "가려진 옷"이다(`find_occluded`).

그래도 이 모듈이 만드는 건 실루엣까지다. 그 안의 질감·주름·무늬를 실제로 그리는 건
다음 단계(FLUX)의 몫이고, 여기서는 거기에 넣기 좋은 입력과 "얼마나, 무엇에 가려져
있었는가"라는 근거를 만든다(`generation_hint`가 그 근거를 프롬프트로 넘긴다).
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
# 가림 메우기 안전장치: 옷 면적의 이 비율을 넘는 조각 하나는 채우지 않는다.
#
# 통 넓은 바지에 덮인 부츠처럼, 가려진 자리가 옷 자체보다 클 수 있는 조합이 있다.
# 그런 걸 그대로 합치면 신발 마스크가 바짓단까지 삼킨다. 넘으면 버리지 않고
# "후보였지만 적용하지 않음"으로 리포트에 남긴다 — 기준을 실측으로 고치기 위해.
OCCLUSION_MAX_RATIO = 0.35
# 마무리 스무딩 커널. 가림을 메우면 팔 실루엣 모양의 톱니가 경계에 남는다.
SMOOTH_PX = 3
# 스무딩이 이보다 많이 깎아내면 되돌린다 — 목걸이 줄처럼 가는 것이 통째로 사라진다.
SMOOTH_MAX_LOSS = 0.03
# 경계색 전파를 돌리는 축소 해상도(긴 변). 원본 해상도로 돌릴 이유가 없다 —
# 메운 자리에 필요한 건 정확한 픽셀이 아니라 이어지는 명암이다.
PROPAGATE_PX = 192
PROPAGATE_ROUNDS = 256
# 알파 경계를 이만큼 흐린다. 이진 알파를 흰 배경에 얹으면 계단이 그대로 보인다.
EDGE_FEATHER_PX = 0.8
# 알파를 흐리기 전에 옷 색을 바깥으로 이만큼 번지게 한다. 안 하면 반투명 경계에
# 마스크 바깥 픽셀(팔·배경)이 섞여 살색 후광이 생긴다.
EDGE_BLEED_PX = 2

DEFAULT_REPAIR = {
    "close": True,
    "fillHoles": True,
    "fillOccluded": True,
    "dropStrays": True,
    "smooth": True,
    # 커널 크기를 크롭 짧은 변의 비율로 잡는다 — 사진 해상도가 달라도 같은 세기로 닫히게.
    "closeScale": 0.012,
    # 가장 큰 조각 대비 이 비율보다 작은 조각은 버린다.
    #
    # "가장 큰 것만 남긴다"가 아니라 비율인 이유: 신발은 왼발·오른발이 정상적으로 두
    # 조각이고, 가방도 본체와 끈이 끊어져 보일 수 있다. 최대 조각만 남기면 멀쩡한
    # 한 짝이 통째로 사라진다.
    "strayRatio": 0.08,
    # 가림 후보 조각의 테두리 중 이 비율 이상이 옷에 닿아야 "가려진 옷"으로 본다.
    #
    # 팔 전체를 삼키지 않기 위한 기준이다. 옷 앞을 가로지르는 팔 토막은 테두리가
    # 대부분 옷에 닿지만, 옷 옆에 나란히 있을 뿐인 팔은 테두리가 배경에 닿는다.
    "occlusionEnclosure": 0.6,
}

HOLE_COLOR = (232, 76, 76)
STRAY_COLOR = (66, 133, 244)
OCCLUDED_COLOR = (245, 166, 35)


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


def _cross(origin, first, second) -> int:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0])


def convex_hull_mask(mask: np.ndarray) -> np.ndarray:
    """마스크를 감싸는 최소 볼록 다각형을 채운 마스크.

    구멍(`find_holes`)이 못 보는 결함을 보기 위해 필요하다. 팔이 옷 앞을 가로지르면
    파먹힌 자리가 바깥과 이어져 버려서 "둘러싸인 빈 곳"이 아니게 된다 — 그 만(灣)
    모양 결손은 볼록껍질과의 차이로만 드러난다.

    껍질은 행마다 가장 왼쪽·오른쪽 픽셀만 후보로 넣어도 같은 결과가 나온다(같은 행의
    나머지는 두 점 사이에 있다). 그래서 점 개수가 마스크 크기가 아니라 높이에 비례한다.
    """
    if not mask.any():
        return np.zeros_like(mask)
    points = set()
    for y in np.flatnonzero(mask.any(axis=1)).tolist():
        xs = np.flatnonzero(mask[y])
        points.add((int(xs[0]), y))
        points.add((int(xs[-1]), y))

    ordered = sorted(points)
    if len(ordered) < 3:
        return mask.copy()
    lower: list[tuple[int, int]] = []
    for point in ordered:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[int, int]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]

    # 볼록 다각형이라 행마다 x 구간 하나로 채워진다 — 변을 훑으며 행별 최소·최대만 모은다.
    height, width = mask.shape
    low = np.full(height, width, dtype=np.int64)
    high = np.full(height, -1, dtype=np.int64)
    for (x0, y0), (x1, y1) in zip(hull, hull[1:] + hull[:1]):
        span = max(abs(y1 - y0), abs(x1 - x0), 1)
        ys = np.rint(np.linspace(y0, y1, span + 1)).astype(np.int64)
        xs = np.linspace(x0, x1, span + 1)
        np.minimum.at(low, ys, np.floor(xs).astype(np.int64))
        np.maximum.at(high, ys, np.ceil(xs).astype(np.int64))
    columns = np.arange(width)
    return (columns[None, :] >= low.clip(0, width)[:, None]) & (columns[None, :] <= high.clip(-1, width - 1)[:, None])


def _shift(array: np.ndarray, axis: int, delta: int, fill) -> np.ndarray:
    """한 칸 민 배열. 바깥에서 들어오는 자리는 fill로 채운다(= 크롭 밖은 배경)."""
    out = np.full_like(array, fill)
    target, source = [slice(None)] * array.ndim, [slice(None)] * array.ndim
    if delta > 0:
        target[axis], source[axis] = slice(delta, None), slice(None, -delta)
    else:
        target[axis], source[axis] = slice(None, delta), slice(-delta, None)
    out[tuple(target)] = array[tuple(source)]
    return out


def _bracketed(mask: np.ndarray) -> np.ndarray:
    """가로나 세로로 옷이 양쪽에서 감싸고 있는 자리.

    볼록껍질만으로는 오목한 실루엣에서 과하게 삼킨다 — 소매와 허리 사이처럼 원래
    파인 자리를 껍질이 가로지르고, 거기 팔이 지나가면 옷이 아닌 곳까지 옷이 된다.
    실제로 옷에 가려진 자리는 그 옷이 양쪽에 남아 있다("팔이 가로질렀다"는 말 자체가
    가로지른 양쪽에 옷이 있다는 뜻이다). 삼킬 위험을 줄이려고 모자라는 쪽을 택한 판정 —
    귀퉁이가 조금 덜 메워지는 건 FLUX가 잇지만, 부풀어 버린 실루엣은 되돌리지 못한다.
    """
    left = np.maximum.accumulate(mask, axis=1)
    right = np.maximum.accumulate(mask[:, ::-1], axis=1)[:, ::-1]
    top = np.maximum.accumulate(mask, axis=0)
    bottom = np.maximum.accumulate(mask[::-1], axis=0)[::-1]
    return (left & right) | (top & bottom)


def _contacts(labels: np.ndarray, count: int, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """조각마다 테두리가 옷에 닿은 횟수와 그 밖에 닿은 횟수.

    "이 빈 자리가 옷에 둘러싸여 있는가"를 판정하는 근거다. 크롭 바깥으로 나가는
    면은 배경으로 센다 — 화면 밖으로 열린 자리는 가려진 옷이라고 볼 수 없다.
    """
    garment = np.zeros(count + 1, dtype=np.int64)
    other = np.zeros(count + 1, dtype=np.int64)
    inside = labels > 0
    for axis, delta in ((0, 1), (0, -1), (1, 1), (1, -1)):
        neighbour = _shift(labels, axis, delta, 0)
        neighbour_mask = _shift(mask, axis, delta, False)
        crossing = inside & (neighbour != labels)
        if not crossing.any():
            continue
        source = labels[crossing]
        touching = neighbour_mask[crossing]
        garment += np.bincount(source[touching], minlength=count + 1)
        other += np.bincount(source[~touching], minlength=count + 1)
    return garment, other


def _where(mask: np.ndarray, region: np.ndarray) -> str:
    """가려진 자리가 옷의 어디쯤인지 — 생성 프롬프트와 리드아웃에 쓰는 위치 코드."""
    ys, xs = np.where(mask)
    ry, rx = np.where(region)
    if not ys.size or not ry.size:
        return ""
    def band(value, low, high, names):
        span = max(1, high - low)
        ratio = (value - low) / span
        return names[0] if ratio < 0.38 else (names[2] if ratio > 0.62 else names[1])
    vertical = band(ry.mean(), ys.min(), ys.max(), ("upper", "middle", "lower"))
    horizontal = band(rx.mean(), xs.min(), xs.max(), ("left", "center", "right"))
    if vertical == "middle" and horizontal == "center":
        return "center"
    return f"{vertical}-{horizontal}"


def find_occluded(
    mask: np.ndarray,
    occluders: np.ndarray | None,
    options: dict | None = None,
    parse: np.ndarray | None = None,
    names: dict | None = None,
    holes: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """팔·머리·가방·다른 옷에 가려진 옷 자리를 찾는다. (채울 영역, 리포트).

    세그멘테이션은 이미 그 팔을 `Left-arm`으로 정확히 찍어 놓았다. 옷 카테고리로
    매핑되지 않는다는 이유로 버리던 그 라벨을 여기서 증거로 쓴다 — "여기는 배경이
    아니라 사람의 다른 부위에 덮인 자리"라는 사실은 형태만 봐서는 알 수 없다.

    네 겹으로 좁힌다:
      1. 볼록껍질 안쪽 — 팔 전체가 아니라 옷을 가로지르는 토막만 남는다
      2. 테두리의 `occlusionEnclosure` 이상이 옷에 닿을 것 — 옷 옆에 나란한 팔 제외
      3. 조각 하나가 옷 면적의 OCCLUSION_MAX_RATIO를 넘지 않을 것 — 과잉 확장 방지
      4. 통과한 조각에서 옷이 양쪽에 남은 자리만 (`_bracketed`) — 소매와 허리 사이처럼
         원래 파인 자리를 껍질이 가로질러 삼키는 것을 막는다
    """
    options = {**DEFAULT_REPAIR, **(options or {})}
    pixel_count = int(mask.sum())
    hull = convex_hull_mask(mask)
    hull_count = int(hull.sum())
    report = {
        "available": occluders is not None,
        # 볼록껍질을 얼마나 채우는가. 팔에 파먹힌 상의는 구멍이 없어도 여기서 티가 난다.
        "solidity": round(pixel_count / hull_count, 4) if hull_count else 0.0,
        "deficitRatio": round((hull_count - pixel_count) / hull_count, 4) if hull_count else 0.0,
        "candidatePixels": 0,
        "occludedPixels": 0,
        "occludedRatio": 0.0,
        "acceptedCount": 0,
        "openCount": 0,
        "oversizedCount": 0,
        "trimmedPixels": 0,
        "by": [],
        "where": "",
    }
    empty = np.zeros_like(mask)
    if occluders is None or not pixel_count:
        return empty, report

    # 둘러싸인 구멍은 빼고 센다. 그것도 대개 팔에 가려진 자리지만 구멍으로 이미
    # 세었고 fillHoles가 메운다 — 두 번 세면 "구멍 10% · 가려짐 19%"처럼 겹쳐 읽힌다.
    candidate = hull & ~mask & occluders & ~(find_holes(mask) if holes is None else holes)
    report["candidatePixels"] = int(candidate.sum())
    if not candidate.any():
        return empty, report

    labels, count = label_components(candidate)
    sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
    sizes[0] = 0
    garment, other = _contacts(labels, count, mask)
    enclosure = garment / np.maximum(1, garment + other)
    enclosed = enclosure >= float(options.get("occlusionEnclosure", DEFAULT_REPAIR["occlusionEnclosure"]))
    oversized = sizes > pixel_count * OCCLUSION_MAX_RATIO
    keep = enclosed & ~oversized
    keep[0] = False

    # 조각을 통째로 판정한 뒤에 양쪽이 옷인 자리만 남긴다. 순서를 바꿔 잘라 놓고
    # 판정하면 잘라낸 단면이 "배경에 닿은 테두리"로 세어져 멀쩡한 조각이 탈락한다.
    accepted = keep[labels]
    occluded = accepted & _bracketed(mask)
    # 통과한 조각이 통째로 잘려 나갈 수도 있다 — 남은 것만 "적용"으로 센다.
    report["acceptedCount"] = int(np.unique(labels[occluded]).size)
    report["openCount"] = int((~enclosed[1:]).sum())
    report["oversizedCount"] = int((enclosed & oversized)[1:].sum())
    # 오목한 실루엣이라 잘라낸 양. 기준을 만질 때 어디까지 손해 보는지 보이려고 남긴다.
    report["trimmedPixels"] = int(accepted.sum() - occluded.sum())
    report["occludedPixels"] = int(occluded.sum())
    # 분모는 "옷이 온전했다면 차지했을 넓이" — holeRatio와 같은 기준으로 읽히게 한다.
    report["occludedRatio"] = round(report["occludedPixels"] / (pixel_count + report["occludedPixels"]), 4)
    if occluded.any():
        report["where"] = _where(mask, occluded)
        if parse is not None:
            counts = np.bincount(parse[occluded].reshape(-1))
            total = int(occluded.sum())
            report["by"] = [
                {"label": (names or {}).get(int(label), str(label)), "ratio": round(int(counts[label]) / total, 3)}
                for label in np.argsort(counts)[::-1][:3].tolist()
                if counts[label]
            ]
    return occluded, report


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


def repair(mask: np.ndarray, options: dict, occluded: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """마스크를 정리한다. (보수된 마스크, 단계별 변화).

    `occluded`는 `find_occluded`가 찾아 둔 "가려진 옷 자리"다. 형태학 연산만으로는
    바깥과 이어진 결손을 되돌릴 수 없어서, 그 판정은 밖에서 받아 여기서는 합치기만 한다.

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

    if options.get("fillOccluded", True) and occluded is not None and occluded.any():
        added = occluded & ~working
        steps.append({"step": "fillOccluded", "detail": f"가려진 자리 {int(added.sum()):,}px", "delta": int(added.sum())})
        working = working | occluded

    if options.get("smooth", True) and working.any():
        # 가림을 메우면 경계에 팔 실루엣 모양의 톱니가 남는다. 중앙값 필터가 그런
        # 한두 픽셀짜리 요철만 골라 없앤다 — 닫기와 달리 전체를 부풀리지 않는다.
        smoothed = _rank(working, SMOOTH_PX, ImageFilter.MedianFilter)
        delta = int(smoothed.sum()) - int(working.sum())
        # 가는 것(목걸이 줄, 가방 끈)은 중앙값 필터에 통째로 지워진다. 많이 깎이면 되돌린다.
        if -delta <= working.sum() * SMOOTH_MAX_LOSS:
            if delta:
                steps.append({"step": "smooth", "detail": f"{SMOOTH_PX}px 중앙값", "delta": delta})
            working = smoothed

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


def _bleed(rgb: np.ndarray, known: np.ndarray, rounds: int) -> np.ndarray:
    """옷 색을 마스크 바깥으로 몇 픽셀 번지게 한다.

    알파를 흐려 계단을 없애면 반투명해진 경계에서 마스크 '바깥' 픽셀이 섞인다 —
    그게 팔이면 살색 후광이 생긴다. 미리 그 자리를 옷 색으로 덮어두면 경계가
    옷에서 투명으로만 흐려진다.
    """
    filled = rgb.astype(np.float32)
    valid = known.copy()
    for _ in range(max(0, rounds)):
        total = np.zeros_like(filled)
        counts = np.zeros(valid.shape, dtype=np.float32)
        for axis, delta in ((0, 1), (0, -1), (1, 1), (1, -1)):
            total += _shift(filled * valid[..., None], axis, delta, 0.0)
            counts += _shift(valid.astype(np.float32), axis, delta, 0.0)
        grow = ~valid & (counts > 0)
        if not grow.any():
            break
        filled[grow] = total[grow] / counts[grow][:, None]
        valid |= grow
    return filled.clip(0, 255).astype(np.uint8)


def _cutout(rgb: np.ndarray, mask: np.ndarray, feather: float = 0.0) -> Image.Image:
    alpha = (mask * 255).astype(np.uint8)
    if feather:
        rgb = _bleed(rgb, mask, EDGE_BLEED_PX)
        alpha = np.array(Image.fromarray(alpha, mode="L").filter(ImageFilter.GaussianBlur(feather)))
    return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")


def _propagate(rgb: np.ndarray, known: np.ndarray, target: np.ndarray) -> np.ndarray | None:
    """옷 경계의 색을 빈 자리 안쪽으로 퍼뜨린 색면. 못 채우면 None.

    대표색 한 가지로 평평하게 칠하면 FLUX가 그 자리를 "무지 패널"로 읽고 그대로
    그려낸다. 경계에서 색을 이어 받으면 팔 아래 그림자나 옷의 명암이 끊기지 않아,
    모델이 채울 일이 "이 자리의 질감과 주름"으로 좁아진다.

    축소판에서 돌린다 — 필요한 건 정확한 픽셀이 아니라 이어지는 명암이고, 원본
    해상도로 수백 번 확산시키면 랩 응답이 눈에 띄게 느려진다.
    """
    height, width = known.shape
    step = max(1, int(np.ceil(max(height, width) / PROPAGATE_PX)))
    pad = ((0, -height % step), (0, -width % step))
    small_shape = ((height + pad[0][1]) // step, (width + pad[1][1]) // step)

    def blocks(array, dtype):
        padded = np.pad(array.astype(dtype), pad + (((0, 0),) if array.ndim == 3 else ()))
        shape = (small_shape[0], step, small_shape[1], step) + ((3,) if array.ndim == 3 else ())
        return padded.reshape(shape).sum(axis=(1, 3))

    counts = blocks(known, np.float32)
    colors = blocks(rgb * known[..., None], np.float32)
    valid = counts > 0
    if not valid.any():
        return None
    small = np.zeros_like(colors)
    small[valid] = colors[valid] / counts[valid][:, None]
    needed = blocks(target, np.float32) > 0

    for _ in range(PROPAGATE_ROUNDS):
        if valid[needed].all():
            break
        total = np.zeros_like(small)
        neighbours = np.zeros(valid.shape, dtype=np.float32)
        for axis, delta in ((0, 1), (0, -1), (1, 1), (1, -1)):
            total += _shift(small * valid[..., None], axis, delta, 0.0)
            neighbours += _shift(valid.astype(np.float32), axis, delta, 0.0)
        grow = ~valid & (neighbours > 0)
        if not grow.any():
            break
        small[grow] = total[grow] / neighbours[grow][:, None]
        valid |= grow
    if not valid[needed].all():
        return None

    # 확산은 계단 모양 경계를 남긴다. 새로 채운 칸에서만 몇 번 평균해 부드럽게 편다.
    seeded = counts > 0
    for _ in range(2):
        total = small.copy()
        neighbours = np.ones(valid.shape, dtype=np.float32)
        for axis, delta in ((0, 1), (0, -1), (1, 1), (1, -1)):
            total += _shift(small, axis, delta, 0.0)
            neighbours += _shift(np.ones_like(neighbours), axis, delta, 0.0)
        blurred = total / neighbours[..., None]
        small = np.where(seeded[..., None], small, blurred)

    resized = Image.fromarray(small.clip(0, 255).astype(np.uint8), mode="RGB").resize((width, height), Image.BILINEAR)
    return np.asarray(resized)


def _patch(rgb: np.ndarray, before: np.ndarray, after: np.ndarray, propagate: bool = True) -> tuple[np.ndarray, dict]:
    """보수로 새로 덮은 자리를 옷의 색으로 칠한다.

    구멍을 알파로만 채우면 그 자리의 원래 픽셀 — 팔, 머리카락, 뒷배경 — 이 그대로
    옷 안에 들어간다. 그 이미지를 FLUX에 넣으면 살색을 옷의 일부로 읽고 그려낸다.
    """
    added = after & ~before
    count = int(added.sum())
    info = {"patchedPixels": count, "patchColor": "", "patchFill": "none"}
    if not count or not before.any():
        return rgb, info

    color = np.median(rgb[before], axis=0)
    info["patchColor"] = "#%02x%02x%02x" % tuple(int(value) for value in color)
    patched = rgb.copy()
    filled = _propagate(rgb, before, added) if propagate else None
    if filled is None:
        patched[added] = color
        info["patchFill"] = "median"
    else:
        patched[added] = filled[added]
        info["patchFill"] = "propagate"
    return patched, info


def _defect_view(
    rgb: np.ndarray, mask: np.ndarray, holes: np.ndarray, strays: np.ndarray, occluded: np.ndarray
) -> Image.Image:
    """구멍은 빨강, 가려진 자리는 주황, 버릴 조각은 파랑. 배경은 어둡게 깔아 옷만 뜨게 한다."""
    grey = rgb.mean(axis=2, keepdims=True)
    canvas = (grey * 0.45 + 26).clip(0, 255).repeat(3, axis=2)
    canvas[mask] = rgb[mask]
    for area, color in ((occluded, OCCLUDED_COLOR), (holes, HOLE_COLOR), (strays, STRAY_COLOR)):
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


def generation_hint(diagnosis: dict) -> str:
    """FLUX 프롬프트에 붙일 한 문장. 없으면 빈 문자열.

    2단계가 알아낸 것을 5단계에 넘기지 않으면 모델은 파먹힌 실루엣을 "원래 그런
    디자인"으로 읽는다. 어디가 무엇에 가려져 비었는지 말해 주면 그 자리를 원단으로
    잇는다. 메운 자리가 눈에 띄지 않을 만큼 작으면 붙이지 않는다 — 없는 결함을
    말해 주면 멀쩡한 곳을 고쳐 그린다.
    """
    ratio = float(diagnosis.get("occludedRatio") or 0.0)
    if ratio < 0.02:
        return ""
    where = str(diagnosis.get("occludedWhere") or "").replace("-", " ")
    by = [str(entry["label"]).replace("-", " ").lower() for entry in diagnosis.get("occludedBy") or []][:2]
    covered = f" by the model's {' and '.join(by)}" if by else ""
    place = f" on its {where} side" if where and where != "center" else ""
    return (
        f" About {ratio:.0%} of this garment{place} was covered{covered} in the source photo and is filled with a "
        "flat approximate colour in the reference, so redraw that area as continuous fabric with the same colour, "
        "pattern and folds as the rest of the item."
    )


def build_stages(
    image_np: np.ndarray, mask: np.ndarray, options: dict | None = None, parse: dict | None = None
) -> dict:
    """마스크 하나에 대한 파이프라인 2~4단계 전부.

    `parse`는 세그멘테이션이 남긴 라벨맵이다(`{"pred", "names", "background"}`).
    없으면 가림 진단만 빠지고 나머지는 그대로 돈다 — 예전 백엔드나 테스트 스텁처럼
    라벨맵을 주지 않는 호출부가 있어서 선택 인자로 둔다.

    반환의 `image`는 FLUX에 넣을 PIL 이미지다 — 호출부가 다시 디코딩하지 않도록
    바이트가 아니라 이미지로 돌려준다.
    """
    options = {**DEFAULT_REPAIR, **(options or {})}
    margin = _kernel_size(image_np.shape[:2], options["closeScale"])
    x0, y0, x1, y1 = crop_window(mask, image_np.shape[:2], margin * 2)
    rgb = image_np[y0:y1, x0:x1]
    window_mask = mask[y0:y1, x0:x1]

    # 가림 판정은 이 옷이 아닌 '사람 픽셀'이 근거다 — 배경은 진짜 옷 경계이므로 건드리지 않는다.
    # 배경 라벨을 모르면 "사람 픽셀"을 가를 수 없다. 그때 전부 사람으로 치면 진짜
    # 배경까지 가려진 옷으로 읽혀 실루엣이 부푼다 — 차라리 가림 진단을 통째로 뺀다.
    background = list((parse or {}).get("background") or [])
    pred = parse["pred"][y0:y1, x0:x1] if parse and background else None
    occluders = None if pred is None else ~np.isin(pred, background) & ~window_mask

    diagnosis = diagnose(window_mask)
    occluded, occlusion = find_occluded(
        window_mask, occluders, options, pred, (parse or {}).get("names"), diagnosis["_holes"]
    )
    diagnosis.update({
        "solidity": occlusion["solidity"],
        "occludedPixels": occlusion["occludedPixels"],
        "occludedRatio": occlusion["occludedRatio"],
        "occludedBy": occlusion["by"],
        "occludedWhere": occlusion["where"],
        "occlusionAvailable": occlusion["available"],
    })
    repaired, report = repair(window_mask, options, occluded)
    report["occlusion"] = occlusion
    holes = diagnosis.pop("_holes")
    dropped = report.pop("_dropped")
    patched, patch_info = _patch(rgb, window_mask, repaired)
    report.update(patch_info)

    normalized = normalize(_cutout(patched, repaired, EDGE_FEATHER_PX))
    return {
        "window": [x0, y0, x1, y1],
        "diagnosis": diagnosis,
        "repair": report,
        "cropPng": _png(_cutout(rgb, window_mask)),
        "defectPng": _png(_defect_view(rgb, window_mask, holes, dropped, occluded)),
        "repairedPng": _png(_cutout(patched, repaired, EDGE_FEATHER_PX)),
        "normalizedPng": _png(normalized),
        "image": normalized,
    }
