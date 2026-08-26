"""전신샷에서 의류 아이템을 자동으로 분리한다 (human-parsing 세그멘테이션).

두 가지 진입점이 있다.
  - segment(): 프로덕션 경로. 품질 필터를 통과한 아이템만 잘라서 돌려준다.
  - analyze(): 개발용 비교 경로. 걸러진 후보와 그 이유, 매핑되지 않은 원본 라벨,
    오버레이, 소요 시간까지 전부 돌려준다.

transformers+torch는 첫 요청 때만 불러와 캐싱한다 — 모델 로딩이 무겁기 때문이다
(가중치는 ~/.cache/huggingface에 자동 캐싱). 모델을 바꿔가며 비교할 수 있도록
여러 개를 동시에 들고 있되, SEGMENT_MODEL_CACHE_SIZE개를 넘으면 가장 오래
쓰지 않은 것부터 내린다.
"""

import io
import os
import threading
import time
from collections import OrderedDict

import segment_models
from segment_models import CATEGORY_COLORS, MODELS, PRODUCTION_MODEL

# 프로덕션 기본 모델. 예전 코드가 참조하던 이름을 유지한다.
MODEL_ID = MODELS[PRODUCTION_MODEL].model_id
LABEL_TO_CATEGORY = MODELS[PRODUCTION_MODEL].label_to_category

CACHE_SIZE = max(1, int(os.getenv("SEGMENT_MODEL_CACHE_SIZE", "3")))
# 이 비율 미만으로 잡힌 라벨은 노이즈로 보고 rawLabels에서 뺀다.
RAW_LABEL_MIN_RATIO = 0.0005
# analyze()는 화면에 늘어놓고 눈으로 비교하는 용도라 원본 해상도를 보낼 이유가 없다.
# 모델 3개면 크롭만 15장이라 원본 크기로는 응답이 수십 MB가 된다. 세그멘테이션
# 자체는 원본으로 돌리고, 응답에 실을 이미지만 줄인다.
PREVIEW_OVERLAY_PX = 860
PREVIEW_CROP_PX = 420

_lock = threading.Lock()
_loaded: "OrderedDict[str, tuple]" = OrderedDict()
_device = None


class ModelUnavailable(RuntimeError):
    pass


def _resolve_device():
    global _device
    if _device is None:
        import torch

        requested = os.getenv("SEGMENTATION_DEVICE", "auto").lower()
        _device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    return _device


def get_model(model_key: str | None = None):
    """(model, processor, spec, device, load_seconds)를 돌려준다.

    load_seconds는 이번 호출에서 실제로 적재한 시간이다. 캐시에 있었으면 0.0 —
    비교 결과에서 콜드 로딩 비용과 추론 비용을 섞지 않기 위해 구분한다.
    """
    spec = segment_models.resolve(model_key)
    with _lock:
        if spec.key in _loaded:
            _loaded.move_to_end(spec.key)
            model, processor = _loaded[spec.key]
            return model, processor, spec, _resolve_device(), 0.0

        try:
            from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise ModelUnavailable(
                "세그멘테이션 모델(transformers/torchvision)이 설치되어 있지 않습니다. "
                "requirements.txt를 설치한 뒤 서버를 다시 실행하세요."
            ) from exc

        device = _resolve_device()
        started = time.perf_counter()
        try:
            processor = SegformerImageProcessor.from_pretrained(spec.model_id)
            model = AutoModelForSemanticSegmentation.from_pretrained(spec.model_id).to(device)
        except Exception as exc:  # 네트워크 없음, 저장소 삭제, 디스크 부족 등
            raise ModelUnavailable(f"'{spec.model_id}' 적재 실패: {exc}") from exc
        model.eval()
        load_seconds = time.perf_counter() - started

        _loaded[spec.key] = (model, processor)
        while len(_loaded) > CACHE_SIZE:
            _loaded.popitem(last=False)
        return model, processor, spec, device, load_seconds


def loaded_keys() -> list[str]:
    with _lock:
        return list(_loaded)


def unload(model_key: str | None = None) -> list[str]:
    """캐시에서 모델을 내린다. key가 없으면 전부. 내린 key 목록을 돌려준다."""
    with _lock:
        removed = [model_key] if model_key and model_key in _loaded else ([] if model_key else list(_loaded))
        for key in removed:
            _loaded.pop(key, None)
    if removed:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return removed


def _infer(image, spec):
    """이미지 -> (클래스맵, 픽셀별 확신도, id2label, 추론 초, 로딩 초, 디바이스)."""
    import torch

    model, processor, spec, device, load_seconds = get_model(spec.key)
    inputs = {key: value.to(device) for key, value in processor(images=image, return_tensors="pt").items()}
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    probs = torch.softmax(upsampled, dim=1)[0]
    confidence = probs.max(dim=0).values.float().cpu().numpy()  # 픽셀별 예측 클래스의 확신도
    pred = upsampled.argmax(dim=1)[0].cpu().numpy()
    inference_seconds = time.perf_counter() - started
    return pred, confidence, model.config.id2label, inference_seconds, load_seconds, device


def _category_masks(pred, id2label, spec):
    """같은 카테고리로 매핑되는 라벨(왼쪽/오른쪽 신발 등)의 마스크를 합친다.

    합치지 않으면 신발 한 켤레가 아이템 두 개로 등록된다.
    """
    masks, labels = {}, {}
    for label_id, label_name in id2label.items():
        category = spec.label_to_category.get(label_name)
        if category is None:
            continue
        mask = pred == int(label_id)
        if not mask.any():
            continue
        if category in masks:
            masks[category] |= mask
            labels[category].append(label_name)
        else:
            masks[category] = mask
            labels[category] = [label_name]
    return masks, labels


def _measure(mask, confidence, img_shape):
    """마스크 -> bbox, 픽셀 수, 면적 비율, 채움 비율, 평균 확신도."""
    import numpy as np

    height, width = img_shape[0], img_shape[1]
    pixel_count = int(mask.sum())
    ys, xs = np.where(mask)
    y0, x0 = max(0, int(ys.min()) - 10), max(0, int(xs.min()) - 10)
    y1 = min(height, int(ys.max()) + 10)
    x1 = min(width, int(xs.max()) + 10)
    bbox_area = (y1 - y0) * (x1 - x0)
    return {
        "bbox": (x0, y0, x1, y1),
        "pixelCount": pixel_count,
        "areaRatio": pixel_count / (height * width),
        "fillRatio": pixel_count / bbox_area if bbox_area else 0.0,
        "confidence": float(confidence[mask].mean()),
    }


def _reject_reason(stats, thresholds):
    """품질 필터에 걸린 이유를 사람이 읽을 수 있게. 통과하면 None."""
    # 소수점 둘째 자리까지 쓴다 — 한 자리면 경계에서 걸린 값이 "1.2% < 기준 1.2%"로
    # 보여서 왜 걸러졌는지 알 수가 없다.
    if stats["areaRatio"] < thresholds["minArea"]:
        return f"면적 {stats['areaRatio']:.2%} < 기준 {thresholds['minArea']:.2%}"
    if stats["fillRatio"] < thresholds["minFill"]:
        return f"채움 {stats['fillRatio']:.0%} < 기준 {thresholds['minFill']:.0%}"
    if stats["confidence"] < thresholds["minConfidence"]:
        return f"확신도 {stats['confidence']:.2f} < 기준 {thresholds['minConfidence']:.2f}"
    return None


def _shrink(image, max_px):
    """긴 변이 max_px를 넘으면 줄인다. max_px가 없으면 그대로."""
    if not max_px or max(image.size) <= max_px:
        return image
    from PIL import Image

    scale = max_px / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def _crop_png(img_np, mask, bbox, max_px=None):
    """마스크를 알파 채널로 쓴 투명 PNG 바이트."""
    import numpy as np
    from PIL import Image

    x0, y0, x1, y1 = bbox
    rgba = np.dstack([img_np[y0:y1, x0:x1], (mask[y0:y1, x0:x1] * 255).astype(np.uint8)])
    buf = io.BytesIO()
    _shrink(Image.fromarray(rgba, mode="RGBA"), max_px).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def segment(image_bytes: bytes, model_key: str | None = None) -> list[dict]:
    """전신샷 바이트 -> 품질 필터를 통과한, 카테고리별로 합쳐 크롭한 아이템 목록."""
    import numpy as np
    from PIL import Image

    spec = segment_models.resolve(model_key)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pred, confidence, id2label, _, _, _ = _infer(image, spec)
    img_np = np.array(image)
    masks, labels = _category_masks(pred, id2label, spec)

    results = []
    for category, mask in masks.items():
        stats = _measure(mask, confidence, img_np.shape)
        if _reject_reason(stats, spec.thresholds_for(category)):
            continue
        results.append({
            "category": category,
            "label": "+".join(labels[category]),
            "pixelCount": stats["pixelCount"],
            "fillRatio": round(stats["fillRatio"], 3),
            "confidence": round(stats["confidence"], 3),
            "png_bytes": _crop_png(img_np, mask, stats["bbox"]),
        })
    return results


def _overlay_png(img_np, masks):
    """원본을 어둡게 깔고 카테고리별 색으로 칠한 오버레이.

    색은 모델과 무관하게 카테고리로 정해진다 — 모델 A와 B의 오버레이를 나란히
    놓았을 때 색이 흔들리면 눈으로 비교할 수가 없다.
    """
    import numpy as np
    from PIL import Image

    grey = img_np.mean(axis=2, keepdims=True)
    canvas = (grey * 0.55 + 30).clip(0, 255).repeat(3, axis=2)
    for category, mask in masks.items():
        color = np.array(CATEGORY_COLORS.get(category, (150, 150, 150)), dtype=float)
        canvas[mask] = img_np[mask] * 0.4 + color * 0.6
    buf = io.BytesIO()
    # 마스크 경계가 비교 대상이라 JPEG 링잉을 피해 PNG로 두고, 크기는 축소로 줄인다.
    _shrink(Image.fromarray(canvas.astype(np.uint8), mode="RGB"), PREVIEW_OVERLAY_PX).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _raw_labels(pred, id2label, spec, total_pixels):
    """모델이 실제로 예측한 모든 라벨의 점유율. 매핑 여부와 상관없이 전부 싣는다.

    Fashionpedia 계열에서 sleeve/collar가 본체 마스크를 얼마나 파먹었는지는
    여기서만 보인다.
    """
    import numpy as np

    counts = np.bincount(pred.reshape(-1), minlength=max(int(key) for key in id2label) + 1)
    rows = []
    for label_id, label_name in id2label.items():
        count = int(counts[int(label_id)]) if int(label_id) < len(counts) else 0
        ratio = count / total_pixels
        if label_name in ("Background", "unlabelled") or ratio < RAW_LABEL_MIN_RATIO:
            continue
        rows.append({
            "label": label_name,
            "category": spec.label_to_category.get(label_name),
            "isPart": label_name in spec.part_labels,
            "pixelRatio": round(ratio, 4),
        })
    return sorted(rows, key=lambda row: -row["pixelRatio"])


def analyze(image_bytes: bytes, model_key: str) -> dict:
    """개발용 진단. 걸러진 후보와 이유, 원본 라벨 분포, 오버레이, 소요 시간까지.

    `_masks`는 모델 간 IoU 계산용 numpy 마스크라 직렬화 전에 호출부가 걷어낸다.
    """
    import numpy as np
    from PIL import Image

    spec = segment_models.resolve(model_key)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pred, confidence, id2label, inference_seconds, load_seconds, device = _infer(image, spec)
    img_np = np.array(image)
    height, width = img_np.shape[0], img_np.shape[1]
    masks, labels = _category_masks(pred, id2label, spec)

    items = []
    for category, mask in masks.items():
        stats = _measure(mask, confidence, img_np.shape)
        thresholds = spec.thresholds_for(category)
        reason = _reject_reason(stats, thresholds)
        items.append({
            "category": category,
            "label": "+".join(labels[category]),
            "accepted": reason is None,
            "rejectReason": reason,
            "pixelCount": stats["pixelCount"],
            # 표시용으로는 두세 자리면 충분하지만, scripts/segment_eval.py가 이 값으로
            # 기준을 바꿔가며 재판정한다. 자리를 아끼면 기준선에 걸친 후보의 판정이
            # 뒤집혀서 실험 결과를 믿을 수 없게 된다.
            "areaRatio": round(stats["areaRatio"], 6),
            "fillRatio": round(stats["fillRatio"], 5),
            "confidence": round(stats["confidence"], 5),
            "thresholds": thresholds,
            "png_bytes": _crop_png(img_np, mask, stats["bbox"], PREVIEW_CROP_PX),
        })
    items.sort(key=lambda item: (not item["accepted"], -item["areaRatio"]))

    return {
        "model": spec.as_dict(),
        "device": device,
        "loadSeconds": round(load_seconds, 2),
        "inferenceSeconds": round(inference_seconds, 2),
        "imageSize": {"width": width, "height": height},
        "acceptedCount": sum(1 for item in items if item["accepted"]),
        "items": items,
        "rawLabels": _raw_labels(pred, id2label, spec, height * width),
        "overlay_png_bytes": _overlay_png(img_np, masks),
        "_masks": masks,
    }


def mask_iou(left, right) -> float:
    """두 모델이 같은 카테고리를 얼마나 같은 픽셀로 봤는지."""
    union = int((left | right).sum())
    return round(int((left & right).sum()) / union, 3) if union else 0.0
