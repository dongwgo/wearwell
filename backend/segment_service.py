"""전신샷에서 의류 아이템을 자동으로 분리한다 (human-parsing 세그멘테이션).

scripts/segment_prototype.py로 검증한 파이프라인의 정식 버전. transformers+torch를
첫 요청 때만 불러와 전역에 캐싱한다 — 모델
로딩이 무겁기 때문이다(가중치는 ~/.cache/huggingface에 자동 캐싱).
"""

import io
import os
import threading

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

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
    "하의": {"minArea": 0.012, "minFill": 0.25, "minConfidence": 0.55},
    "원피스": {"minArea": 0.02, "minFill": 0.25, "minConfidence": 0.55},
    "신발": {"minArea": 0.004, "minFill": 0.08, "minConfidence": 0.5},
    "가방": {"minArea": 0.004, "minFill": 0.08, "minConfidence": 0.5},
    "액세서리": {"minArea": 0.003, "minFill": 0.08, "minConfidence": 0.5},
}

# ATR 데이터셋 라벨 -> 이 프로젝트 옷장 카테고리(app.js의 `categories`) 매핑.
LABEL_TO_CATEGORY = {
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

_lock = threading.Lock()
_model = None
_processor = None
_device = None


class ModelUnavailable(RuntimeError):
    pass


def get_model():
    global _model, _processor, _device
    with _lock:
        if _model is None:
            try:
                from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
            except ImportError as exc:
                raise ModelUnavailable(
                    "세그멘테이션 모델(transformers/torchvision)이 설치되어 있지 않습니다. "
                    "requirements.txt를 설치한 뒤 서버를 다시 실행하세요."
                ) from exc
            import torch

            requested = os.getenv("SEGMENTATION_DEVICE", "auto").lower()
            _device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
            _processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
            _model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).to(_device)
            _model.eval()
    return _model, _processor


def segment(image_bytes: bytes) -> list[dict]:
    """전신샷 바이트 -> 카테고리별로 합쳐 크롭한 아이템 목록.

    같은 카테고리로 매핑되는 라벨(왼쪽/오른쪽 신발 등)은 마스크를 합친 뒤
    한 번만 크롭한다 — 신발 한 켤레가 아이템 두 개로 등록되는 것을 막는다.
    """
    import numpy as np
    import torch
    from PIL import Image

    model, processor = get_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = {key: value.to(_device) for key, value in processor(images=image, return_tensors="pt").items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    probs = torch.softmax(upsampled, dim=1)[0]
    confidence = probs.max(dim=0).values.float().cpu().numpy()  # 픽셀별 예측 클래스의 확신도
    pred = upsampled.argmax(dim=1)[0].cpu().numpy()

    id2label = model.config.id2label
    img_np = np.array(image)
    total_pixels = img_np.shape[0] * img_np.shape[1]

    category_masks = {}
    category_labels = {}
    for label_id, label_name in id2label.items():
        category = LABEL_TO_CATEGORY.get(label_name)
        if category is None:
            continue
        mask = pred == int(label_id)
        if not mask.any():
            continue
        if category in category_masks:
            category_masks[category] |= mask
            category_labels[category].append(label_name)
        else:
            category_masks[category] = mask
            category_labels[category] = [label_name]

    results = []
    for category, mask in category_masks.items():
        thresholds = QUALITY_THRESHOLDS[category]
        pixel_count = int(mask.sum())
        if pixel_count < total_pixels * thresholds["minArea"]:
            continue

        ys, xs = np.where(mask)
        y0, x0 = max(0, ys.min() - 10), max(0, xs.min() - 10)
        y1 = min(img_np.shape[0], ys.max() + 10)
        x1 = min(img_np.shape[1], xs.max() + 10)
        bbox_area = (y1 - y0) * (x1 - x0)
        fill_ratio = pixel_count / bbox_area if bbox_area else 0
        mean_confidence = float(confidence[mask].mean())
        if fill_ratio < thresholds["minFill"] or mean_confidence < thresholds["minConfidence"]:
            continue

        crop_rgb = img_np[y0:y1, x0:x1]
        crop_mask = mask[y0:y1, x0:x1]
        rgba = np.dstack([crop_rgb, (crop_mask * 255).astype(np.uint8)])

        buf = io.BytesIO()
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
        results.append({
            "category": category,
            "label": "+".join(category_labels[category]),
            "pixelCount": pixel_count,
            "fillRatio": round(fill_ratio, 3),
            "confidence": round(mean_confidence, 3),
            "png_bytes": buf.getvalue(),
        })

    return results
