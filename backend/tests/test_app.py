from __future__ import annotations

import base64
import importlib
import io
import sys
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


@pytest.fixture()
def backend_app(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B")
    monkeypatch.setenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    monkeypatch.setenv("VLM_LOAD_IN_4BIT", "1")
    monkeypatch.setenv("WEARWELL_DEV_TOOLS", "1")
    monkeypatch.setenv("WEARWELL_API_TOKEN", "test-token")
    monkeypatch.setenv("IMAGE_WIDTH", "768")
    monkeypatch.setenv("IMAGE_HEIGHT", "1152")
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def image_payload(color: str = "white") -> str:
    output = io.BytesIO()
    Image.new("RGB", (64, 96), color).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode()


def test_health_describes_unified_flux_runtime(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))

    result = backend_app.health()

    assert result["ok"] is True
    assert result["gpu"] == "NVIDIA L4"
    assert result["model"] == "black-forest-labs/FLUX.2-klein-4B"
    assert result["avatarModel"] == result["tryonModel"] == result["model"]
    assert result["resolution"] == "768x1152"
    assert result["modelLoaded"] is False
    assert result["vlmModel"] == "Qwen/Qwen3-VL-8B-Instruct"
    assert result["vlmLoaded"] is False
    assert result["embeddingModel"] == "google/siglip-base-patch16-224"
    assert result["embeddingLoaded"] is False
    assert result["embeddingWarmupVerified"] is False
    assert result["vlmQuantization"] == "nf4"
    assert result["segmentationModel"] == "sayeed99/segformer_b3_clothes"
    assert result["segmentationModelKey"] == "b3_clothes"
    assert result["segmentationLoaded"] is False
    assert result["segmentationLoadedModels"] == []
    assert result["segmentationDevice"] is None
    assert result["segmentationWarmupVerified"] is False
    assert result["queueTimeoutSeconds"] == 300
    assert result["gpuConcurrency"] == 2
    assert result["imageWorkersLoaded"] == 0
    assert result["rateLimitPerMinute"] == 60


def test_default_gpu_gate_allows_two_parallel_requests(backend_app):
    assert backend_app.INFERENCE_GATE.acquire(timeout=0)
    assert backend_app.INFERENCE_GATE.acquire(timeout=0)
    try:
        assert not backend_app.INFERENCE_GATE.acquire(timeout=0)
    finally:
        backend_app.INFERENCE_GATE.release()
        backend_app.INFERENCE_GATE.release()


def test_inference_slot_waits_instead_of_returning_busy(backend_app, monkeypatch: pytest.MonkeyPatch):
    class Gate:
        def __init__(self):
            self.timeout = None

        def acquire(self, *, timeout):
            self.timeout = timeout
            return True

    gate = Gate()
    monkeypatch.setattr(backend_app, "INFERENCE_GATE", gate)
    backend_app.acquire_inference_slot()
    assert gate.timeout == 300


def test_qwen_json_parser_accepts_fenced_json(backend_app):
    result = backend_app.QwenVLMEngine._parse_json(
        '```json\n{"pieces":[{"pieceId":"top-1","colors":["white"]}]}\n```'
    )
    assert result["pieces"][0]["pieceId"] == "top-1"


def test_vlm_routes_decode_image_and_use_distinct_prompts(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    calls = []

    def analyze(image, prompt, max_new_tokens):
        calls.append((image.size, prompt, max_new_tokens))
        return {"pieces": [{"pieceId": "piece-1", "colors": ["white"]}], "engine": "Qwen3-VL-8B-Instruct"}

    monkeypatch.setattr(backend_app.vlm_engine, "analyze", analyze)
    client = TestClient(backend_app.app)
    auth = {"Authorization": "Bearer test-token"}
    payload = {"image": image_payload(), "name": "white shirt", "category": "upper", "gender": "men"}

    garment = client.post("/api/vlm/garment", headers=auth, json=payload)
    lookbook = client.post("/api/vlm/lookbook", headers=auth, json=payload)
    body = client.post("/api/vlm/body", headers=auth, json=payload)

    assert garment.status_code == lookbook.status_code == body.status_code == 200
    assert [call[2] for call in calls] == [640, 900, 320]
    assert all(call[0] == (64, 96) for call in calls)
    assert len({call[1] for call in calls}) == 3
    assert "bbox" in backend_app.QwenVLMEngine.LOOKBOOK_PROMPT


def test_embedding_route_returns_siglip_vector(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    calls = []
    monkeypatch.setattr(
        backend_app.embedding_engine,
        "embed",
        lambda image: calls.append(image.size) or [0.25, -0.5, 0.75],
    )
    response = TestClient(backend_app.app).post(
        "/api/embedding",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload()},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "google/siglip-base-patch16-224",
        "vector": [0.25, -0.5, 0.75],
    }
    assert calls == [(64, 96)]


def test_siglip_v5_pooling_output_is_unwrapped(backend_app):
    pooled = object()
    output = SimpleNamespace(
        pooler_output=pooled,
        last_hidden_state=object(),
    )

    assert backend_app.SigLIPEmbeddingEngine._feature_tensor(output) is pooled


def test_avatar_prompt_contains_every_supplied_measurement(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    calls = []
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(
        engine,
        "_run",
        lambda **kwargs: calls.append(kwargs) or Image.new("RGB", backend_app.INFERENCE_SIZE),
    )
    data = backend_app.Measurements(
        gender="men",
        height=181,
        weight=76,
        body_shape="역삼각형",
        shoulder=49,
        chest=103,
        waist=82,
        hip=96,
        inseam=83,
        seed=77,
    )

    monkeypatch.setattr(backend_app, "AVATAR_BODY_REFERENCE", False)
    images, engine_name, report = engine.generate_avatar(data)

    assert images["front"].size == (768, 1152)
    assert engine_name.startswith("flux2-klein-4b")
    assert engine_name.endswith("text-only")
    assert report["bodyReference"] == "text-only"
    assert calls[0]["seed"] == 77
    for value in ("181 cm", "76 kg", "49 cm", "103 cm", "82 cm", "96 cm", "83 cm"):
        assert value in calls[0]["prompt"]
    for composition_instruction in ("soles of both feet", "floor beneath them", "75 percent", "Do not crop"):
        assert composition_instruction in calls[0]["prompt"]


def test_avatar_passes_a_body_silhouette_as_the_reference_image(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    monkeypatch.setattr(backend_app, "AVATAR_BODY_REFERENCE", True)
    calls = []
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(
        engine,
        "_run",
        lambda **kwargs: calls.append(kwargs) or Image.new("RGB", backend_app.INFERENCE_SIZE),
    )
    data = backend_app.Measurements(gender="women", height=163, weight=52, waist=68, hip=92)

    _, engine_name, report = engine.generate_avatar(data)
    assert report["views"] == ["front"]

    # 치수는 문장이 아니라 참조 이미지로 들어간다.
    assert len(calls[0]["images"]) == 1
    assert calls[0]["images"][0].size == backend_app.INFERENCE_SIZE
    assert "Reference image 1 is a body-shape guide" in calls[0]["prompt"]
    assert "163" not in calls[0]["prompt"]
    # 입력하지 않은 둘레도 추정치로 채워져 목표에 들어간다.
    assert report["targetMeasurements"]["waist"] == 68
    assert report["targetMeasurements"]["chest"] > 0
    assert engine_name.endswith(report["bodyReference"])


def test_segmentation_route_returns_transparent_crops(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "segment_service",
        SimpleNamespace(analyze=lambda raw, model_key: calls.append(model_key) or {
            "model": {"key": model_key}, "inferenceSeconds": 0.1,
            "items": [{"category": "상의", "label": "Upper-clothes", "confidence": 0.91,
                       "accepted": True, "png_bytes": b"transparent-png"}],
            "_masks": {}, "_parse": None,
        }),
    )
    response = TestClient(backend_app.app).post(
        "/api/closet/segment",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "name": "outfit"},
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert item["category"] == "상의"
    assert item["image"].startswith("data:image/png;base64,")
    assert item["confidence"] == 0.91
    # 모델을 지정하지 않으면 프로덕션 기본값으로 돈다.
    import segment_models

    assert calls == [segment_models.PRODUCTION_MODEL]
    assert response.json()["model"]["key"] == segment_models.PRODUCTION_MODEL


def test_segmentation_preview_is_reused_by_refine(backend_app, monkeypatch: pytest.MonkeyPatch):
    calls = []
    analysis = {"model": {"key": "b3_clothes"}, "items": [], "_masks": {}, "_parse": None}
    monkeypatch.setitem(
        sys.modules, "segment_service",
        SimpleNamespace(analyze=lambda raw, key: calls.append(key) or analysis),
    )
    raw = b"same-normalized-photo"

    first, first_hit = backend_app.segmentation_analysis(raw)
    second, second_hit = backend_app.segmentation_analysis(raw)

    assert first is second
    assert (first_hit, second_hit) == (False, True)
    assert calls == ["b3_clothes"]


def test_segmentation_route_rejects_unknown_model(backend_app):
    from fastapi.testclient import TestClient

    response = TestClient(backend_app.app).post(
        "/api/closet/segment",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "name": "outfit", "model": "does-not-exist"},
    )

    assert response.status_code == 422


def test_only_fashionpedia_model_can_produce_outer_category():
    """ATR 라벨에는 아우터가 없다 — 코트도 Upper-clothes로 나온다.

    비교 탭에서 아우터 컬럼이 한 모델에만 뜨는 게 버그가 아니라는 근거.
    """
    import segment_models

    outer_capable = {
        key for key, spec in segment_models.MODELS.items()
        if "아우터" in spec.label_to_category.values()
    }
    assert outer_capable == {"b3_fashion"}
    # 아우터를 만들 수 있는 모델이 있으니 품질 기준에도 아우터가 있어야 한다.
    assert set(segment_models.QUALITY_THRESHOLDS) >= set(segment_models.CATEGORIES)


def test_threshold_overrides_only_replace_named_keys():
    """모델별 보정은 지정한 키만 덮고 나머지는 기본값을 유지해야 한다.

    통째로 갈아끼우면 보정 하나 넣을 때마다 나머지 두 축이 조용히 사라진다.
    """
    import segment_models

    spec = segment_models.MODELS["b3_clothes"]
    base = segment_models.QUALITY_THRESHOLDS["신발"]
    resolved = spec.thresholds_for("신발")

    assert resolved["minConfidence"] == 0.42 != base["minConfidence"]
    assert resolved["minArea"] == base["minArea"]
    assert resolved["minFill"] == base["minFill"]
    # 보정이 없는 카테고리와 모델은 기본값 그대로.
    assert spec.thresholds_for("상의") == segment_models.QUALITY_THRESHOLDS["상의"]
    assert segment_models.MODELS["b2_clothes"].thresholds_for("신발") == base


def test_dev_model_listing_describes_every_registered_model(backend_app):
    from fastapi.testclient import TestClient

    payload = TestClient(backend_app.app).get("/api/dev/segment/models").json()

    import segment_models

    assert [model["key"] for model in payload["models"]] == list(segment_models.MODELS)
    assert payload["production"] in [model["key"] for model in payload["models"]]
    assert len(payload["default"]) == 3
    # 오버레이 색은 모델이 아니라 카테고리로 정해져야 나란히 놓고 비교할 수 있다.
    assert set(payload["categoryColors"]) == set(segment_models.CATEGORY_COLORS)


def test_dev_routes_are_hidden_when_dev_tools_are_disabled(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "DEV_TOOLS_ENABLED", False)
    client = TestClient(backend_app.app)

    assert client.get("/api/dev/segment/models").status_code == 404
    assert client.post(
        "/api/dev/segment/compare",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload()},
    ).status_code == 404
    # 모델 목록 자체는 상수라 dev 도구와 함께 닫을 이유가 없다. Refine Lab이 여기서 읽는다.
    assert client.get("/api/closet/models").status_code == 200


def stub_analysis(category: str, mask):
    return {
        "model": {"key": "stub", "title": "Stub"},
        "device": "cpu",
        "loadSeconds": 0.0,
        "inferenceSeconds": 0.1,
        "imageSize": {"width": 4, "height": 4},
        "acceptedCount": 1,
        "items": [{"category": category, "accepted": True, "png_bytes": b"crop"}],
        "rawLabels": [],
        "overlay_png_bytes": b"overlay",
        "_masks": {category: mask},
    }


def test_compare_route_returns_one_result_per_model_with_pairwise_iou(backend_app, monkeypatch: pytest.MonkeyPatch):
    import numpy as np
    from fastapi.testclient import TestClient

    import segment_service as real_service

    left = np.array([[True, True], [False, False]])
    right = np.array([[True, False], [False, False]])
    analyses = {"b2_clothes": stub_analysis("상의", left), "b3_clothes": stub_analysis("상의", right)}
    monkeypatch.setitem(
        sys.modules,
        "segment_service",
        SimpleNamespace(
            analyze=lambda raw, key: analyses[key],
            mask_iou=real_service.mask_iou,
            loaded_keys=lambda: [],
        ),
    )

    payload = TestClient(backend_app.app).post(
        "/api/dev/segment/compare",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "models": ["b2_clothes", "b3_clothes"]},
    ).json()

    assert len(payload["results"]) == 2
    # 마스크는 IoU 계산용이라 응답에 실려 나가면 안 된다(직렬화도 안 된다).
    assert all("_masks" not in result for result in payload["results"])
    assert payload["results"][0]["overlay"].startswith("data:image/png;base64,")
    assert payload["results"][0]["items"][0]["image"].startswith("data:image/png;base64,")
    assert payload["agreement"] == [{"left": "b2_clothes", "right": "b3_clothes", "categories": {"상의": 0.5}}]


def test_compare_route_keeps_working_when_one_model_fails(backend_app, monkeypatch: pytest.MonkeyPatch):
    """모델 하나가 못 떠도 나머지 비교 결과는 살려야 한다 — 비교가 목적이므로."""
    import numpy as np
    from fastapi.testclient import TestClient

    import segment_service as real_service

    def analyze(raw, key):
        if key == "b3_clothes":
            raise RuntimeError("가중치를 내려받지 못했습니다")
        return stub_analysis("하의", np.array([[True, False]]))

    monkeypatch.setitem(
        sys.modules,
        "segment_service",
        SimpleNamespace(analyze=analyze, mask_iou=real_service.mask_iou, loaded_keys=lambda: []),
    )

    payload = TestClient(backend_app.app).post(
        "/api/dev/segment/compare",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "models": ["b2_clothes", "b3_clothes"]},
    ).json()

    assert payload["results"][0]["acceptedCount"] == 1
    assert payload["results"][1]["error"] == "가중치를 내려받지 못했습니다"
    # 한쪽이 실패했으면 비교할 쌍이 없다.
    assert payload["agreement"] == []


def test_agreement_flags_categories_only_one_model_detected(backend_app):
    """한쪽만 검출한 카테고리를 빠뜨리면 '아우터를 얘만 찾았다'는 걸 놓친다."""
    import numpy as np

    import segment_service

    rows = backend_app.pairwise_agreement(
        {
            "b2_clothes": {"상의": np.array([[True]])},
            "b3_fashion": {"상의": np.array([[True]]), "아우터": np.array([[True]])},
        },
        segment_service.mask_iou,
    )

    assert rows[0]["categories"]["상의"] == 1.0
    assert rows[0]["categories"]["아우터"] == {"onlyIn": "b3_fashion"}


ARM = (slice(30, 45), slice(22, 40))       # 몸통을 가려 마스크에 구멍으로 남은 자리
FOREARM = (slice(50, 70), slice(14, 28))   # 아랫자락을 가로질러 바깥까지 이어진 자리


def damaged_mask(category_offset: int = 0):
    """세그멘테이션이 실제로 내놓는 모양 — 구멍, 가려서 파먹힌 자리, 떨어져 나온 조각.

    image_payload()가 만드는 64x96 사진과 크기를 맞춘다(numpy는 (높이, 너비)).
    """
    import numpy as np

    mask = np.zeros((96, 64), dtype=bool)
    mask[10 + category_offset:70 + category_offset, 8:56] = True
    mask[ARM] = False       # 팔이 가려서 뚫린 구멍
    mask[FOREARM] = False   # 옷 밖까지 이어져 '구멍'으로는 세지지 않는 파먹힌 자리
    mask[90:94, 2:6] = True  # 경계 부스러기
    return mask


def stub_parse(mask):
    """세그멘테이션 라벨맵. 마스크에서 빠진 두 자리가 팔로 찍혀 있다.

    이게 없으면 파먹힌 자리를 배경과 구분할 수 없어 가림 진단이 통째로 빠진다.
    """
    import numpy as np

    arm = np.zeros_like(mask)
    arm[ARM] = True
    arm[FOREARM] = True
    return {
        "pred": np.where(arm, 14, np.where(mask, 4, 0)).astype(np.int16),
        "names": {0: "Background", 4: "Upper-clothes", 14: "Left-arm"},
        "background": [0],
    }


def stub_refine_analysis(category: str = "상의"):
    mask = damaged_mask()
    analysis = stub_analysis(category, mask)
    analysis["_parse"] = stub_parse(mask)
    analysis["items"][0].update(
        label="Upper-clothes", rejectReason=None, areaRatio=0.42, fillRatio=0.61, confidence=0.91,
        thresholds={"minArea": 0.012, "minFill": 0.25, "minConfidence": 0.55},
    )
    return analysis


def install_segment_stub(monkeypatch: pytest.MonkeyPatch, build=stub_refine_analysis):
    """analyze()만 갈아끼운다.

    호출마다 새 분석 결과를 만든다 — 라우트가 응답을 만들면서 analyze()가 준 dict를
    비워 쓰기 때문에(마스크와 크롭 바이트) 같은 dict를 두 번 주면 두 번째 호출이 깨진다.
    refine_service는 진짜 모듈이라 먼저 적재해 둔다 — stub으로 덮은 뒤에 처음
    import하면 refine_service가 shrink를 못 찾는다.
    """
    import refine_service  # noqa: F401

    monkeypatch.setitem(
        sys.modules,
        "segment_service",
        SimpleNamespace(analyze=lambda raw, key: build(), loaded_keys=lambda: []),
    )


def test_refine_route_returns_every_pipeline_stage_for_each_garment(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    install_segment_stub(monkeypatch)
    calls = []
    monkeypatch.setattr(
        backend_app.image_engine,
        "refine_garment",
        lambda garment, category, name, seed, steps, hint="", gender=None, label=None: calls.append((garment.size, category, name, seed, steps, hint, gender, label))
        or (Image.new("RGB", (768, 768), "white"), "stub-flux"),
    )

    response = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "name": "스냅", "seed": 7},
    )

    assert response.status_code == 200
    payload = response.json()
    item = payload["items"][0]
    # 네 단계가 전부 실려야 파이프라인을 눈으로 따라갈 수 있다.
    assert item["stages"]["crop"].startswith("data:image/png;base64,")
    assert item["stages"]["defects"].startswith("data:image/png;base64,")
    assert item["stages"]["repaired"].startswith("data:image/png;base64,")
    assert item["stages"]["normalized"].startswith("data:image/png;base64,")
    assert item["stages"]["closet"].startswith("data:image/jpeg;base64,")
    # 세그멘테이션이 남긴 결함과 보수 결과가 숫자로 함께 와야 왜 다시 그렸는지 설명된다.
    assert item["diagnosis"]["holeCount"] == 1
    assert item["diagnosis"]["componentCount"] == 2
    assert item["repair"]["holesAfter"] == 0
    assert item["repair"]["componentsAfter"] == 1
    assert item["generation"] == {"engine": "stub-flux", "seed": 7, "steps": 4, "seconds": item["generation"]["seconds"]}
    assert item["generationError"] is None
    # 생성 모델에는 보수·정규화를 마친 정사각 이미지가 들어간다.
    assert [call[:5] for call in calls] == [((768, 768), "상의", "스냅 상의", 7, None)]
    # 어디가 무엇에 가려져 비었는지 함께 넘겨야 모델이 메운 자리를 원단으로 잇는다.
    assert "left arm" in calls[0][5]
    assert payload["overlay"].startswith("data:image/png;base64,")
    # 같은 그림을 두 번 싣지 않는다 — analyze()가 만든 크롭은 버리고 스테이지 크롭만 쓴다.
    assert "png_bytes" not in item


def test_refine_route_reports_and_fills_what_the_arm_covered(backend_app, monkeypatch: pytest.MonkeyPatch):
    """구멍 진단만으로는 '멀쩡함'이 나오던 사진. 라벨맵이 있으면 파먹힌 자리가 잡힌다."""
    from fastapi.testclient import TestClient

    install_segment_stub(monkeypatch)

    item = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "generate": False},
    ).json()["items"][0]

    assert item["diagnosis"]["occludedRatio"] > 0.02
    assert item["diagnosis"]["occludedBy"][0]["label"] == "Left-arm"
    assert item["diagnosis"]["solidity"] < 1.0
    assert [step["step"] for step in item["repair"]["steps"]].count("fillOccluded") == 1
    assert item["repair"]["occlusion"]["acceptedCount"] >= 1
    # 가림을 끄면 같은 사진에서 그 단계만 빠져야 한다 — 무엇이 무슨 일을 했는지 비교가 된다.
    without = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "generate": False, "repair": {"fillOccluded": False}},
    ).json()["items"][0]

    assert "fillOccluded" not in [step["step"] for step in without["repair"]["steps"]]
    assert without["repair"]["pixelsAfter"] < item["repair"]["pixelsAfter"]


def test_refine_route_stays_open_when_dev_tools_are_disabled(backend_app, monkeypatch: pytest.MonkeyPatch):
    """Colab 공개 터널은 dev 도구를 끄고 뜬다. 거기서도 Refine Lab이 돌아야 한다."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "DEV_TOOLS_ENABLED", False)
    install_segment_stub(monkeypatch)

    response = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "generate": False},
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["stages"]["normalized"].startswith("data:image/png;base64,")


def test_closet_model_listing_matches_the_dev_listing(backend_app):
    """Refine Lab은 공개 경로를, Seg Lab은 dev 경로를 읽는다 — 내용이 갈리면 안 된다."""
    from fastapi.testclient import TestClient

    client = TestClient(backend_app.app)

    assert client.get("/api/closet/models").json() == client.get("/api/dev/segment/models").json()


def test_refine_route_keeps_the_earlier_stages_when_generation_fails(backend_app, monkeypatch: pytest.MonkeyPatch):
    """FLUX가 실패해도 세그멘테이션·보수 결과는 살려야 어디서 깨졌는지 알 수 있다."""
    from fastapi.testclient import TestClient

    install_segment_stub(monkeypatch)

    def explode(*args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(backend_app.image_engine, "refine_garment", explode)

    payload = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload()},
    ).json()

    item = payload["items"][0]
    assert item["generationError"] == "CUDA out of memory"
    assert item["stages"]["closet"] is None
    assert item["stages"]["repaired"].startswith("data:image/png;base64,")


def test_refine_route_can_stop_before_the_generation_step(backend_app, monkeypatch: pytest.MonkeyPatch):
    """GPU 없이 마스크 보수만 확인할 수 있어야 한다 — 로컬 CPU에서 쓰는 경로."""
    from fastapi.testclient import TestClient

    install_segment_stub(monkeypatch)
    monkeypatch.setattr(
        backend_app.image_engine, "refine_garment",
        lambda *args: pytest.fail("generate=false인데 생성 모델을 불렀다"),
    )

    payload = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "generate": False},
    ).json()

    assert payload["items"][0]["stages"]["closet"] is None
    assert payload["items"][0]["generation"] is None
    assert payload["items"][0]["repair"]["pixelsAfter"] > 0


def test_refine_route_skips_rejected_candidates_unless_asked(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    def rejected():
        analysis = stub_refine_analysis()
        analysis["items"][0].update(accepted=False, rejectReason="면적 0.30% < 기준 1.20%")
        return analysis

    install_segment_stub(monkeypatch, rejected)
    monkeypatch.setattr(
        backend_app.image_engine, "refine_garment",
        lambda *args: (Image.new("RGB", (8, 8), "white"), "stub-flux"),
    )
    client = TestClient(backend_app.app)

    default = client.post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload()},
    ).json()
    asked = client.post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "includeRejected": True},
    ).json()

    assert default["items"] == []
    assert asked["items"][0]["rejectReason"].startswith("면적")


def test_refine_route_rejects_unknown_category(backend_app):
    from fastapi.testclient import TestClient

    response = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "categories": ["모자"]},
    )

    assert response.status_code == 422


def test_refine_route_applies_category_override_to_generation(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    install_segment_stub(monkeypatch)
    calls = []
    monkeypatch.setattr(
        backend_app.image_engine,
        "refine_garment",
        lambda garment, category, name, seed, steps, hint="", gender=None, label=None: calls.append((category, name))
        or (Image.new("RGB", (32, 32), "white"), "stub-flux"),
    )

    payload = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "categoryOverrides": {"상의": "아우터"}},
    ).json()

    assert payload["items"][0]["sourceCategory"] == "상의"
    assert payload["items"][0]["category"] == "아우터"
    assert calls == [("아우터", "full-body photo 아우터")]


def test_bottoms_are_never_described_as_maybe_a_skirt(backend_app):
    noun = backend_app.FluxImageEngine.garment_noun

    assert noun("하의", "Pants", "men") == "men's pair of trousers"
    assert "skirt" not in noun("하의", "Pants", "men")
    assert noun("하의", "Skirt", "women") == "women's skirt"
    assert noun("하의", "Skirt+Pants", "men") == "men's lower-body garment (pants or skirt)"
    assert noun("상의", None, None) == "upper-body garment (top)"


def test_generation_prompt_carries_the_wearers_gender(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    captured = {}
    monkeypatch.setattr(
        backend_app.image_engine, "_run",
        lambda **kwargs: captured.update(kwargs) or Image.new("RGB", (8, 8)),
    )

    backend_app.image_engine.refine_garment(
        Image.new("RGB", (768, 768)), "하의", "스냅 하의", 7, None, "", "men", "Pants",
    )

    assert "men's pair of trousers" in captured["prompt"]
    assert "This is menswear" in captured["prompt"]
    assert "unless the reference clearly shows one" in captured["prompt"]


def test_generation_without_a_gender_still_forbids_restyling(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    captured = {}
    monkeypatch.setattr(
        backend_app.image_engine, "_run",
        lambda **kwargs: captured.update(kwargs) or Image.new("RGB", (8, 8)),
    )

    backend_app.image_engine.refine_garment(Image.new("RGB", (768, 768)), "하의", "스냅 하의", 7)

    assert "menswear" not in captured["prompt"] and "womenswear" not in captured["prompt"]
    assert "Do not restyle the garment into a different cut" in captured["prompt"]


def test_refine_route_passes_gender_and_label_to_generation(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    install_segment_stub(monkeypatch)
    calls = []
    monkeypatch.setattr(
        backend_app.image_engine, "refine_garment",
        lambda garment, category, name, seed, steps, hint="", gender=None, label=None:
        calls.append((gender, label)) or (Image.new("RGB", (32, 32), "white"), "stub-flux"),
    )

    TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "gender": "men"},
    )

    assert calls == [("men", "Upper-clothes")]


def test_refine_route_rejects_an_unknown_gender(backend_app):
    from fastapi.testclient import TestClient

    response = TestClient(backend_app.app).post(
        "/api/closet/refine",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "gender": "남성"},
    )

    assert response.status_code == 422


def test_garment_generation_uses_a_square_canvas(backend_app, monkeypatch: pytest.MonkeyPatch):
    """아바타용 768x1152를 그대로 쓰면 옷이 세로로 늘어난 채 생성된다."""
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    captured = {}
    monkeypatch.setattr(
        backend_app.image_engine,
        "_run",
        lambda **kwargs: captured.update(kwargs) or Image.new("RGB", (8, 8)),
    )

    backend_app.image_engine.refine_garment(Image.new("RGB", (768, 768)), "상의", "스냅 상의", 7, None)

    assert captured["size"] == backend_app.REFINE_SIZE == (768, 768)
    # 옷을 "다시 상상"하지 않도록 원본 유지 지시가 프롬프트에 남아 있어야 한다.
    assert "same color" in captured["prompt"]
    assert "upper-body garment" in captured["prompt"]


def test_tryon_uses_person_and_all_garments_in_one_generation(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    calls = []
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(
        engine,
        "_run",
        lambda **kwargs: calls.append(kwargs) or Image.new("RGB", backend_app.INFERENCE_SIZE),
    )
    request = backend_app.TryOnRequest(
        avatar=image_payload("gray"),
        garments=[
            {"image": image_payload("white"), "category": "upper", "name": "흰 셔츠"},
            {"image": image_payload("navy"), "category": "lower", "name": "청바지"},
        ],
        seed=31,
    )

    images, _, _ = engine.generate_tryon(request)

    assert images["front"].size == (768, 1152)
    assert len(calls) == 1
    assert len(calls[0]["images"]) == 3
    assert calls[0]["seed"] == 31
    assert calls[0]["steps"] == backend_app.TRYON_STEPS
    prompt = calls[0]["prompt"]
    assert "reference image 1 is the person" in prompt.lower()
    # 참조 번호는 옷장 순서가 아니라 레이어 순서를 따른다: 하의(2) 다음에 상의(3).
    assert "Reference image 2 is the lower-body garment" in prompt
    assert "Reference image 3 is the upper-body top" in prompt


def test_tryon_reorders_layers_and_keeps_every_selected_item(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    calls = []
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(
        engine,
        "_run",
        lambda **kwargs: calls.append(kwargs) or Image.new("RGB", backend_app.INFERENCE_SIZE),
    )
    # 예전 파이프라인은 앞 4개만 잘라서 가방과 액세서리를 버렸다.
    request = backend_app.TryOnRequest(
        avatar=image_payload("gray"),
        garments=[
            {"image": image_payload("black"), "category": "accessory", "name": "볼캡"},
            {"image": image_payload("brown"), "category": "bag", "name": "크로스백"},
            {"image": image_payload("white"), "category": "shoes", "name": "스니커즈"},
            {"image": image_payload("navy"), "category": "outer", "name": "블레이저"},
            {"image": image_payload("white"), "category": "upper", "name": "셔츠"},
            {"image": image_payload("gray"), "category": "lower", "name": "슬랙스"},
        ],
    )

    engine.generate_tryon(request)

    prompt = calls[0]["prompt"]
    assert len(calls[0]["images"]) == 7  # 사람 1 + 아이템 6
    positions = [
        prompt.index(needle)
        for needle in ("'슬랙스'", "'셔츠'", "'블레이저'", "'스니커즈'", "'크로스백'", "'볼캡'")
    ]
    assert positions == sorted(positions)
    assert "the top then the outerwear" in prompt
    assert "cross-body" in prompt  # 가방 종류에 맞는 착용 지점
    assert "on the head" in prompt  # 액세서리 착용 지점


def test_tryon_request_rejects_oversized_images(backend_app):
    with pytest.raises(ValueError):
        backend_app.TryOnRequest(
            avatar="a" * 8_000_001,
            garments=[{"image": "small", "category": "upper"}],
        )


def test_tryon_accepts_full_outfit_categories(backend_app):
    request = backend_app.TryOnRequest(
        avatar=image_payload(),
        garments=[
            {"image": image_payload(), "category": "outer"},
            {"image": image_payload(), "category": "shoes"},
            {"image": image_payload(), "category": "bag"},
            {"image": image_payload(), "category": "accessory"},
        ],
    )
    assert [item.category for item in request.garments] == ["outer", "shoes", "bag", "accessory"]


def test_decode_image_rejects_excessive_dimensions(backend_app):
    image = Image.new("RGB", (4097, 1), "white")
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    encoded = base64.b64encode(payload.getvalue()).decode()
    with pytest.raises(ValueError, match="dimensions"):
        backend_app.decode_image(encoded)


def test_public_app_requires_token_and_hides_backend_source(backend_app):
    from fastapi.testclient import TestClient

    client = TestClient(backend_app.app)
    assert client.post("/api/avatar", json={}).status_code == 401
    assert client.get("/backend/app.py").status_code == 404

    auth = {"Authorization": "Bearer test-token"}
    invalid = client.post(
        "/api/tryon",
        headers=auth,
        json={"avatar": "not-base64", "garments": [{"image": "not-base64", "category": "upper"}]},
    )
    assert invalid.status_code == 422
    assert invalid.json() == {"detail": "Invalid image payload"}

    oversized = client.post(
        "/api/avatar",
        headers={**auth, "Content-Length": str(backend_app.MAX_REQUEST_BYTES + 1)},
        content=b"{}",
    )
    assert oversized.status_code == 413


def test_backend_exposes_only_api_routes(backend_app):
    from fastapi.testclient import TestClient

    client = TestClient(backend_app.app)
    assert client.get("/").status_code == 404
    assert client.get("/index.html").status_code == 404
    assert client.get("/app.js").status_code == 404
    assert client.get("/assets/lookbook/look-001.jpg").status_code == 404
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_cors_allows_local_frontend_only(backend_app):
    from fastapi.testclient import TestClient

    client = TestClient(backend_app.app)
    headers = {
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }
    local = client.options("/api/avatar", headers={**headers, "Origin": "http://localhost:8000"})
    assert local.status_code == 200
    assert local.headers["access-control-allow-origin"] == "http://localhost:8000"

    remote = client.options(
        "/api/avatar",
        headers={**headers, "Origin": "https://frontend.example.com"},
    )
    assert "access-control-allow-origin" not in remote.headers


def test_gpu_api_fails_closed_without_server_token(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "API_TOKEN", "")
    response = TestClient(backend_app.app).post("/api/avatar", json={})
    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication is not configured"}


def test_tryon_judge_route_scores_a_result_image(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    seen = {}

    def fake_analyze(image, prompt, max_new_tokens):
        seen["prompt"] = prompt
        return {"layering_ok": True, "items_present": ["셔츠"], "identity_ok": True}

    monkeypatch.setattr(backend_app.vlm_engine, "analyze", fake_analyze)
    response = TestClient(backend_app.app).post(
        "/api/vlm/tryon-judge",
        headers={"Authorization": "Bearer test-token"},
        json={"image": image_payload(), "manifest": "- 화이트 셔츠 (상의)\\n- 네이비 블레이저 (아우터)"},
    )

    assert response.status_code == 200
    assert response.json()["layering_ok"] is True
    # 심판 프롬프트에 채점 대상 목록이 실제로 들어가야 한다.
    assert "네이비 블레이저" in seen["prompt"]
    assert "layering_ok" in seen["prompt"]


def test_avatar_generates_side_and_back_from_the_finished_front(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    monkeypatch.setattr(backend_app, "AVATAR_BODY_REFERENCE", True)
    calls = []
    engine = backend_app.FluxImageEngine()

    def fake_run(**kwargs):
        calls.append(kwargs)
        # 호출마다 구분되는 이미지를 돌려줘야 front가 재사용됐는지 확인할 수 있다.
        return Image.new("RGB", backend_app.INFERENCE_SIZE, (len(calls) * 40, 0, 0))

    monkeypatch.setattr(engine, "_run", fake_run)
    data = backend_app.Measurements(gender="men", height=178, weight=72, views=["side", "back"])

    images, _, report = engine.generate_avatar(data)

    # front를 요청하지 않아도 먼저 만들어진다 — 나머지 시점의 인물 기준이기 때문.
    assert report["views"] == ["front", "side", "back"]
    assert set(images) == {"front", "side", "back"}
    assert [len(call["images"]) for call in calls] == [1, 2, 2]
    # 측면·후면은 완성된 정면을 참조 이미지 2로 받는다.
    for call in calls[1:]:
        assert call["images"][1] is images["front"]
    assert "seen from the side" in calls[1]["prompt"]
    assert "rotated to the side view" in calls[1]["prompt"]
    assert "seen from the back" in calls[2]["prompt"]


def test_avatar_view_list_is_deduplicated(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    monkeypatch.setattr(backend_app, "AVATAR_BODY_REFERENCE", True)
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(engine, "_run", lambda **k: Image.new("RGB", backend_app.INFERENCE_SIZE))
    data = backend_app.Measurements(gender="men", height=178, weight=72, views=["front", "front", "side"])

    _, _, report = engine.generate_avatar(data)

    assert report["views"] == ["front", "side"]


def test_avatar_route_returns_every_requested_view(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    monkeypatch.setattr(
        backend_app.image_engine,
        "generate_avatar",
        lambda data: (
            {view: Image.new("RGB", (64, 96)) for view in ("front", "back")},
            "stub-engine",
            {"bodyReference": "silhouette-fallback", "views": ["front", "back"]},
        ),
    )
    response = TestClient(backend_app.app).post(
        "/api/avatar",
        headers={"Authorization": "Bearer test-token"},
        json={"gender": "men", "height": 178, "weight": 72, "views": ["front", "back"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body["views"]) == {"front", "back"}
    assert body["image"] == body["views"]["front"]


def test_avatar_route_rejects_an_unknown_view(backend_app):
    from fastapi.testclient import TestClient

    response = TestClient(backend_app.app).post(
        "/api/avatar",
        headers={"Authorization": "Bearer test-token"},
        json={"gender": "men", "height": 178, "weight": 72, "views": ["top-down"]},
    )
    assert response.status_code == 422


def test_tryon_pads_the_person_reference_so_the_feet_survive(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    calls = []
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(
        engine, "_run",
        lambda **kwargs: calls.append(kwargs) or Image.new("RGB", backend_app.INFERENCE_SIZE),
    )
    request = backend_app.TryOnRequest(
        avatar=image_payload("gray"),
        garments=[{"image": image_payload("white"), "category": "upper", "name": "셔츠"}],
    )

    engine.generate_tryon(request)

    # 여백을 붙여도 입력 사진 비율은 유지하고, 생성 크기도 그 비율을 따른다.
    person = calls[0]["images"][0]
    assert abs(person.width / person.height - 64 / 96) < 0.01
    width, height = calls[0]["size"]
    assert abs(width / height - person.width / person.height) < 0.02


def test_tryon_rotates_the_finished_front_for_other_views(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    calls = []
    engine = backend_app.FluxImageEngine()

    def fake_run(**kwargs):
        calls.append(kwargs)
        return Image.new("RGB", backend_app.INFERENCE_SIZE, (len(calls) * 30, 0, 0))

    monkeypatch.setattr(engine, "_run", fake_run)
    request = backend_app.TryOnRequest(
        avatar=image_payload("gray"),
        garments=[
            {"image": image_payload("white"), "category": "upper", "name": "셔츠"},
            {"image": image_payload("navy"), "category": "outer", "name": "코트"},
        ],
        views=["front", "side", "back"],
        avatarViews={"side": image_payload("gray"), "back": image_payload("gray")},
    )

    images, _, _ = engine.generate_tryon(request)

    assert set(images) == {"front", "side", "back"}
    assert [len(call["images"]) for call in calls] == [3, 3, 3]
    for call in calls[1:]:
        assert call["images"][0] is images["front"]
        assert "Reference image 1 is the finished photograph" in call["prompt"]
    assert "exact left profile" in calls[1]["prompt"]
    assert "directly behind" in calls[2]["prompt"]
    assert len({call["size"] for call in calls}) == 1


def test_tryon_rotates_without_a_separate_side_avatar(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(engine, "_run", lambda **k: Image.new("RGB", backend_app.INFERENCE_SIZE))
    # 완성된 정면과 옷 참조가 있으므로 별도 측면 아바타 없이도 회전한다.
    request = backend_app.TryOnRequest(
        avatar=image_payload("gray"),
        garments=[{"image": image_payload("white"), "category": "upper"}],
        views=["front", "side"],
    )

    images, _, _ = engine.generate_tryon(request)

    assert set(images) == {"front", "side"}


def test_tryon_route_returns_every_generated_view(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    monkeypatch.setattr(
        backend_app.image_engine,
        "generate_tryon",
        lambda request: (
            {v: Image.new("RGB", (64, 96)) for v in ("front", "back")},
            "stub",
            list(request.garments),
        ),
    )
    response = TestClient(backend_app.app).post(
        "/api/tryon",
        headers={"Authorization": "Bearer test-token"},
        json={
            "avatar": image_payload(),
            "garments": [{"image": image_payload(), "category": "upper", "name": "셔츠"}],
            "views": ["front", "back"],
            "avatarViews": {"back": image_payload()},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body["views"]) == {"front", "back"}
    assert body["image"] == body["views"]["front"]


def test_decode_image_flattens_transparency_instead_of_reviving_the_background(backend_app):
    """segment_service가 만드는 옷 PNG는 알파로만 옷을 표시하고 RGB 채널에는
    원본 사진이 그대로 남아 있다. 그냥 convert("RGB")하면 잘라냈다고 생각한
    사람과 배경이 통째로 되살아난다."""
    import numpy as np

    photo = np.zeros((40, 30, 3), np.uint8)
    photo[:, :] = (200, 40, 40)          # 배경(사람·스튜디오)
    photo[10:30, 8:22] = (30, 30, 200)   # 실제 옷
    mask = np.zeros((40, 30), bool)
    mask[10:30, 8:22] = True
    payload = io.BytesIO()
    Image.fromarray(np.dstack([photo, (mask * 255).astype(np.uint8)]), "RGBA").save(payload, format="PNG")

    decoded = backend_app.decode_image(base64.b64encode(payload.getvalue()).decode())

    assert decoded.mode == "RGB"
    assert decoded.getpixel((2, 2)) == (255, 255, 255)   # 투명했던 곳은 흰 배경
    assert decoded.getpixel((15, 20)) == (30, 30, 200)   # 옷은 그대로


def test_decode_image_leaves_opaque_images_alone(backend_app):
    payload = io.BytesIO()
    Image.new("RGB", (20, 20), (12, 34, 56)).save(payload, format="PNG")
    decoded = backend_app.decode_image(base64.b64encode(payload.getvalue()).decode())
    assert decoded.getpixel((5, 5)) == (12, 34, 56)


def test_tryon_response_names_the_items_it_had_to_drop(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "cuda_info", lambda: (False, None))
    response = TestClient(backend_app.app).post(
        "/api/tryon",
        headers={"Authorization": "Bearer test-token"},
        json={
            "avatar": image_payload(),
            "garments": [
                {"image": image_payload(), "category": "upper", "name": "티셔츠"},
                {"image": image_payload(), "category": "upper", "name": "셔츠"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    # 같은 자리를 다투면 안쪽 한 벌만 남는다. 무엇이 빠졌는지 알려줘야 사용자가
    # "옷이 반영되지 않았다"고 오해하지 않는다.
    assert [item["name"] for item in body["appliedGarments"]] == ["티셔츠"]
    assert [item["name"] for item in body["droppedGarments"]] == ["셔츠"]


def test_tryon_reports_nothing_dropped_for_a_clean_outfit(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "cuda_info", lambda: (False, None))
    response = TestClient(backend_app.app).post(
        "/api/tryon",
        headers={"Authorization": "Bearer test-token"},
        json={
            "avatar": image_payload(),
            "garments": [
                {"image": image_payload(), "category": "upper", "name": "셔츠"},
                {"image": image_payload(), "category": "lower", "name": "슬랙스"},
            ],
        },
    )
    assert response.json()["droppedGarments"] == []


def test_tryon_survives_an_unreadable_side_avatar(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(engine, "_run", lambda **k: Image.new("RGB", backend_app.INFERENCE_SIZE))
    request = backend_app.TryOnRequest(
        avatar=image_payload("gray"),
        garments=[{"image": image_payload("white"), "category": "upper", "name": "셔츠"}],
        views=["front", "side"],
        avatarViews={"side": "not-a-valid-image"},
    )

    images, _, _ = engine.generate_tryon(request)

    # 측면 아바타가 깨져도 완성 정면을 기준으로 측면을 만든다.
    assert set(images) == {"front", "side"}


def test_pipeline_is_only_built_once_under_concurrent_load(backend_app, monkeypatch: pytest.MonkeyPatch):
    builds = []
    engine = backend_app.FluxImageEngine()

    def slow_build():
        time.sleep(0.05)
        builds.append(1)
        engine.pipe = object()

    monkeypatch.setattr(engine, "_build_pipeline", slow_build)
    threads = [threading.Thread(target=engine._load) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(builds) == 1
