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
    assert result["vlmQuantization"] == "nf4"
    assert result["queueTimeoutSeconds"] == 300
    assert result["rateLimitPerMinute"] == 60


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

    monkeypatch.setitem(
        sys.modules,
        "segment_service",
        SimpleNamespace(segment=lambda raw: [{
            "category": "상의",
            "label": "Upper-clothes",
            "confidence": 0.91,
            "png_bytes": b"transparent-png",
        }]),
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

    # 사람 참조는 원본 그대로가 아니라 위아래 여백을 덧댄 뒤 목표 크기로 들어간다.
    person = calls[0]["images"][0]
    assert person.size == backend_app.INFERENCE_SIZE
    assert person is not None


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
    # 정면은 사람 + 옷 2장, 측면·후면은 체형 가이드 + 완성된 정면 결과만 받는다.
    assert [len(call["images"]) for call in calls] == [3, 2, 2]
    for call in calls[1:]:
        assert call["images"][1] is images["front"]
        assert "Reference image 2 is the finished photograph" in call["prompt"]
    assert "exact left profile" in calls[1]["prompt"]
    assert "directly behind" in calls[2]["prompt"]


def test_tryon_skips_views_without_an_avatar_image(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))
    engine = backend_app.FluxImageEngine()
    monkeypatch.setattr(engine, "_run", lambda **k: Image.new("RGB", backend_app.INFERENCE_SIZE))
    # 측면 아바타 없이 측면 착장을 요청해도 조용히 정면만 만든다.
    request = backend_app.TryOnRequest(
        avatar=image_payload("gray"),
        garments=[{"image": image_payload("white"), "category": "upper"}],
        views=["front", "side"],
    )

    images, _, _ = engine.generate_tryon(request)

    assert set(images) == {"front"}


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

    # 측면 아바타가 깨져도 정면 착장은 살아남는다.
    assert set(images) == {"front"}


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
