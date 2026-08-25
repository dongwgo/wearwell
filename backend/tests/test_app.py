from __future__ import annotations

import base64
import importlib
import io
import sys
from pathlib import Path

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

    result, engine_name = engine.generate_avatar(data)

    assert result.size == (768, 1152)
    assert engine_name.startswith("flux2-klein-4b")
    assert calls[0]["seed"] == 77
    for value in ("181 cm", "76 kg", "49 cm", "103 cm", "82 cm", "96 cm", "83 cm"):
        assert value in calls[0]["prompt"]


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

    result, _ = engine.generate_tryon(request)

    assert result.size == (768, 1152)
    assert len(calls) == 1
    assert len(calls[0]["images"]) == 3
    assert calls[0]["seed"] == 31
    assert "reference image 1 is the person" in calls[0]["prompt"].lower()
    assert "reference image 2" in calls[0]["prompt"]
    assert "reference image 3" in calls[0]["prompt"]


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
