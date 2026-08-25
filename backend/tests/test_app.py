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
def backend_app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("FASHN_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("AVATAR_MODEL", "stabilityai/sdxl-turbo")
    monkeypatch.setenv("WEARWELL_API_TOKEN", "test-token")
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_health_describes_colab_fashn_runtime(backend_app, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "cuda_info", lambda: (True, "NVIDIA L4"))

    result = backend_app.health()

    assert result["ok"] is True
    assert result["gpu"] == "NVIDIA L4"
    assert result["model"] == "fashn-ai/fashn-vton-1.5"
    assert result["avatarModel"] == "stabilityai/sdxl-turbo"
    assert result["resolution"] == "576x864"


def test_fashn_engine_maps_api_category_and_parameters(backend_app, tmp_path: Path):
    calls = []

    class FakeResult:
        images = [Image.new("RGB", (576, 864), "navy")]

    class FakePipeline:
        inference_dtype = "torch.bfloat16"

        def __init__(self, *, weights_dir: str, device: str):
            calls.append(("init", Path(weights_dir), device))

        def __call__(self, **kwargs):
            calls.append(("call", kwargs))
            return FakeResult()

    engine = backend_app.TryOnEngine(pipeline_factory=FakePipeline)
    person = Image.new("RGB", (384, 512), "white")
    garment = Image.new("RGB", (384, 512), "black")

    result = engine.apply_one(person, garment, "upper", seed=17)

    assert result.size == (576, 864)
    assert calls[0] == ("init", tmp_path / "weights", "cuda")
    request = calls[1][1]
    assert request["category"] == "tops"
    assert request["garment_photo_type"] == "model"
    assert request["num_timesteps"] == 30
    assert request["seed"] == 17
    assert request["segmentation_free"] is True
    assert engine.last_dtype == "bfloat16"


def test_fashn_engine_maps_all_supported_categories(backend_app):
    assert backend_app.TryOnEngine.CATEGORY_MAP == {
        "upper": "tops",
        "lower": "bottoms",
        "overall": "one-pieces",
    }


def test_avatar_engine_uses_sdxl_turbo_default(backend_app):
    assert backend_app.AVATAR_MODEL == "stabilityai/sdxl-turbo"


def test_tryon_request_rejects_oversized_images(backend_app):
    with pytest.raises(ValueError):
        backend_app.TryOnRequest(
            avatar="a" * 8_000_001,
            garments=[{"image": "small", "category": "upper"}],
        )


def test_weights_ready_requires_every_fashn_file(backend_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backend_app, "FASHN_WEIGHTS_DIR", tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"model")
    assert backend_app.fashn_weights_ready() is False
    (tmp_path / "dwpose").mkdir()
    (tmp_path / "dwpose/yolox_l.onnx").write_bytes(b"pose")
    (tmp_path / "dwpose/dw-ll_ucoco_384.onnx").write_bytes(b"pose")
    assert backend_app.fashn_weights_ready() is True


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


def test_gpu_api_fails_closed_without_server_token(backend_app, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(backend_app, "API_TOKEN", "")
    response = TestClient(backend_app.app).post("/api/avatar", json={})
    assert response.status_code == 503
    assert response.json() == {"detail": "API authentication is not configured"}
