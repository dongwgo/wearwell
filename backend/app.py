from __future__ import annotations

import base64
import gc
import hmac
import io
import logging
import os
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
FASHN_MODEL = "fashn-ai/fashn-vton-1.5"
FASHN_WEIGHTS_DIR = Path(os.getenv("FASHN_WEIGHTS_DIR", ROOT / "weights" / "fashn-vton-1.5"))
AVATAR_MODEL = os.getenv("AVATAR_MODEL", "stabilityai/sdxl-turbo")
API_TOKEN = os.getenv("WEARWELL_API_TOKEN", "")
INFERENCE_SIZE = (576, 864)
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 32_000_000
GPU_LOCK = threading.RLock()
INFERENCE_GATE = threading.Lock()
RATE_LOCK = threading.Lock()
REQUEST_TIMES: dict[str, deque[float]] = defaultdict(deque)
WARMUP_VERIFIED = False


class BodyLimitMiddleware:
    def __init__(self, app, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            if int(headers.get(b"content-length", b"0")) > self.max_bytes:
                await JSONResponse({"detail": "Request too large"}, status_code=413)(scope, receive, send)
                return
        except ValueError:
            await JSONResponse({"detail": "Invalid content length"}, status_code=400)(scope, receive, send)
            return
        total = 0

        async def limited_receive():
            nonlocal total
            message = await receive()
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                raise HTTPException(status_code=413, detail="Request too large")
            return message

        await self.app(scope, limited_receive, send)

app = FastAPI(
    title="오늘옷 GPU 스타일링 API",
    version="0.3.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^http://(?:127\.0\.0\.1|localhost)(?::[1-9]\d{0,4})?$",
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def protect_gpu_api(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/"):
        if not API_TOKEN:
            return JSONResponse({"detail": "API authentication is not configured"}, status_code=503)
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ")
        if not hmac.compare_digest(supplied, API_TOKEN):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with RATE_LOCK:
            times = REQUEST_TIMES[key]
            while times and times[0] < now - 60:
                times.popleft()
            if len(times) >= 12:
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
            times.append(now)
    return await call_next(request)


class Measurements(BaseModel):
    gender: Literal["women", "men"]
    height: float = Field(ge=130, le=210)
    weight: float = Field(ge=35, le=180)
    body_shape: str = "보통"
    shoulder: float | None = Field(default=None, ge=30, le=70)
    chest: float | None = Field(default=None, ge=60, le=160)
    waist: float | None = Field(default=None, ge=45, le=160)
    hip: float | None = Field(default=None, ge=60, le=180)
    inseam: float | None = Field(default=None, ge=50, le=110)
    seed: int = 20260825


class AvatarResponse(BaseModel):
    image: str
    engine: str
    gpu: str | None
    disclaimer: str = "측정값을 시각적으로 근사한 2D 이미지이며 실제 체형·사이즈를 보증하지 않습니다."


class GarmentInput(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    category: Literal["upper", "lower", "overall"] = "upper"
    name: str = "옷"


class TryOnRequest(BaseModel):
    avatar: str = Field(min_length=1, max_length=8_000_000)
    garments: list[GarmentInput] = Field(min_length=1, max_length=4)
    seed: int = 42


def decode_image(value: str) -> Image.Image:
    try:
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        image = Image.open(io.BytesIO(base64.b64decode(encoded, validate=True)))
        width, height = image.size
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            raise ValueError("Image dimensions exceed the limit")
        if width * height > MAX_IMAGE_PIXELS:
            raise ValueError("Image pixel count exceeds the limit")
        image.load()
        return image.convert("RGB")
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Invalid image payload") from error


def encode_image(image: Image.Image, mime: str = "image/jpeg") -> str:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
    return f"data:{mime};base64,{base64.b64encode(output.getvalue()).decode()}"


def cuda_info() -> tuple[bool, str | None]:
    try:
        import torch
        return torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return False, None


def fashn_weights_ready() -> bool:
    required = (
        FASHN_WEIGHTS_DIR / "model.safetensors",
        FASHN_WEIGHTS_DIR / "dwpose" / "yolox_l.onnx",
        FASHN_WEIGHTS_DIR / "dwpose" / "dw-ll_ucoco_384.onnx",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


class AvatarEngine:
    def __init__(self) -> None:
        self.pipe = None

    def _load(self):
        if self.pipe is not None:
            return self.pipe
        import torch
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            AVATAR_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to("cuda")
        self.pipe = pipe
        return pipe

    def unload(self) -> None:
        if self.pipe is None:
            return
        self.pipe = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _shape_prompt(data: Measurements) -> str:
        bmi = data.weight / ((data.height / 100) ** 2)
        build = "slim build" if bmi < 20 else "average build" if bmi < 25 else "soft sturdy build" if bmi < 30 else "plus size build"
        gender = "Korean woman" if data.gender == "women" else "Korean man"
        ratios = []
        if data.waist and data.hip:
            ratios.append("defined waist" if data.waist / data.hip < .78 else "straight waistline")
        if data.shoulder and data.chest:
            ratios.append("balanced shoulders" if data.shoulder < data.chest * .55 else "broad shoulders")
        return ", ".join([gender, f"{data.height:.0f}cm visual proportion", build, data.body_shape, *ratios])

    def generate(self, data: Measurements) -> tuple[Image.Image, str]:
        has_cuda, _ = cuda_info()
        if has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1":
            import torch
            pipe = self._load()
            prompt = (
                f"full body studio catalog photo of a {self._shape_prompt(data)}, standing naturally in a neutral front pose, "
                "wearing seamless fitted gray base garments, feet visible, soft even light, plain warm gray background, "
                "realistic proportions, Korean fashion fitting model, high detail"
            )
            negative = "cropped, close-up, extra limbs, deformed body, asymmetric pose, loose coat, text, watermark, busy background"
            generator = torch.Generator(device="cuda").manual_seed(data.seed)
            with GPU_LOCK, torch.inference_mode():
                image = pipe(
                    prompt,
                    negative_prompt=negative,
                    width=INFERENCE_SIZE[0],
                    height=INFERENCE_SIZE[1],
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    generator=generator,
                ).images[0]
                torch.cuda.empty_cache()
            return image, "sdxl-turbo-cuda-fp16"
        return self.fallback(data), "measurement-preview-fallback"

    @staticmethod
    def fallback(data: Measurements) -> Image.Image:
        canvas = Image.new("RGB", (384, 512), "#eeeae5")
        draw = ImageDraw.Draw(canvas)
        bmi = data.weight / ((data.height / 100) ** 2)
        body_width = int(75 + max(-15, min(45, (bmi - 20) * 4)))
        shoulder_width = int((data.shoulder or (40 if data.gender == "women" else 46)) * 2.05)
        hip_width = int((data.hip or (92 if data.gender == "women" else 94)) * .78)
        cx = 192
        skin = "#d2ab94"
        cloth = "#838990"
        draw.ellipse((cx - 32, 38, cx + 32, 102), fill=skin)
        draw.rounded_rectangle((cx - shoulder_width // 2, 103, cx + shoulder_width // 2, 285), 35, fill=cloth)
        draw.polygon([(cx - body_width // 2, 200), (cx + body_width // 2, 200), (cx + hip_width // 2, 326), (cx - hip_width // 2, 326)], fill=cloth)
        leg_gap = 8
        draw.rounded_rectangle((cx - hip_width // 2, 310, cx - leg_gap, 475), 18, fill="#5e646a")
        draw.rounded_rectangle((cx + leg_gap, 310, cx + hip_width // 2, 475), 18, fill="#5e646a")
        draw.rounded_rectangle((cx - shoulder_width // 2 - 18, 115, cx - shoulder_width // 2 + 12, 325), 15, fill=skin)
        draw.rounded_rectangle((cx + shoulder_width // 2 - 12, 115, cx + shoulder_width // 2 + 18, 325), 15, fill=skin)
        return canvas.resize(INFERENCE_SIZE, Image.Resampling.LANCZOS)


class TryOnEngine:
    CATEGORY_MAP = {"upper": "tops", "lower": "bottoms", "overall": "one-pieces"}

    def __init__(self, pipeline_factory=None) -> None:
        self.pipeline = None
        self.pipeline_factory = pipeline_factory
        self.last_dtype: str | None = None

    def _load(self):
        if self.pipeline is not None:
            return self.pipeline
        if self.pipeline_factory is None:
            from fashn_vton import TryOnPipeline

            self.pipeline_factory = TryOnPipeline
        self.pipeline = self.pipeline_factory(weights_dir=str(FASHN_WEIGHTS_DIR), device="cuda")
        dtype = getattr(self.pipeline, "inference_dtype", None)
        self.last_dtype = str(dtype).removeprefix("torch.") if dtype is not None else "unknown"
        return self.pipeline

    def unload(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def apply_one(self, person: Image.Image, garment: Image.Image, category: str, seed: int) -> Image.Image:
        pipeline = self._load()
        with GPU_LOCK:
            result = pipeline(
                person_image=person,
                garment_image=garment,
                category=self.CATEGORY_MAP[category],
                garment_photo_type="model",
                num_samples=1,
                num_timesteps=int(os.getenv("FASHN_STEPS", "30")),
                guidance_scale=1.5,
                seed=seed,
                segmentation_free=True,
            ).images[0]
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
        return result.resize(INFERENCE_SIZE, Image.Resampling.LANCZOS)

    @staticmethod
    def fallback(person: Image.Image, garments: list[GarmentInput]) -> Image.Image:
        person = person.resize(INFERENCE_SIZE)
        draw = ImageDraw.Draw(person, "RGBA")
        colors = [(255, 113, 91, 110), (55, 83, 120, 115), (238, 238, 232, 125), (45, 45, 45, 100)]
        regions = {"upper": (168, 203, 408, 473), "lower": (180, 447, 396, 793), "overall": (158, 194, 418, 776)}
        for index, garment in enumerate(garments):
            draw.rounded_rectangle(regions[garment.category], radius=24, fill=colors[index % len(colors)])
        return person

    def generate(self, request: TryOnRequest) -> tuple[Image.Image, str]:
        person = decode_image(request.avatar)
        has_cuda, _ = cuda_info()
        if has_cuda and fashn_weights_ready() and os.getenv("ONEULOUT_GPU", "1") == "1":
            for index, garment in enumerate(request.garments):
                person = self.apply_one(person, decode_image(garment.image), garment.category, request.seed + index)
            return person, f"fashn-vton-1.5-cuda-{self.last_dtype or 'unknown'}"
        return self.fallback(person, request.garments), "tryon-preview-fallback"


avatar_engine = AvatarEngine()
tryon_engine = TryOnEngine()


@app.get("/api/health")
def health():
    available, name = cuda_info()
    return {
        "ok": True,
        "cuda": available,
        "gpu": name,
        "model": FASHN_MODEL,
        "avatarModel": AVATAR_MODEL,
        "weightsInstalled": fashn_weights_ready(),
        "warmupVerified": WARMUP_VERIFIED,
        "dtype": tryon_engine.last_dtype,
        "resolution": f"{INFERENCE_SIZE[0]}x{INFERENCE_SIZE[1]}",
    }


@app.post("/api/avatar", response_model=AvatarResponse)
def generate_avatar(data: Measurements):
    if not INFERENCE_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="GPU is busy")
    try:
        with GPU_LOCK:
            tryon_engine.unload()
            image, engine = avatar_engine.generate(data)
        _, gpu = cuda_info()
        return AvatarResponse(image=encode_image(image), engine=engine, gpu=gpu)
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid image payload") from error
    except Exception as error:
        logging.exception("Avatar generation failed")
        raise HTTPException(status_code=500, detail="Avatar generation failed") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/tryon")
def generate_tryon(request: TryOnRequest):
    if not INFERENCE_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="GPU is busy")
    try:
        with GPU_LOCK:
            avatar_engine.unload()
            image, engine = tryon_engine.generate(request)
        _, gpu = cuda_info()
        return {"image": encode_image(image), "engine": engine, "gpu": gpu, "garmentCount": len(request.garments)}
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid image payload") from error
    except Exception as error:
        logging.exception("Try-on generation failed")
        raise HTTPException(status_code=500, detail="Try-on generation failed") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/warmup")
def warmup_models():
    global WARMUP_VERIFIED
    if not fashn_weights_ready():
        raise HTTPException(status_code=503, detail="FASHN weights are incomplete")
    if not INFERENCE_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="GPU is busy")
    try:
        with GPU_LOCK:
            avatar_engine._load()
            avatar_engine.unload()
            tryon_engine._load()
            tryon_engine.unload()
            WARMUP_VERIFIED = True
        return {"ok": True, "avatarModel": AVATAR_MODEL, "tryonModel": FASHN_MODEL}
    except Exception as error:
        logging.exception("Model warmup failed")
        raise HTTPException(status_code=500, detail="Model warmup failed") from error
    finally:
        INFERENCE_GATE.release()
