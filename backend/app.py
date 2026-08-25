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
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B")
API_TOKEN = os.getenv("WEARWELL_API_TOKEN", "")
INFERENCE_SIZE = (
    int(os.getenv("IMAGE_WIDTH", "768")),
    int(os.getenv("IMAGE_HEIGHT", "1152")),
)
INFERENCE_STEPS = int(os.getenv("FLUX_STEPS", "4"))
GUIDANCE_SCALE = float(os.getenv("FLUX_GUIDANCE", "1.0"))
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
    title="오늘옷 생성형 이미지 API",
    version="0.4.0",
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
    disclaimer: str = "입력한 치수를 시각적으로 근사한 이미지이며 실제 체형이나 의류 사이즈를 보증하지 않습니다."


class GarmentInput(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    category: Literal["upper", "lower", "overall", "outer", "shoes", "bag", "accessory"] = "upper"
    name: str = Field(default="옷", max_length=100)


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
    image.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
    return f"data:{mime};base64,{base64.b64encode(output.getvalue()).decode()}"


def cuda_info() -> tuple[bool, str | None]:
    try:
        import torch

        available = torch.cuda.is_available()
        return available, torch.cuda.get_device_name(0) if available else None
    except Exception:
        return False, None


class FluxImageEngine:
    CATEGORY_LABELS = {
        "upper": "upper-body garment",
        "lower": "lower-body garment",
        "overall": "one-piece or full-body garment",
        "outer": "outerwear layer",
        "shoes": "pair of shoes",
        "bag": "bag",
        "accessory": "fashion accessory",
    }

    def __init__(self, pipeline_factory=None) -> None:
        self.pipe = None
        self.pipeline_factory = pipeline_factory
        self.dtype: str | None = None

    def _load(self):
        if self.pipe is not None:
            return self.pipe
        import torch

        if self.pipeline_factory is None:
            from diffusers import Flux2KleinPipeline

            self.pipeline_factory = Flux2KleinPipeline

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.dtype = str(dtype).removeprefix("torch.")
        pipe = self.pipeline_factory.from_pretrained(IMAGE_MODEL, torch_dtype=dtype)
        if os.getenv("FLUX_CPU_OFFLOAD", "0") == "1":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        if hasattr(pipe, "set_progress_bar_config"):
            pipe.set_progress_bar_config(disable=True)
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
    def _shape_description(data: Measurements) -> str:
        bmi = data.weight / ((data.height / 100) ** 2)
        build = "lean" if bmi < 20 else "average" if bmi < 24 else "solid" if bmi < 28 else "fuller"
        gender = "adult Korean woman" if data.gender == "women" else "adult Korean man"
        details = [
            gender,
            f"{data.height:.0f} cm tall",
            f"{data.weight:.0f} kg",
            f"a {build} build",
            f"body-shape description: {data.body_shape}",
        ]
        if data.shoulder:
            details.append(f"shoulder width {data.shoulder:.0f} cm")
        if data.chest:
            details.append(f"chest circumference {data.chest:.0f} cm")
        if data.waist:
            details.append(f"waist circumference {data.waist:.0f} cm")
        if data.hip:
            details.append(f"hip circumference {data.hip:.0f} cm")
        if data.inseam:
            details.append(f"inseam {data.inseam:.0f} cm")
        if data.waist and data.hip:
            details.append(f"waist-to-hip ratio {data.waist / data.hip:.2f}")
        return ", ".join(details)

    def _run(self, *, prompt: str, seed: int, images: list[Image.Image] | None = None) -> Image.Image:
        import torch

        pipe = self._load()
        inputs = {
            "prompt": prompt,
            "height": INFERENCE_SIZE[1],
            "width": INFERENCE_SIZE[0],
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": INFERENCE_STEPS,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
        }
        if images:
            inputs["image"] = images
        with GPU_LOCK, torch.inference_mode():
            output = pipe(**inputs).images[0]
            torch.cuda.empty_cache()
        return output.resize(INFERENCE_SIZE, Image.Resampling.LANCZOS)

    def generate_avatar(self, data: Measurements) -> tuple[Image.Image, str]:
        has_cuda, _ = cuda_info()
        if has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1":
            prompt = (
                "Create a photorealistic full-body studio fitting avatar. The subject is "
                f"{self._shape_description(data)}. Accurately reflect the stated height, weight, body build, "
                "and body proportions without slimming or exaggeration. Front-facing relaxed symmetrical A-pose, "
                "arms slightly away from the torso, both feet fully visible, eye-level camera, 85 mm catalog lens. "
                "Wear a plain fitted charcoal crew-neck top and fitted mid-thigh charcoal shorts so the body outline "
                "is clearly visible. Clean warm-gray seamless background, soft even studio lighting, realistic skin, "
                "natural Korean facial features, no accessories, no outerwear, no text, no watermark."
            )
            return self._run(prompt=prompt, seed=data.seed), self.engine_name
        return self.fallback_avatar(data), "measurement-preview-fallback"

    def generate_tryon(self, request: TryOnRequest) -> tuple[Image.Image, str]:
        person = decode_image(request.avatar)
        garments = [decode_image(item.image) for item in request.garments]
        has_cuda, _ = cuda_info()
        if has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1":
            garment_roles = "; ".join(
                f"reference image {index + 2}: {self.CATEGORY_LABELS[item.category]} named '{item.name}'"
                for index, item in enumerate(request.garments)
            )
            prompt = (
                "Virtual try-on edit using all reference images together. Reference image 1 is the person and must "
                f"remain the same person with the same face, hair, body proportions, pose, hands, legs, and background. {garment_roles}. "
                "Dress the person in exactly those referenced garments. Preserve each garment's actual color, fabric, "
                "pattern, logo placement, neckline, sleeve length, cut, and silhouette; remove any original model or "
                "background visible in the garment references. Replace only the clothing categories provided and keep "
                "the remaining base clothing unchanged. Make the fit physically plausible with correct layering, folds, "
                "occlusion, body contact, and consistent studio lighting. Photorealistic Korean fashion e-commerce "
                "full-body result, no extra garments, no invented accessories, no text, no collage, no split screen."
            )
            return self._run(prompt=prompt, seed=request.seed, images=[person, *garments]), self.engine_name
        return self.fallback_tryon(person, request.garments), "tryon-preview-fallback"

    @property
    def engine_name(self) -> str:
        return f"flux2-klein-4b-cuda-{self.dtype or 'auto'}"

    @staticmethod
    def fallback_avatar(data: Measurements) -> Image.Image:
        canvas = Image.new("RGB", (512, 768), "#eeeae5")
        draw = ImageDraw.Draw(canvas)
        bmi = data.weight / ((data.height / 100) ** 2)
        body_width = int(90 + max(-18, min(55, (bmi - 20) * 4.5)))
        shoulder_width = int((data.shoulder or (40 if data.gender == "women" else 46)) * 2.35)
        hip_width = int((data.hip or (92 if data.gender == "women" else 94)) * 0.93)
        cx = 256
        skin = "#d2ab94"
        cloth = "#696f75"
        draw.ellipse((cx - 38, 55, cx + 38, 131), fill=skin)
        draw.rounded_rectangle((cx - shoulder_width // 2, 132, cx + shoulder_width // 2, 410), 42, fill=cloth)
        draw.polygon(
            [(cx - body_width // 2, 280), (cx + body_width // 2, 280), (cx + hip_width // 2, 500), (cx - hip_width // 2, 500)],
            fill=cloth,
        )
        draw.rounded_rectangle((cx - hip_width // 2, 480, cx - 10, 715), 22, fill="#555b61")
        draw.rounded_rectangle((cx + 10, 480, cx + hip_width // 2, 715), 22, fill="#555b61")
        draw.rounded_rectangle((cx - shoulder_width // 2 - 23, 150, cx - shoulder_width // 2 + 14, 475), 18, fill=skin)
        draw.rounded_rectangle((cx + shoulder_width // 2 - 14, 150, cx + shoulder_width // 2 + 23, 475), 18, fill=skin)
        return canvas.resize(INFERENCE_SIZE, Image.Resampling.LANCZOS)

    @staticmethod
    def fallback_tryon(person: Image.Image, garments: list[GarmentInput]) -> Image.Image:
        person = person.resize(INFERENCE_SIZE)
        draw = ImageDraw.Draw(person, "RGBA")
        colors = [(255, 113, 91, 110), (55, 83, 120, 115), (238, 238, 232, 125), (45, 45, 45, 100)]
        width, height = INFERENCE_SIZE
        regions = {
            "upper": (int(width * .29), int(height * .23), int(width * .71), int(height * .54)),
            "lower": (int(width * .31), int(height * .51), int(width * .69), int(height * .91)),
            "overall": (int(width * .27), int(height * .22), int(width * .73), int(height * .90)),
            "outer": (int(width * .24), int(height * .21), int(width * .76), int(height * .65)),
            "shoes": (int(width * .25), int(height * .88), int(width * .75), int(height * .98)),
            "bag": (int(width * .68), int(height * .38), int(width * .88), int(height * .68)),
            "accessory": (int(width * .35), int(height * .08), int(width * .65), int(height * .22)),
        }
        for index, garment in enumerate(garments):
            draw.rounded_rectangle(regions[garment.category], radius=28, fill=colors[index % len(colors)])
        return person


image_engine = FluxImageEngine()


@app.get("/api/health")
def health():
    available, name = cuda_info()
    return {
        "ok": True,
        "cuda": available,
        "gpu": name,
        "model": IMAGE_MODEL,
        "avatarModel": IMAGE_MODEL,
        "tryonModel": IMAGE_MODEL,
        "modelLoaded": image_engine.pipe is not None,
        "warmupVerified": WARMUP_VERIFIED,
        "dtype": image_engine.dtype,
        "resolution": f"{INFERENCE_SIZE[0]}x{INFERENCE_SIZE[1]}",
    }


@app.post("/api/avatar", response_model=AvatarResponse)
def generate_avatar(data: Measurements):
    if not INFERENCE_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="GPU is busy")
    try:
        image, engine = image_engine.generate_avatar(data)
        _, gpu = cuda_info()
        return AvatarResponse(image=encode_image(image), engine=engine, gpu=gpu)
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
        image, engine = image_engine.generate_tryon(request)
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
def warmup_model():
    global WARMUP_VERIFIED
    available, _ = cuda_info()
    if not available:
        raise HTTPException(status_code=503, detail="CUDA GPU is unavailable")
    if not INFERENCE_GATE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="GPU is busy")
    try:
        image_engine._load()
        WARMUP_VERIFIED = True
        return {"ok": True, "model": IMAGE_MODEL, "dtype": image_engine.dtype}
    except Exception as error:
        logging.exception("Model warmup failed")
        raise HTTPException(status_code=500, detail="Model warmup failed") from error
    finally:
        INFERENCE_GATE.release()
