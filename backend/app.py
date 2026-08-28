from __future__ import annotations

import base64
import gc
import hashlib
import hmac
import io
import json
import logging
import os
import queue
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

import segment_models
from avatar_body import (
    VIEW_YAW,
    BodyTarget,
    build_avatar_prompt,
    build_body_reference,
    build_photo_avatar_prompt,
    fit_generation_size,
    pad_for_full_body,
)
from tryon_prompt import build_tryon_prompt, build_tryon_view_prompt, order_garments

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
SIGLIP_MODEL = os.getenv("SIGLIP_MODEL", "google/siglip-base-patch16-224")
SEGMENTATION_MODEL = os.getenv("SEGMENTATION_MODEL", "sayeed99/segformer_b3_clothes")
VLM_LOAD_IN_4BIT = os.getenv("VLM_LOAD_IN_4BIT", "1") == "1"
VLM_MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", str(1024 * 1024)))
API_TOKEN = os.getenv("WEARWELL_API_TOKEN", "")
HF_TOKEN = os.getenv("HF_TOKEN") or None
INFERENCE_SIZE = (
    int(os.getenv("IMAGE_WIDTH", "768")),
    int(os.getenv("IMAGE_HEIGHT", "1152")),
)
INFERENCE_STEPS = int(os.getenv("FLUX_STEPS", "4"))
# 다중 참조 편집은 단일 텍스트 생성보다 스텝이 더 필요하다. klein은 4스텝 증류
# 모델이지만 참조가 5~7장으로 늘면 4스텝에서 레이어가 뭉개진다.
TRYON_STEPS = int(os.getenv("FLUX_TRYON_STEPS", "8"))
GUIDANCE_SCALE = float(os.getenv("FLUX_GUIDANCE", "1.0"))
# 파인튜닝한 try-on LoRA 경로(.safetensors). 비어 있으면 base 모델로만 돈다.
TRYON_LORA_PATH = os.getenv("FLUX_TRYON_LORA", "")
TRYON_LORA_SCALE = float(os.getenv("FLUX_TRYON_LORA_SCALE", "1.0"))
MAX_TRYON_GARMENTS = int(os.getenv("MAX_TRYON_GARMENTS", "6"))
# 0으로 두면 예전처럼 치수를 텍스트로만 넘긴다. 개선 전/후 A/B 비교용 스위치.
AVATAR_BODY_REFERENCE = os.getenv("AVATAR_BODY_REFERENCE", "1") == "1"
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 32_000_000
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
DEV_TOOLS_ENABLED = os.getenv("WEARWELL_DEV_TOOLS", "0") == "1"
CORS_ORIGIN_REGEX = os.getenv(
    "WEARWELL_CORS_ORIGIN_REGEX",
    r"^http://(?:127\.0\.0\.1|localhost)(?::[1-9]\d{0,4})?$",
)
MAX_COMPARE_MODELS = 4
# 옷장 한 벌씩 /api/embedding을 부르면 200벌에서 분당 요청 제한에 먼저 걸린다.
MAX_EMBEDDING_BATCH = int(os.getenv("MAX_EMBEDDING_BATCH", "32"))
# 옷 한 벌은 정사각에 가깝다. 아바타용 768x1152로 만들면 옷이 세로로 늘어난다.
REFINE_SIZE = (
    int(os.getenv("REFINE_WIDTH", "768")),
    int(os.getenv("REFINE_HEIGHT", "768")),
)
# 옷 한 벌마다 FLUX를 한 번씩 돌린다. 한 요청 안에서 무한정 돌리면 GPU 큐가 막힌다.
MAX_REFINE_ITEMS = int(os.getenv("MAX_REFINE_ITEMS", "4"))
GPU_QUEUE_TIMEOUT = float(os.getenv("GPU_QUEUE_TIMEOUT", "300"))
GPU_CONCURRENCY = max(1, int(os.getenv("GPU_CONCURRENCY", "2")))
INFERENCE_GATE = threading.BoundedSemaphore(GPU_CONCURRENCY)
VLM_LOCK = threading.RLock()
RATE_LOCK = threading.Lock()
REQUEST_TIMES: dict[str, deque[float]] = defaultdict(deque)
WARMUP_VERIFIED = False
VLM_WARMUP_VERIFIED = False
EMBEDDING_WARMUP_VERIFIED = False
SEGMENTATION_WARMUP_VERIFIED = False
SEGMENTATION_CACHE_MAX = max(1, int(os.getenv("SEGMENTATION_CACHE_MAX", "4")))
SEGMENTATION_CACHE: dict[str, dict] = {}
SEGMENTATION_CACHE_ORDER: deque[str] = deque()
SEGMENTATION_CACHE_LOCK = threading.Lock()


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
    allow_origin_regex=CORS_ORIGIN_REGEX,
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
            if len(times) >= RATE_LIMIT_PER_MINUTE:
                retry_after = max(1, int(60 - (now - times[0])))
                return JSONResponse(
                    {"detail": "Rate limit exceeded"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
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
    # 만들 시점 목록. front는 항상 먼저 만들어지고 측면·후면의 인물 기준이 된다.
    views: list[Literal["front", "side", "back"]] = Field(default_factory=lambda: ["front"], max_length=3)


class AvatarResponse(BaseModel):
    image: str
    engine: str
    gpu: str | None
    # 치수 목표 대비 실제 달성치와 cm 오차. 텍스트 전용 경로에서는 비어 있다.
    fit: dict = Field(default_factory=dict)
    # 시점 이름 -> 이미지. image 필드는 항상 views[0](= front)와 같다.
    views: dict[str, str] = Field(default_factory=dict)
    disclaimer: str = "입력한 치수를 시각적으로 근사한 이미지이며 실제 체형이나 의류 사이즈를 보증하지 않습니다."


class GarmentInput(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    category: Literal["upper", "lower", "overall", "outer", "shoes", "bag", "accessory"] = "upper"
    name: str = Field(default="옷", max_length=100)


class TryOnRequest(BaseModel):
    avatar: str = Field(min_length=1, max_length=8_000_000)
    garments: list[GarmentInput] = Field(min_length=1, max_length=MAX_TRYON_GARMENTS)
    seed: int = 42
    # 만들 시점. front는 항상 먼저 만들어지고 나머지 시점의 기준이 된다.
    views: list[Literal["front", "side", "back"]] = Field(default_factory=lambda: ["front"], max_length=3)
    # 아바타를 시점별로 만들어 뒀다면 여기에 담아 보낸다. 없으면 체형 가이드
    # 없이 정면 결과만 보고 돌린다.
    avatarViews: dict[Literal["front", "side", "back"], str] = Field(default_factory=dict)
    # 체형 가이드를 다시 그리기 위한 치수. 없으면 아바타 이미지만으로 회전한다.
    measurements: Measurements | None = None


class VLMImageRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="이미지", max_length=120)
    category: str | None = Field(default=None, max_length=30)
    gender: Literal["women", "men"] | None = None
    # Qwen Lab처럼 "무엇을 물어봤고 무엇이 그대로 돌아왔는가"를 봐야 하는 호출자용.
    # 앱은 파싱된 JSON만 쓰므로 기본은 꺼 둔다 — 프롬프트와 원문은 응답을 두 배로 만든다.
    debug: bool = False


class EmbeddingRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)


class EmbeddingBatchRequest(BaseModel):
    images: list[str] = Field(min_length=1, max_length=MAX_EMBEDDING_BATCH)


class PhotoAvatarRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    seed: int = 20260825


class TryOnJudgeRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    # 입히기로 한 아이템 목록을 사람이 읽는 문장으로. 심판은 이 목록과 사진만 본다.
    manifest: str = Field(min_length=1, max_length=2_000)
    debug: bool = False


class SegmentationRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    model: str | None = Field(default=None, max_length=60)


class SegmentCompareRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    models: list[str] = Field(default_factory=list, max_length=MAX_COMPARE_MODELS)


class RepairOptions(BaseModel):
    """마스크 보수 단계 스위치. 랩에서 하나씩 꺼 보면서 어떤 단계가 무슨 일을 하는지 본다."""

    close: bool = True
    fillHoles: bool = True
    # 팔·가방에 가려진 자리를 세그멘테이션 라벨을 근거로 메운다. 구멍과 달리 바깥과
    # 이어진 결손이라 형태학 연산으로는 되돌릴 수 없다.
    fillOccluded: bool = True
    dropStrays: bool = True
    smooth: bool = True
    closeScale: float = Field(default=0.012, ge=0.0, le=0.05)
    strayRatio: float = Field(default=0.08, ge=0.0, le=1.0)
    occlusionEnclosure: float = Field(default=0.6, ge=0.0, le=1.0)


class ClosetRefineRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    model: str | None = Field(default=None, max_length=60)
    # 비우면 품질 필터를 통과한 카테고리 전부.
    categories: list[str] = Field(default_factory=list, max_length=len(segment_models.CATEGORIES))
    # 검출 카테고리 -> 옷장에 저장하고 FLUX 프롬프트에 사용할 카테고리.
    # 일반 옷장 UI에서 사용자가 잘못 분류된 옷을 고칠 때 쓴다.
    categoryOverrides: dict[str, str] = Field(default_factory=dict)
    # 걸러진 후보까지 파이프라인에 태운다. 왜 걸러졌는지를 생성 결과로 확인할 때 쓴다.
    includeRejected: bool = False
    # 입고 있던 사람의 성별. 프롬프트에 넣지 않으면 모델이 상품컷 기본값인 여성복
    # 재단으로 그려서, 남자 바지가 하이웨이스트 큐롯으로 돌아온다.
    gender: Literal["men", "women"] | None = None
    repair: RepairOptions = Field(default_factory=RepairOptions)
    generate: bool = True
    seed: int = 42
    steps: int | None = Field(default=None, ge=1, le=50)


def flatten_transparency(image: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """투명 배경을 흰색으로 메운 뒤 RGB로 바꾼다.

    `.convert("RGB")`를 바로 부르면 알파 채널만 버리고 그 아래 RGB 값은 그대로
    남는다. segment_service가 만드는 옷 PNG는 알파로만 옷 영역을 표시하고 RGB
    채널에는 원본 사진이 그대로 들어 있어서, 그냥 변환하면 잘라냈다고 생각한
    사람과 배경이 통째로 되살아난다. 지시문에는 "옷만 남기고 원래 모델과
    배경은 지워라"라고 써 놓고 실제로는 그 배경을 다시 넣어 주고 있었던 셈이다.
    """
    if image.mode not in ("RGBA", "LA") and not (image.mode == "P" and "transparency" in image.info):
        return image.convert("RGB")
    rgba = image.convert("RGBA")
    canvas = Image.new("RGB", rgba.size, background)
    canvas.paste(rgba, mask=rgba.split()[3])
    return canvas


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
        return flatten_transparency(image)
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("Invalid image payload") from error


def encode_image(image: Image.Image, mime: str = "image/jpeg") -> str:
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=92, optimize=True)
    return f"data:{mime};base64,{base64.b64encode(output.getvalue()).decode()}"


def encode_png(data: bytes) -> str:
    """이미 PNG 바이트인 결과물(투명 크롭·오버레이)을 그대로 data URL로."""
    return "data:image/png;base64," + base64.b64encode(data).decode()


def cuda_info() -> tuple[bool, str | None]:
    try:
        import torch

        available = torch.cuda.is_available()
        return available, torch.cuda.get_device_name(0) if available else None
    except Exception:
        return False, None


class FluxImageEngine:
    # 세그멘테이션 결과의 한국어 카테고리를 생성 프롬프트용 설명으로 바꾼다.
    CLOSET_CATEGORY_LABELS = {
        "상의": "upper-body garment (top)",
        "아우터": "outerwear jacket or coat",
        "하의": "lower-body garment (pants or skirt)",
        "원피스": "one-piece dress",
        "신발": "pair of shoes",
        "가방": "bag",
        "액세서리": "fashion accessory",
    }
    # 세그멘테이션 라벨 -> 프롬프트에 쓸 옷 이름.
    #
    # 카테고리만 쓰면 하의가 "pants or skirt"가 된다. 둘 중 뭘 그릴지 모델이 고르게
    # 되고, 상품컷 학습 분포가 여성복 쪽으로 기울어 있어서 남자 바지가 치마·큐롯으로
    # 나온다(실측: 남성 하의 3벌 전부). 세그멘테이션은 이미 Pants인지 Skirt인지
    # 알고 있으므로 그 라벨을 그대로 쓴다.
    SEGMENT_LABEL_NOUNS = {
        "Pants": "pair of trousers",
        "Skirt": "skirt",
        "Dress": "dress",
        "Upper-clothes": "upper-body garment (top)",
        "Left-shoe": "pair of shoes",
        "Right-shoe": "pair of shoes",
        "Bag": "bag",
        "Scarf": "scarf",
        "Belt": "belt",
    }
    # 성별을 말해 주지 않으면 모델이 여성복 쪽 기본값으로 그린다. 다만 참조에 없는
    # 디테일을 만들어내지 않도록 "레퍼런스에 있는 경우"라는 단서를 반드시 붙인다.
    GENDER_CUT = {
        "men": " This is menswear: keep a men's cut with straight side seams and broad square shoulders, and do not "
               "give it a nipped waist, a high elasticated waistband or a cropped culotte hem unless the reference "
               "clearly shows one.",
        "women": " This is womenswear: keep the women's cut exactly as the reference shows it.",
    }
    def __init__(self, pipeline_factory=None) -> None:
        self.pipe = None
        self.pipeline_factory = pipeline_factory
        self.dtype: str | None = None
        self.lora_name: str | None = None
        self.pipes: list = []
        self._lora_workers: set[int] = set()
        self._available: queue.LifoQueue = queue.LifoQueue()
        self._load_lock = threading.Lock()

    def _load(self):
        if self.pipe is not None:
            return self.pipe
        with self._load_lock:
            if self.pipe is None:
                self._build_pipeline()
        return self.pipe

    def _build_pipeline(self) -> None:
        """VRAM 여유에 맞춘 독립 FLUX worker들을 한 번만 구성한다."""
        import torch

        if self.pipeline_factory is None:
            from diffusers import Flux2KleinPipeline

            self.pipeline_factory = Flux2KleinPipeline

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.dtype = str(dtype).removeprefix("torch.")
        for worker_id in range(GPU_CONCURRENCY):
            load_options = {"torch_dtype": dtype}
            if HF_TOKEN:
                load_options["token"] = HF_TOKEN
            pipe = self.pipeline_factory.from_pretrained(IMAGE_MODEL, **load_options)
            if os.getenv("FLUX_CPU_OFFLOAD", "0") == "1":
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)
            if TRYON_LORA_PATH:
                try:
                    pipe.load_lora_weights(TRYON_LORA_PATH, adapter_name="wearwell_tryon")
                    if hasattr(pipe, "disable_lora"):
                        pipe.disable_lora()
                    self.lora_name = os.path.basename(TRYON_LORA_PATH)
                    self._lora_workers.add(worker_id)
                except Exception:
                    logging.exception("Try-on LoRA load failed on worker %s; using base model", worker_id)
            self.pipes.append(pipe)
            self._available.put(worker_id)
        self.pipe = self.pipes[0]

    def unload(self) -> None:
        if not self.pipes:
            return
        self.pipe = None
        self.pipes.clear()
        self._lora_workers.clear()
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except queue.Empty:
                break
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

    def _run(
        self,
        *,
        prompt: str,
        seed: int,
        images: list[Image.Image] | None = None,
        size: tuple[int, int] | None = None,
        steps: int | None = None,
        use_tryon_lora: bool = False,
    ) -> Image.Image:
        """독립 FLUX worker에서 추론한다. size는 (너비, 높이)다."""
        import torch

        width, height = size or INFERENCE_SIZE
        self._load()
        inputs = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": steps or INFERENCE_STEPS,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
        }
        if images:
            inputs["image"] = images
        try:
            worker_id = self._available.get(timeout=GPU_QUEUE_TIMEOUT)
        except queue.Empty as error:
            raise RuntimeError("FLUX worker pool timeout") from error
        pipe = self.pipes[worker_id]
        try:
            with torch.inference_mode():
                if worker_id in self._lora_workers:
                    if use_tryon_lora:
                        if hasattr(pipe, "set_adapters"):
                            pipe.set_adapters("wearwell_tryon", adapter_weights=TRYON_LORA_SCALE)
                        elif hasattr(pipe, "enable_lora"):
                            pipe.enable_lora()
                    elif hasattr(pipe, "disable_lora"):
                        pipe.disable_lora()
                output = pipe(**inputs).images[0]
        finally:
            self._available.put(worker_id)
        return output.resize((width, height), Image.Resampling.LANCZOS)

    def generate_avatar(self, data: Measurements) -> tuple[dict[str, Image.Image], str, dict]:
        """치수 -> 시점별 아바타 이미지, 엔진 이름, 치수 정확도 리포트.

        치수를 프롬프트 문장으로만 넘기면 확산 모델이 숫자를 길이로 해석하지
        못한다. 먼저 치수를 체형 실루엣 이미지로 바꿔 참조 이미지 1로 넣고,
        프롬프트는 "이 실루엣을 그대로 따라라"만 시킨다.

        측면·후면은 정면을 만든 뒤 그 결과를 참조 이미지 2로 함께 넘긴다.
        체형 가이드만 보고 각 시점을 독립 생성하면 매번 다른 사람이 나온다.
        """
        has_cuda, _ = cuda_info()
        if not (has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1"):
            return {"front": self.fallback_avatar(data)}, "measurement-preview-fallback", {}

        if AVATAR_BODY_REFERENCE:
            target = BodyTarget(
                gender=data.gender,
                height=data.height,
                weight=data.weight,
                chest=data.chest,
                waist=data.waist,
                hip=data.hip,
                shoulder=data.shoulder,
                inseam=data.inseam,
            )
            # front가 나머지 시점의 인물 기준이므로 반드시 먼저, 반드시 포함한다.
            requested = list(dict.fromkeys(["front", *data.views]))
            images: dict[str, Image.Image] = {}
            report: dict = {}
            for view in requested:
                reference = build_body_reference(target, view=view)
                references = [reference.image]
                if view != "front":
                    references.append(images["front"])
                images[view] = self._run(
                    prompt=build_avatar_prompt(
                        target, view=view, identity_reference=view != "front"
                    ),
                    # 시점마다 시드를 흘리면 같은 사람이 나올 확률이 떨어진다.
                    seed=data.seed,
                    images=references,
                )
                if view == "front":
                    report = {
                        "bodyReference": reference.source,
                        "targetMeasurements": reference.target,
                        "achievedMeasurements": reference.achieved,
                        "measurementErrorCm": reference.errors(),
                        "meanAbsoluteErrorCm": reference.mean_absolute_error(),
                        "betas": reference.betas,
                    }
            report["views"] = requested
            return images, f"{self.engine_name}+{report['bodyReference']}", report

        # 텍스트 전용 경로 (비교 기준선)
        prompt = (
            "Create a photorealistic full-body studio fitting avatar. The subject is "
            f"{self._shape_description(data)}. Accurately reflect the stated height, weight, body build, "
            "and body proportions without slimming or exaggeration. Front-facing relaxed symmetrical A-pose, "
            "arms slightly away from the torso. Compose a wide full-length catalog shot showing the person from "
            "the top of the hair through both ankles to the soles of both feet. Dress both feet in plain charcoal "
            "low-top studio shoes. Both complete shoes and the floor beneath them must be visible. Leave generous "
            "empty margin above the head, below the shoes, and on both sides. The person must occupy no more than "
            "75 percent of the image height. Use an eye-level camera far enough away with an 85 mm catalog lens. "
            "Do not crop the head, hands, legs, ankles, feet, or shoes. "
            "Wear a plain fitted charcoal crew-neck top and fitted mid-thigh charcoal shorts so the body outline "
            "is clearly visible. Clean warm-gray seamless background, soft even studio lighting, realistic skin, "
            "natural Korean facial features, bare arms and legs, single frame, clean image with no lettering."
        )
        return (
            {"front": self._run(prompt=prompt, seed=data.seed)},
            f"{self.engine_name}+text-only",
            {"bodyReference": "text-only", "views": ["front"]},
        )

    def generate_tryon(self, request: TryOnRequest) -> tuple[dict[str, Image.Image], str, list[GarmentInput]]:
        """착장 요청 -> 시점별 결과 이미지.

        측면·후면은 옷 사진을 다시 넣지 않고 **완성된 정면 결과**를 참조로 돌린다.
        착장이 이미 조립돼 있으니 참조 수도 줄고 옷 일관성도 낫다.
        """
        # 참조 이미지 번호가 레이어 순서와 어긋나면 모델이 번호를 레이어 힌트로
        # 오해한다. 디코딩 전에 안쪽 레이어부터 정렬하고 슬롯 충돌을 정리한다.
        ordered = order_garments(list(request.garments), limit=MAX_TRYON_GARMENTS)
        garments = [decode_image(item.image) for item in ordered]
        # 정면은 필수라 실패하면 그대로 올린다. 측면·후면은 부가 정보이므로
        # 하나가 깨져도 정면 착장까지 같이 죽이지 않는다.
        # 정면의 출처는 언제나 request.avatar다. avatarViews에 front가 섞여
        # 들어와도 무시해서 "어느 쪽이 이기는가"를 고민할 일을 없앤다.
        person_views = {"front": decode_image(request.avatar)}
        for view, payload in request.avatarViews.items():
            if view == "front":
                continue
            try:
                person_views[view] = decode_image(payload)
            except ValueError:
                logging.warning("Skipping unreadable avatar view: %s", view)

        has_cuda, _ = cuda_info()
        if not (has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1"):
            return {"front": self.fallback_tryon(person_views["front"], ordered)}, "tryon-preview-fallback", ordered

        requested = list(dict.fromkeys(["front", *request.views]))
        person = pad_for_full_body(person_views["front"])
        size = fit_generation_size(person)
        results: dict[str, Image.Image] = {}
        for view in requested:
            if view == "front":
                results["front"] = self._run(
                    prompt=build_tryon_prompt(ordered),
                    seed=request.seed,
                    images=[person, *garments],
                    steps=TRYON_STEPS,
                    size=size,
                    use_tryon_lora=True,
                )
                continue
            results[view] = self._run(
                prompt=build_tryon_view_prompt(view, ordered),
                seed=request.seed,
                images=[results["front"], *garments],
                steps=TRYON_STEPS,
                size=size,
                use_tryon_lora=True,
            )
        engine = self.engine_name
        if self.lora_name:
            engine += f"+lora:{self.lora_name}"
        return results, engine, ordered

    def avatarize_photo(self, request: PhotoAvatarRequest) -> tuple[Image.Image, str]:
        photo = pad_for_full_body(decode_image(request.image))
        has_cuda, _ = cuda_info()
        if not (has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1"):
            return photo, "photo-passthrough"
        image = self._run(
            prompt=build_photo_avatar_prompt(), seed=request.seed,
            images=[photo], size=fit_generation_size(photo),
        )
        return image, f"{self.engine_name}+photo-avatar"

    @classmethod
    def garment_noun(cls, category: str, label: str | None, gender: str | None) -> str:
        """프롬프트에 쓸 옷 이름. 구체적인 라벨이 하나일 때만 그 이름을 쓴다.

        `Skirt+Pants`처럼 서로 다른 옷 라벨이 합쳐져 오면 무엇인지 확정할 수
        없으므로 카테고리 이름으로 돌아간다.
        """
        nouns = {
            cls.SEGMENT_LABEL_NOUNS[part]
            for part in str(label or "").split("+")
            if part in cls.SEGMENT_LABEL_NOUNS
        }
        noun = nouns.pop() if len(nouns) == 1 else cls.CLOSET_CATEGORY_LABELS.get(category, "garment")
        return f"{'men' if gender == 'men' else 'women'}'s {noun}" if gender in ("men", "women") else noun

    def refine_garment(
        self, garment: Image.Image, category: str, name: str, seed: int,
        steps: int | None = None, hint: str = "", gender: str | None = None,
        label: str | None = None,
    ):
        """세그멘테이션으로 오려낸 옷 -> 옷장에 넣을 상품컷.

        입력은 이미 흰 배경에 얹은 정규화 이미지지만, 가려졌던 자리는 실루엣만
        메워져 대표색으로 평평하게 칠해져 있다. 그 안의 질감·주름을 그리는 게 여기
        일이다. 그래서 프롬프트가 "구멍을 메우고 끊어진 부분을 잇되 색·패턴·재단은
        그대로"라는 말을 반드시 담아야 한다 — 자유롭게 그리라고 하면 다른 옷이 온다.

        `hint`는 2단계 진단이 만든 한 문장이다(`refine_service.generation_hint`).
        "어디가 무엇에 가려져 비었는지"를 알려주지 않으면 모델은 평평한 색면을
        무지 패널로, 파먹힌 실루엣을 원래 디자인으로 읽는다.
        """
        has_cuda, _ = cuda_info()
        if has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1":
            prompt = (
                f"Reference image 1 is a {self.garment_noun(category, label, gender)} named '{name}' that was "
                "automatically cut out of a full-body photo, so its shape is damaged: the cutout has holes where arms, "
                "hair, straps or other garments covered it, some pieces are detached, and the edges are ragged. "
                "Redraw it as one clean e-commerce product photo of that same single item. Fill every hole, reconnect "
                "the detached pieces, and rebuild the parts that were occluded so the silhouette is complete and "
                "naturally symmetric. Keep exactly the same color, print, pattern placement, fabric texture, sheen, "
                "neckline, sleeve length, hem, closure, and overall cut as the reference — do not restyle it, do not "
                "change its proportions, and do not add garments or details that are not visible in the reference. "
                "Present the item alone, centered and upright, gently filled out as if worn by an invisible body, on a "
                "plain pure white background with soft even studio lighting and a subtle contact shadow. "
                "No person, no skin, no face, no hands, no visible mannequin, no hanger, no props, no background scene, "
                "no text, no watermark, no collage, no split screen, no duplicated item."
                # 성별을 모르면 최소한 "레퍼런스와 다른 성별의 재단으로 바꾸지 말라"고는 말해 둔다.
                f"{self.GENDER_CUT.get(gender, ' Do not restyle the garment into a different cut than the reference shows.')}"
                f"{hint}"
            )
            return self._run(prompt=prompt, seed=seed, images=[garment], size=REFINE_SIZE, steps=steps), self.engine_name
        # GPU가 없어도 랩이 돌아야 한다. 정규화 이미지를 그대로 최종본으로 쓰면
        # 구멍은 남지만 흰 배경·정사각 규격이라는 나머지 단계는 눈으로 확인된다.
        return garment.resize(REFINE_SIZE, Image.Resampling.LANCZOS), "refine-passthrough-fallback"

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
        person = person.convert("RGB")
        draw = ImageDraw.Draw(person, "RGBA")
        colors = [(255, 113, 91, 110), (55, 83, 120, 115), (238, 238, 232, 125), (45, 45, 45, 100)]
        width, height = person.size
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


class QwenVLMEngine:
    GARMENT_PROMPT = """이 사진에서 주된 옷 한 벌만 분석해 한국어 JSON으로만 답해. 보이지 않는 정보는 추측하지 말고 '확인 어려움'으로 써. 스키마: {"category":"상의/하의/아우터/원피스/신발/가방/액세서리","subcategory":"구체 종류","primaryColor":"주색","secondaryColors":["보조색"],"material":"소재 추정","texture":"표면 질감","fit":"슬림/레귤러/세미오버/오버/스트레이트/세미와이드/와이드/커브드","sleeveLength":"민소매/반팔/긴팔/확인 어려움","length":"상의:크롭/기본/롱, 하의:반바지/긴바지/확인 어려움","neckline":"라운드넥/카라·폴로/브이넥/터틀넥/확인 어려움","silhouette":"실루엣","wrinkle":"주름의 정도와 형태","finish":"광택·워싱·표면 마감","construction":["봉제선·단추·지퍼·포켓·밑단 등 보이는 디테일"],"pattern":"패턴","season":["계절"],"weather":["어울리는 날씨"],"summary":"한 문장 요약"}"""
    LOOKBOOK_PROMPT = """이 패션 룩북에서 가장 크게 보이는 한 사람의 착장을 분석해. 착장 전체를 한 항목으로 요약하지 말고 눈에 보이는 옷을 실제 경계대로 각각 분리해 한국어 JSON으로만 답해. 재킷 안에 셔츠가 있으면 아우터와 상의를 별도 pieces 항목으로 쓰고 같은 카테고리의 레이어도 합치지 마. 보이지 않는 옷이나 색은 추측하지 말고 색상은 각 옷 자체의 주조색부터 최대 2개만 써. 신발·가방도 충분히 보일 때만 별도 항목으로 써. bbox는 전체 이미지 왼쪽 위를 0,0, 오른쪽 아래를 1000,1000으로 본 옷의 경계야. 스키마: {"summary":"색·실루엣·레이어링 한 문장","mood":"스타일 무드","pieces":[{"pieceId":"고유 번호","label":"화이트 셔츠처럼 옷을 구별하는 이름","layer":"아우터/이너/단독/하의/신발/가방","category":"상의/하의/아우터/원피스/신발/가방/액세서리","bbox":[0,0,1000,1000],"colors":["정확한 주조색"],"materials":["소재 추정"],"fits":["핏"],"sleeveLength":"민소매/반팔/긴팔/확인 어려움","length":"상의:크롭/기본/롱, 하의:반바지/긴바지/확인 어려움","subcategory":"폴로 셔츠/티셔츠/쇼츠/데님 팬츠처럼 구체 종류","pattern":"무지/스트라이프/체크/그래픽/확인 어려움","neckline":"라운드넥/카라·폴로/브이넥/확인 어려움","details":["주름·마감·봉제·형태 디테일"],"confidence":0.0}]}"""
    # 착장 결과 자동 채점용. 정답 이미지가 없는 태스크라 픽셀 지표(SSIM/LPIPS)로는
    # "상의가 아우터 안쪽인가"를 잴 수 없어서 VLM 이진 판정을 쓴다.
    # 근거를 먼저 쓰게 하면 판정이 눈에 띄게 안정된다 — 결론부터 내라고 하면
    # 첫 토큰에 걸려 뒤 근거를 결론에 끼워 맞춘다.
    TRYON_JUDGE_PROMPT = """너는 가상 착장 결과를 채점하는 심판이야. 사진은 한 사람에게 옷을 합성한 결과야.
입히기로 한 아이템 목록:
{manifest}

각 항목을 보이는 대로만 판정하고 한국어 JSON으로만 답해. 확신이 서지 않으면 false로 둬. 스키마: {{"reasons":"보이는 근거 두 문장","layering_ok":true,"items_present":["실제로 보이는 아이템 이름"],"extra_items":["목록에 없는데 추가로 생긴 옷·가방·액세서리"],"merged_items":["두 벌이 한 벌로 합쳐져 보이는 아이템 이름"],"accessories_placed_ok":true,"identity_ok":true,"artifacts":["손가락 뭉개짐·팔 개수 이상 등 눈에 띄는 결함"]}}

판정 기준:
- layering_ok: 상의와 아우터가 함께 있을 때 상의가 안쪽, 아우터가 바깥이면 true. 아우터가 상의 밑으로 들어갔거나 둘이 한 벌로 합쳐졌으면 false. 몸통 레이어가 한 겹뿐이면 true.
- accessories_placed_ok: 가방과 액세서리가 각각 맞는 부위에 있으면 true. 모자가 몸에 붙어 있거나 가방끈이 옷 속을 통과하면 false. 목록에 가방·액세서리가 없으면 true.
- identity_ok: 얼굴, 체형, 포즈가 원래 사람과 같으면 true.
- 속옷이 목록에 있으면 겉옷 아래로 완전히 가려져 보이지 않아야 layering_ok가 true다. 바지 위에 팬티가 보이면 false.
- 양말과 신발이 함께 있으면 양말이 신발 안에 들어가 있어야 true. 양말이 신발 위로 나와 있으면 false."""

    BODY_PROMPT = """전신사진에서 의상과 포즈의 영향을 감안해 보이는 체형 특징만 한국어 JSON으로 답해. 키나 몸무게 숫자를 추측하지 마. 스키마: {"body_shape":"보통/마른 체형/탄탄한 체형/상체가 발달한 체형/하체가 발달한 체형/통통한 체형 중 하나","proportion":"상하체 비율 설명","shoulderLine":"어깨선 설명","silhouette":"전체 실루엣 설명"}"""

    def __init__(self, model_factory=None, processor_factory=None) -> None:
        self.model = None
        self.processor = None
        self.model_factory = model_factory
        self.processor_factory = processor_factory
        self.dtype: str | None = None

    def _load(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        model_factory = self.model_factory or AutoModelForImageTextToText
        processor_factory = self.processor_factory or AutoProcessor
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quantization_config = None
        if VLM_LOAD_IN_4BIT:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        load_options = {
            "dtype": dtype,
            "device_map": "auto",
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }
        if HF_TOKEN:
            load_options["token"] = HF_TOKEN
        if quantization_config is not None:
            load_options["quantization_config"] = quantization_config
        self.model = model_factory.from_pretrained(VLM_MODEL, **load_options)
        processor_options = {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": VLM_MAX_PIXELS,
        }
        if HF_TOKEN:
            processor_options["token"] = HF_TOKEN
        self.processor = processor_factory.from_pretrained(VLM_MODEL, **processor_options)
        self.dtype = str(dtype).removeprefix("torch.")
        return self.model, self.processor

    def unload(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _parse_json(value: str) -> dict:
        cleaned = re.sub(r"```(?:json)?|```", "", value, flags=re.IGNORECASE).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("VLM response did not contain JSON")
        result = json.loads(cleaned[start : end + 1])
        if not isinstance(result, dict):
            raise ValueError("VLM response must be a JSON object")
        return result

    def analyze(self, image: Image.Image, prompt: str, max_new_tokens: int, debug: bool = False) -> dict:
        import torch

        started = time.monotonic()
        model, processor = self._load()
        loaded = time.monotonic()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(model.device)
        with VLM_LOCK, torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        result = self._parse_json(text)
        result.update({
            "engine": "Qwen3-VL-8B-Instruct",
            "model": VLM_MODEL,
            "quantization": "nf4" if VLM_LOAD_IN_4BIT else self.dtype,
            "seconds": round(time.monotonic() - loaded, 1),
            "loadSeconds": round(loaded - started, 1),
        })
        if debug:
            # 모델이 실제로 뱉은 문자열. 파싱된 JSON만 보면 왜 필드가 비었는지
            # ("확인 어려움"인지, 코드펜스에 잘렸는지, 토큰 예산에서 끊겼는지) 알 수 없다.
            result["rawText"] = text
            result["outputTokens"] = int(len(trimmed[0]))
            result["truncated"] = len(trimmed[0]) >= max_new_tokens
            result["imageSize"] = {"width": image.width, "height": image.height}
        return result

    @property
    def engine_name(self) -> str:
        mode = "nf4" if VLM_LOAD_IN_4BIT else self.dtype or "auto"
        return f"qwen3-vl-8b-{mode}-cuda"


class SigLIPEmbeddingEngine:
    """Lazy, thread-safe SigLIP image encoder used by wardrobe similarity."""

    def __init__(self, model_factory=None, processor_factory=None) -> None:
        self.model = None
        self.processor = None
        self.model_factory = model_factory
        self.processor_factory = processor_factory
        self.device: str | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor
        with self._load_lock:
            if self.model is not None and self.processor is not None:
                return self.model, self.processor
            import torch
            from transformers import AutoModel, AutoProcessor

            model_factory = self.model_factory or AutoModel
            processor_factory = self.processor_factory or AutoProcessor
            options = {"low_cpu_mem_usage": True}
            if HF_TOKEN:
                options["token"] = HF_TOKEN
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.processor = processor_factory.from_pretrained(
                SIGLIP_MODEL, **({"token": HF_TOKEN} if HF_TOKEN else {})
            )
            self.model = model_factory.from_pretrained(SIGLIP_MODEL, **options).to(self.device).eval()
        return self.model, self.processor

    def embed(self, image: Image.Image) -> list[float]:
        return self.embed_many([image])[0]

    def embed_many(self, images: list[Image.Image]) -> list[list[float]]:
        """여러 장을 한 번의 forward로. 옷장 전체를 벡터로 만들 때 이 경로를 쓴다."""
        import torch

        model, processor = self._load()
        inputs = processor(
            images=[image.convert("RGB") for image in images], return_tensors="pt"
        ).to(self.device)
        with self._inference_lock, torch.inference_mode():
            if hasattr(model, "get_image_features"):
                features = model.get_image_features(**inputs)
            else:
                features = model(**inputs)
            features = self._feature_tensor(features)
            # 정규화해서 돌려주므로 클라이언트의 코사인 유사도는 내적과 같다.
            features = torch.nn.functional.normalize(features.float(), dim=-1)
        return features.detach().cpu().tolist()

    @staticmethod
    def _feature_tensor(output):
        """transformers 4.x의 Tensor와 5.x의 ModelOutput을 모두 받는다."""
        if hasattr(output, "float") and hasattr(output, "shape"):
            return output
        for name in ("image_embeds", "pooler_output"):
            value = getattr(output, name, None)
            if value is not None:
                return value
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            return hidden.mean(dim=1)
        if isinstance(output, (tuple, list)):
            # BaseModelOutputWithPooling의 tuple 순서는 hidden state, pooled output이다.
            if len(output) > 1 and output[1] is not None:
                return output[1]
            if output:
                value = output[0]
                return value.mean(dim=1) if getattr(value, "ndim", 0) == 3 else value
        raise TypeError(f"Unsupported SigLIP output: {type(output).__name__}")


image_engine = FluxImageEngine()
vlm_engine = QwenVLMEngine()
embedding_engine = SigLIPEmbeddingEngine()


def acquire_inference_slot() -> None:
    if not INFERENCE_GATE.acquire(timeout=GPU_QUEUE_TIMEOUT):
        raise HTTPException(
            status_code=503,
            detail="GPU queue timeout",
            headers={"Retry-After": "10"},
        )


@app.get("/api/health")
def health():
    import segment_service

    available, name = cuda_info()
    segmentation = segment_service.status()
    return {
        "ok": True,
        "cuda": available,
        "gpu": name,
        "model": IMAGE_MODEL,
        "avatarModel": IMAGE_MODEL,
        "segmentationModel": segmentation["model"],
        "segmentationModelKey": segmentation["modelKey"],
        "segmentationLoaded": segmentation["loaded"],
        "segmentationLoadedModels": segmentation["loadedModels"],
        "segmentationDevice": segmentation["device"],
        "segmentationWarmupVerified": SEGMENTATION_WARMUP_VERIFIED,
        "devTools": DEV_TOOLS_ENABLED,
        "tryonModel": IMAGE_MODEL,
        "vlmModel": VLM_MODEL,
        "embeddingModel": SIGLIP_MODEL,
        "modelLoaded": image_engine.pipe is not None,
        "vlmLoaded": vlm_engine.model is not None,
        "embeddingLoaded": embedding_engine.model is not None,
        "warmupVerified": WARMUP_VERIFIED,
        "vlmWarmupVerified": VLM_WARMUP_VERIFIED,
        "embeddingWarmupVerified": EMBEDDING_WARMUP_VERIFIED,
        "embeddingDevice": embedding_engine.device,
        "dtype": image_engine.dtype,
        "vlmDtype": vlm_engine.dtype,
        "vlmQuantization": "nf4" if VLM_LOAD_IN_4BIT else "none",
        "queueTimeoutSeconds": GPU_QUEUE_TIMEOUT,
        "gpuConcurrency": GPU_CONCURRENCY,
        "imageWorkersLoaded": len(image_engine.pipes),
        "rateLimitPerMinute": RATE_LIMIT_PER_MINUTE,
        "maxEmbeddingBatch": MAX_EMBEDDING_BATCH,
        "resolution": f"{INFERENCE_SIZE[0]}x{INFERENCE_SIZE[1]}",
        "tryonSteps": TRYON_STEPS,
        "maxTryonGarments": MAX_TRYON_GARMENTS,
        "tryonLora": os.path.basename(TRYON_LORA_PATH) if TRYON_LORA_PATH else None,
        "avatarBodyReference": AVATAR_BODY_REFERENCE,
        "avatarViews": sorted(VIEW_YAW),
    }


@app.post("/api/avatar", response_model=AvatarResponse)
def generate_avatar(data: Measurements):
    acquire_inference_slot()
    try:
        images, engine, report = image_engine.generate_avatar(data)
        _, gpu = cuda_info()
        encoded = {view: encode_image(image) for view, image in images.items()}
        return AvatarResponse(
            image=encoded["front"], engine=engine, gpu=gpu, fit=report, views=encoded
        )
    except Exception as error:
        logging.exception("Avatar generation failed")
        raise HTTPException(status_code=500, detail="Avatar generation failed") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/avatar/from-photo")
def avatarize_photo(request: PhotoAvatarRequest):
    acquire_inference_slot()
    try:
        image, engine = image_engine.avatarize_photo(request)
        _, gpu = cuda_info()
        return {"image": encode_image(image), "engine": engine, "gpu": gpu}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("Photo avatar generation failed")
        raise HTTPException(status_code=500, detail="Photo avatar generation failed") from error
    finally:
        INFERENCE_GATE.release()


def segmentation_analysis(image_bytes: bytes, model_key: str | None = None) -> tuple[dict, bool]:
    """같은 업로드의 preview/refine 사이에서 비싼 segmentation 추론을 재사용한다."""
    import segment_service

    resolved = segment_models.resolve(model_key).key
    cache_key = hashlib.sha256(resolved.encode() + b"\0" + image_bytes).hexdigest()
    with SEGMENTATION_CACHE_LOCK:
        cached = SEGMENTATION_CACHE.get(cache_key)
        if cached is not None:
            try:
                SEGMENTATION_CACHE_ORDER.remove(cache_key)
            except ValueError:
                pass
            SEGMENTATION_CACHE_ORDER.append(cache_key)
            return cached, True

    analysis = segment_service.analyze(image_bytes, resolved)
    with SEGMENTATION_CACHE_LOCK:
        SEGMENTATION_CACHE[cache_key] = analysis
        SEGMENTATION_CACHE_ORDER.append(cache_key)
        while len(SEGMENTATION_CACHE_ORDER) > SEGMENTATION_CACHE_MAX:
            expired = SEGMENTATION_CACHE_ORDER.popleft()
            SEGMENTATION_CACHE.pop(expired, None)
    return analysis, False


@app.post("/api/closet/segment")
def segment_closet_photo(request: SegmentationRequest):
    acquire_inference_slot()
    try:
        image = decode_image(request.image)
        source = io.BytesIO()
        image.save(source, format="PNG")
        analysis, cache_hit = segmentation_analysis(source.getvalue(), request.model)
        detections = [item for item in analysis["items"] if item["accepted"]]
        return {
            "model": analysis["model"],
            "segmentSeconds": 0 if cache_hit else analysis["inferenceSeconds"],
            "segmentationCacheHit": cache_hit,
            "items": [
                {
                    "id": f"segment-{int(time.time() * 1000)}-{index}",
                    "name": f"{request.name} - {item['category']}",
                    "category": item["category"],
                    "color": item.get("color", "색상 미분류"),
                    "image": encode_png(item["png_bytes"]),
                    "label": item["label"],
                    "confidence": item["confidence"],
                }
                for index, item in enumerate(detections)
            ]
        }
    except KeyError as error:
        raise HTTPException(status_code=422, detail=str(error.args[0])) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("Garment segmentation failed")
        raise HTTPException(status_code=503, detail="Segmentation model is unavailable") from error
    finally:
        INFERENCE_GATE.release()


def segmentation_registry() -> dict:
    """모델 목록·카테고리 색·품질 기준. 모델을 적재하지 않으므로 torch 없이도 응답한다."""
    loaded: list[str] = []
    try:
        import segment_service
        loaded = segment_service.loaded_keys()
    except Exception:
        logging.exception("Unable to read segmentation model registry status")

    return {
        "models": [spec.as_dict() for spec in segment_models.MODELS.values()],
        "default": segment_models.DEFAULT_COMPARE,
        "production": segment_models.PRODUCTION_MODEL,
        "loaded": loaded,
        "categoryColors": {name: f"#{r:02x}{g:02x}{b:02x}" for name, (r, g, b) in segment_models.CATEGORY_COLORS.items()},
        # 기본 기준. 모델별 최종 기준은 각 model의 thresholds에 들어 있다.
        "thresholds": segment_models.QUALITY_THRESHOLDS,
    }


@app.get("/api/closet/models")
def list_closet_models():
    """세그멘테이션 메타데이터. 옷장·Refine Lab이 카테고리 색과 기준을 읽어 간다.

    적재도 추론도 하지 않는 상수 응답이라 dev 도구와 함께 닫을 이유가 없다.
    """
    return segmentation_registry()


@app.post("/api/closet/refine")
def refine_closet_photo(request: ClosetRefineRequest):
    """전신샷 -> 세그멘테이션 -> 마스크 보수 -> FLUX 재생성을 한 요청에서 전부 돌린다.

    /api/closet/segment과 같은 급의 제품 경로다 — 모델을 여러 개 올려 비교하는
    /api/dev/*와 달리 프로덕션 모델 하나만 쓰므로, dev 도구를 끈 공개 터널에서도
    열어 둔다. Refine Lab은 이 경로 하나로 단계별 중간 산출물을 받아 늘어놓는다.

    옷 한 벌씩 독립 실패를 허용한다 — 한 벌의 생성이 실패했다고 나머지 단계 결과까지
    버리면 무엇이 문제였는지 볼 수가 없다.
    """
    unknown = [category for category in request.categories if category not in segment_models.CATEGORIES]
    unknown += [
        category
        for pair in request.categoryOverrides.items()
        for category in pair
        if category not in segment_models.CATEGORIES
    ]
    if unknown:
        raise HTTPException(status_code=422, detail=f"알 수 없는 카테고리: {', '.join(dict.fromkeys(unknown))}")

    acquire_inference_slot()
    started = time.perf_counter()
    try:
        import numpy as np

        image = decode_image(request.image)
        source = io.BytesIO()
        image.save(source, format="PNG")
        import refine_service
        analysis, segmentation_cache_hit = segmentation_analysis(
            source.getvalue(), request.model or segment_models.PRODUCTION_MODEL
        )
        masks = analysis["_masks"]
        # 라벨맵이 있어야 "여기는 배경이 아니라 팔에 가려진 옷"을 가릴 수 있다.
        # 예전 백엔드·테스트 스텁은 주지 않으므로 없으면 가림 진단만 빠진다.
        parse = analysis.get("_parse")
        image_np = np.array(image)

        wanted = [
            item for item in analysis["items"]
            if (item["accepted"] or request.includeRejected)
            and (not request.categories or item["category"] in request.categories)
        ]
        skipped = len(wanted) - MAX_REFINE_ITEMS
        options = request.repair.model_dump()

        items = []
        generation_jobs = []
        for item in wanted[:MAX_REFINE_ITEMS]:
            source_category = item["category"]
            target_category = request.categoryOverrides.get(source_category, source_category)
            item["sourceCategory"] = source_category
            item["category"] = target_category
            item.pop("png_bytes", None)  # 스테이지 크롭을 따로 만든다 — 같은 그림을 두 번 싣지 않는다
            repair_started = time.perf_counter()
            stages = refine_service.build_stages(image_np, masks[source_category], options, parse)
            item["diagnosis"] = stages["diagnosis"]
            item["repair"] = stages["repair"]
            item["repairSeconds"] = round(time.perf_counter() - repair_started, 2)
            item["stages"] = {
                "crop": encode_png(stages["cropPng"]),
                "defects": encode_png(stages["defectPng"]),
                "repaired": encode_png(stages["repairedPng"]),
                "normalized": encode_png(stages["normalizedPng"]),
                "closet": None,
            }
            item["generation"] = None
            item["generationError"] = None
            if request.generate:
                generation_jobs.append((item, stages["image"], target_category))
            items.append(item)

        def generate_one(job):
            item, normalized, category = job
            generate_started = time.perf_counter()
            closet, engine = image_engine.refine_garment(
                normalized, category, f"{request.name} {category}", request.seed, request.steps,
                # 진단과 원본 라벨, 성별을 함께 넘겨야 가려진 바지가 다른 재단으로 바뀌지 않는다.
                refine_service.generation_hint(item["diagnosis"]),
                request.gender, item.get("label"),
            )
            return item, closet, engine, round(time.perf_counter() - generate_started, 2)

        # G4에서는 독립 FLUX pipeline 두 개가 이미 준비돼 있다. 한 요청 안에서도
        # 두 벌씩 보내야 worker pool이 실제로 병렬 사용된다.
        with ThreadPoolExecutor(max_workers=min(GPU_CONCURRENCY, 2, len(generation_jobs) or 1)) as executor:
            futures = {executor.submit(generate_one, job): job[0] for job in generation_jobs}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    item, closet, engine, seconds = future.result()
                    item["stages"]["closet"] = encode_image(closet)
                    item["generation"] = {
                        "engine": engine, "seed": request.seed,
                        "steps": request.steps or INFERENCE_STEPS, "seconds": seconds,
                    }
                except Exception as error:
                    logging.exception("Closet refinement failed for %s", item["category"])
                    item["generationError"] = str(error) or "생성에 실패했습니다"

        return {
            "name": request.name,
            "model": analysis["model"],
            "device": analysis["device"],
            "imageSize": analysis["imageSize"],
            "segmentSeconds": 0 if segmentation_cache_hit else analysis["inferenceSeconds"],
            "segmentationCacheHit": segmentation_cache_hit,
            "loadSeconds": analysis["loadSeconds"],
            "overlay": encode_png(analysis["overlay_png_bytes"]),
            "detectedCount": len(analysis["items"]),
            "skippedCount": max(0, skipped),
            "items": items,
            "totalSeconds": round(time.perf_counter() - started, 2),
        }
    except KeyError as error:
        raise HTTPException(status_code=422, detail=str(error.args[0])) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logging.exception("Closet refinement pipeline failed")
        raise HTTPException(status_code=503, detail="Refinement pipeline is unavailable") from error
    finally:
        INFERENCE_GATE.release()


def require_dev_tools() -> None:
    if not DEV_TOOLS_ENABLED:
        raise HTTPException(status_code=404, detail="Dev tools are disabled")


@app.get("/api/dev/segment/models")
def list_segmentation_models():
    """비교 탭이 띄울 모델 목록. /api/closet/models와 같은 내용이다."""
    require_dev_tools()
    return segmentation_registry()


def pairwise_agreement(masks_by_model: dict, iou) -> list[dict]:
    rows = []
    keys = list(masks_by_model)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            left_masks, right_masks = masks_by_model[left], masks_by_model[right]
            categories = sorted(set(left_masks) | set(right_masks), key=segment_models.CATEGORIES.index)
            scores = {}
            for category in categories:
                if category in left_masks and category in right_masks:
                    scores[category] = iou(left_masks[category], right_masks[category])
                else:
                    scores[category] = {"onlyIn": left if category in left_masks else right}
            rows.append({"left": left, "right": right, "categories": scores})
    return rows


@app.post("/api/dev/segment/compare")
def compare_segmentation_models(request: SegmentCompareRequest):
    require_dev_tools()
    keys = request.models or segment_models.DEFAULT_COMPARE
    unknown = [key for key in keys if key not in segment_models.MODELS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"알 수 없는 모델: {', '.join(unknown)}")
    acquire_inference_slot()
    try:
        image = decode_image(request.image)
        source = io.BytesIO()
        image.save(source, format="PNG")
        raw = source.getvalue()
        import segment_service

        results, masks_by_model = [], {}
        for key in keys:
            try:
                analysis = segment_service.analyze(raw, key)
            except Exception as error:
                logging.exception("Segmentation comparison failed for %s", key)
                results.append({"model": segment_models.MODELS[key].as_dict(), "error": str(error) or "모델 실행에 실패했습니다"})
                continue
            masks_by_model[key] = analysis.pop("_masks")
            analysis.pop("_parse", None)  # 라벨맵은 refine 경로에서만 쓴다 — 직렬화도 되지 않는다
            analysis["overlay"] = encode_png(analysis.pop("overlay_png_bytes"))
            for item in analysis["items"]:
                item["image"] = encode_png(item.pop("png_bytes"))
            results.append(analysis)
        return {
            "name": request.name,
            "requested": keys,
            "results": results,
            "agreement": pairwise_agreement(masks_by_model, segment_service.mask_iou),
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logging.exception("Segmentation comparison failed")
        raise HTTPException(status_code=503, detail="Segmentation model is unavailable") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/tryon")
def generate_tryon(request: TryOnRequest):
    acquire_inference_slot()
    try:
        images, engine, applied = image_engine.generate_tryon(request)
        _, gpu = cuda_info()
        encoded = {view: encode_image(image) for view, image in images.items()}
        applied_keys = {id(item) for item in applied}
        return {
            "image": encoded["front"],
            "views": encoded,
            "engine": engine,
            "gpu": gpu,
            "garmentCount": len(applied),
            "requestedCount": len(request.garments),
            # 레이어 순서대로 정렬된 결과. 요청 순서와 다를 수 있다.
            "appliedGarments": [
                {"name": item.name, "category": item.category} for item in applied
            ],
            # 같은 자리를 다투다 빠진 아이템. 화면에서 조용히 사라지면 사용자는
            # "옷이 반영되지 않았다"고 읽으므로 무엇이 빠졌는지 함께 돌려준다.
            "droppedGarments": [
                {"name": item.name, "category": item.category}
                for item in request.garments
                if id(item) not in applied_keys
            ],
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Invalid image payload") from error
    except Exception as error:
        logging.exception("Try-on generation failed")
        raise HTTPException(status_code=500, detail="Try-on generation failed") from error
    finally:
        INFERENCE_GATE.release()


def run_vlm_request(request: VLMImageRequest, prompt: str, max_new_tokens: int, debug: bool = False) -> dict:
    acquire_inference_slot()
    try:
        result = vlm_engine.analyze(decode_image(request.image), prompt, max_new_tokens, debug=debug)
        if debug:
            # 랩이 프롬프트를 자기 쪽에 복사해 두면 백엔드를 고칠 때마다 조용히 어긋난다.
            # 실제로 보낸 문자열을 그대로 돌려주는 편이 유일한 정본이다.
            result["prompt"] = prompt
            result["maxNewTokens"] = max_new_tokens
        return result
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("VLM analysis failed")
        raise HTTPException(status_code=500, detail="VLM analysis failed") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/embedding")
def create_embedding(request: EmbeddingRequest):
    acquire_inference_slot()
    try:
        vector = embedding_engine.embed(decode_image(request.image))
        return {"model": SIGLIP_MODEL, "vector": vector}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("SigLIP embedding failed")
        raise HTTPException(status_code=500, detail="SigLIP embedding failed") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/embeddings")
def create_embeddings(request: EmbeddingBatchRequest):
    """옷장 여러 벌을 한 요청으로. 한 벌씩 부르면 분당 요청 제한에 먼저 걸린다."""
    acquire_inference_slot()
    try:
        images = [decode_image(image) for image in request.images]
        return {"model": SIGLIP_MODEL, "vectors": embedding_engine.embed_many(images)}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("SigLIP batch embedding failed")
        raise HTTPException(status_code=500, detail="SigLIP embedding failed") from error
    finally:
        INFERENCE_GATE.release()


def garment_prompt(name: str, category: str | None) -> str:
    return QwenVLMEngine.GARMENT_PROMPT + (
        "\n판정 기준: 가방은 물건을 넣어 들거나 메는 수납 용품(백팩·토트·크로스백·클러치)만 뜻해. "
        "벨트·모자·안경·시계·목도리·장갑·주얼리는 액세서리이고, 조끼·베스트·민소매 옷은 상의 또는 아우터야. "
        "primaryColor는 흰/투명 배경, 피부와 그림자를 제외하고 아이템 본체에서 가장 넓은 색으로 정해."
        f"\n사용자 레이블: name={name}, category={category or '미지정'}. 레이블은 참고만 하고 사진을 우선해."
    )


def body_prompt(gender: str | None) -> str:
    return QwenVLMEngine.BODY_PROMPT + f"\n사용자가 선택한 성별은 {gender or '미지정'}이야."


# Qwen이 앱에서 맡은 네 가지 일. 태스크마다 프롬프트도 토큰 예산도 다르고,
# 이 표가 Qwen Lab의 목록이 된다 — 랩이 목록을 따로 들고 있으면 여기에 태스크를
# 하나 더해도 랩에는 나타나지 않는다.
VLM_TASKS = {
    "garment": {
        "title": "옷 한 벌 분석",
        "role": "옷장에 넣을 사진 한 장에서 카테고리·색·소재·핏·디테일을 뽑아 JSON으로 만든다. 옷장 검색과 추천이 이 필드를 읽는다.",
        "maxNewTokens": 640,
        "path": "/api/vlm/garment",
    },
    "lookbook": {
        "title": "룩북 분해",
        "role": "한 사람이 입은 착장을 옷 단위로 쪼갠다. 아우터 안의 셔츠까지 별도 항목으로 내고 bbox(0~1000)로 위치까지 준다.",
        "maxNewTokens": 900,
        "path": "/api/vlm/lookbook",
    },
    "body": {
        "title": "체형 특징",
        "role": "전신사진에서 보이는 체형·비율·어깨선만 말한다. 키·몸무게 숫자는 추측하지 않는다.",
        "maxNewTokens": 320,
        "path": "/api/vlm/body",
    },
    "tryon-judge": {
        "title": "착장 결과 심판",
        "role": "합성된 착장 사진을 채점한다. 정답 이미지가 없어 픽셀 지표로는 잴 수 없는 '레이어 순서가 맞나'를 VLM 이진 판정으로 대신한다.",
        "maxNewTokens": 700,
        "path": "/api/vlm/tryon-judge",
    },
}


@app.get("/api/vlm/prompts")
def list_vlm_prompts():
    """태스크별 프롬프트 본문. Qwen Lab이 실행 전에도 무엇을 물어보는지 띄운다.

    적재도 추론도 하지 않는 상수 응답이다. garment·body는 요청값이 뒤에 한 줄
    붙으므로 여기 것은 그 앞부분이고, 실제로 보낸 문자열은 debug 응답의 prompt에 온다.
    """
    prompts = {
        "garment": QwenVLMEngine.GARMENT_PROMPT,
        "lookbook": QwenVLMEngine.LOOKBOOK_PROMPT,
        "body": QwenVLMEngine.BODY_PROMPT,
        "tryon-judge": QwenVLMEngine.TRYON_JUDGE_PROMPT,
    }
    return {
        "model": VLM_MODEL,
        "quantization": "nf4" if VLM_LOAD_IN_4BIT else "none",
        "maxPixels": VLM_MAX_PIXELS,
        "tasks": [{"key": key, **meta, "prompt": prompts[key]} for key, meta in VLM_TASKS.items()],
    }


@app.post("/api/vlm/garment")
def analyze_garment(request: VLMImageRequest):
    prompt = garment_prompt(request.name, request.category)
    return run_vlm_request(request, prompt, VLM_TASKS["garment"]["maxNewTokens"], request.debug)


@app.post("/api/vlm/lookbook")
def analyze_lookbook(request: VLMImageRequest):
    return run_vlm_request(request, QwenVLMEngine.LOOKBOOK_PROMPT, VLM_TASKS["lookbook"]["maxNewTokens"], request.debug)


@app.post("/api/vlm/body")
def analyze_body(request: VLMImageRequest):
    return run_vlm_request(request, body_prompt(request.gender), VLM_TASKS["body"]["maxNewTokens"], request.debug)


@app.post("/api/vlm/tryon-judge")
def judge_tryon(request: TryOnJudgeRequest):
    """착장 결과 자동 채점. scripts/eval_tryon.py가 개선 전/후 수치를 낼 때 쓴다."""
    prompt = QwenVLMEngine.TRYON_JUDGE_PROMPT.format(manifest=request.manifest)
    return run_vlm_request(
        VLMImageRequest(image=request.image, name="tryon-judge"),
        prompt, VLM_TASKS["tryon-judge"]["maxNewTokens"], request.debug,
    )


@app.post("/api/warmup")
def warmup_model():
    global WARMUP_VERIFIED, VLM_WARMUP_VERIFIED, EMBEDDING_WARMUP_VERIFIED, SEGMENTATION_WARMUP_VERIFIED
    available, _ = cuda_info()
    if not available:
        raise HTTPException(status_code=503, detail="CUDA GPU is unavailable")
    acquire_inference_slot()
    try:
        vlm_engine._load()
        VLM_WARMUP_VERIFIED = True
        embedding_engine._load()
        EMBEDDING_WARMUP_VERIFIED = True
        image_engine._load()
        WARMUP_VERIFIED = True
        import segment_service

        segment_service.get_model()
        SEGMENTATION_WARMUP_VERIFIED = True
        segmentation = segment_service.status()
        return {
            "ok": True,
            "model": IMAGE_MODEL,
            "dtype": image_engine.dtype,
            "vlmModel": VLM_MODEL,
            "embeddingModel": SIGLIP_MODEL,
            "embeddingDevice": embedding_engine.device,
            "vlmDtype": vlm_engine.dtype,
            "vlmQuantization": "nf4" if VLM_LOAD_IN_4BIT else "none",
            "segmentationModel": segmentation["model"],
            "segmentationDevice": segmentation["device"],
        }
    except Exception as error:
        logging.exception("Model warmup failed")
        raise HTTPException(status_code=500, detail="Model warmup failed") from error
    finally:
        INFERENCE_GATE.release()
