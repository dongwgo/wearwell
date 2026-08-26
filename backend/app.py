from __future__ import annotations

import base64
import gc
import hmac
import io
import json
import logging
import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

import segment_models

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
VLM_LOAD_IN_4BIT = os.getenv("VLM_LOAD_IN_4BIT", "1") == "1"
VLM_MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", str(1024 * 1024)))
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
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
# 세그멘테이션 모델 비교 도구(/api/dev/*). 로컬 개발용이라 기본은 켜짐이고,
# 공개 터널에 붙일 때는 0으로 끈다 — 모델을 여러 개 올려 메모리를 크게 쓴다.
DEV_TOOLS_ENABLED = os.getenv("WEARWELL_DEV_TOOLS", "1") == "1"
MAX_COMPARE_MODELS = 4
# 옷 한 벌은 정사각에 가깝다. 아바타용 768x1152로 만들면 옷이 세로로 늘어난다.
REFINE_SIZE = (
    int(os.getenv("REFINE_WIDTH", "768")),
    int(os.getenv("REFINE_HEIGHT", "768")),
)
# 옷 한 벌마다 FLUX를 한 번씩 돌린다. 한 요청 안에서 무한정 돌리면 GPU 큐가 막힌다.
MAX_REFINE_ITEMS = int(os.getenv("MAX_REFINE_ITEMS", "4"))
GPU_QUEUE_TIMEOUT = float(os.getenv("GPU_QUEUE_TIMEOUT", "300"))
GPU_LOCK = threading.RLock()
INFERENCE_GATE = threading.Lock()
RATE_LOCK = threading.Lock()
REQUEST_TIMES: dict[str, deque[float]] = defaultdict(deque)
WARMUP_VERIFIED = False
VLM_WARMUP_VERIFIED = False


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


class VLMImageRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="이미지", max_length=120)
    category: str | None = Field(default=None, max_length=30)
    gender: Literal["women", "men"] | None = None


class SegmentationRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    # 비우면 프로덕션 기본 모델. segment_models.MODELS의 key를 넣으면 그 모델로 돈다.
    model: str | None = Field(default=None, max_length=60)


class SegmentCompareRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    models: list[str] = Field(default_factory=list, max_length=MAX_COMPARE_MODELS)


class RepairOptions(BaseModel):
    """마스크 보수 단계 스위치. 랩에서 하나씩 꺼 보면서 어떤 단계가 무슨 일을 하는지 본다."""

    close: bool = True
    fillHoles: bool = True
    dropStrays: bool = True
    closeScale: float = Field(default=0.012, ge=0.0, le=0.05)
    strayRatio: float = Field(default=0.08, ge=0.0, le=1.0)


class ClosetRefineRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    model: str | None = Field(default=None, max_length=60)
    # 비우면 품질 필터를 통과한 카테고리 전부.
    categories: list[str] = Field(default_factory=list, max_length=len(segment_models.CATEGORIES))
    # 걸러진 후보까지 파이프라인에 태운다. 왜 걸러졌는지를 생성 결과로 확인할 때 쓴다.
    includeRejected: bool = False
    repair: RepairOptions = Field(default_factory=RepairOptions)
    generate: bool = True
    seed: int = 42
    steps: int | None = Field(default=None, ge=1, le=50)


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
    CATEGORY_LABELS = {
        "upper": "upper-body garment",
        "lower": "lower-body garment",
        "overall": "one-piece or full-body garment",
        "outer": "outerwear layer",
        "shoes": "pair of shoes",
        "bag": "bag",
        "accessory": "fashion accessory",
    }
    # 세그멘테이션은 옷장 카테고리(한국어)로 결과를 낸다. 위 CATEGORY_LABELS는
    # 프론트가 보내는 영문 키용이라 그대로 쓸 수 없어 따로 둔다.
    CLOSET_CATEGORY_LABELS = {
        "상의": "upper-body garment (top)",
        "아우터": "outerwear jacket or coat",
        "하의": "lower-body garment (pants or skirt)",
        "원피스": "one-piece dress",
        "신발": "pair of shoes",
        "가방": "bag",
        "액세서리": "fashion accessory",
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

    def _run(
        self,
        *,
        prompt: str,
        seed: int,
        images: list[Image.Image] | None = None,
        size: tuple[int, int] | None = None,
        steps: int | None = None,
    ) -> Image.Image:
        """size는 (너비, 높이). 전신 결과물은 세로로 길지만 옷 한 벌은 정사각이라
        아바타 해상도를 그대로 쓰면 옷이 늘어난 채 생성된다."""
        import torch

        width, height = size or INFERENCE_SIZE
        pipe = self._load()
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
        with GPU_LOCK, torch.inference_mode():
            output = pipe(**inputs).images[0]
            torch.cuda.empty_cache()
        return output.resize((width, height), Image.Resampling.LANCZOS)

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

    def refine_garment(self, garment: Image.Image, category: str, name: str, seed: int, steps: int | None = None):
        """세그멘테이션으로 오려낸 옷 -> 옷장에 넣을 상품컷.

        입력은 이미 흰 배경에 얹은 정규화 이미지지만, 팔에 가려졌던 자리는 여전히
        비어 있다. 형태학 연산으로는 없는 픽셀을 만들 수 없어서 그 복원을 여기서
        시킨다. 그래서 프롬프트가 "구멍을 메우고 끊어진 부분을 잇되 색·패턴·재단은
        그대로"라는 말을 반드시 담아야 한다 — 자유롭게 그리라고 하면 다른 옷이 온다.
        """
        has_cuda, _ = cuda_info()
        if has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1":
            prompt = (
                f"Reference image 1 is a {self.CLOSET_CATEGORY_LABELS.get(category, 'garment')} named '{name}' that was "
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


class QwenVLMEngine:
    GARMENT_PROMPT = """이 사진에서 주된 옷 한 벌만 분석해 한국어 JSON으로만 답해. 보이지 않는 정보는 추측하지 말고 '확인 어려움'으로 써. 스키마: {"category":"상의/하의/아우터/원피스/신발/가방/액세서리","subcategory":"구체 종류","primaryColor":"주색","secondaryColors":["보조색"],"material":"소재 추정","texture":"표면 질감","fit":"슬림/레귤러/세미오버/오버/스트레이트/세미와이드/와이드/커브드","silhouette":"실루엣","wrinkle":"주름의 정도와 형태","finish":"광택·워싱·표면 마감","construction":["봉제선·단추·지퍼·포켓·밑단 등 보이는 디테일"],"pattern":"패턴","season":["계절"],"weather":["어울리는 날씨"],"summary":"한 문장 요약"}"""
    LOOKBOOK_PROMPT = """이 패션 룩북에서 가장 크게 보이는 한 사람의 착장을 분석해. 착장 전체를 한 항목으로 요약하지 말고 눈에 보이는 옷을 실제 경계대로 각각 분리해 한국어 JSON으로만 답해. 재킷 안에 셔츠가 있으면 아우터와 상의를 별도 pieces 항목으로 쓰고 같은 카테고리의 레이어도 합치지 마. 보이지 않는 옷이나 색은 추측하지 말고 색상은 각 옷 자체의 주조색부터 최대 2개만 써. 신발·가방도 충분히 보일 때만 별도 항목으로 써. bbox는 전체 이미지 왼쪽 위를 0,0, 오른쪽 아래를 1000,1000으로 본 옷의 경계야. 스키마: {"summary":"색·실루엣·레이어링 한 문장","mood":"스타일 무드","pieces":[{"pieceId":"고유 번호","label":"화이트 셔츠처럼 옷을 구별하는 이름","layer":"아우터/이너/단독/하의/신발/가방","category":"상의/하의/아우터/원피스/신발/가방/액세서리","bbox":[0,0,1000,1000],"colors":["정확한 주조색"],"materials":["소재 추정"],"fits":["핏"],"details":["주름·마감·봉제·형태 디테일"],"confidence":0.0}]}"""
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
        if quantization_config is not None:
            load_options["quantization_config"] = quantization_config
        self.model = model_factory.from_pretrained(VLM_MODEL, **load_options)
        self.processor = processor_factory.from_pretrained(
            VLM_MODEL,
            min_pixels=256 * 28 * 28,
            max_pixels=VLM_MAX_PIXELS,
        )
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

    def analyze(self, image: Image.Image, prompt: str, max_new_tokens: int) -> dict:
        import torch

        model, processor = self._load()
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
        with GPU_LOCK, torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(source) :] for source, output in zip(inputs.input_ids, generated)]
        text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        result = self._parse_json(text)
        result.update({
            "engine": "Qwen3-VL-8B-Instruct",
            "model": VLM_MODEL,
            "quantization": "nf4" if VLM_LOAD_IN_4BIT else self.dtype,
        })
        return result

    @property
    def engine_name(self) -> str:
        mode = "nf4" if VLM_LOAD_IN_4BIT else self.dtype or "auto"
        return f"qwen3-vl-8b-{mode}-cuda"


image_engine = FluxImageEngine()
vlm_engine = QwenVLMEngine()


def acquire_inference_slot() -> None:
    if not INFERENCE_GATE.acquire(timeout=GPU_QUEUE_TIMEOUT):
        raise HTTPException(
            status_code=503,
            detail="GPU queue timeout",
            headers={"Retry-After": "10"},
        )


@app.get("/api/health")
def health():
    available, name = cuda_info()
    return {
        "ok": True,
        "cuda": available,
        "gpu": name,
        "model": IMAGE_MODEL,
        "avatarModel": IMAGE_MODEL,
        "segmentationModel": segment_models.MODELS[segment_models.PRODUCTION_MODEL].model_id,
        "segmentationModelKey": segment_models.PRODUCTION_MODEL,
        "devTools": DEV_TOOLS_ENABLED,
        "tryonModel": IMAGE_MODEL,
        "vlmModel": VLM_MODEL,
        "modelLoaded": image_engine.pipe is not None,
        "vlmLoaded": vlm_engine.model is not None,
        "warmupVerified": WARMUP_VERIFIED,
        "vlmWarmupVerified": VLM_WARMUP_VERIFIED,
        "dtype": image_engine.dtype,
        "vlmDtype": vlm_engine.dtype,
        "vlmQuantization": "nf4" if VLM_LOAD_IN_4BIT else "none",
        "queueTimeoutSeconds": GPU_QUEUE_TIMEOUT,
        "rateLimitPerMinute": RATE_LIMIT_PER_MINUTE,
        "resolution": f"{INFERENCE_SIZE[0]}x{INFERENCE_SIZE[1]}",
    }


@app.post("/api/avatar", response_model=AvatarResponse)
def generate_avatar(data: Measurements):
    acquire_inference_slot()
    try:
        image, engine = image_engine.generate_avatar(data)
        _, gpu = cuda_info()
        return AvatarResponse(image=encode_image(image), engine=engine, gpu=gpu)
    except Exception as error:
        logging.exception("Avatar generation failed")
        raise HTTPException(status_code=500, detail="Avatar generation failed") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/closet/segment")
def segment_closet_photo(request: SegmentationRequest):
    acquire_inference_slot()
    try:
        image = decode_image(request.image)
        source = io.BytesIO()
        image.save(source, format="PNG")
        import segment_service

        detections = segment_service.segment(source.getvalue(), request.model)
        return {
            "model": segment_models.resolve(request.model).as_dict(),
            "items": [
                {
                    "id": f"segment-{int(time.time() * 1000)}-{index}",
                    "name": f"{request.name} - {item['category']}",
                    "category": item["category"],
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


def require_dev_tools() -> None:
    if not DEV_TOOLS_ENABLED:
        raise HTTPException(status_code=404, detail="Dev tools are disabled")


@app.get("/api/dev/segment/models")
def list_segmentation_models():
    """비교 탭이 띄울 모델 목록. 모델을 적재하지 않으므로 torch 없이도 응답한다."""
    require_dev_tools()
    loaded: list[str] = []
    try:
        import segment_service

        loaded = segment_service.loaded_keys()
    except Exception:  # torch 미설치 등 — 목록 자체는 그대로 내려준다
        logging.debug("segment_service unavailable while listing models", exc_info=True)
    return {
        "models": [spec.as_dict() for spec in segment_models.MODELS.values()],
        "default": segment_models.DEFAULT_COMPARE,
        "production": segment_models.PRODUCTION_MODEL,
        "loaded": loaded,
        "categoryColors": {name: f"#{r:02x}{g:02x}{b:02x}" for name, (r, g, b) in segment_models.CATEGORY_COLORS.items()},
        # 기본 기준. 모델별 최종 기준은 각 model의 thresholds에 들어 있다.
        "thresholds": segment_models.QUALITY_THRESHOLDS,
    }


@app.post("/api/dev/segment/compare")
def compare_segmentation_models(request: SegmentCompareRequest):
    """같은 사진을 여러 모델에 돌려 결과를 나란히 돌려준다.

    GPU/CPU를 순차로 쓰도록 추론 게이트 안에서 한 모델씩 처리한다. 모델별로
    독립 실패를 허용한다 — 하나가 못 뜬다고 비교 전체를 버리면 쓸모가 없다.
    """
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
                results.append({
                    "model": segment_models.MODELS[key].as_dict(),
                    "error": str(error) or "모델 실행에 실패했습니다",
                })
                continue
            masks_by_model[key] = analysis.pop("_masks")
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


def pairwise_agreement(masks_by_model: dict, iou) -> list[dict]:
    """모델 쌍마다 카테고리별 IoU. 어느 옷에서 의견이 갈리는지 보려는 것이다.

    한쪽만 검출한 카테고리는 IoU가 0이므로 빠뜨리지 않고 함께 싣는다.
    """
    keys = list(masks_by_model)
    rows = []
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


@app.post("/api/dev/closet/refine")
def refine_closet_photo(request: ClosetRefineRequest):
    """전신샷 -> 세그멘테이션 -> 마스크 보수 -> FLUX 재생성을 한 요청에서 전부 돌린다.

    Refine Lab이 단계별 중간 산출물을 나란히 놓고 보기 위한 개발용 경로다. 옷 한 벌씩
    독립 실패를 허용한다 — 한 벌의 생성이 실패했다고 나머지 단계 결과까지 버리면
    무엇이 문제였는지 볼 수가 없다.
    """
    require_dev_tools()
    unknown = [category for category in request.categories if category not in segment_models.CATEGORIES]
    if unknown:
        raise HTTPException(status_code=422, detail=f"알 수 없는 카테고리: {', '.join(unknown)}")

    acquire_inference_slot()
    started = time.perf_counter()
    try:
        import numpy as np

        image = decode_image(request.image)
        source = io.BytesIO()
        image.save(source, format="PNG")
        import refine_service
        import segment_service

        analysis = segment_service.analyze(source.getvalue(), request.model or segment_models.PRODUCTION_MODEL)
        masks = analysis.pop("_masks")
        image_np = np.array(image)

        wanted = [
            item for item in analysis["items"]
            if (item["accepted"] or request.includeRejected)
            and (not request.categories or item["category"] in request.categories)
        ]
        skipped = len(wanted) - MAX_REFINE_ITEMS
        options = request.repair.model_dump()

        items = []
        for item in wanted[:MAX_REFINE_ITEMS]:
            item.pop("png_bytes", None)  # 스테이지 크롭을 따로 만든다 — 같은 그림을 두 번 싣지 않는다
            repair_started = time.perf_counter()
            stages = refine_service.build_stages(image_np, masks[item["category"]], options)
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
                generate_started = time.perf_counter()
                try:
                    closet, engine = image_engine.refine_garment(
                        stages["image"], item["category"], f"{request.name} {item['category']}",
                        request.seed, request.steps,
                    )
                    item["stages"]["closet"] = encode_image(closet)
                    item["generation"] = {
                        "engine": engine,
                        "seed": request.seed,
                        "steps": request.steps or INFERENCE_STEPS,
                        "seconds": round(time.perf_counter() - generate_started, 2),
                    }
                except Exception as error:
                    logging.exception("Closet refinement failed for %s", item["category"])
                    item["generationError"] = str(error) or "생성에 실패했습니다"
            items.append(item)

        return {
            "name": request.name,
            "model": analysis["model"],
            "device": analysis["device"],
            "imageSize": analysis["imageSize"],
            "segmentSeconds": analysis["inferenceSeconds"],
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


@app.post("/api/tryon")
def generate_tryon(request: TryOnRequest):
    acquire_inference_slot()
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


def run_vlm_request(request: VLMImageRequest, prompt: str, max_new_tokens: int) -> dict:
    acquire_inference_slot()
    try:
        return vlm_engine.analyze(decode_image(request.image), prompt, max_new_tokens)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logging.exception("VLM analysis failed")
        raise HTTPException(status_code=500, detail="VLM analysis failed") from error
    finally:
        INFERENCE_GATE.release()


@app.post("/api/vlm/garment")
def analyze_garment(request: VLMImageRequest):
    context = f"\n사용자 레이블: name={request.name}, category={request.category or '미지정'}. 레이블은 참고만 하고 사진을 우선해."
    return run_vlm_request(request, QwenVLMEngine.GARMENT_PROMPT + context, 640)


@app.post("/api/vlm/lookbook")
def analyze_lookbook(request: VLMImageRequest):
    return run_vlm_request(request, QwenVLMEngine.LOOKBOOK_PROMPT, 900)


@app.post("/api/vlm/body")
def analyze_body(request: VLMImageRequest):
    context = f"\n사용자가 선택한 성별은 {request.gender or '미지정'}이야."
    return run_vlm_request(request, QwenVLMEngine.BODY_PROMPT + context, 320)


@app.post("/api/warmup")
def warmup_model():
    global WARMUP_VERIFIED, VLM_WARMUP_VERIFIED
    available, _ = cuda_info()
    if not available:
        raise HTTPException(status_code=503, detail="CUDA GPU is unavailable")
    acquire_inference_slot()
    try:
        vlm_engine._load()
        VLM_WARMUP_VERIFIED = True
        image_engine._load()
        WARMUP_VERIFIED = True
        return {
            "ok": True,
            "model": IMAGE_MODEL,
            "dtype": image_engine.dtype,
            "vlmModel": VLM_MODEL,
            "vlmDtype": vlm_engine.dtype,
            "vlmQuantization": "nf4" if VLM_LOAD_IN_4BIT else "none",
        }
    except Exception as error:
        logging.exception("Model warmup failed")
        raise HTTPException(status_code=500, detail="Model warmup failed") from error
    finally:
        INFERENCE_GATE.release()
