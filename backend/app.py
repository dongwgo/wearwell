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
from avatar_body import VIEW_YAW, BodyTarget, build_avatar_prompt, build_body_reference
from tryon_prompt import build_tryon_prompt, order_garments

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "black-forest-labs/FLUX.2-klein-4B")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
SEGMENTATION_MODEL = os.getenv("SEGMENTATION_MODEL", "sayeed99/segformer_b3_clothes")
VLM_LOAD_IN_4BIT = os.getenv("VLM_LOAD_IN_4BIT", "1") == "1"
VLM_MAX_PIXELS = int(os.getenv("VLM_MAX_PIXELS", str(1024 * 1024)))
API_TOKEN = os.getenv("WEARWELL_API_TOKEN", "")
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
MAX_COMPARE_MODELS = 4
GPU_QUEUE_TIMEOUT = float(os.getenv("GPU_QUEUE_TIMEOUT", "300"))
GPU_LOCK = threading.RLock()
INFERENCE_GATE = threading.Lock()
RATE_LOCK = threading.Lock()
REQUEST_TIMES: dict[str, deque[float]] = defaultdict(deque)
WARMUP_VERIFIED = False
VLM_WARMUP_VERIFIED = False
SEGMENTATION_WARMUP_VERIFIED = False


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


class VLMImageRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="이미지", max_length=120)
    category: str | None = Field(default=None, max_length=30)
    gender: Literal["women", "men"] | None = None


class TryOnJudgeRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    # 입히기로 한 아이템 목록을 사람이 읽는 문장으로. 심판은 이 목록과 사진만 본다.
    manifest: str = Field(min_length=1, max_length=2_000)


class SegmentationRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    model: str | None = Field(default=None, max_length=60)


class SegmentCompareRequest(BaseModel):
    image: str = Field(min_length=1, max_length=8_000_000)
    name: str = Field(default="full-body photo", max_length=120)
    models: list[str] = Field(default_factory=list, max_length=MAX_COMPARE_MODELS)


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
    def __init__(self, pipeline_factory=None) -> None:
        self.pipe = None
        self.pipeline_factory = pipeline_factory
        self.dtype: str | None = None
        self.lora_name: str | None = None

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
        # 파인튜닝한 try-on edit-LoRA가 지정돼 있으면 얹는다. 실패해도 base 모델로
        # 서비스는 계속 돌아가야 하므로 예외를 삼키고 로그만 남긴다.
        if TRYON_LORA_PATH:
            try:
                # 아바타 생성과 try-on이 같은 파이프라인을 공유하므로 LoRA를 fuse하면
                # 아바타까지 try-on 학습 가중치의 영향을 받는다. 이름 있는 어댑터로
                # 올린 뒤 try-on 호출에서만 켠다.
                pipe.load_lora_weights(TRYON_LORA_PATH, adapter_name="wearwell_tryon")
                if hasattr(pipe, "disable_lora"):
                    pipe.disable_lora()
                self.lora_name = os.path.basename(TRYON_LORA_PATH)
            except Exception:
                logging.exception("Try-on LoRA load failed; continuing with the base model")
                self.lora_name = None
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
        steps: int | None = None,
        use_tryon_lora: bool = False,
    ) -> Image.Image:
        import torch

        pipe = self._load()
        inputs = {
            "prompt": prompt,
            "height": INFERENCE_SIZE[1],
            "width": INFERENCE_SIZE[0],
            "guidance_scale": GUIDANCE_SCALE,
            "num_inference_steps": steps or INFERENCE_STEPS,
            "generator": torch.Generator(device="cuda").manual_seed(seed),
        }
        if images:
            inputs["image"] = images
        with GPU_LOCK, torch.inference_mode():
            if self.lora_name:
                if use_tryon_lora:
                    if hasattr(pipe, "set_adapters"):
                        pipe.set_adapters("wearwell_tryon", adapter_weights=TRYON_LORA_SCALE)
                    elif hasattr(pipe, "enable_lora"):
                        pipe.enable_lora()
                elif hasattr(pipe, "disable_lora"):
                    pipe.disable_lora()
            output = pipe(**inputs).images[0]
            torch.cuda.empty_cache()
        return output.resize(INFERENCE_SIZE, Image.Resampling.LANCZOS)

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

    def generate_tryon(self, request: TryOnRequest) -> tuple[Image.Image, str]:
        person = decode_image(request.avatar)
        # 참조 이미지 번호가 레이어 순서와 어긋나면 모델이 번호를 레이어 힌트로
        # 오해한다. 디코딩 전에 안쪽 레이어부터 정렬하고 슬롯 충돌을 정리한다.
        ordered = order_garments(list(request.garments), limit=MAX_TRYON_GARMENTS)
        garments = [decode_image(item.image) for item in ordered]
        has_cuda, _ = cuda_info()
        if has_cuda and os.getenv("ONEULOUT_GPU", "1") == "1":
            prompt = build_tryon_prompt(ordered)
            image = self._run(
                prompt=prompt,
                seed=request.seed,
                images=[person, *garments],
                steps=TRYON_STEPS,
                use_tryon_lora=True,
            )
            engine = self.engine_name
            if self.lora_name:
                engine += f"+lora:{self.lora_name}"
            return image, engine
        return self.fallback_tryon(person, ordered), "tryon-preview-fallback"

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
- identity_ok: 얼굴, 체형, 포즈가 원래 사람과 같으면 true."""

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
                    "image": "data:image/png;base64," + base64.b64encode(item["png_bytes"]).decode(),
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
    require_dev_tools()
    import segment_service

    return {
        "models": [spec.as_dict() for spec in segment_models.MODELS.values()],
        "default": segment_models.DEFAULT_COMPARE,
        "production": segment_models.PRODUCTION_MODEL,
        "loaded": segment_service.loaded_keys(),
        "categoryColors": {name: f"#{r:02x}{g:02x}{b:02x}" for name, (r, g, b) in segment_models.CATEGORY_COLORS.items()},
        # 기본 기준. 모델별 최종 기준은 각 model의 thresholds에 들어 있다.
        "thresholds": segment_models.QUALITY_THRESHOLDS,
    }


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
            analysis["overlay"] = "data:image/png;base64," + base64.b64encode(analysis.pop("overlay_png_bytes")).decode()
            for item in analysis["items"]:
                item["image"] = "data:image/png;base64," + base64.b64encode(item.pop("png_bytes")).decode()
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
        applied = order_garments(list(request.garments), limit=MAX_TRYON_GARMENTS)
        image, engine = image_engine.generate_tryon(request)
        _, gpu = cuda_info()
        return {
            "image": encode_image(image),
            "engine": engine,
            "gpu": gpu,
            "garmentCount": len(applied),
            "requestedCount": len(request.garments),
            # 레이어 순서대로 정렬된 결과. 요청 순서와 다를 수 있고, 같은 부위가
            # 겹치면 안쪽 레이어만 남으므로 실제로 입힌 목록을 그대로 돌려준다.
            "appliedGarments": [
                {"name": item.name, "category": item.category} for item in applied
            ],
        }
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


@app.post("/api/vlm/tryon-judge")
def judge_tryon(request: TryOnJudgeRequest):
    """착장 결과 자동 채점. scripts/eval_tryon.py가 개선 전/후 수치를 낼 때 쓴다."""
    prompt = QwenVLMEngine.TRYON_JUDGE_PROMPT.format(manifest=request.manifest)
    return run_vlm_request(VLMImageRequest(image=request.image, name="tryon-judge"), prompt, 700)


@app.post("/api/warmup")
def warmup_model():
    global WARMUP_VERIFIED, VLM_WARMUP_VERIFIED, SEGMENTATION_WARMUP_VERIFIED
    available, _ = cuda_info()
    if not available:
        raise HTTPException(status_code=503, detail="CUDA GPU is unavailable")
    acquire_inference_slot()
    try:
        vlm_engine._load()
        VLM_WARMUP_VERIFIED = True
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
