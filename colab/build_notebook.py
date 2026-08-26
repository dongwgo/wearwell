#!/usr/bin/env python3
"""colab/wearwell_tryon_lora.ipynb 를 생성한다.

노트북 JSON을 직접 손으로 쓰면 escape 때문에 깨지기 쉬워서, 셀을 파이썬
리스트로 정의하고 여기서 직렬화한다. 노트북을 고칠 때는 이 파일을 고치고
다시 실행하는 게 안전하다.
"""

from __future__ import annotations

import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = []


def add(kind: str, source: str) -> None:
    CELLS.append((kind, source.strip("\n")))


add(MD, r"""
# Wearwell — 가상 착장 edit-LoRA 파인튜닝 (L4 24GB)

이 노트북이 고치려는 것은 세 가지 실패 유형이다.

| 실패 | 증상 | 이 노트북의 처방 |
|---|---|---|
| 레이어 뒤집힘 | 아우터가 상의 밑으로 들어가거나 둘이 한 벌로 합쳐짐 | 레이어 순서가 명시된 캡션으로 학습 |
| 아이템 누락·환각 | 고른 옷이 안 나오거나 없던 옷이 생김 | 재구성(reconstruction) 목적 학습 |
| 가방·액세서리 오배치 | 모자가 몸통에 붙고 가방끈이 옷 속을 통과 | 착용 지점(anchor) 문장이 들어간 캡션 |

**핵심 설계**: 학습 캡션을 `backend/tryon_prompt.py`의 `build_tryon_prompt()`로
만든다. 추론 때 백엔드가 보내는 문장과 **글자 그대로 같은 형식**이어야 LoRA가
전이된다. 캡션 형식과 추론 프롬프트가 다르면 학습은 수렴해도 서비스에서는
효과가 없다 — edit-LoRA 파인튜닝에서 가장 흔한 실패 원인이다.

**학습 대상은 `FLUX.2-klein-base-4B`** (증류 전 base)이고, 추론은 기존대로
`FLUX.2-klein-4B`(4-step 증류판)에 LoRA만 얹어서 한다. BFL 공식 가이드가
권장하는 조합이고, L4 24GB에서 1시간 안쪽으로 끝난다.

**런타임**: `런타임 → 런타임 유형 변경 → L4` (A100/H100도 가능)
""")

add(MD, "## 1. 환경 준비")

add(CODE, r"""
import os, subprocess, sys, textwrap

WORK = "/content/wearwell-lora"
REPO = "/content/wearwell"
os.makedirs(WORK, exist_ok=True)

# 프로젝트 저장소 — build_tryon_prompt()를 캡션 생성에 그대로 쓴다.
if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/dongwgo/wearwell.git", REPO], check=True)
sys.path.insert(0, f"{REPO}/backend")

if not os.path.isdir("/content/ai-toolkit"):
    subprocess.run(["git", "clone", "--depth", "1", "--recursive",
                    "https://github.com/ostris/ai-toolkit.git", "/content/ai-toolkit"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                "/content/ai-toolkit/requirements.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "huggingface_hub", "pillow", "numpy", "pyyaml", "requests"], check=True)

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("VRAM", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB"
      if torch.cuda.is_available() else "")
""")

add(CODE, r"""
from huggingface_hub import notebook_login
# FLUX.2-klein-base-4B 는 gated 저장소라 로그인 + 라이선스 동의가 필요하다.
# https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B 에서 먼저 동의할 것.
notebook_login()
""")

add(MD, r"""
## 2. 학습 데이터 만들기

VTON 학습의 표준 구성은 **재구성(reconstruction)**이다.

    (옷이 지워진 사람, 옷 조각들) --> 원본 사진

원본 사진이 곧 정답이므로 별도 라벨링이 필요 없다. 여기서는 저장소에 이미 있는
룩북 200장 + 인플루언서 룩 100장을 그대로 학습 데이터로 바꾼다.

한 장에서 세 가지를 뽑는다.

- **target**: 원본 사진 (모델이 만들어내야 할 결과)
- **control_1**: 옷 영역을 회색으로 지운 사람 = cloth-agnostic person
- **control_2 / control_3**: 잘라낸 옷 조각

옷의 경계와 **레이어(아우터/이너)** 는 이미 백엔드에 있는
`/api/vlm/lookbook`(Qwen3-VL)이 잡아준다. SegFormer는 ATR 라벨 구조상
`Upper-clothes` 하나로 상의와 아우터를 구분하지 못하므로, 정확히 이 프로젝트가
고치려는 레이어 문제에는 VLM 쪽 경계를 써야 한다.
""")

add(CODE, r"""
# 백엔드 주소. wearwell_backend_l4.ipynb를 먼저 띄우고 그 주소/토큰을 넣는다.
# 같은 런타임에서 백엔드를 돌리고 있다면 127.0.0.1:8787 그대로 두면 된다.
API_BASE  = "http://127.0.0.1:8787"
API_TOKEN = ""

# 몇 장으로 학습할지. 50~200쌍이면 edit-LoRA에 충분하다(BFL 권장 범위).
MAX_PAIRS = 160
""")

add(CODE, r"""
import base64, io, json, glob, random
from pathlib import Path
import requests
from PIL import Image, ImageDraw, ImageFilter

HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

def to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO(); image.convert("RGB").save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def analyze_lookbook(image: Image.Image) -> dict:
    response = requests.post(f"{API_BASE}/api/vlm/lookbook",
                             json={"image": to_data_url(image)},
                             headers=HEADERS, timeout=300)
    response.raise_for_status()
    return response.json()

# VLM이 돌려주는 layer/category 를 백엔드 슬롯 이름으로 옮긴다.
LAYER_TO_SLOT = {"아우터": "outer", "이너": "upper", "단독": "upper",
                 "하의": "lower", "신발": "shoes", "가방": "bag"}
CATEGORY_TO_SLOT = {"상의": "upper", "하의": "lower", "원피스": "overall", "아우터": "outer",
                    "신발": "shoes", "가방": "bag", "액세서리": "accessory"}

def piece_slot(piece: dict) -> str | None:
    return LAYER_TO_SLOT.get(piece.get("layer")) or CATEGORY_TO_SLOT.get(piece.get("category"))

def bbox_to_pixels(bbox, size):
    '''VLM bbox(0..1000 정규화) -> 픽셀 좌표. 경계를 살짝 넉넉히 잡는다.'''
    width, height = size
    x0, y0, x1, y1 = bbox
    pad = 0.01
    return (
        max(0, int((x0 / 1000 - pad) * width)),  max(0, int((y0 / 1000 - pad) * height)),
        min(width, int((x1 / 1000 + pad) * width)), min(height, int((y1 / 1000 + pad) * height)),
    )

sources = sorted(glob.glob(f"{REPO}/assets/lookbook/*.*")) + \
          sorted(glob.glob(f"{REPO}/assets/influencers/*.*"))
sources = [p for p in sources if Path(p).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
random.Random(0).shuffle(sources)
print(len(sources), "장의 원본 후보")
""")

add(CODE, r"""
from dataclasses import dataclass
from tryon_prompt import build_tryon_prompt, order_garments, LAYER_RANK

@dataclass
class Item:
    category: str
    name: str

ROOT = Path(WORK) / "data"
DIRS = {key: ROOT / key for key in ("target", "control_1", "control_2", "control_3")}
for path in DIRS.values():
    path.mkdir(parents=True, exist_ok=True)

# ai-toolkit은 control 폴더를 3개까지 받으므로 사람 1장 + 옷 2장이 상한이다.
# 학습에서 노리는 건 두 아이템 사이의 **관계**(상의-아우터 안팎, 옷-가방 걸침)라
# 2장이면 충분하고, 추론 때는 FLUX.2가 참조를 더 많이 받아도 규칙이 전이된다.
MAX_CONTROL_GARMENTS = 2

# 관계를 가르치는 데 쓸모 있는 조합만 남긴다. 상의 한 벌짜리 샘플을 아무리 넣어도
# 레이어 순서는 학습되지 않는다.
def is_useful(slots: list[str]) -> bool:
    has_layering = "upper" in slots and "outer" in slots
    has_carried = any(s in slots for s in ("bag", "accessory")) and len(slots) >= 2
    return has_layering or has_carried

def make_agnostic(image: Image.Image, boxes: list[tuple]) -> Image.Image:
    '''옷 영역을 지운 사람 이미지.

    단색으로 덮으면 경계가 너무 또렷해서 모델이 그 사각형을 그대로 따라 그린다.
    흐린 회색으로 덮어 "여기에 옷이 온다"는 힌트만 남긴다.
    '''
    agnostic = image.copy()
    overlay = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        draw.rectangle(box, fill=255)
    overlay = overlay.filter(ImageFilter.GaussianBlur(image.width * 0.012))
    grey = Image.new("RGB", image.size, (150, 150, 150))
    return Image.composite(grey, agnostic, overlay)

pairs, skipped = 0, 0
for path in sources:
    if pairs >= MAX_PAIRS:
        break
    try:
        image = Image.open(path).convert("RGB")
        analysis = analyze_lookbook(image)
        pieces = [p for p in analysis.get("pieces", []) if p.get("bbox") and piece_slot(p)]
        # 신뢰도가 낮은 검출은 학습 데이터를 오염시킨다.
        pieces = [p for p in pieces if float(p.get("confidence") or 0) >= 0.6]
        pieces.sort(key=lambda p: LAYER_RANK.get(piece_slot(p), 99))
        slots = [piece_slot(p) for p in pieces]
        if not is_useful(slots):
            skipped += 1
            continue

        chosen = pieces[:MAX_CONTROL_GARMENTS]
        boxes = [bbox_to_pixels(p["bbox"], image.size) for p in chosen]
        if any((b[2] - b[0]) < 40 or (b[3] - b[1]) < 40 for b in boxes):
            skipped += 1
            continue

        stem = f"{pairs:04d}"
        image.save(DIRS["target"] / f"{stem}.jpg", quality=95)
        make_agnostic(image, boxes).save(DIRS["control_1"] / f"{stem}.jpg", quality=95)
        for index, (piece, box) in enumerate(zip(chosen, boxes), start=2):
            image.crop(box).save(DIRS[f"control_{index}"] / f"{stem}.jpg", quality=95)

        # 캡션 = 추론 때 백엔드가 보내는 그 문장. 이게 이 노트북의 핵심이다.
        items = [Item(piece_slot(p), p.get("label") or p.get("category") or "아이템") for p in chosen]
        caption = build_tryon_prompt(order_garments(items, limit=MAX_CONTROL_GARMENTS))
        (DIRS["target"] / f"{stem}.txt").write_text(caption, encoding="utf-8")
        pairs += 1
        if pairs % 20 == 0:
            print(f"  {pairs}쌍 완성")
    except Exception as error:
        skipped += 1
        print("  건너뜀:", Path(path).name, type(error).__name__, error)

print(f"\n학습쌍 {pairs}개 / 건너뜀 {skipped}개 -> {ROOT}")
""")

add(CODE, r"""
# control_3 이 비어 있는 샘플이 있으면 ai-toolkit이 그 폴더를 통째로 무시한다.
# (파일명 기준으로 매칭하므로 한 폴더라도 비면 그 샘플의 control 개수가 달라진다)
# 아이템이 하나뿐인 샘플은 여기서 제외해 control 개수를 3으로 통일한다.
import shutil
targets = sorted(p.stem for p in DIRS["target"].glob("*.jpg"))
dropped = 0
for stem in targets:
    if not all((DIRS[key] / f"{stem}.jpg").exists() for key in ("control_1", "control_2", "control_3")):
        for key in DIRS:
            for suffix in (".jpg", ".txt"):
                (DIRS[key] / f"{stem}{suffix}").unlink(missing_ok=True)
        dropped += 1
kept = len(list(DIRS["target"].glob("*.jpg")))
print(f"control 3장이 모두 있는 샘플 {kept}개 (제외 {dropped}개)")
assert kept >= 30, "학습쌍이 너무 적다. MAX_PAIRS를 늘리거나 신뢰도 기준(0.6)을 낮춰볼 것."
""")

add(CODE, r"""
# 눈으로 한 번 확인 — 데이터가 잘못되면 학습은 조용히 망가진다.
import numpy as np
import matplotlib.pyplot as plt
stem = sorted(DIRS["target"].glob("*.jpg"))[0].stem
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
for ax, key in zip(axes, ("control_1", "control_2", "control_3", "target")):
    ax.imshow(Image.open(DIRS[key] / f"{stem}.jpg")); ax.set_title(key); ax.axis("off")
plt.tight_layout(); plt.show()
print(textwrap.fill((DIRS["target"] / f"{stem}.txt").read_text(encoding="utf-8"), 110))
""")

add(MD, r"""
### (선택) Garments2Look로 갈아타기

시간이 더 있으면 [Garments2Look](https://huggingface.co/datasets/ArtmeScienceLab/Garments2Look)
(CVPR 2026, Apache 2.0, 80K 착장쌍)이 훨씬 좋다. 레이어 순서와 액세서리가 라벨로
붙어 있고, 위에서 만든 재구성 데이터와 달리 **참조 옷 사진이 착장 사진과 다른
사진**이라 실제 서비스 조건과 같다.

전체는 284GB라 스트리밍으로 필요한 만큼만 받는다. 아래 셀은 필드 이름을 먼저
찍어보고 매핑을 결정한다 — 데이터셋 스키마를 추측해서 하드코딩하면 조용히
엉뚱한 컬럼으로 학습된다.
""")

add(CODE, r"""
# 스키마 확인용. 실행해서 필드 이름을 본 뒤 아래 매핑을 채운다.
from datasets import load_dataset

probe = load_dataset("ArtmeScienceLab/Garments2Look", split="train", streaming=True)
sample = next(iter(probe))
for key, value in sample.items():
    preview = value if isinstance(value, (str, int, float)) else type(value).__name__
    print(f"{key:24} {str(preview)[:120]}")
""")

add(MD, "## 3. 학습 설정")

add(CODE, r"""
import yaml

RUN_NAME = "wearwell_tryon_v1"
CONFIG = {
    "job": "extension",
    "config": {
        "name": RUN_NAME,
        "process": [{
            "type": "diffusion_trainer",
            "training_folder": f"{WORK}/output",
            "device": "cuda:0",
            "network": {"type": "lora", "linear": 32, "linear_alpha": 32},
            "save": {"dtype": "float16", "save_every": 250, "max_step_saves_to_keep": 6},
            "datasets": [{
                "folder_path": str(DIRS["target"]),
                # 파일명 stem으로 매칭된다. control_1=사람, 2·3=옷 조각.
                "control_path": [str(DIRS["control_1"]), str(DIRS["control_2"]), str(DIRS["control_3"])],
                "caption_ext": "txt",
                # 캡션을 가끔 지워야 모델이 이미지 조건에도 의존하게 된다.
                # 0으로 두면 문장만 외우고 참조 이미지를 덜 본다.
                "caption_dropout_rate": 0.05,
                "resolution": [768, 1024],
            }],
            "train": {
                "batch_size": 1,
                "gradient_accumulation": 4,
                "steps": 1500,
                "cache_text_embeddings": True,
                "train_unet": True,
                "train_text_encoder": False,
                "gradient_checkpointing": True,
                "noise_scheduler": "flowmatch",
                "timestep_type": "weighted",
                "optimizer": "adamw8bit",
                "lr": 1e-4,
                "dtype": "bf16",
            },
            "model": {
                # 증류판(FLUX.2-klein-4B)이 아니라 base에 학습한다. 증류 모델에
                # 직접 LoRA를 얹으면 4-step 스케줄이 깨져 결과가 뭉개진다.
                "name_or_path": "black-forest-labs/FLUX.2-klein-base-4B",
                "arch": "flux2_klein_4b",
                "quantize": True,
                "qtype": "qfloat8",
                "quantize_te": True,
                "qtype_te": "qfloat8",
                "low_vram": True,
            },
            "sample": {
                "sampler": "flowmatch",
                "sample_every": 250,
                "width": 768, "height": 1152,
                "samples": [],  # 아래에서 실제 학습 데이터로 채운다
            },
        }],
    },
    "meta": {"name": RUN_NAME, "version": "1.0"},
}

# 샘플 프롬프트는 학습에 안 쓴 데이터로 만들어야 과적합이 보인다.
holdout = sorted(DIRS["target"].glob("*.jpg"))[-3:]
CONFIG["config"]["process"][0]["sample"]["samples"] = [
    {
        "prompt": (DIRS["target"] / f"{p.stem}.txt").read_text(encoding="utf-8"),
        "ctrl_img_1": str(DIRS["control_1"] / f"{p.stem}.jpg"),
        "ctrl_img_2": str(DIRS["control_2"] / f"{p.stem}.jpg"),
        "ctrl_img_3": str(DIRS["control_3"] / f"{p.stem}.jpg"),
    }
    for p in holdout
]

CONFIG_PATH = f"{WORK}/{RUN_NAME}.yaml"
Path(CONFIG_PATH).write_text(yaml.safe_dump(CONFIG, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(CONFIG_PATH)
""")

add(MD, r"""
## 4. 학습

L4 24GB에서 1500 스텝에 대략 50~70분. **loss가 아니라 sample 이미지를 봐라** —
BFL 가이드도 같은 얘기를 한다. edit-LoRA는 보통 750~1500 스텝 사이에 시각적
정점을 찍고 그 뒤로는 참조 사진의 배경까지 베끼기 시작한다.
""")

add(CODE, r"""
%cd /content/ai-toolkit
!python run.py {CONFIG_PATH}
""")

add(CODE, r"""
# 저장된 체크포인트 목록. 250스텝마다 하나씩 나온다.
import glob
import numpy as np
import matplotlib.pyplot as plt
checkpoints = sorted(glob.glob(f"{WORK}/output/{RUN_NAME}/*.safetensors"))
for path in checkpoints:
    print(Path(path).name)

# 샘플 이미지를 나란히 보고 고른다.
sample_dirs = sorted(glob.glob(f"{WORK}/output/{RUN_NAME}/samples/*.jpg"))
if sample_dirs:
    picks = sample_dirs[::max(1, len(sample_dirs)//6)][:6]
    fig, axes = plt.subplots(1, len(picks), figsize=(4*len(picks), 6))
    for ax, path in zip(np.atleast_1d(axes), picks):
        ax.imshow(Image.open(path)); ax.set_title(Path(path).stem[-18:], fontsize=8); ax.axis("off")
    plt.tight_layout(); plt.show()
""")

add(MD, "## 5. 추론에 얹어서 확인")

add(CODE, r"""
# 눈으로 고른 체크포인트 (loss 최저가 아니라 sample이 가장 나은 것)
LORA_PATH = checkpoints[-1]   # 예: .../wearwell_tryon_v1_000001000.safetensors
print("사용할 LoRA:", Path(LORA_PATH).name)

import torch
from diffusers import Flux2KleinPipeline

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights(LORA_PATH)
pipe.set_progress_bar_config(disable=True)

stem = holdout[0].stem
refs = [Image.open(DIRS[key] / f"{stem}.jpg") for key in ("control_1", "control_2", "control_3")]
prompt = (DIRS["target"] / f"{stem}.txt").read_text(encoding="utf-8")

result = pipe(prompt=prompt, image=refs, height=1152, width=768,
              guidance_scale=1.0, num_inference_steps=8,
              generator=torch.Generator("cuda").manual_seed(42)).images[0]

fig, axes = plt.subplots(1, 3, figsize=(13, 6))
for ax, (img, title) in zip(axes, [(refs[0], "입력(사람)"), (result, "LoRA 결과"),
                                   (Image.open(DIRS["target"] / f"{stem}.jpg"), "정답")]):
    ax.imshow(img); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
""")

add(MD, r"""
## 6. 서비스에 붙이기

체크포인트를 백엔드가 볼 수 있는 곳에 두고 환경변수만 지정하면 된다.
`backend/app.py`가 시작할 때 자동으로 얹고, 실패하면 base 모델로 조용히 폴백한다.

```bash
export FLUX_TRYON_LORA=/content/wearwell-lora/output/wearwell_tryon_v1/wearwell_tryon_v1_000001000.safetensors
export FLUX_TRYON_LORA_SCALE=1.0
```

`/api/health`의 `tryonLora` 필드와 `/api/tryon` 응답의 `engine` 문자열에
LoRA 이름이 찍히므로, 결과 이미지가 base인지 파인튜닝판인지 항상 구분된다.

## 7. 개선폭 측정

발표에 넣을 수치는 여기서 나온다. 같은 케이스·같은 시드로 두 번 돌려 비교한다.

```bash
# base
FLUX_TRYON_LORA= python -m uvicorn app:app --port 8787 &
python scripts/eval_tryon.py --cases eval/cases.json --out eval/before.json

# LoRA
FLUX_TRYON_LORA=/path/to.safetensors python -m uvicorn app:app --port 8787 &
python scripts/eval_tryon.py --cases eval/cases.json --out eval/after.json

python scripts/eval_tryon.py --compare eval/before.json eval/after.json
```

채점 항목은 `layering_accuracy`, `item_accuracy`, `accessory_accuracy`,
`identity_preservation` 네 가지이고, 각각 이 노트북이 고치려던 실패 유형과
1:1로 대응한다.
""")


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == CODE else {}),
            }
            for kind, source in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "L4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "wearwell_tryon_lora.ipynb"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")
