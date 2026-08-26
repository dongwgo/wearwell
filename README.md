# Wearwell

날씨·체형·취향 기반 스타일 추천 프론트엔드와 GPU 이미지 API를 분리한 프로젝트입니다.

- 로컬 컴퓨터: `index.html`, CSS, JavaScript, `assets/`
- GPU 런타임: FLUX.2 [klein] 4B + Qwen3-VL-8B-Instruct(NF4) + SegFormer, FastAPI
- Cloudflare Quick Tunnel: 로컬 브라우저의 모델 요청만 전달

## 권장 실행 환경

- Google Colab 유료 런타임
- Python 3.10 이상
- NVIDIA L4 24GB 이상 권장

L4에서 기본 설정으로 실행할 수 있고 A100/H100도 호환됩니다. VRAM이 20GB보다 작으면 노트북이 CPU offload를 자동으로 켭니다.

## 모델과 처리 방식

[FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)는 Apache 2.0 공개 가중치이며 텍스트 생성, 이미지 편집, 다중 참조 편집을 하나의 파이프라인에서 지원합니다.

- 아바타: 치수를 문장이 아니라 **체형 실루엣 이미지**로 바꿔 참조 이미지로 넣고, FLUX가 그 실루엣을 따라 768×1152 전신 이미지를 그립니다(`backend/avatar_body.py`). `views`로 정면·측면·후면을 요청할 수 있고, 미리보기에서 가로로 드래그하면 시점이 돌아갑니다. 측면·후면은 완성된 정면을 참조 이미지로 함께 받아 같은 인물을 유지합니다. `SMPLX_MODEL_PATH`를 지정하면 SMPL-X 메시를 목표 치수에 맞춰 피팅하고 `/api/avatar` 응답의 `fit.measurementErrorCm`으로 cm 단위 오차를 함께 돌려줍니다. 가중치가 없으면 인체 계측 비율 실루엣으로 폴백합니다.
- 가상 착장: 아바타와 선택한 의류 사진을 한 번에 전달해 얼굴·체형을 유지하면서 모든 옷을 동시에 반영합니다. 참조 이미지는 피부에서 바깥으로 나가는 **레이어 순서**로 번호가 매겨지고(하의 → 상의 → 아우터 → 신발 → 가방 → 액세서리), 지시문에 몸통 레이어 스택과 가방·액세서리의 착용 지점이 문장으로 들어갑니다. 자세한 근거는 `backend/tryon_prompt.py` 참고.
- 옷 분리: `sayeed99/segformer_b3_clothes`가 전신샷에서 의류를 투명 PNG로 분리합니다. 모델 선정과 품질 임계값 설계는 [docs/segmentation.md](docs/segmentation.md)에 정리했습니다. 설정된 Colab API를 우선 사용하고, 연결 실패 시 같은 FastAPI를 `127.0.0.1:8787`에서 실행 중이면 로컬로 폴백합니다.
- 기본 추론: BF16, 4 steps, guidance 1.0

측정값으로 만든 아바타는 시각적 근사치이며 실제 신체 스캔이나 의류 사이즈 판정 결과가 아닙니다.

## 1. GPU 모델 API 실행

[Colab에서 Wearwell notebook 열기](https://colab.research.google.com/github/dongwgo/wearwell/blob/main/colab/wearwell_backend_l4.ipynb)

1. `런타임 → 런타임 유형 변경`에서 L4, A100 또는 H100을 선택합니다.
2. Colab 왼쪽의 열쇠 모양 **Secrets**에서 이름이 `HF_TOKEN`인 secret을 만들고 Hugging Face 토큰을 값으로 넣은 뒤, 노트북 액세스를 켭니다. 토큰은 Git에 올리지 않습니다.
3. 위에서부터 모든 셀을 실행합니다.
4. 마지막 셀이 내려주는 `local-config.js`를 로컬 Wearwell 프로젝트 루트에 저장합니다.

노트북은 저장소와 의존성을 준비하고, 모델을 미리 적재한 다음 세션별 API 토큰과 Quick Tunnel 주소가 담긴 설정 파일을 생성합니다. GPU 런타임에서는 `/api/*`만 제공하며 프론트엔드 파일은 제공하지 않습니다.

## 2. 로컬 프론트엔드 실행 (처음 하는 사람용)

아래 과정은 **내 컴퓨터에서 Wearwell 웹 화면을 여는 방법**입니다. 코드를 수정할 필요 없이 명령어를 한 줄씩 복사해서 실행하면 됩니다.

### 2-1. Git 설치 확인

먼저 [Git](https://git-scm.com/downloads)을 설치합니다. 설치가 끝나면 Windows에서는 **PowerShell**, macOS에서는 **터미널**을 열고 다음 명령어를 입력합니다.

```bash
git --version
```

`git version 2.x.x`처럼 버전이 표시되면 준비가 된 것입니다. 명령어를 찾을 수 없다는 메시지가 나오면 Git을 설치한 뒤 터미널을 완전히 닫았다가 다시 여세요.

### 2-2. 프로젝트 다운로드

터미널에서 아래 명령어를 **한 줄씩** 실행합니다.

```bash
git clone https://github.com/dongwgo/wearwell.git
cd wearwell
```

- `git clone`은 GitHub에 있는 프로젝트를 내 컴퓨터로 복사합니다.
- `cd wearwell`은 방금 받은 프로젝트 폴더로 이동합니다.
- 두 번째 명령에서 폴더를 찾을 수 없다고 나오면 첫 번째 명령이 오류 없이 끝났는지 확인하세요.

이미 프로젝트를 clone했다면 다시 받을 필요 없이, 터미널에서 기존 `wearwell` 폴더로 이동하면 됩니다.

### 2-3. uv 설치

`uv`는 이 프로젝트를 간단하게 실행할 수 있게 도와주는 프로그램입니다. 사용하는 운영체제에 맞는 명령어 **하나만** 실행하세요.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS 또는 Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

설치가 끝나면 터미널을 완전히 닫았다가 다시 열고, `wearwell` 폴더로 다시 이동합니다. 설치 확인은 다음과 같이 합니다.

```bash
uv --version
```

`uv 0.x.x`처럼 버전이 표시되면 정상입니다.

### 2-4. Python 3.11 설치

프로젝트 폴더(`wearwell`) 안에서 다음 명령어를 실행합니다.

```bash
uv python install 3.11
```

이미 Python 3.11이 있어도 이 명령어를 실행해도 괜찮습니다. 이 프론트엔드는 HTML, CSS, JavaScript로 되어 있으므로 `npm install`은 필요하지 않습니다.

### 2-5. 프론트엔드 서버 실행

같은 터미널에서 다음 명령어를 실행합니다.

```bash
uv run --python 3.11 python -m http.server 8000 --bind 127.0.0.1
```

`Serving HTTP on 127.0.0.1 port 8000`과 비슷한 문구가 나오면 성공입니다. 서버를 사용하는 동안에는 **이 터미널을 닫지 마세요.**

### 2-6. 브라우저에서 열기

Chrome, Edge, Safari 같은 웹 브라우저를 열고 주소창에 다음 주소를 입력합니다.

<http://127.0.0.1:8000>

Wearwell 화면이 나오면 로컬 프론트엔드 실행이 완료된 것입니다. 서버를 종료하려면 실행 중인 터미널을 클릭하고 `Ctrl+C`를 누릅니다.

> 화면은 열리지만 AI 이미지 생성 기능이 동작하지 않는다면 위의 **1. GPU 모델 API 실행**도 완료했는지 확인하세요. Colab에서 받은 `local-config.js` 파일은 `index.html`과 같은 `wearwell` 폴더에 있어야 합니다.

### `local-config.js` 형식

`local-config.js`는 다음 형식입니다.

```js
window.WEARWELL_CONFIG = {
  API_BASE: "https://example.trycloudflare.com",
  API_TOKEN: "session-token",
  LOCAL_API_BASE: "http://127.0.0.1:8787",
  LOCAL_API_TOKEN: "optional-local-token"
};
```

이 파일에는 세션 인증 정보가 있으므로 commit하거나 공유하지 마세요. 런타임을 재시작하면 새 파일로 교체해야 합니다.

옷 사진과 룩북의 색상·아이템 경계·주름·핏·소재·마감·봉제 디테일은 GPU 백엔드의 `Qwen3-VL-8B-Instruct`가 NF4 4비트로 분석합니다. 프론트엔드는 `/api/vlm/*`를 호출하고 결과를 브라우저 IndexedDB(`oneulout-fashion`)에 저장합니다. 백엔드가 잠시 연결되지 않으면 저장된 분석이나 상품명 기반 기본 분석으로 추천 기능을 계속 사용할 수 있습니다.

## API

- `GET /api/health`
- `POST /api/avatar` — `views: ["front","side","back"]`로 시점을 요청 (기본 `["front"]`)
- `POST /api/closet/segment`
- `POST /api/tryon`
- `POST /api/vlm/garment`
- `POST /api/vlm/lookbook`
- `POST /api/vlm/body`
- `POST /api/vlm/tryon-judge`
- `POST /api/warmup`
- `GET /api/dev/segment/models`, `POST /api/dev/segment/compare` (개발용 Seg Lab)

모델 추론 endpoint는 `Authorization: Bearer <token>`이 필요합니다. CORS는 `localhost`와 `127.0.0.1`에서 실행되는 프론트엔드만 허용합니다.

환경변수:

- `IMAGE_MODEL`: 기본값 `black-forest-labs/FLUX.2-klein-4B`
- `VLM_MODEL`: 기본값 `Qwen/Qwen3-VL-8B-Instruct`
- `VLM_LOAD_IN_4BIT`: L4에서는 `1` 권장
- `VLM_MAX_PIXELS`: VLM 입력 이미지 최대 픽셀 수, 기본값 `1048576`
- `GPU_QUEUE_TIMEOUT`: 앞선 GPU 작업을 기다리는 최대 시간(초), 기본값 `300`
- `RATE_LIMIT_PER_MINUTE`: 세션당 분당 POST 요청 제한, 기본값 `60`
- `IMAGE_WIDTH`, `IMAGE_HEIGHT`: 기본값 `768`, `1152`
- `FLUX_STEPS`: 기본값 `4`
- `FLUX_TRYON_STEPS`: 가상 착장 전용 스텝 수, 기본값 `8`. 참조가 여러 장인 편집은 4스텝에서 레이어가 뭉개집니다
- `FLUX_TRYON_LORA`: 파인튜닝한 try-on edit-LoRA(`.safetensors`) 경로. 비어 있으면 base 모델로 동작
- `FLUX_TRYON_LORA_SCALE`: 기본값 `1.0`
- `MAX_TRYON_GARMENTS`: 한 번에 입힐 수 있는 아이템 수, 기본값 `6`
- `AVATAR_BODY_REFERENCE`: `1`(기본)이면 체형 실루엣을 참조로 사용, `0`이면 예전 텍스트 전용 경로(A/B 비교용)
- `SMPLX_MODEL_PATH`: SMPL-X 모델 폴더 경로. 지정하면 메시 기반 체형 피팅 활성화 (측면·후면은 메시를 실제로 회전시켜 렌더)
- `FLUX_GUIDANCE`: 기본값 `1.0`
- `FLUX_CPU_OFFLOAD`: `1`이면 일부 모델을 CPU RAM으로 이동
- `HF_HOME`: Hugging Face cache 경로
- `HF_TOKEN`: Hugging Face 인증 토큰. FLUX, Qwen, SegFormer 다운로드에 사용하며 코드나 Git에 직접 적지 않음
- `ONEULOUT_GPU`: `1`이면 GPU 추론 활성화
- `SEGMENTATION_DEVICE`: `auto`(기본), `cuda`, `cpu` 중 하나. `auto`는 GPU가 있으면 GPU를 사용
- `WEARWELL_DEV_TOOLS`: `1`이면 개발용 Seg Lab API를 노출. 기본값은 `0`
- `SEGMENT_MODEL_CACHE_SIZE`: 메모리에 유지할 세그멘테이션 모델 수. 기본값은 `3`
- `WEARWELL_API_TOKEN`: API bearer token
- `GPU_CONCURRENCY`: 동시에 처리할 GPU 요청 및 FLUX 파이프라인 수 (기본 `2`; 96GB VRAM 권장값)

## Seg Lab — 옷 분리 모델 비교

전신사진 한 장을 여러 모델에 실행해 크롭, 오버레이, 품질 필터 탈락 이유와 모델 간 IoU를 비교하는 개발 도구입니다. `WEARWELL_DEV_TOOLS=1`로 backend를 시작하고 로컬 프론트엔드에 `?dev=1`을 붙이면 상단에 **Seg Lab** 탭이 표시됩니다.

모델 목록은 [`backend/segment_models.py`](backend/segment_models.py)에서 관리합니다.

| key | 모델 | 라벨 체계 | 가중치 | 특징 |
| --- | --- | --- | --- | --- |
| `b2_clothes` | `mattmdjaga/segformer_b2_clothes` | ATR 18 | 110MB | 가장 가볍고 빠름 |
| `b3_clothes` | `sayeed99/segformer_b3_clothes` | ATR 18 | 189MB | **프로덕션 기본값.** B2와 라벨이 같고 인코더만 큼 |
| `b3_fashion` | `sayeed99/segformer-b3-fashion` | Fashionpedia 46 | 189MB | 유일하게 상의와 아우터를 구분 |
| `b5_human_parsing` | `matei-dorian/segformer-b5-finetuned-human-parsing` | ATR 18 | 339MB | 가장 큰 ATR 모델. 기본 선택에는 빠져 있음 |

ATR 라벨에는 아우터가 없어 코트도 상의로 분류될 수 있습니다. Fashionpedia 모델은 아우터를 구분하지만 소매·카라를 별도 부속 라벨로 분리합니다.

```bash
cd backend
WEARWELL_DEV_TOOLS=1 WEARWELL_API_TOKEN=wearwell-local-dev uvicorn app:app --host 127.0.0.1 --port 8787
```

`local-config.js`의 `LOCAL_API_TOKEN`을 같은 토큰으로 설정하고 <http://127.0.0.1:8000/?dev=1>을 엽니다. 처음 실행하는 모델은 가중치를 다운로드하므로 시간이 걸릴 수 있습니다.

## 개발 검증

```bash
python -m pytest backend/tests -q
node scripts/config-test.mjs
node scripts/smoke-test.mjs
node scripts/avatar-view-test.mjs
python scripts/validate-notebook.py colab/wearwell_backend_l4.ipynb
```

브라우저 테스트는 Chrome 실행 파일을 찾습니다. Windows 기본 경로가 아니면 `CHROME_PATH`로 지정하세요.

`avatar-view-test.mjs`는 두 가지 회귀를 막습니다 — 전신 이미지가 잘리지 않는지(`object-fit: contain`)와 시점 여러 개일 때 드래그 회전이 도는지. 발이 잘리던 문제는 CSS `object-fit: cover`가 768×1152 이미지를 390×510 상자에 폭 기준으로 확대하면서 위아래를 잘라낸 것이 원인이었고, 생성 프롬프트에도 머리 위·발밑 여백 요구를 추가했습니다.

## 착장 품질 측정

가상 착장에는 정답 이미지가 없어 SSIM/LPIPS로는 "상의가 아우터 안쪽인가"를 잴 수 없습니다. `scripts/eval_tryon.py`는 백엔드의 Qwen3-VL을 심판으로 써서 네 항목을 이진 채점합니다 — 레이어 정확도, 아이템 일치도, 가방·액세서리 배치, 인물 보존.

```bash
python scripts/eval_tryon.py --cases eval/cases.json --out eval/before.json --api $API --token $TOKEN
# 개선/LoRA 적용 후 다시 돌리고
python scripts/eval_tryon.py --cases eval/cases.json --out eval/after.json --api $API --token $TOKEN
python scripts/eval_tryon.py --compare eval/before.json eval/after.json
```

## 착장 모델 파인튜닝

`colab/wearwell_tryon_lora.ipynb`가 `FLUX.2-klein-base-4B`에 edit-LoRA를 학습시킵니다(L4 24GB, 약 1시간). 학습 캡션은 `build_tryon_prompt()`로 만들어 추론 지시문과 형식을 맞춥니다 — 이 둘이 다르면 학습이 수렴해도 서비스에 전이되지 않습니다. 학습 데이터는 저장소의 룩북 이미지에서 cloth-agnostic 사람 + 옷 조각 쌍으로 자동 생성합니다.

[Colab에서 파인튜닝 notebook 열기](https://colab.research.google.com/github/dongwgo/wearwell/blob/main/colab/wearwell_tryon_lora.ipynb)

## 데이터 갱신

옷장 데이터는 무신사 남성 랭킹 상품 200개로 구성되며 상품 ID, 카테고리, 세부 분류, 색상, 원문 URL과 이미지 출처를 함께 보관합니다.

```bash
node scripts/fetch-korean-influencer-lookbook.mjs
node scripts/fetch-lookbook.mjs
```

외부 데이터 수집이 실패하면 마지막 정상 생성물을 유지합니다.
