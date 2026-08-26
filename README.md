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
- 가상 착장: 아바타와 선택한 의류 사진을 한 번에 전달해 얼굴·체형을 유지하면서 모든 옷을 동시에 반영합니다. 참조 이미지는 피부에서 바깥으로 나가는 **레이어 순서**로 번호가 매겨지고, 지시문에 몸통 레이어 스택과 가방·액세서리의 착용 지점이 문장으로 들어갑니다.

  착용 순서는 카테고리가 아니라 **아이템 종류**로 정해집니다 — 양말과 모자는 둘 다 '액세서리'지만 양말은 신발보다 먼저 신고 모자는 맨 마지막에 씁니다. **속옷(5)** → 하의(20) → 상의(30) → 벨트·넥타이·목걸이(35) → 아우터(40) → 시계·반지(44) → **양말(45) → 신발(50)** → 가방(60) → 목도리·장갑(65) → 모자·안경(72) 순입니다.

  속옷은 브라가 '상의', 팬티가 '하의'로 들어오기 때문에 카테고리로는 잡을 수 없어 **이름**으로 판정합니다. 이름 매칭이 카테고리를 가로지르므로 오탐이 위험해서, `브라운`·`슬립온`·`브레이슬릿` 같은 닮은꼴 낱말을 먼저 걷어낸 뒤 판정합니다 — 통째로 거부하면 '브라운 드로즈'까지 놓치기 때문입니다. 속옷은 겉옷과 자리를 다투지 않는 별도 슬롯(`underwear-legs` 등)을 쓰므로 팬티와 바지가 함께 살아남습니다. 안경·시계·귀걸이처럼 화면에서 작아 통째로 빠지기 쉬운 아이템은 지시문 끝에서 이름을 한 번 더 불러 줍니다. 자세한 근거는 `backend/tryon_prompt.py` 참고.

  `views`로 착장 결과도 정면·측면·후면을 만들 수 있습니다. 측면·후면은 옷 사진을 다시 넣지 않고 **완성된 정면 결과**를 참조로 돌립니다 — 착장이 이미 조립돼 있으니 참조 수도 줄고 옷 일관성도 낫습니다.
- 옷 분리: `sayeed99/segformer_b3_clothes`가 전신샷에서 의류를 투명 PNG로 분리합니다. 모델 선정과 품질 임계값 설계는 [docs/segmentation.md](docs/segmentation.md)에 정리했습니다. 설정된 Colab API를 우선 사용하고, 연결 실패 시 같은 FastAPI를 `127.0.0.1:8787`에서 실행 중이면 로컬로 폴백합니다.
- 기본 추론: BF16, 4 steps, guidance 1.0

측정값으로 만든 아바타는 시각적 근사치이며 실제 신체 스캔이나 의류 사이즈 판정 결과가 아닙니다.

## 1. GPU 모델 API 실행

[Colab에서 Wearwell notebook 열기](https://colab.research.google.com/github/dongwgo/wearwell/blob/main/colab/wearwell_backend_l4.ipynb)

1. `런타임 → 런타임 유형 변경`에서 L4, A100 또는 H100을 선택합니다.
2. Colab 왼쪽의 열쇠 모양 **Secrets**에서 이름이 `HF_TOKEN`인 secret을 만들고 Hugging Face 토큰을 값으로 넣은 뒤, 노트북 액세스를 켭니다. 토큰은 Git에 올리지 않습니다.
3. 위에서부터 모든 셀을 실행합니다.
4. 마지막 셀이 내려주는 `local-config.js`를 로컬 Wearwell 프로젝트 루트에 저장합니다.

노트북은 저장소와 의존성을 준비하고 모델을 미리 적재한 다음, backend와 현재 frontend에 각각 Cloudflare Quick Tunnel 주소를 발급합니다. **공개 프론트엔드 실행** 셀에 출력되는 `FRONTEND_URL`을 공유하면 다른 사람도 바로 접속할 수 있습니다. 링크를 아는 사람은 해당 Colab 세션의 GPU 기능을 사용할 수 있으므로 데모가 끝나면 런타임을 종료하세요.

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
- `POST /api/tryon` — `views`, `avatarViews`로 시점별 착장 생성
- `POST /api/vlm/garment`
- `POST /api/vlm/lookbook`
- `POST /api/vlm/body`
- `POST /api/vlm/tryon-judge`
- `POST /api/warmup`
- `GET /api/dev/segment/models`, `POST /api/dev/segment/compare` (아래 Seg Lab 참고)
- `GET /api/closet/models`, `POST /api/closet/refine` (아래 Refine Lab 참고)

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
- `WEARWELL_DEV_TOOLS`: `1`이면 Seg Lab endpoint(`/api/dev/*`)와 Seg Lab·Refine Lab 탭을 노출. 공개 Colab에서는 기본 `0`
- `REFINE_WIDTH`, `REFINE_HEIGHT`: 옷 상품컷 생성 해상도, 기본값 `768`, `768`(정사각)
- `MAX_REFINE_ITEMS`: Refine Lab 요청 한 번에 재생성할 옷의 최대 개수, 기본값 `4`

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

# 2) 정적 프론트엔드
python -m http.server 8000 --bind 127.0.0.1
```

`WEARWELL_DEV_TOOLS=1`을 빼면 `/api/dev/*`가 404가 되어 두 랩 탭의 모델 목록이 비어 있습니다.

`local-config.js`의 `LOCAL_API_TOKEN`을 위 `WEARWELL_API_TOKEN`과 같게 맞춘 뒤
`http://127.0.0.1:8000`을 열면 상단에 **Seg Lab** 탭이 나옵니다. 탭은 `localhost`/`127.0.0.1`
에서 열었거나 `?dev=1`을 붙였을 때만 보입니다.

비교 결과에는 통과한 아이템뿐 아니라 **품질 필터에 걸러진 후보와 그 이유**(면적·채움·확신도
중 어느 기준을 못 넘었는지), 모델이 예측한 원본 라벨 분포, 카테고리별 오버레이,
모델 쌍별 IoU가 함께 나옵니다. 오버레이 색은 모델이 아니라 카테고리로 고정되어 있어
나란히 놓고 눈으로 비교할 수 있습니다.

세그멘테이션 자체는 원본 해상도로 돌리고, 응답에 싣는 이미지만 줄입니다(오버레이 860px,
크롭 420px). 처음 쓰는 모델은 가중치를 내려받느라 오래 걸리며, 적재 시간은 추론 시간과
구분해서 표시합니다.

### 테스트

```bash
cd backend && pytest tests -q          # 백엔드 라우트와 레지스트리
node scripts/seg-lab-test.mjs          # 두 서버가 뜬 상태에서 탭 전체 E2E
```

## Refine Lab — 잘라낸 옷을 옷장 이미지로 (개발용)

세그멘테이션 결과를 그대로 옷장에 넣기 어렵다는 문제를 다루는 개발 탭입니다.
argmax는 픽셀마다 라벨을 하나만 주기 때문에 **팔이 몸통을 가리면 상의 마스크에 구멍이
뚫리고, 가려진 옷은 조각으로 끊어집니다.** 이 탭은 전신샷 한 장을 다음 순서로 돌리고
각 단계의 중간 산출물을 나란히 보여줍니다.

```
전신샷
  → 세그멘테이션 (Seg Lab과 같은 모델 레지스트리)
  → 결함 진단      구멍·조각을 세고 빨강/파랑으로 칠한다
  → 마스크 보수    닫기 → 구멍 메우기 → 부스러기 조각 정리
  → 정규화        메운 자리를 옷 대표색으로 칠하고 흰 배경 정사각으로
  → FLUX 재생성    가려졌던 형태와 질감을 채워 상품컷으로
```

**보수와 재생성의 역할이 다릅니다.** 형태학 연산은 없는 픽셀을 만들지 못하므로
팔에 가려졌던 자리는 여전히 비어 있습니다. 보수 단계는 실루엣을 온전하게 만들어
생성 모델에게 좋은 입력을 주는 것이 목적이고, 실제로 그 자리를 그리는 것은 FLUX입니다.
그래서 프롬프트가 색·패턴·재단을 그대로 유지하라고 반복해서 못박습니다 —
자유롭게 그리라고 하면 다른 옷이 나옵니다.

메운 자리를 옷의 중앙값 색으로 칠하는 이유도 같습니다. 알파로만 채우면 그 자리의
팔·머리카락·뒷배경 픽셀이 옷 안에 남고, 생성 모델이 그 살색을 옷의 일부로 읽습니다.

관련 코드: [`backend/refine_service.py`](backend/refine_service.py)(진단·보수·정규화, torch 불필요),
`FluxImageEngine.refine_garment`([`backend/app.py`](backend/app.py)), 탭: [`refine-lab.js`](refine-lab.js)

### 어느 백엔드를 쓰나

5단계(FLUX)는 GPU가 있어야 하므로 **`local-config.js`의 `API_BASE`(Colab 터널)를 기본값으로 씁니다.**
탭 상단에서 `Colab GPU` / `로컬 백엔드`를 눌러 바꿀 수 있고, 주소마다 토큰이 다르므로
(`API_TOKEN` vs `LOCAL_API_TOKEN`) 선택한 주소에 맞는 토큰을 자동으로 실어 보냅니다.
Seg Lab은 세그멘테이션만 보므로 기존대로 로컬이 기본이며, 두 탭은 주소를 따로 기억합니다.

Colab에서는 `WEARWELL_DEV_TOOLS=0`이어도 `/api/closet/models`와
`/api/closet/refine`이 동작합니다. 공개 프론트엔드에서 Refine Lab 탭을 보려면 주소 끝에
`?dev=1`을 붙이세요. `/api/dev/*`의 모델 비교 기능만 `WEARWELL_DEV_TOOLS=1`이 필요합니다.
노트북은 저장소의 기본 브랜치를 `git pull`하므로, 이 기능이 main에 반영된 이후의 코드를
받아야 합니다.

### 조작

- **마스크 보수 단계**를 하나씩 꺼 보면서 각 단계가 무슨 일을 하는지 확인합니다
- **닫기 커널**은 크롭 짧은 변에 대한 비율입니다(기본 1.2%). 키우면 옷 모양이 뭉개집니다
- **조각 기준**은 가장 큰 조각 대비 비율입니다(기본 8%). "가장 큰 것만 남기기"가 아닌
  이유는 신발 한 켤레가 정상적으로 두 조각이기 때문입니다
- **FLUX로 다시 그리기**를 끄면 4단계까지만 돌아 GPU 없이도 사용할 수 있습니다

한 요청에서 옷 한 벌마다 FLUX를 한 번씩 돌리므로 `MAX_REFINE_ITEMS`(기본 4)까지만
처리하고, 카테고리 칩으로 대상을 좁힐 수 있습니다.

### 테스트

```bash
node scripts/refine-lab-test.mjs                       # 1~4단계 (GPU 불필요)
REFINE_LAB_GENERATE=1 node scripts/refine-lab-test.mjs # 5단계까지
```

## 개발 검증

```bash
python -m pytest backend/tests -q
node scripts/config-test.mjs
node scripts/smoke-test.mjs
node scripts/avatar-view-test.mjs
python scripts/validate-notebook.py colab/wearwell_backend_l4.ipynb
```

### 착용 순서 규칙이 두 곳에 있는 이유

순서·자리 규칙은 `backend/tryon_prompt.py`가 최종 판단을 하지만, 프론트엔드도 6벌로 자르기 전에 무엇을 남길지 같은 기준으로 골라야 합니다 — 그러지 않으면 상의를 두 벌 고른 탓에 신발이 백엔드에 도달하지도 못하고 밀려납니다. 두 구현이 갈라지지 않도록 `eval/ordering-cases.json` 한 파일로 양쪽을 묶어 뒀습니다. 파이썬은 `test_tryon_prompt.py`가, 자바스크립트는 `scripts/avatar-view-test.mjs`가 같은 픽스처를 읽고 같은 결과를 요구합니다.

브라우저 테스트는 Chrome 실행 파일을 찾습니다. Windows 기본 경로가 아니면 `CHROME_PATH`로 지정하세요.

`avatar-view-test.mjs`는 아바타 미리보기와 착장 결과 양쪽에서 드래그 회전이 도는지, 그리고 전신 이미지가 잘리지 않는지를 검사합니다.

### 안정성 관련 수정

착장이 드물게 반영되지 않던 원인 몇 가지를 정리했습니다.

- **투명 PNG의 배경 부활** (가장 큰 원인): `segment_service`가 만드는 옷 조각은 알파 채널로만 옷 영역을 표시하고 RGB 채널에는 원본 사진이 그대로 남아 있습니다. `decode_image`가 `.convert("RGB")`를 바로 부르면서 알파만 버리고 그 아래 사진이 되살아나, 지시문에는 "옷만 남기고 원래 모델과 배경은 지워라"라고 써 놓고 실제로는 그 배경을 다시 넣어 주고 있었습니다. `flatten_transparency()`가 투명 영역을 흰색으로 메웁니다.
- **응답 경쟁**: 연달아 다른 조합을 고르면 먼저 보낸 요청이 늦게 도착해 최신 결과를 덮어썼습니다. 요청 번호로 최신 것만 화면에 올립니다.
- **조용한 아이템 탈락**: 같은 부위를 두 벌 고르면 안쪽 한 벌만 남는데 아무 설명이 없었습니다. `/api/tryon`이 `droppedGarments`를 돌려주고 화면이 이를 알립니다.
- **아바타 생성 실패 후 크래시**: `avatarImage`가 null인 채로 `.slice()`를 불러 터졌습니다.
- **부분 실패의 전면 확대**: 옷 사진 한 장을 못 읽으면 `Promise.all`이 거부돼 착장 전체가 실패했습니다. 이제 읽힌 것만 입히고 몇 장을 못 읽었는지 알립니다. 측면 아바타가 깨져도 정면 착장은 살아남습니다.
- **무한 대기**: GPU 호출에 타임아웃이 없어 응답이 오지 않으면 영원히 매달렸습니다(240초).
- **캐시 무한 증가**: 착장 결과(시점당 최대 3장)를 무한정 쌓았습니다. 12개로 제한합니다.
- **파이프라인 이중 로딩**: `_load()`가 락 없이 `self.pipe`를 검사했습니다. 지금은 모든 호출부가 `INFERENCE_GATE`를 먼저 잡아 우연히 안전하지만, 그 규약이 깨지면 VRAM이 터집니다.

### 발 잘림에 대해

원인이 두 개였고 둘 다 막아 뒀습니다.

1. **화면**: `.tryon-stage`가 고정 높이(510px)에 `object-fit: cover`여서, 768×1152 이미지를 폭 기준으로 확대하며 위아래를 잘라냈습니다. 상자에 `aspect-ratio: 2 / 3`을 줘서 이미지 비율과 맞췄습니다 — 잘림도 레터박스도 없습니다.
2. **생성**: FLUX.2는 참조 이미지의 **구도**를 강하게 따라갑니다. 넣어준 사람 사진이 종아리에서 끝나면 결과도 거기서 끝납니다. 문장으로 부탁하는 대신 `pad_for_full_body()`가 참조 이미지 위아래에 배경색 여백을 실제로 덧대서 "여기까지가 화면"이라고 보여줍니다. 배경색은 가장자리 띠의 **바깥쪽 열**에서 뽑습니다 — 아래쪽 띠 가운데에는 다리와 신발이 있어서 통째로 평균 내면 배경이 아니라 살색이 나옵니다.

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
