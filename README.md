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
- 옷 분리: `mattmdjaga/segformer_b2_clothes`가 전신샷에서 의류를 투명 PNG로 분리합니다. 설정된 Colab API를 우선 사용하고, 연결 실패 시 같은 FastAPI를 `127.0.0.1:8787`에서 실행 중이면 로컬로 폴백합니다.
- 기본 추론: BF16, 4 steps, guidance 1.0

측정값으로 만든 아바타는 시각적 근사치이며 실제 신체 스캔이나 의류 사이즈 판정 결과가 아닙니다.

## 1. GPU 모델 API 실행

[Colab에서 Wearwell notebook 열기](https://colab.research.google.com/github/dongwgo/wearwell/blob/main/colab/wearwell_backend_l4.ipynb)

1. `런타임 → 런타임 유형 변경`에서 L4, A100 또는 H100을 선택합니다.
2. 위에서부터 모든 셀을 실행합니다.
3. 마지막 셀이 내려주는 `local-config.js`를 로컬 Wearwell 프로젝트 루트에 저장합니다.

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
- `POST /api/avatar/from-photo` — 전신사진을 같은 인물의 스튜디오 아바타로 변환
- `POST /api/closet/segment`
- `POST /api/tryon` — `views`, `avatarViews`로 시점별 착장 생성
- `POST /api/vlm/garment`
- `POST /api/vlm/lookbook`
- `POST /api/vlm/body`
- `POST /api/vlm/tryon-judge`
- `POST /api/warmup`

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
- `ONEULOUT_GPU`: `1`이면 GPU 추론 활성화
- `SEGMENTATION_DEVICE`: `auto`(기본), `cuda`, `cpu` 중 하나. `auto`는 GPU가 있으면 GPU를 사용
- `WEARWELL_API_TOKEN`: API bearer token

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

### 비율 왜곡에 대해

발 잘림을 막으려고 넣은 `pad_for_full_body()`가 **세로로만** 여백을 붙인 뒤 고정 크기(768×1152)로 되돌리고 있었습니다. 계산해 보면 세로가 0.855배로 눌려서 사람이 **17% 넓어 보입니다** — 아바타가 찌그러져 보이던 원인입니다. 세 곳을 고쳤습니다.

1. **패딩이 비율을 지킵니다.** 세로에 붙인 만큼 가로에도 같은 비율로 붙입니다.
2. **생성 크기가 원본 비율을 따릅니다.** `fit_generation_size()`가 사람 사진의 비율에 맞는 16의 배수 크기를 계산합니다. 3:4 휴대폰 사진을 2:3으로 뽑으면 사람과 배경이 세로로 늘어납니다(test4의 증상). 한 요청 안의 모든 시점은 같은 크기를 씁니다 — 드래그로 돌릴 때 화면이 튀지 않아야 하니까요.
3. **화면 상자가 이미지 비율을 따릅니다.** `--stage-aspect`를 이미지의 `naturalWidth/Height`에서 채웁니다.

패딩은 이제 **조건부**입니다. 맨 아랫줄이 배경 한 가지 색으로 균일하면(= 인물이 프레임 안에서 끝났으면) 붙이지 않습니다. 실제 사진에서는 덧댄 띠가 눈에 보이는 회색 줄로 남기 때문입니다.

### 측면·후면에서 옷이 바뀌던 문제

참조 **순서**가 원인이었습니다. 회전 지시문이 체형 가이드(회색 마네킹)를 참조 1번에, 완성된 정면을 2번에 두고 있었는데, FLUX.2는 1번 참조를 가장 강하게 따라갑니다. "옷 없는 회색 인체"가 착장을 밀어내서 후면에서 패딩이 통째로 사라지고 토트백이 백팩으로 바뀌었습니다.

- **완성된 정면이 참조 1번**이 됩니다.
- **옷 사진을 다시 붙입니다.** 낭비 같지만, 회전 중에 아이템이 다른 물건으로 바뀌는 걸 막는 유일한 근거입니다. 각 아이템의 착용 지점 문장도 함께 다시 줍니다.
- **마네킹 가이드는 뺐습니다.** 체형 정보는 이미 정면 사진 안에 다 들어 있습니다.

### 전신사진의 두 가지 모드

전신사진으로 시작하면 기본은 **사진에 직접 입히기**입니다 — 배경과 포즈가 그대로라 현실감이 좋습니다. 착장 결과 창의 `✦ 아바타로 보기`를 누르면 `/api/avatar/from-photo`가 같은 인물의 깨끗한 스튜디오 컷을 만들고 거기에 다시 입힙니다. 아바타화는 한 번만 하고 이후 옷 조합이 바뀌어도 재사용합니다. 버튼은 전신사진으로 시작한 경우에만 뜹니다 — 치수로 만든 아바타는 이미 스튜디오 컷이라 바꿀 것이 없습니다.

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
