# Wearwell

날씨·체형·취향 기반 스타일 추천과 GPU 가상 착장을 제공하는 정적 프론트엔드 + FastAPI 프로젝트입니다.

## 권장 실행 환경

- Google Colab 유료 런타임
- Python 3.10 이상(Colab 기본 Python 사용)
- NVIDIA L4 24GB 권장, A100/H100 지원

Colab에는 일반적으로 `G4`라는 GPU 선택지가 없습니다. 요청한 G4급 구성은 Colab의 **L4 GPU**를 기준으로 설계했습니다. T4에서도 서버는 시작할 수 있지만 BF16을 지원하지 않아 느리고 메모리 여유가 적습니다.

## 모델

- 가상 착장: [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5)
  - 2026년 공개, Apache-2.0
  - maskless pixel-space VTON, 약 2GB weights
  - L4/A100에서 BF16 자동 사용
  - tops, bottoms, one-pieces 지원
- 체형 아바타: `stabilityai/sdxl-turbo`
  - FP16, 4-step 생성

두 모델은 동시에 VRAM에 상주하지 않습니다. 아바타와 착장 endpoint가 전환될 때 이전 pipeline을 해제해 L4 24GB 안에서 안정적으로 동작하도록 구성했습니다.

## Colab에서 실행

[Colab에서 Wearwell L4 notebook 열기](https://colab.research.google.com/github/dongwgo/wearwell/blob/main/colab/wearwell_backend_l4.ipynb)

1. 위 링크 또는 `colab/wearwell_backend_l4.ipynb`를 Google Colab에서 엽니다.
2. `런타임 → 런타임 유형 변경 → L4 GPU`를 선택합니다.
3. 위에서부터 모든 셀을 실행합니다.
4. 마지막 셀의 **인증된 Wearwell 열기** 링크(`APP_URL`)를 엽니다. URL fragment의 세션 토큰은 서버로 전송되지 않고 브라우저에만 저장됩니다.

Notebook이 자동으로 수행하는 작업:

- `https://github.com/dongwgo/wearwell.git` clone/update
- Python 3.10+와 GPU 확인
- API 및 FASHN VTON 설치
- FASHN weights 다운로드
- Uvicorn 실행 후 FASHN·SDXL 실제 로드 warm-up 검증
- 세션별 API 토큰 생성
- Cloudflare Quick Tunnel 생성
- 공개 URL을 clone된 `config.js`에 기록
- 실행 가능한 프론트엔드 링크 출력

Quick Tunnel URL은 Colab 세션마다 바뀝니다. 영구 배포 주소가 필요한 경우 Cloudflare Named Tunnel 또는 별도 GPU 서버가 필요합니다.

## 프론트엔드 API URL

`config.js`에서 주소를 지정합니다.

```js
window.WEARWELL_CONFIG = { API_BASE: "https://example.trycloudflare.com" };
```

주소 결정 우선순위:

1. 브라우저 `localStorage["오늘옷-api"]`
2. `config.js`의 검증된 HTTP(S) `API_BASE`
3. HTTP(S)로 제공되면 현재 origin
4. 로컬 파일이면 `http://127.0.0.1:8787`

외부 주소를 임의로 주입할 수 있는 `?api=` query override는 지원하지 않습니다. Colab notebook은 `API_BASE`를 자동으로 갱신하며, 같은 Cloudflare URL에서 프론트와 API를 함께 제공하므로 마지막 인증 링크만 열면 됩니다.

## API

- `GET /api/health`
- `POST /api/avatar`
- `POST /api/tryon`
- `POST /api/warmup` (notebook 모델 검증용)
- `GET /` 정적 프론트엔드

환경변수:

- `FASHN_WEIGHTS_DIR`: FASHN weights 경로
- `HF_HOME`: Hugging Face cache 경로
- `AVATAR_MODEL`: 기본값 `stabilityai/sdxl-turbo`
- `FASHN_STEPS`: 기본값 `30` (`20` 빠름, `30` 균형, `50` 품질)
- `ONEULOUT_GPU`: `1`이면 GPU 추론 활성화
- `WEARWELL_API_TOKEN`: 공개 tunnel의 POST API를 보호하는 bearer token

## 개발 검증

```bash
python -m pytest backend/tests -q
node scripts/config-test.mjs
node scripts/smoke-test.mjs
```

Colab notebook 자체는 다음으로 JSON과 모든 Python 셀의 문법을 검사할 수 있습니다.

```bash
python scripts/validate-notebook.py colab/wearwell_backend_l4.ipynb
```

## 데이터 갱신

```bash
node scripts/fetch-trends.mjs
node scripts/fetch-lookbook.mjs
```

외부 데이터 fetch가 실패하면 마지막 정상 생성물을 유지합니다.
