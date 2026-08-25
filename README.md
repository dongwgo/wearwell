# Wearwell

날씨·체형·취향 기반 스타일 추천 프론트엔드와 Colab GPU 모델 API를 분리한 프로젝트입니다.

- 로컬 컴퓨터: `index.html`, CSS, JavaScript, `assets/`
- Google Colab: FASHN VTON v1.5, SDXL Turbo, FastAPI
- Cloudflare Quick Tunnel: 로컬 브라우저에서 Colab API로 보내는 요청만 전달

## 권장 실행 환경

- Google Colab 유료 런타임
- Python 3.10 이상
- NVIDIA L4 24GB 권장

계정에 G4가 표시될 수 있지만 이 프로젝트에는 L4면 충분합니다. A100/H100도 호환되며 T4는 느리고 메모리 여유가 적습니다.

## 모델

- 가상 착장: [FASHN VTON v1.5](https://github.com/fashn-AI/fashn-vton-1.5)
  - Apache-2.0
  - maskless pixel-space VTON
  - tops, bottoms, one-pieces 지원
- 체형 아바타: `stabilityai/sdxl-turbo`
  - FP16, 4-step 생성

두 모델은 동시에 VRAM에 상주하지 않습니다. endpoint가 전환될 때 이전 pipeline을 해제하고 단일 GPU 요청을 직렬 처리합니다.

## 1. Colab 모델 API 실행

[Colab에서 Wearwell L4 notebook 열기](https://colab.research.google.com/github/dongwgo/wearwell/blob/main/colab/wearwell_backend_l4.ipynb)

1. `런타임 → 런타임 유형 변경 → L4 GPU`를 선택합니다.
2. 위에서부터 모든 셀을 실행합니다.
3. 마지막 셀이 다운로드하는 `local-config.js`를 로컬 Wearwell 프로젝트 루트에 저장합니다.

Notebook이 수행하는 작업:

- 저장소 clone/update
- FASHN VTON과 backend 의존성 설치
- FASHN weights 다운로드 및 필수 파일 확인
- FastAPI 실행 후 FASHN·SDXL 실제 load warm-up
- 세션별 API token 생성
- Cloudflare Quick Tunnel 생성
- 로컬 전용 `local-config.js` 다운로드

Colab은 `/api/*`만 제공합니다. HTML, CSS, JavaScript 및 `assets/`는 Tunnel에서 제공하지 않습니다.

## 2. 로컬 프론트엔드 실행

프로젝트 루트에서:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

브라우저에서 다음 주소를 엽니다.

```text
http://127.0.0.1:8000
```

`local-config.js`는 다음 런타임 설정을 담습니다.

```js
window.WEARWELL_CONFIG = {
  API_BASE: "https://example.trycloudflare.com",
  API_TOKEN: "colab-session-token"
};
```

직접 만들려면 `local-config.example.js`를 복사해 값을 채웁니다. `local-config.js`는 `.gitignore`에 포함되어 있으므로 commit하지 않습니다.

Quick Tunnel 주소와 token은 Colab 세션을 재시작할 때마다 바뀝니다. 새 세션에서는 마지막 셀이 생성한 파일로 기존 `local-config.js`를 교체하세요.

## API

- `GET /api/health`
- `POST /api/avatar`
- `POST /api/tryon`
- `POST /api/warmup` — notebook 모델 검증용

POST endpoint는 `Authorization: Bearer <token>`이 필요합니다. CORS는 `localhost`와 `127.0.0.1`에서 실행되는 로컬 프론트엔드만 허용합니다.

환경변수:

- `FASHN_WEIGHTS_DIR`: FASHN weights 경로
- `HF_HOME`: Hugging Face cache 경로
- `AVATAR_MODEL`: 기본값 `stabilityai/sdxl-turbo`
- `FASHN_STEPS`: 기본값 `30`
- `ONEULOUT_GPU`: `1`이면 GPU 추론 활성화
- `WEARWELL_API_TOKEN`: 공개 API의 bearer token

## 개발 검증

```bash
python -m pytest backend/tests -q
node scripts/config-test.mjs
node scripts/smoke-test.mjs
python scripts/validate-notebook.py colab/wearwell_backend_l4.ipynb
```

## 데이터 갱신

```bash
node scripts/fetch-trends.mjs
node scripts/fetch-lookbook.mjs
```

외부 데이터 fetch가 실패하면 마지막 정상 생성물을 유지합니다.
