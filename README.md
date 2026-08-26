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

- 아바타: 성별, 키, 몸무게, 체형과 선택 입력한 상세 치수를 포함해 기본 FLUX 백엔드가 768×1152 전신 이미지를 생성합니다.
- 가상 착장: 아바타와 선택한 의류 사진을 한 번에 전달해 얼굴·체형을 유지하면서 모든 옷을 동시에 반영합니다.
- 옷 분리: `mattmdjaga/segformer_b2_clothes`가 전신샷에서 의류를 투명 PNG로 분리합니다. 설정된 Colab API를 우선 사용하고, 연결 실패 시 같은 FastAPI를 `127.0.0.1:8787`에서 실행 중이면 로컬로 폴백합니다.
- 기본 추론: BF16, 4 steps, guidance 1.0

측정값으로 만든 아바타는 시각적 근사치이며 실제 신체 스캔이나 의류 사이즈 판정 결과가 아닙니다.

## 1. GPU 모델 API 실행

[Colab에서 Wearwell notebook 열기](https://colab.research.google.com/github/dongwgo/wearwell/blob/main/colab/wearwell_backend_l4.ipynb)

1. `런타임 → 런타임 유형 변경`에서 L4, A100 또는 H100을 선택합니다.
2. 위에서부터 모든 셀을 실행합니다.
3. 마지막 셀이 내려주는 `local-config.js`를 로컬 Wearwell 프로젝트 루트에 저장합니다.

노트북은 저장소와 의존성을 준비하고, 모델을 미리 적재한 다음 세션별 API 토큰과 Quick Tunnel 주소가 담긴 설정 파일을 생성합니다. GPU 런타임에서는 `/api/*`만 제공하며 프론트엔드 파일은 제공하지 않습니다.

## 2. 로컬 프론트엔드 실행

프로젝트 루트에서 다음 명령을 실행합니다.

```bash
uv run --python 3.11 python -m http.server 8000 --bind 127.0.0.1
```

브라우저에서 `http://127.0.0.1:8000`을 엽니다. `local-config.js`는 다음 형식입니다.

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
- `POST /api/avatar`
- `POST /api/closet/segment`
- `POST /api/tryon`
- `POST /api/vlm/garment`
- `POST /api/vlm/lookbook`
- `POST /api/vlm/body`
- `POST /api/warmup`
- `GET /api/dev/segment/models`, `POST /api/dev/segment/compare` (아래 Seg Lab 참고)

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
- `FLUX_GUIDANCE`: 기본값 `1.0`
- `FLUX_CPU_OFFLOAD`: `1`이면 일부 모델을 CPU RAM으로 이동
- `HF_HOME`: Hugging Face cache 경로
- `ONEULOUT_GPU`: `1`이면 GPU 추론 활성화
- `SEGMENTATION_DEVICE`: `auto`(기본), `cuda`, `cpu` 중 하나. `auto`는 GPU가 있으면 GPU를 사용
- `WEARWELL_API_TOKEN`: API bearer token
- `WEARWELL_DEV_TOOLS`: `1`(기본)이면 Seg Lab endpoint를 노출. 공개 터널에서는 `0` 권장
- `SEGMENT_MODEL_CACHE_SIZE`: 동시에 메모리에 올려둘 세그멘테이션 모델 수, 기본값 `3`

## Seg Lab — 옷 분리 모델 비교 (개발용)

전신샷 한 장을 여러 세그멘테이션 모델에 돌려 결과를 나란히 비교하는 개발 탭입니다.
모델을 바꿔가며 옷 분리 성능을 보기 위한 것으로, 일반 사용자 화면에는 나오지 않습니다.

등록된 모델은 [`backend/segment_models.py`](backend/segment_models.py)에 있습니다.

| key | 모델 | 라벨 체계 | 가중치 | 특징 |
| --- | --- | --- | --- | --- |
| `b2_clothes` | `mattmdjaga/segformer_b2_clothes` | ATR 18 | 110MB | 프로덕션 기본값. 가장 가볍고 빠름 |
| `b3_clothes` | `sayeed99/segformer_b3_clothes` | ATR 18 | 189MB | B2와 라벨이 같고 인코더만 큼. 용량 효과만 비교 |
| `b3_fashion` | `sayeed99/segformer-b3-fashion` | Fashionpedia 46 | 189MB | 유일하게 상의와 아우터를 구분 |
| `b5_human_parsing` | `matei-dorian/segformer-b5-finetuned-human-parsing` | ATR 18 | 339MB | 가장 큰 ATR 모델. 기본 선택에는 빠져 있음 |

ATR 라벨에는 아우터 클래스가 없어 코트도 `Upper-clothes`로 나옵니다. 아우터 컬럼이
`b3_fashion`에만 뜨는 것은 버그가 아니라 라벨 체계 차이입니다. 반대로 Fashionpedia는
소매·카라를 별도 클래스로 떼어내 본체 마스크에 구멍을 냅니다 — 결과 카드의
"모델이 예측한 원본 라벨"에서 그 비율을 확인할 수 있습니다.

### 실행

```bash
# 1) 백엔드 (GPU가 없으면 SEGMENTATION_DEVICE=cpu로 자동 폴백)
cd backend
WEARWELL_API_TOKEN=wearwell-local-dev uvicorn app:app --host 127.0.0.1 --port 8787

# 2) 정적 프론트엔드
python -m http.server 8000 --bind 127.0.0.1
```

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

## 개발 검증

```bash
python -m pytest backend/tests -q
node scripts/config-test.mjs
node scripts/smoke-test.mjs
python scripts/validate-notebook.py colab/wearwell_backend_l4.ipynb
```

## 데이터 갱신

옷장 데이터는 무신사 남성 랭킹 상품 200개로 구성되며 상품 ID, 카테고리, 세부 분류, 색상, 원문 URL과 이미지 출처를 함께 보관합니다.

```bash
node scripts/fetch-korean-influencer-lookbook.mjs
node scripts/fetch-lookbook.mjs
```

외부 데이터 수집이 실패하면 마지막 정상 생성물을 유지합니다.
