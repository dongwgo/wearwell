# 세 모델이 하는 일

Wearwell은 GPU 위에 모델 세 개를 올려 둔다. 셋은 서로를 호출하지 않는다 — **서로 다른
종류의 질문에 답하고**, 앱이 그 답을 이어 붙인다.

| | 답하는 질문 | 입력 → 출력 | 못 하는 것 |
|---|---|---|---|
| **SegFormer** | 이 픽셀은 **어느 옷인가** | 사진 → 픽셀별 라벨(마스크) | 소재·핏을 모른다. 라벨 18개가 아는 전부 |
| **Qwen3-VL** | 이건 **무슨 옷인가** | 사진 → 한국어 JSON | 픽셀 경계를 못 준다. bbox까지가 한계 |
| **FLUX.2** | **이렇게 보일 것이다** | 프롬프트 + 참조 이미지 → 새 이미지 | 아무것도 판정하지 않는다. 그리기만 한다 |

한 문장으로: **SegFormer가 오려내고, Qwen이 설명하고, FLUX가 그린다.**

---

## SegFormer — 어디에 있나

`sayeed99/segformer_b3_clothes` (ATR 18 라벨, 189MB) · [segment_service.py](../backend/segment_service.py)

전신샷의 **픽셀마다** 라벨 하나를 배정한다(argmax). 그걸 옷장 카테고리로 묶어 마스크를
만들고, 투명 PNG로 잘라낸다.

- **프로덕션**: `/api/closet/segment` — 전신샷 한 장에서 옷을 카테고리별로 분리해 옷장에 넣는다
- **Refine Lab**: `/api/closet/refine` — 위 마스크를 진단·보수한 뒤 FLUX에 넘긴다
- **Seg Lab**: `/api/dev/segment/compare` — 등록된 모델 4종 중 골라(한 번에 최대 4개)
  같은 사진에 나란히 돌리고 쌍별 IoU까지 낸다

두 가지가 이 모델의 성격을 결정한다:

1. **"없다"고 말하지 못한다.** argmax는 모든 픽셀에 반드시 클래스를 준다. 사진에 가방이
   없어도 가방 확률이 가장 높은 픽셀을 어딘가에서 찾아낸다. 그래서 면적·채움·확신도
   **3중 품질 필터**로 "이번 검출을 믿을 수 있는가"를 따로 판정한다 →
   [segmentation.md](segmentation.md)
2. **픽셀당 라벨이 하나뿐이다.** 팔이 몸통을 가리면 그 픽셀은 팔로 가고 상의 마스크에는
   구멍이 남는다. 이 구조적 한계를 메우는 것이 Refine 파이프라인이다 →
   [refine-lab-pipeline.md](refine-lab-pipeline.md)

## Qwen3-VL — 무엇인가

`Qwen/Qwen3-VL-8B-Instruct` (NF4 양자화, 최대 100만 픽셀) · [app.py `QwenVLMEngine`](../backend/app.py#L610)

사진을 보고 **말로 된 속성**을 뽑는다. 프롬프트 네 개가 전부이고, 넷 다 한국어 JSON만
답하게 강제한다. 스키마를 프롬프트에 박아 두는 이유는 응답을 그대로 IndexedDB에 넣고
필터·추천에 쓰기 때문이다 — 자유 문장이면 파싱이 매번 깨진다.

| 프롬프트 | 쓰는 곳 | 뽑는 것 |
|---|---|---|
| `GARMENT_PROMPT` | 옷장 아이템 등록 시 ([app.js `analyzeGarment`](../app.js#L819)) | 소재·질감·핏·주름·마감·봉제 디테일·계절·날씨 |
| `LOOKBOOK_PROMPT` | 인플루언서 룩북 분석 ([app.js `analyzeLookbooks`](../app.js#L867)) | 착장을 옷 단위로 분해 + bbox + 레이어(아우터/이너) |
| `BODY_PROMPT` | 전신사진으로 아바타 만들 때 ([app.js:336](../app.js#L336)) | 체형 특징 (키·몸무게 **숫자는 추측 금지**) |
| `TRYON_JUDGE_PROMPT` | 착장 자동 평가 ([eval_tryon.py](../scripts/eval_tryon.py)) | 레이어 순서·누락·중복·정체성 유지 판정 |

이 JSON이 앱의 어느 계산에 들어가는지는 필드 단위로 따로 정리해 두었다 →
[vlm-json-usage.md](vlm-json-usage.md)

**Qwen Lab**(개발용 탭, `?dev=1`)에서 이 네 프롬프트를 직접 돌려볼 수 있다. 실제로 전송된
프롬프트 전문과 모델이 뱉은 원문, 그 원문을 파싱한 필드를 나란히 보여준다 — 필드가 빈
이유가 "확인 어려움"인지 토큰 예산에서 잘린 것인지를 여기서 가른다 →
[`qwen-lab.js`](../qwen-lab.js), `GET /api/vlm/prompts`

두 가지 설계 결정이 눈에 띈다:

- **`TRYON_JUDGE`는 근거를 먼저 쓰게 한다.** 착장 결과에는 정답 이미지가 없어서
  SSIM·LPIPS 같은 픽셀 지표로는 "상의가 아우터 안쪽인가"를 잴 수 없다. 그래서 VLM에게
  이진 판정을 시키는데, 결론부터 내라고 하면 첫 토큰에 걸려 뒤 근거를 결론에 끼워 맞춘다
- **`BODY_PROMPT`는 숫자를 금지한다.** 사진에서 키·몸무게를 추정하면 그럴듯한 숫자가
  나오지만 근거가 없다. 체형 **특징**만 받아 아바타 프롬프트에 쓴다

## FLUX.2 — 이렇게 보일 것이다

`black-forest-labs/FLUX.2-klein-4B` (Apache 2.0, BF16, 4스텝 증류) · [app.py `FluxImageEngine`](../backend/app.py)

텍스트 생성·이미지 편집·다중 참조 편집을 한 파이프라인에서 하므로, 세 가지 일을 같은
엔진이 맡는다. 워커 2개를 올려 두고 요청을 나눠 받는다.

| 용도 | 참조 이미지 | steps | 핵심 |
|---|---|---|---|
| **아바타** `/api/avatar` | 체형 실루엣 1장 | 4 | 치수를 문장이 아니라 **그림**으로 넣는다 |
| **가상 착장** `/api/tryon` | 아바타 + 옷 최대 6장 | 8 | 피부에서 바깥으로 **레이어 순서**대로 번호를 매긴다 |
| **옷 재생성** `/api/closet/refine` | 정규화한 옷 1장 | 4 | 가려졌던 자리를 원단으로 잇되 **색·재단은 유지** |

- **아바타**: "shoulder width 46 cm"를 프롬프트에 써도 확산 모델은 숫자를 길이로 해석하지
  못한다. 숫자만 바꿔가며 생성하면 체형은 그대로고 얼굴과 조명만 바뀐다. 그래서
  SMPL-X 메시를 치수에 맞춰 피팅하고 실루엣을 렌더해 **구조적 조건**으로 넣는다.
  덤으로 맞춰진 메시를 다시 재서 cm 단위 오차를 응답에 실을 수 있다 →
  [avatar_body.py](../backend/avatar_body.py)
- **착장**: 참조 번호가 레이어 순서와 어긋나면 모델이 번호를 레이어 힌트로 오해한다.
  그래서 디코딩 전에 안쪽부터 정렬한다 → [tryon_prompt.py](../backend/tryon_prompt.py)
- **재생성**: 목적이 생성이 아니라 **복원**이라 트레이드오프 방향이 반대다. 자유롭게
  그리라고 하면 다른 옷이 온다 → [refine-lab-pipeline.md](refine-lab-pipeline.md)

`FLUX_GUIDANCE=1.0`으로 CFG를 꺼 두었고 기본 4스텝이다. 증류 모델이라 스텝은
"올릴수록 좋아지는" 손잡이가 아니다 — 근거는 refine 문서의 steps 절에 있다.

---

## 셋이 한 GPU를 나눠 쓰는 방식

```
요청 → 토큰 검사 + 레이트리밋(60/분)  [POST만]
     → INFERENCE_GATE (동시 2개)
     → FLUX 워커 2개 · Qwen 1개(VLM_LOCK) · SegFormer 최대 3개 캐시
```

- **`INFERENCE_GATE`**: `GPU_CONCURRENCY=2`짜리 세마포어. 세 모델이 같은 VRAM을 쓰므로
  요청 종류와 무관하게 동시 실행을 막는다. 넘치면 큐에서 최대 300초 기다리고 503
- **Qwen만 NF4 양자화**: 8B를 BF16으로 올리면 FLUX 워커 2개와 같이 못 산다.
  이미지 생성 품질에는 양자화를 쓰지 않고, 판정·분석 쪽만 정밀도를 내준 것이다
- **SegFormer는 LRU 캐시**: 비교 탭이 모델을 바꿔가며 부르므로 3개까지 들고 있다가
  오래된 것부터 내린다. 가중치가 100~340MB로 작아서 가능한 사치다

## 셋이 만나는 지점

대부분의 경로에서 세 모델은 서로를 모른다. 앱이 결과를 이어 붙일 뿐이다.

```
전신샷 ─ SegFormer ─→ 옷 크롭 ─ Qwen ─→ 소재·핏 JSON ─→ 옷장 DB
                                                          │
아바타 ← FLUX ← 체형 실루엣 ← 치수/체형 JSON ← Qwen ← 전신사진
   └────────────→ FLUX(착장) ←── 옷 사진들 ────────────────┘
```

**단 한 곳, Refine 파이프라인만 두 모델이 한 요청 안에서 이어진다.** SegFormer가
`Left-arm`으로 찍었지만 옷 카테고리가 없어 **버리던** 라벨을, 가림 판정의 근거로 쓰고
그 결과를 문장으로 만들어 FLUX 프롬프트에 넘긴다:

> *"About 18% of this garment on its lower left side was covered by the model's left arm…"*

세그멘테이션이 이미 알고 있던 것을 생성 모델에게 말로 전달하는 셈이다. 자세한 내용은
[refine-lab-pipeline.md](refine-lab-pipeline.md)의 2단계를 참고.
