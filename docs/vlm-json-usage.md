# Qwen이 뱉은 JSON은 어디로 가나

Qwen3-VL은 그림을 그리지 않는다. 사진을 읽어 **스키마가 고정된 한국어 JSON**을 채우는
일만 한다([models.md](models.md)). 그 JSON이 실제로 앱의 어느 계산에 들어가는지를 필드
단위로 적어 둔다 — 프롬프트를 고칠 때 "이 필드를 바꾸면 무엇이 흔들리나"를 알기 위해서다.

프롬프트 네 개, 소비처 네 개. **서로 겹치지 않는다.**

| 프롬프트 | 응답이 도착하는 곳 | 최종 소비 |
|---|---|---|
| `GARMENT_PROMPT` | `item.analysis` (메모리 + IndexedDB) | 옷장 검색·표시, **룩북 매칭 점수** |
| `LOOKBOOK_PROMPT` | `look.pieces` (정규화 후) | 매칭의 **요구사항 쪽**, bbox 크롭 |
| `BODY_PROMPT` | `avatarMeasurements` | 룩 추천의 체형 점수 |
| `TRYON_JUDGE_PROMPT` | `scripts/eval_tryon.py` | 착장 품질 지표 (앱 런타임 아님) |

**착장 생성(`/api/tryon`)은 이 JSON을 하나도 보지 않는다.** 옷 이미지·카테고리·이름만
보낸다([app.js](../app.js#L685), `TryOnRequest`). 분석은 *무엇을 입힐지 고르는 단계*까지만
쓰이고, 그리는 단계에는 넘어가지 않는다.

---

## 공통 — JSON이 되기까지

```
사진 + 프롬프트 → Qwen 생성(greedy) → _parse_json → dict → 라우트 응답
                                        │
                                        └ 코드펜스 제거 후 첫 { ~ 마지막 } 만 취함
                                          객체가 아니면 ValueError → HTTP 422
```

- **파싱 실패는 500이 아니라 422다.** 서버가 아픈 게 아니라 모델이 JSON이 아닌 말을 한
  것이고, 프롬프트 쪽 문제라는 뜻이다 ([app.py `run_vlm_request`](../backend/app.py))
- **앱은 실패해도 멈추지 않는다.** 옷 분석은 `seedGarmentAnalysis`(이름 문자열 규칙,
  `engine: "catalog-rules"`)가 항상 먼저 채워져 있고, Qwen 응답이 오면 그 위에 덮인다.
  옷장 카드의 `Qwen 분석` / `특징 저장됨` 배지가 이 둘을 가른다
- 실제로 오간 프롬프트와 생성 원문은 **Qwen Lab** 탭에서 볼 수 있다(`debug: true`) →
  [`qwen-lab.js`](../qwen-lab.js)

---

## 1. 옷 한 벌 분석 — 옷장의 모든 계산이 여기서 나온다

`/api/vlm/garment` → [app.js `analyzeGarment`](../app.js#L1198)

```
Qwen JSON ─→ item.analysis ─┬─→ category/primaryColor로 사용자 입력을 덮어씀
                            ├─→ IndexedDB (userAdded 옷만)
                            ├─→ 옷장 카드·상세 패널 표시
                            ├─→ 검색어 매칭
                            └─→ garmentSimilarityDetail() ← 이게 본론
```

### 필드가 실제로 하는 일

| 필드 | 쓰이는 곳 | 무게 |
|---|---|---|
| `category` | 화이트리스트(7종)에 있으면 **아이템의 카테고리를 덮어쓴다** | 카테고리 불일치는 매칭에서 즉시 탈락 |
| `primaryColor` | "확인 어려움"이 아니면 **아이템 색을 덮어쓴다** | 색 점수의 입력 |
| `primaryColor`·`secondaryColors` | `knownColors` → RGB 거리 비교 | **0.49** (가장 큼) |
| `material`·`texture` | `setSimilarity` (코튼=면, 데님=진 동의어 묶음) | 0.11 |
| `fit`·`silhouette` | `setSimilarity` (오버/세미오버, 와이드/커브드 묶음) | 0.10 |
| `subcategory`·`pattern`·`finish`·`construction` | `setSimilarity` | 0.07 |
| `sleeveLength`·`length` | `canonicalGarmentForm`로 정규화 후 **하드 필터** | 불일치면 **총점 0** |
| `neckline`(+`subcategory`,`construction`) | 폴로/티셔츠 오판 **하드 필터** | 불일치면 **총점 0** |
| `summary`·`wrinkle`·`finish`·`material`·`fit`·`construction` | 옷 상세 패널의 "옷 사진 텍스트 분석" | 표시 전용 |
| `material`·`fit`·`finish` | 옷장 검색어 매칭 | 표시 전용 |
| `engine` | `Qwen 분석` / `특징 저장됨` 배지 | 표시 전용 |
| `season`·`weather` | **현재 아무 데서도 읽지 않는다** | — |

> 날씨 점수는 옷이 아니라 **룩 데이터의 `look.weather`**에서 온다. 옷의 `season`·`weather`는
> 스키마에만 있고 소비처가 없다 — 프롬프트에서 빼면 토큰이 준다.

### 여기서 중요한 두 가지

**하드 필터가 점수보다 세다.** 색·소재·핏 점수가 아무리 높아도 `sleeveLength`가 반팔인데
룩이 긴팔을 요구하면 총점이 0이 된다. 즉 **`sleeveLength`·`length`·`neckline`의 오답은
가중치와 무관하게 그 옷을 후보에서 지운다.** 프롬프트가 이 세 필드에 "확인 어려움"을
허용하는 이유가 여기 있다 — 모르면 모른다고 해야 필터가 통과된다(`unknown`은 무조건 통과).

**총점의 20%는 Qwen이 만든 값이 아니다.** `visual`은 SigLIP 임베딩의 코사인 유사도다.
같은 사진에 가까우면(≥0.92) Qwen의 유형 오판을 **구제**하고, ≥0.95면 카테고리 오판까지
구제한다. Qwen이 틀려도 사진이 같으면 살아남게 하는 안전장치다.

```
total = 0.03 + color×0.49 + material×0.11 + fit×0.10 + detail×0.07 + visual×0.20
        (같은 룩북에서 직접 잘라 등록한 옷이면 최소 0.985로 고정)
```

---

## 2. 룩북 분해 — 매칭의 "요구사항" 쪽을 만든다

`/api/vlm/lookbook` → [app.js `applyLookAnalysis`](../app.js#L1218)

옷 분석이 **내 옷**을 설명한다면, 이쪽은 **찾아야 할 옷**을 설명한다. 두 JSON이 같은
필드 이름으로 만나 위의 점수표가 계산된다.

```
pieces[] ─→ 정규화 ─→ 필터 ─→ look.pieces ─→ 옷장 전체와 대조해 후보 조합
                       │
                       └ 카테고리가 7종 화이트리스트에 없거나
                         colors가 비었거나 confidence < 0.5 → 버림
```

- **배열을 그대로 쓴다.** 재킷 안의 셔츠를 아우터·상의 **두 항목**으로 유지한다. 같은
  카테고리라고 첫 항목으로 합치면 레이어드 룩이 한 벌짜리 룩이 된다
- **`confidence`는 앱이 쓰는 유일한 자기평가 값이다.** 0.5 미만은 매칭에 올리지 않는다
- **`bbox`는 두 가지 일을 한다.** ① 그 좌표로 룩 사진을 잘라 SigLIP 임베딩을 만든다
  (총점의 20%를 만드는 그 값이다) ② 룩 썸네일의 `object-position`을 정해 옷이 화면
  가운데 오게 한다. **bbox가 없으면 룩 사진 전체가 임베딩되어** 옷 단위 비교가 흐려진다
- `summary`는 룩 카드 문구로, `mood`는 현재 표시에 쓰지 않는다

### 두 종류의 룩 분석이 있다

| 출처 | 언제 | pieces의 모양 |
|---|---|---|
| `assets/influencer-data.js` (저장소 배포) | 기본 제공 룩북 | 스크립트가 만든 템플릿. **bbox·confidence 없음** |
| Qwen 실행 | 사용자가 룩북을 올릴 때 | 전체 스키마 |

기본 룩북을 저장소에 캐시해 두는 이유는 **모든 사용자의 브라우저에서 같은 사진을 다시
분석하지 않기 위해서**다. 사용자가 올린 룩은 IndexedDB에 `analysisVersion`과 함께 저장하고,
[app.js](../app.js#L111)의 `LOOK_ANALYSIS_VERSION`을 올리면 캐시가 통째로 무효화된다 —
프롬프트를 고쳤는데 옛 결과가 남아 있으면 무엇이 달라졌는지 알 수 없기 때문이다.

---

## 3. 체형 특징 — 네 필드 중 하나만 계산에 들어간다

`/api/vlm/body` → [app.js `runGenerateAvatar`](../app.js#L377)

응답은 `avatarMeasurements`에 병합되지만, **실제로 숫자를 바꾸는 것은 `body_shape` 하나다.**

```
photoBased 경로:  look.bodyShapes에 body_shape가 있으면 96점, 없으면 72점
```

`proportion`·`shoulderLine`·`silhouette`은 화면 문구로도 쓰지 않는다. 아바타 생성 프롬프트의
`body-shape description`은 **치수 입력 경로**(사용자가 키·몸무게를 적는 쪽)에서 오는
`Measurements.body_shape`이지 이 응답이 아니다.

프롬프트가 **키·몸무게 숫자를 금지**하는 것도 같은 맥락이다. 사진에서 추정하면 그럴듯한
숫자가 나오지만 근거가 없고, 그 숫자가 아바타 치수로 들어가면 틀린 체형이 만들어진다.
그래서 사진 경로는 성별 기본값(남 175/70, 여 165/55)을 쓰고 `body_shape`만 받아 쓴다.

---

## 4. 착장 심판 — 앱이 아니라 평가 스크립트가 읽는다

`/api/vlm/tryon-judge` → [scripts/eval_tryon.py](../scripts/eval_tryon.py)

사용자 화면에는 절대 나오지 않는다. 개선 전/후 수치를 낼 때만 돈다.

| 응답 필드 | 만들어지는 지표 |
|---|---|
| `layering_ok` | `layering_accuracy` |
| `items_present` | `item_recall`, 그리고 `extra_items`·`merged_items`가 비어야 `items_ok` |
| `accessories_placed_ok` | `accessory_accuracy` |
| `identity_ok` | `identity_preservation` |
| `artifacts` | `mean_artifacts` (개수만) |
| `reasons` | 사람이 읽는 근거. 지표에는 안 들어감 |

`items_ok`는 **세 조건의 AND**다 — 요청한 옷이 전부 보이고, 없던 옷이 생기지 않고, 두 벌이
한 벌로 합쳐지지 않아야 한다. 그래서 프롬프트가 `extra_items`와 `merged_items`를 따로 묻는다.
`reasons`를 먼저 쓰게 하는 이유는 [models.md](models.md)에 있다.

---

## 전체 그림

```
       옷 사진 ─ Qwen(garment) ─→ item.analysis ┐
                                                ├─→ garmentSimilarity ─→ 룩 추천·후보 조합
   룩북 사진 ─ Qwen(lookbook) ─→ look.pieces ───┘         ↑
                     └─ bbox ─→ 크롭 ─ SigLIP ─→ visual (20%)

   전신사진 ─ Qwen(body) ─→ body_shape ─→ 룩 추천의 체형 점수

   착장 결과 ─ Qwen(judge) ─→ eval_tryon.py 지표          [앱 런타임 밖]
```

옷 쪽과 룩북 쪽이 **같은 필드 이름으로 만나는 것**이 이 설계의 전부다. 한쪽 프롬프트에서
필드 이름이나 어휘("와이드", "레귤러", "확인 어려움")를 바꾸면 반대쪽 `setSimilarity`가
조용히 못 맞추게 된다 — 점수가 0이 되는 게 아니라 **낮아지기만 해서 눈에 잘 띄지 않는다.**
어휘를 손볼 때는 두 프롬프트를 같이 봐야 한다.
