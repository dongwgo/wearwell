(function () {
  const MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct";

  const includes = (value, pattern) => pattern.test(String(value || "").toLowerCase());
  const unique = values => [...new Set(values.filter(Boolean))];

  function subcategory(item) {
    const name = item.name || "";
    if (includes(name, /셔츠|shirt/)) return "셔츠";
    if (includes(name, /니트|knit|스웨트/)) return "니트·스웨트";
    if (includes(name, /후드|hood/)) return "후드";
    if (includes(name, /티셔츠|tee|저지|반팔|롱 슬리브/)) return "티셔츠";
    if (includes(name, /데님|jean/)) return "데님 팬츠";
    if (includes(name, /슬랙스/)) return "슬랙스";
    if (includes(name, /치노|코튼 팬츠/)) return "치노 팬츠";
    if (includes(name, /쇼츠|반바지/)) return "쇼츠";
    if (includes(name, /패딩|다운/)) return "패딩";
    if (includes(name, /윈드|쉘/)) return "윈드브레이커";
    if (includes(name, /자켓|재킷|jacket|트랙탑/)) return "재킷";
    if (includes(name, /스니커|에어 포스|젤-|아디스타|아미/)) return "스니커즈";
    if (includes(name, /더비|1461/)) return "더비";
    if (includes(name, /백팩|backpack|pack/)) return "백팩";
    return item.category || "의류";
  }

  function seedGarmentAnalysis(item) {
    const name = item.name || "";
    const material = includes(name, /데님|jean/) ? "데님"
      : includes(name, /니트|knit|와플|스웨트/) ? "니트"
      : includes(name, /린넨/) ? "린넨"
      : includes(name, /레더|가죽|홀스하이드|스웨이드/) ? "레더"
      : includes(name, /윈드|쉘|나일론|패딩|다운/) ? "나일론"
      : includes(name, /셔츠|티셔츠|tee|코튼|치노/) ? "코튼"
      : includes(name, /메쉬|러닝|젤-|스니커/) ? "메쉬"
      : "혼방";
    const fit = includes(name, /엑스트라 와이드|와이드/) ? "와이드"
      : includes(name, /세미 와이드/) ? "세미와이드"
      : includes(name, /커브드/) ? "커브드"
      : includes(name, /스트레이트/) ? "스트레이트"
      : includes(name, /오버핏|오버/) ? "오버"
      : includes(name, /세미크롭|크롭/) ? "크롭"
      : includes(name, /릴렉스/) ? "세미오버"
      : includes(name, /머슬핏/) ? "슬림"
      : "레귤러";
    const texture = material === "데님" ? "탄탄한 능직"
      : material === "니트" ? "조직감 있음"
      : material === "린넨" ? "드라이하고 성긴 조직"
      : material === "레더" ? "매끈하고 단단함"
      : material === "나일론" ? "얇고 바스락거림"
      : "평직·매끈함";
    const wrinkle = material === "린넨" ? "주름이 자연스럽게 생김"
      : material === "코튼" ? "잔주름이 생길 수 있음"
      : ["나일론", "레더"].includes(material) ? "주름이 적음"
      : "보통";
    const finish = includes(name, /워시|피그먼트|washed/) ? "워싱·빈티지 마감"
      : material === "레더" ? "은은한 광택 마감"
      : includes(name, /유광|메탈/) ? "광택 마감"
      : "무광·깔끔한 마감";
    const construction = unique([
      includes(name, /원턱|one tuck/) && "원턱",
      includes(name, /투 턱|two tuck/) && "투턱",
      includes(name, /밴딩/) && "허리 밴딩",
      includes(name, /크롭/) && "짧은 기장",
      includes(name, /레이어드/) && "레이어드용 밑단",
      includes(name, /투웨이|2way/) && "투웨이 지퍼",
      includes(name, /후드/) && "후드 디테일",
      includes(name, /카고/) && "카고 포켓",
      includes(name, /롤업/) && "롤업 소매"
    ]);
    const season = material === "나일론" && includes(name, /패딩|다운/) ? ["겨울", "초봄"]
      : includes(name, /반팔|쇼츠|쿨|린넨/) ? ["여름"]
      : ["봄", "가을"];
    return {
      category: item.category,
      subcategory: subcategory(item),
      primaryColor: item.color || "색상 미분류",
      secondaryColors: [],
      material,
      texture,
      fit,
      silhouette: ["와이드", "세미와이드", "커브드"].includes(fit) ? "여유 있는 실루엣" : fit === "크롭" ? "짧고 정돈된 실루엣" : "기본 실루엣",
      wrinkle,
      finish,
      construction: construction.length ? construction : ["기본 봉제"],
      pattern: includes(name, /그래픽|레터링|로고/) ? "그래픽·로고" : "무지",
      season,
      weather: material === "나일론" ? ["바람", "약한 비"] : includes(name, /쿨|린넨|반팔|메쉬/) ? ["더움", "습함"] : ["선선함"],
      summary: `${item.color || "색상 미분류"} ${subcategory(item)}. ${fit} 핏, ${texture}, ${finish}, ${wrinkle}.`,
      engine: "catalog-rules",
      analyzedAt: new Date().toISOString()
    };
  }

  function emit(state, detail = "") {
    window.dispatchEvent(new CustomEvent("wearwell:vlm-status", { detail: { state, detail, model: MODEL_ID } }));
  }

  async function imageToDataUrl(imageSource) {
    if (String(imageSource || "").startsWith("data:")) return imageSource;
    const response = await fetch(imageSource);
    if (!response.ok) throw new Error("분석할 사진을 읽지 못했어요.");
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(new Error("사진을 변환하지 못했어요."));
      reader.readAsDataURL(blob);
    });
  }

  async function apiRequest(path, payload = null) {
    const base = window.resolveWearwellApiBase();
    const token = window.resolveWearwellApiToken();
    const response = await fetch(`${base}${path}`, {
      method: payload ? "POST" : "GET",
      headers: payload ? { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) } : {},
      ...(payload ? { body: JSON.stringify(payload) } : {})
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || `Qwen API 오류 (${response.status})`);
    return result;
  }

  async function loadModel() {
    emit("loading", "스타일 분석 AI를 준비하고 있어요");
    const health = await apiRequest("/api/health");
    if (!health.vlmModel) throw new Error("백엔드에 Qwen VLM이 준비되지 않았어요.");
    emit("ready", `${health.vlmModel} · ${health.vlmQuantization || "GPU"} 준비 완료`);
    return health;
  }

  async function analyzeImage(imageSource, item = {}) {
    emit("analyzing", `${item.name || "옷 사진"}의 색상과 디테일을 분석 중`);
    const result = await apiRequest("/api/vlm/garment", {
      image: await imageToDataUrl(imageSource),
      name: item.name || "옷 사진",
      category: item.category || null
    });
    const analysis = {
      ...seedGarmentAnalysis(item),
      ...result,
      construction: Array.isArray(result.construction) ? result.construction : [result.construction].filter(Boolean),
      secondaryColors: Array.isArray(result.secondaryColors) ? result.secondaryColors : [],
      season: Array.isArray(result.season) ? result.season : [result.season].filter(Boolean),
      weather: Array.isArray(result.weather) ? result.weather : [result.weather].filter(Boolean),
      engine: "Qwen3-VL-8B-Instruct",
      analyzedAt: new Date().toISOString()
    };
    emit("ready", `${item.name || "옷 사진"} 분석 완료`);
    return analysis;
  }

  async function analyzeLookImage(imageSource, look = {}) {
    emit("analyzing", `${look.creator || "크리에이터"} 룩북의 옷을 하나씩 분석 중`);
    const result = await apiRequest("/api/vlm/lookbook", {
      image: await imageToDataUrl(imageSource),
      name: `${look.creator || "크리에이터"} 룩북`
    });
    emit("ready", `${look.creator || "크리에이터"} 룩북 분석 완료`);
    return { ...result, engine: "Qwen3-VL-8B-Instruct", analyzedAt: new Date().toISOString() };
  }

  async function analyzeBodyImage(imageSource, gender = "women") {
    emit("analyzing", "전신사진에서 체형 특징을 분석 중");
    const result = await apiRequest("/api/vlm/body", {
      image: await imageToDataUrl(imageSource),
      name: "전신사진",
      gender
    });
    emit("ready", "전신사진 체형 특징 분석 완료");
    return { ...result, engine: "Qwen3-VL-8B-Instruct" };
  }

  window.WearwellVLM = { MODEL_ID, seedGarmentAnalysis, loadModel, analyzeImage, analyzeLookImage, analyzeBodyImage };
})();
