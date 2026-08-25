const categories = ["전체", "아우터", "상의", "하의", "신발", "가방", "액세서리"];
const styleOptions = {
  women: [
    { name: "단정한 미니멀", image: 4 }, { name: "편안한 꾸안꾸", image: 14 },
    { name: "러블리 데일리", image: 26 }, { name: "힙한 스트릿", image: 37 },
    { name: "차분한 클래식", image: 48 }
  ],
  men: [
    { name: "깔끔한 댄디", image: 54 }, { name: "편안한 캐주얼", image: 64 },
    { name: "미니멀 시티", image: 76 }, { name: "힙한 스트릿", image: 87 },
    { name: "빈티지 아메카지", image: 98 }
  ]
};
const priorityOptions = [
  ["☁", "편안함이 우선", "오래 입어도 편한 핏과 소재"],
  ["✦", "깔끔해 보이기", "단정한 비율과 정돈된 인상"],
  ["☂", "날씨에 잘 맞기", "기온과 비까지 빠짐없이 반영"],
  ["↗", "새로운 조합 시도", "평소보다 한 걸음 새로운 코디"]
];
const rankingWardrobe = Array.isArray(window.MUSINSA_RANKING) ? window.MUSINSA_RANKING : [];
if (rankingWardrobe.length !== 100) throw new Error(`무신사 랭킹 데이터가 100개 필요합니다. 현재 ${rankingWardrobe.length}개입니다.`);
const influencerLooks = Array.isArray(window.WEARWELL_INFLUENCER_LOOKS) ? window.WEARWELL_INFLUENCER_LOOKS : [];
let wardrobe = rankingWardrobe.map(item => ({ ...item, analysis: window.WearwellVLM.seedGarmentAnalysis(item) }));
const photo = number => wardrobe[((number - 1) % wardrobe.length + wardrobe.length) % wardrobe.length].image;
const itemMatchesGender = item => item.userAdded || !selectedGender || item.gender === "all" || item.gender === selectedGender;

const lookSets = {
  women: [
    { title: "비 오는 날, 단정한 꾸안꾸", mood: "출근룩", score: 94, pieces: [3, 10, 21, 34], reasons: ["☂ 오늘 날씨에 딱", "◌ 좋아하는 차분한 색", "♧ 출근부터 약속까지"] },
    { title: "힘주지 않은 미니멀룩", mood: "미니멀", score: 92, pieces: [5, 17, 30, 42], reasons: ["✦ 비율이 깔끔해요", "◌ 자주 입는 색 조합", "↗ 새로운 레이어드"] },
    { title: "데님으로 가볍게", mood: "꾸안꾸", score: 90, pieces: [8, 20, 31, 43], reasons: ["♧ 하루 종일 편하게", "◌ 잘 입는 데님 활용", "☀ 실내외 모두 좋아요"] },
    { title: "기분 좋아지는 포인트 컬러", mood: "주말룩", score: 87, pieces: [7, 26, 38, 45], reasons: ["✦ 얼굴이 밝아 보여요", "◇ 색 균형이 좋아요", "↗ 취향을 살짝 확장"] },
    { title: "카페 가기 좋은 부드러운 룩", mood: "데이트룩", score: 91, pieces: [2, 18, 35, 47], reasons: ["♧ 여유 있는 핏", "◌ 부드러운 소재", "☀ 가벼운 레이어드"] },
    { title: "퇴근 후 약속까지 한 번에", mood: "모임룩", score: 89, pieces: [11, 23, 39, 49], reasons: ["◇ 차분하고 또렷하게", "◌ 내 취향 컬러", "♧ 갈아입을 필요 없이"] }
  ],
  men: [
    { title: "비 오는 날, 깔끔한 출근룩", mood: "출근룩", score: 95, pieces: [52, 63, 76, 89], reasons: ["☂ 젖어도 부담 없이", "◌ 좋아하는 무채색", "♧ 출근부터 저녁까지"] },
    { title: "셔츠 하나로 완성한 댄디룩", mood: "댄디", score: 92, pieces: [55, 67, 80, 92], reasons: ["✦ 단정한 첫인상", "◌ 자주 입는 실루엣", "↗ 소매를 가볍게 롤업"] },
    { title: "편하지만 갖춰 입은 캐주얼", mood: "꾸안꾸", score: 90, pieces: [58, 70, 81, 94], reasons: ["♧ 하루 종일 편하게", "◌ 익숙한 스니커즈", "☀ 일교차에 좋은 조합"] },
    { title: "오늘은 살짝 힙하게", mood: "스트릿", score: 88, pieces: [60, 72, 85, 97], reasons: ["✦ 포인트가 확실해요", "◇ 상하의 균형", "↗ 새로운 조합 도전"] },
    { title: "주말 카페를 위한 아메카지", mood: "주말룩", score: 91, pieces: [51, 65, 78, 90], reasons: ["♧ 여유 있는 핏", "◌ 따뜻한 색감", "☀ 가볍게 걸치기 좋아요"] },
    { title: "약속 있는 날의 미니멀", mood: "모임룩", score: 89, pieces: [54, 69, 83, 99], reasons: ["◇ 깔끔한 인상", "◌ 내 취향 컬러", "♧ 어디서든 자연스럽게"] }
  ]
};

const trendSets = {
  women: [
    { title: "커프드 데님 + 깔끔한 셔츠", mood: "데님", image: 6, source: "2026 스트리트 스타일", sourceUrl: "https://www.whowhatwear.com/fashion/outfit-ideas/fall-cuffed-jeans-outfit-ideas-2026", requirements: [{ category: "상의", colors: ["화이트", "아이보리"] }, { category: "하의", keyword: "데님", colors: ["네이비"] }, { category: "신발", colors: ["화이트", "아이보리"] }, { category: "가방", colors: ["블랙", "브라운"] }] },
    { title: "볼륨 팬츠 + 심플한 톱", mood: "볼륨 팬츠", image: 12, source: "Vogue 2026 S/S", sourceUrl: "https://www.vogue.com/article/the-spring-2026-trend-report-a-season-of-uniform-dressing-rococo-flourishes-and-much-more", requirements: [{ category: "상의", colors: ["화이트", "블랙", "아이보리"] }, { category: "하의", keyword: "와이드", colors: ["그레이", "블랙", "오트밀"] }, { category: "신발", colors: ["블랙", "화이트"] }, { category: "가방", colors: ["브라운", "블랙"] }] },
    { title: "플랫 슈즈로 가볍게", mood: "플랫 슈즈", image: 19, source: "Vogue Korea 트렌드", sourceUrl: "https://www.vogue.co.kr/2026/01/06/%ED%8C%A8%EC%85%98-%EC%9C%84%ED%81%AC%EC%97%90%EC%84%9C-%ED%99%95%EC%9D%B8%ED%95%9C-2026-%EB%B4%84-%EC%97%AC%EB%A6%84-%ED%95%B5%EC%8B%AC-%ED%8A%B8%EB%A0%8C%EB%93%9C-12%EA%B0%80%EC%A7%80/", requirements: [{ category: "상의", colors: ["아이보리", "화이트"] }, { category: "하의", keyword: "스커트", colors: ["블랙", "그레이"] }, { category: "신발", keyword: "플랫", colors: ["블랙", "브라운"] }, { category: "가방", colors: ["브라운", "블랙"] }] },
    { title: "크림 재킷 + 가벼운 이너", mood: "라이트 레이어", image: 27, source: "Seoul 2026 스타일", sourceUrl: "https://www.marieclaire.com/fashion/celebrity-style/hailey-bieber-seoul-style-chanel-bags-sheer-crochet-top-toteme-heeled-flip-flops/", requirements: [{ category: "아우터", colors: ["아이보리", "오트밀"] }, { category: "상의", colors: ["화이트", "아이보리"] }, { category: "하의", colors: ["그레이", "네이비"] }, { category: "신발", colors: ["브라운", "블랙"] }] },
    { title: "톤온톤 유니폼 드레싱", mood: "톤온톤", image: 34, source: "Vogue 2026 S/S", sourceUrl: "https://www.vogue.com/article/spring-2026-fashion-trends", requirements: [{ category: "아우터", colors: ["그레이", "오트밀", "네이비"] }, { category: "상의", colors: ["그레이", "아이보리"] }, { category: "하의", colors: ["그레이", "네이비"] }, { category: "신발", colors: ["블랙", "화이트"] }] },
    { title: "스카프처럼 쓰는 컬러 포인트", mood: "포인트 컬러", image: 43, source: "2026 S/S 트렌드", sourceUrl: "https://www.vogue.co.kr/2026/01/06/%ED%8C%A8%EC%85%98-%EC%9C%84%ED%81%AC%EC%97%90%EC%84%9C-%ED%99%95%EC%9D%B8%ED%95%9C-2026-%EB%B4%84-%EC%97%AC%EB%A6%84-%ED%95%B5%EC%8B%AC-%ED%8A%B8%EB%A0%8C%EB%93%9C-12%EA%B0%80%EC%A7%80/", requirements: [{ category: "상의", colors: ["코랄", "화이트"] }, { category: "하의", colors: ["네이비", "블랙"] }, { category: "신발", colors: ["화이트", "블랙"] }, { category: "가방", colors: ["코랄", "브라운"] }] }
  ],
  men: [
    { title: "가벼운 셔츠 테일러링", mood: "테일러링", image: 54, source: "WooYoungMi 2026 S/S", sourceUrl: "https://www.wallpaper.com/fashion-beauty/paris-fashion-week-mens-ss-2026-highlights-review", requirements: [{ category: "상의", keyword: "셔츠", colors: ["화이트", "아이보리"] }, { category: "하의", colors: ["그레이", "네이비", "블랙"] }, { category: "아우터", colors: ["그레이", "네이비"] }, { category: "신발", colors: ["블랙", "화이트"] }] },
    { title: "버뮤다 팬츠 + 캐주얼 니트", mood: "버뮤다", image: 61, source: "2026 S/S 트렌드", sourceUrl: "https://www.vogue.co.kr/2026/01/06/%ED%8C%A8%EC%85%98-%EC%9C%84%ED%81%AC%EC%97%90%EC%84%9C-%ED%99%95%EC%9D%B8%ED%95%9C-2026-%EB%B4%84-%EC%97%AC%EB%A6%84-%ED%95%B5%EC%8B%AC-%ED%8A%B8%EB%A0%8C%EB%93%9C-12%EA%B0%80%EC%A7%80/", requirements: [{ category: "상의", keyword: "니트", colors: ["아이보리", "네이비"] }, { category: "하의", keyword: "버뮤다", colors: ["그레이", "블랙", "카키"] }, { category: "신발", colors: ["화이트", "블랙"] }, { category: "가방", colors: ["블랙", "카키"] }] },
    { title: "화이트 셔츠 + 진청 데님", mood: "데님", image: 69, source: "2026 커프드 데님", sourceUrl: "https://www.whowhatwear.com/fashion/outfit-ideas/fall-cuffed-jeans-outfit-ideas-2026", requirements: [{ category: "상의", keyword: "셔츠", colors: ["화이트"] }, { category: "하의", keyword: "데님", colors: ["네이비"] }, { category: "신발", colors: ["화이트"] }, { category: "아우터", colors: ["네이비", "그레이"] }] },
    { title: "올블랙에 소재 차이 주기", mood: "올블랙", image: 77, source: "2026 가을 스트리트", sourceUrl: "https://www.whowhatwear.com/fashion/trends/street-style-trend-predictions-autumn-2026", requirements: [{ category: "상의", colors: ["블랙"] }, { category: "하의", colors: ["블랙", "그레이"] }, { category: "아우터", colors: ["블랙"] }, { category: "신발", colors: ["블랙"] }] },
    { title: "셔츠 레이어드 + 와이드 팬츠", mood: "레이어드", image: 86, source: "Seoul 2026 스트리트", sourceUrl: "https://english.seoul.go.kr/seoul-fashion-week-expands-beyond-ddp-to-lotte-world-tower-turning-seouls-iconic-landmarks-into-runways/", requirements: [{ category: "상의", keyword: "셔츠", colors: ["화이트", "그레이"] }, { category: "하의", keyword: "와이드", colors: ["블랙", "그레이"] }, { category: "아우터", colors: ["카키", "네이비"] }, { category: "신발", colors: ["화이트", "블랙"] }] },
    { title: "스포티한 시티 프레피", mood: "프레피", image: 95, source: "2026 스트리트 스타일", sourceUrl: "https://www.vogue.com/article/spring-2026-fashion-trends", requirements: [{ category: "상의", keyword: "카라", colors: ["네이비", "화이트"] }, { category: "하의", colors: ["그레이", "네이비"] }, { category: "신발", colors: ["화이트"] }, { category: "가방", colors: ["블랙", "네이비"] }] }
  ]
};

let selectedGender = null;
let looks = lookSets.women;
let currentLook = 0;
let visibleWardrobe = 24;
let activeCategory = "전체";
let activeMood = "전체";
let selectedStyles = new Set();
let selectedInfluencerLookIds = new Set();
let selectedPriorities = new Set(["날씨에 잘 맞기"]);
let uploadFiles = [];
let savedLooks = new Set(JSON.parse(localStorage.getItem("오늘옷-saved") || "[]"));
let selectedWardrobeItem = null;
let matchVariation = 0;
let avatarImage = localStorage.getItem("오늘옷-avatar") || null;
let avatarMeasurements = null;
let bodyInputMethod = "measurements";
let fullBodyPhoto = null;
const selectedGarmentIds = new Set();
const tryonCache = new Map();
const lookbookAnalysisQueued = new Set();
const API_BASE = window.resolveWearwellApiBase();
const API_TOKEN = window.resolveWearwellApiToken();
const API_HEADERS = { "Content-Type": "application/json", ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}) };
const todayWeather = { temperature: 22, condition: "약한 비", tags: ["선선함", "약한 비", "간절기", "습함"] };
const LOOK_ANALYSIS_VERSION = 3;
const VLM_ANALYSIS_ENGINE = "Qwen3-VL-8B-Instruct";

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
})[character]);

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => toast.classList.remove("show"), 2200);
}

function openDialog(dialog) { if (!dialog.open) dialog.showModal(); }
function closeDialogs() { $$('dialog[open]').forEach(dialog => dialog.close()); }
function genderLabel() { return selectedGender === "men" ? "남성" : "여성"; }

function readMeasurements() {
  if (bodyInputMethod === "photo" && avatarMeasurements?.photoBased) return { ...avatarMeasurements, gender: selectedGender || avatarMeasurements.gender };
  const optionalNumber = id => $(id).value ? Number($(id).value) : null;
  return {
    gender: selectedGender || "women",
    height: Number($("#bodyHeight").value),
    weight: Number($("#bodyWeight").value),
    body_shape: $("#bodyShape").value,
    shoulder: optionalNumber("#bodyShoulder"), chest: optionalNumber("#bodyChest"),
    waist: optionalNumber("#bodyWaist"), hip: optionalNumber("#bodyHip"), inseam: optionalNumber("#bodyInseam"),
    seed: 20260825
  };
}

function applyMeasurementValues(data) {
  if (!data) return;
  avatarMeasurements = data;
  $("#bodyHeight").value = data.height || (selectedGender === "men" ? 175 : 165);
  $("#bodyWeight").value = data.weight || (selectedGender === "men" ? 70 : 55);
  $("#bodyShape").value = data.body_shape || "보통";
  [["#bodyShoulder", "shoulder"], ["#bodyChest", "chest"], ["#bodyWaist", "waist"], ["#bodyHip", "hip"], ["#bodyInseam", "inseam"]].forEach(([id, key]) => { $(id).value = data[key] || ""; });
}

function localAvatarPreview(data) {
  const canvas = document.createElement("canvas");
  canvas.width = 384; canvas.height = 512;
  const context = canvas.getContext("2d");
  context.fillStyle = "#eeeae5"; context.fillRect(0, 0, 384, 512);
  const bmi = data.weight / ((data.height / 100) ** 2);
  const bodyWidth = 76 + Math.max(-14, Math.min(46, (bmi - 20) * 4));
  const shoulder = (data.shoulder || (data.gender === "men" ? 46 : 40)) * 2;
  const hip = (data.hip || (data.gender === "men" ? 94 : 92)) * .78;
  const cx = 192;
  context.fillStyle = "#d2ab94"; context.beginPath(); context.arc(cx, 70, 32, 0, Math.PI * 2); context.fill();
  context.fillStyle = "#858b92"; context.beginPath(); context.roundRect(cx - shoulder / 2, 103, shoulder, 182, 34); context.fill();
  context.beginPath(); context.moveTo(cx - bodyWidth / 2, 200); context.lineTo(cx + bodyWidth / 2, 200); context.lineTo(cx + hip / 2, 326); context.lineTo(cx - hip / 2, 326); context.fill();
  context.fillStyle = "#5f656c"; context.beginPath(); context.roundRect(cx - hip / 2, 310, hip / 2 - 8, 165, 18); context.roundRect(cx + 8, 310, hip / 2 - 8, 165, 18); context.fill();
  context.fillStyle = "#d2ab94"; context.beginPath(); context.roundRect(cx - shoulder / 2 - 18, 118, 30, 208, 15); context.roundRect(cx + shoulder / 2 - 12, 118, 30, 208, 15); context.fill();
  return canvas.toDataURL("image/jpeg", .9);
}

function showAvatar(image, engine = "") {
  avatarImage = image;
  try { localStorage.setItem("오늘옷-avatar", image); } catch {}
  $("#avatarPreview").innerHTML = `<img src="${image}" alt="내 체형 아바타" />`;
  $("#profileAvatar").classList.add("has-image");
  $("#profileAvatar").style.backgroundImage = `url(${image})`;
  if (engine) $("#avatarEngineStatus").textContent = engine === "photo-reference" ? "전신사진을 아바타 기준으로 준비했어요" : engine.includes("cuda") ? "내 체형 아바타가 완성됐어요" : "기본 아바타 미리보기";
}

async function generateAvatar() {
  if (bodyInputMethod === "photo") {
    if (!fullBodyPhoto) return showToast("먼저 전신사진을 선택해주세요");
    avatarMeasurements = {
      gender: selectedGender || "women", height: selectedGender === "men" ? 175 : 165,
      weight: selectedGender === "men" ? 70 : 55, body_shape: "보통", photoBased: true, seed: 20260825
    };
    showAvatar(fullBodyPhoto, "photo-reference");
    window.WearwellVLM.analyzeBodyImage(fullBodyPhoto, selectedGender).then(profile => {
      avatarMeasurements = { ...avatarMeasurements, ...profile, photoBased: true };
      $("#avatarEngineStatus").textContent = `사진에서 ${profile.body_shape || "체형"} 특징을 정리했어요`;
      if ($('.preference-step[data-step="3"]').classList.contains("active")) {
        renderPreferenceChoices(); updatePreferenceCount();
      }
    }).catch(() => {});
    return;
  }
  const data = readMeasurements();
  avatarMeasurements = data;
  const card = $("#avatarGeneratorCard");
  card.classList.add("generating");
  $("#generateAvatar").disabled = true;
  $("#avatarEngineStatus").textContent = "신체 비율을 계산하고 있어요…";
  try {
    const response = await fetch(`${API_BASE}/api/avatar`, { method: "POST", headers: API_HEADERS, body: JSON.stringify(data) });
    if (!response.ok) throw new Error("GPU backend unavailable");
    const result = await response.json();
    showAvatar(result.image, result.engine);
  } catch {
    showAvatar(localAvatarPreview(data), "fallback");
  } finally {
    card.classList.remove("generating");
    $("#generateAvatar").disabled = false;
  }
}

async function resizeBodyPhoto(file) {
  const source = await fileToDataUrl(file);
  const image = new Image();
  await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = source; });
  const scale = Math.min(1, 1200 / Math.max(image.width, image.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(image.width * scale); canvas.height = Math.round(image.height * scale);
  canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", .88);
}

function setBodyInputMethod(method) {
  bodyInputMethod = method;
  $$("[data-body-method]").forEach(button => button.classList.toggle("active", button.dataset.bodyMethod === method));
  $("#measurementBodyForm").hidden = method !== "measurements";
  $("#fullBodyUpload").hidden = method !== "photo";
  $("#generateAvatar").textContent = method === "photo" ? "✦ 이 사진으로 시작하기" : "✦ 아바타 만들기";
  $("#avatarEngineStatus").textContent = method === "photo" ? "전신사진을 선택해주세요" : "아바타를 만들 준비가 됐어요";
}

async function imageToDataUrl(source) {
  if (source.startsWith("data:")) return source;
  const response = await fetch(source);
  const blob = await response.blob();
  return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob); });
}

function garmentType(category) {
  return ({ 상의: "upper", 하의: "lower", 원피스: "overall", 아우터: "outer", 신발: "shoes", 가방: "bag", 액세서리: "accessory" })[category] || null;
}

function renderTryonStage(reference, output = null, loading = false) {
  const resultPane = loading
    ? '<div class="tryon-compare-loading"><div class="generation-loader"><span></span><strong>내 옷을 입혀보는 중</strong><small>룩북의 비율과 분위기를 비교해요</small></div></div>'
    : `<img src="${escapeHtml(output)}" alt="내 아바타 착장 결과" />`;
  if (!reference) {
    $("#tryonStage").classList.remove("comparison");
    $("#tryonStage").innerHTML = loading ? resultPane : `<img src="${escapeHtml(output)}" alt="AI 가상 착장 결과" />`;
    return;
  }
  $("#tryonStage").classList.add("comparison");
  $("#tryonStage").innerHTML = `
    <figure><div><img src="${escapeHtml(reference.image)}" alt="${escapeHtml(reference.title || "룩북 원본")}" /></div><figcaption><b>룩북 원본</b><span>${escapeHtml(reference.title || "")}</span></figcaption></figure>
    <figure><div>${resultPane}</div><figcaption><b>내 아바타 + 내 옷</b><span>${loading ? "생성 중" : "내 옷장으로 재현"}</span></figcaption></figure>`;
}

async function tryOnItems(items, comparison = null) {
  if (!items.length) return showToast("입혀볼 옷을 먼저 골라주세요");
  if (!avatarImage) await generateAvatar();
  const avatarKey = avatarMeasurements ? JSON.stringify(avatarMeasurements) : avatarImage.slice(-64);
  const cacheKey = `${items.map(item => item.id).join("-")}-${avatarKey}`;
  $("#tryonGarments").innerHTML = items.map(item => `<div><img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.name)}" /><span>${escapeHtml(item.name)}</span></div>`).join("");
  renderTryonStage(comparison, null, true);
  $("#tryonStatusText").textContent = "선택한 옷의 색과 형태를 살려 조합하고 있어요.";
  openDialog($("#avatarTryonDialog"));
  if (tryonCache.has(cacheKey)) {
    const cached = tryonCache.get(cacheKey); renderTryonStage(comparison, cached); return;
  }
  try {
    const wearableItems = items.filter(item => garmentType(item.category)).slice(0, 4);
    if (!wearableItems.length) throw new Error("No supported garments");
    const garments = await Promise.all(wearableItems.map(async item => ({ image: await imageToDataUrl(item.image), category: garmentType(item.category), name: item.name })));
    const response = await fetch(`${API_BASE}/api/tryon`, { method: "POST", headers: API_HEADERS, body: JSON.stringify({ avatar: avatarImage, garments, seed: 42 }) });
    if (!response.ok) throw new Error("Try-on backend unavailable");
    const result = await response.json();
    tryonCache.set(cacheKey, result.image);
    renderTryonStage(comparison, result.image);
    $("#tryonStatusText").textContent = comparison ? "왼쪽은 룩북 원본, 오른쪽은 내 옷장에서 고른 옷을 입힌 결과예요." : "완성됐어요. 옷 사진의 디테일을 아바타 체형에 맞춰 표현했어요.";
    $("#tryonEngineLabel").textContent = comparison ? "원본 ↔ 내 착장 비교" : "AI 착장 결과";
  } catch {
    renderTryonStage(comparison, avatarImage);
    $("#tryonStatusText").textContent = "지금은 착장 이미지를 만들지 못했어요. 잠시 후 다시 시도해주세요.";
    $("#tryonEngineLabel").textContent = "착장 미리보기";
  }
}

function renderCurrentLook() {
  const look = looks[currentLook];
  $("#lookTitle").textContent = look.title;
  $("#matchScore").textContent = look.score;
  $("#reasonList").innerHTML = look.reasons.map(reason => `<span>${reason}</span>`).join("");
  $("#outfitCanvas").innerHTML = look.pieces.map((pieceIndex, slot) => {
    const item = wardrobe[pieceIndex % wardrobe.length];
    return `<div class="outfit-piece" data-slot="${slot}">
      <img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.name)}" />
      <span class="piece-tag">${escapeHtml(item.category)} · ${escapeHtml(item.color)}</span>
      <button class="swap-hint" data-swap="${slot}" aria-label="${escapeHtml(item.name)} 바꾸기">↻</button>
    </div>`;
  }).join("");
  $("#saveLook").classList.toggle("saved", savedLooks.has(`${selectedGender}-${currentLook}`));
  $("#saveLook").textContent = savedLooks.has(`${selectedGender}-${currentLook}`) ? "♥" : "♡";
  $$('[data-swap]').forEach(button => button.addEventListener("click", () => swapPiece(Number(button.dataset.swap))));
}

function swapPiece(slot) {
  const look = looks[currentLook];
  const currentItem = wardrobe[look.pieces[slot]];
  const alternatives = wardrobe.map((item, index) => ({ item, index })).filter(entry =>
    entry.item.category === currentItem.category && itemMatchesGender(entry.item) && !look.pieces.includes(entry.index)
  );
  const next = alternatives[Math.floor(Math.random() * alternatives.length)];
  if (next) look.pieces[slot] = next.index;
  look.score = Math.max(84, Math.min(98, look.score + (Math.random() > .5 ? 1 : -1)));
  renderCurrentLook();
  showToast(`${currentItem.category}를 바꿔서 다시 맞춰봤어요`);
}

function nextLook() {
  currentLook = (currentLook + 1) % looks.length;
  renderCurrentLook();
  $(".look-card").animate([{ opacity: .55 }, { opacity: 1 }], { duration: 260 });
}

function renderRotation() {
  $("#rotationGrid").innerHTML = looks.slice(1, 5).map((look, index) => `
    <article class="mini-look" data-look="${index + 1}">
      <div class="mini-look-image"><img src="${photo(look.pieces[0] + 1)}" alt="${look.title}" loading="lazy" /></div>
      <div class="mini-look-info"><strong>${look.title}</strong><span>${look.score}% 잘 맞아요</span></div>
    </article>`).join("");
  $$('.mini-look').forEach(card => card.addEventListener("click", () => {
    currentLook = Number(card.dataset.look); renderCurrentLook(); window.scrollTo({ top: 0, behavior: "smooth" });
  }));
}

function renderCategoryFilters() {
  $("#categoryFilters").innerHTML = categories.map(category => `<button class="filter-chip ${category === activeCategory ? "active" : ""}" data-category="${category}">${category}</button>`).join("");
  $$('[data-category]').forEach(button => button.addEventListener("click", () => {
    activeCategory = button.dataset.category; visibleWardrobe = 24; renderCategoryFilters(); renderWardrobe();
  }));
}

const matchCategories = {
  아우터: ["상의", "하의", "신발"],
  상의: ["하의", "신발", "아우터"],
  하의: ["상의", "신발", "아우터"],
  원피스: ["아우터", "신발", "가방"],
  신발: ["상의", "하의", "아우터"],
  가방: ["상의", "하의", "신발"],
  액세서리: ["상의", "하의", "신발"]
};
const colorMatches = {
  화이트: ["네이비", "블랙", "그레이", "카키", "화이트", "아이보리"],
  아이보리: ["브라운", "네이비", "카키", "블랙", "오트밀"],
  블랙: ["화이트", "아이보리", "그레이", "코랄", "네이비"],
  네이비: ["화이트", "아이보리", "그레이", "오트밀", "브라운"],
  그레이: ["화이트", "블랙", "네이비", "코랄", "아이보리"],
  카키: ["아이보리", "화이트", "브라운", "블랙", "오트밀"],
  브라운: ["아이보리", "오트밀", "네이비", "화이트", "카키"],
  오트밀: ["브라운", "네이비", "카키", "블랙", "아이보리"],
  코랄: ["화이트", "아이보리", "그레이", "네이비", "블랙"]
};

function partnerScore(anchor, item, desiredCategory, variation) {
  let score = item.category === desiredCategory ? 50 : 0;
  const preferred = colorMatches[anchor.color] || ["화이트", "블랙", "네이비", "그레이"];
  const colorRank = preferred.indexOf(item.color);
  if (colorRank >= 0) score += 24 - colorRank * 3;
  if (anchor.color === "화이트" && item.category === "하의" && /데님/.test(item.name)) score += 22;
  if (anchor.color === "화이트" && item.category === "신발" && item.color === "화이트") score += 20;
  score += Math.min(item.worn, 10) / 2;
  score += ((Number(item.id.replace(/\D/g, "")) || 0) * (variation + 3)) % 9;
  return score;
}

function buildMatch(anchor, variation = 0) {
  const desired = matchCategories[anchor.category] || ["상의", "하의", "신발"];
  const pieces = [anchor];
  desired.forEach((category, categoryIndex) => {
    const pool = wardrobe.filter(item =>
      item.id !== anchor.id && item.category === category &&
      itemMatchesGender(item)
    ).sort((a, b) => partnerScore(anchor, b, category, variation) - partnerScore(anchor, a, category, variation));
    const offset = variation % Math.max(1, Math.min(pool.length, 4));
    if (pool[offset]) pieces.push(pool[offset]);
  });
  const isWhiteTop = anchor.category === "상의" && ["화이트", "아이보리"].includes(anchor.color);
  return {
    pieces,
    score: Math.max(84, 97 - variation * 4),
    title: variation === 0 ? "가장 잘 어울리는 조합" : variation === 1 ? "조금 더 편안한 조합" : "새롭게 입어보는 조합",
    reasons: isWhiteTop
      ? ["진청 데님으로 선명하게", "화이트 스니커즈로 통일", "오늘 날씨에 맞는 겉옷"]
      : [`${anchor.color}과 자연스러운 색 조합`, "내가 자주 입는 아이템", selectedPriorities.has("편안함이 우선") ? "편안한 핏 우선" : "깔끔한 비율"]
  };
}

function buildTrendMatch(trend, variation = 0) {
  const used = new Set();
  const pieces = trend.requirements.map((requirement, requirementIndex) => {
    const pool = wardrobe.filter(item =>
      !used.has(item.id) && item.category === requirement.category &&
      itemMatchesGender(item)
    ).sort((a, b) => {
      const score = item => {
        let value = 0;
        const colorIndex = (requirement.colors || []).indexOf(item.color);
        if (colorIndex >= 0) value += 40 - colorIndex * 6;
        if (requirement.keyword && item.name.includes(requirement.keyword)) value += 35;
        if (item.userAdded) value += 8;
        value += Math.min(item.worn, 12) / 3;
        value += ((Number(item.id.replace(/\D/g, "")) || 0) * (variation + requirementIndex + 2)) % 7;
        return value;
      };
      return score(b) - score(a);
    });
    const offset = variation % Math.max(1, Math.min(pool.length, 3));
    const selected = pool[offset] || pool[0];
    if (selected) used.add(selected.id);
    return selected;
  }).filter(Boolean);
  const fulfilled = pieces.filter((piece, index) => {
    const requirement = trend.requirements[index];
    return (requirement.colors || []).includes(piece.color) || (requirement.keyword && piece.name.includes(requirement.keyword));
  }).length;
  return { pieces, fulfilled, total: trend.requirements.length, score: 86 + Math.min(10, fulfilled * 3) };
}

function textTokens(value) {
  return String(value || "").toLowerCase().split(/[\s·,/()[\]_-]+/).filter(token => token.length > 1);
}

const colorRgb = {
  "블랙": [22, 22, 24], "차콜": [58, 61, 66], "그레이": [128, 130, 134], "실버": [190, 194, 199],
  "화이트": [246, 246, 242], "아이보리": [239, 232, 207], "크림": [244, 229, 190], "오트밀": [205, 191, 164],
  "네이비": [28, 42, 72], "진청": [31, 55, 90], "블루": [53, 105, 176], "연청": [137, 178, 207],
  "브라운": [108, 70, 45], "카멜": [170, 116, 67], "베이지": [205, 184, 145],
  "카키": [105, 105, 67], "올리브": [91, 104, 58], "그린": [46, 126, 79],
  "레드": [190, 42, 45], "버건디": [104, 32, 46], "코랄": [231, 107, 92], "오렌지": [226, 119, 47],
  "옐로우": [227, 194, 53], "핑크": [220, 135, 159], "라벤더": [170, 149, 202], "퍼플": [105, 69, 143]
};

function knownColors(...values) {
  const text = values.flat(Infinity).filter(Boolean).join(" ").toLowerCase();
  return Object.keys(colorRgb).filter(color => text.includes(color.toLowerCase()));
}

function rgbColorSimilarity(left, right) {
  if (left === right) return 1;
  const a = colorRgb[left];
  const b = colorRgb[right];
  if (!a || !b) return 0;
  const distance = Math.sqrt(a.reduce((sum, channel, index) => sum + ((channel - b[index]) ** 2), 0));
  return Math.exp(-((distance ** 2) / (2 * (92 ** 2))));
}

function setSimilarity(actualValues, wantedValues, groups = []) {
  const actual = textTokens((actualValues || []).join(" "));
  const wanted = textTokens((wantedValues || []).join(" "));
  if (!actual.length || !wanted.length) return .18;
  let best = 0;
  for (const left of actual) {
    for (const right of wanted) {
      if (left === right) best = Math.max(best, 1);
      else if (left.includes(right) || right.includes(left)) best = Math.max(best, .82);
      else if (groups.some(group => group.some(token => left.includes(token)) && group.some(token => right.includes(token)))) best = Math.max(best, .68);
    }
  }
  return best;
}

function garmentSimilarityDetail(item, requirement) {
  const analysis = item.analysis || window.WearwellVLM.seedGarmentAnalysis(item);
  if (item.category !== requirement.category) return { total: 0, color: 0, material: 0, fit: 0, detail: 0 };
  const actualColors = knownColors(item.color, item.name, analysis.primaryColor, analysis.secondaryColors, analysis.dominantColors);
  const wantedColors = knownColors(requirement.colors);
  const color = actualColors.length && wantedColors.length
    ? Math.max(...actualColors.flatMap(actual => wantedColors.map(wanted => rgbColorSimilarity(actual, wanted))))
    : .12;
  const material = setSimilarity(
    [analysis.material, analysis.texture, item.name], requirement.materials || [],
    [["코튼", "면"], ["울", "모직"], ["폴리에스터", "나일론", "합성"], ["레더", "가죽"], ["데님", "진"]]
  );
  const fit = setSimilarity(
    [analysis.fit, analysis.silhouette, item.name], requirement.fits || [],
    [["오버", "세미오버", "여유"], ["와이드", "세미와이드", "커브드"], ["레귤러", "스트레이트", "기본"], ["슬림", "크롭"]]
  );
  const detail = setSimilarity(
    [analysis.subcategory, analysis.pattern, analysis.finish, ...(analysis.construction || []), item.name],
    requirement.details || []
  );
  const total = .04 + color * .62 + material * .14 + fit * .12 + detail * .08;
  return { total: Math.min(1, total), color, material, fit, detail };
}

function garmentSimilarity(item, requirement) {
  return garmentSimilarityDetail(item, requirement).total;
}

function buildInfluencerMatch(look, variation = 0) {
  if (!look.analysisReady || !Array.isArray(look.pieces) || !look.pieces.length) {
    return { matches: [], pieces: [], fulfilled: 0, total: 0, similarity: 0, colorSimilarity: 0, score: 0, weatherHits: 0, analyzed: false };
  }
  const candidateLists = look.pieces.map((requirement, requirementIndex) => ({
    requirement,
    requirementIndex,
    candidates: wardrobe
      .filter(item => itemMatchesGender(item) && item.category === requirement.category)
      .map(item => {
        const detail = garmentSimilarityDetail(item, requirement);
        return { item, similarity: detail.total, detail };
      })
      .sort((a, b) => b.similarity - a.similarity || b.detail.color - a.detail.color || Number(Boolean(b.item.userAdded)) - Number(Boolean(a.item.userAdded)) || Number(a.item.sourceRank || 999) - Number(b.item.sourceRank || 999))
  })).map(entry => ({ ...entry, candidates: entry.candidates.filter(candidate => candidate.detail.color >= .56 && candidate.similarity >= .56).slice(0, 5) }));

  // 같은 옷을 두 번 쓰지 않는 최대 가중치 1:1 배정. 후보가 적은 룩북 의류부터 탐색해 탐욕 매칭의 오판을 피한다.
  const searchOrder = [...candidateLists].sort((left, right) => left.candidates.length - right.candidates.length);
  let best = { fulfilled: -1, sum: -1, matches: [] };
  const assign = (position, used, chosen, fulfilled, sum) => {
    if (position === searchOrder.length) {
      if (fulfilled > best.fulfilled || (fulfilled === best.fulfilled && sum > best.sum)) best = { fulfilled, sum, matches: [...chosen] };
      return;
    }
    const entry = searchOrder[position];
    for (const candidate of entry.candidates) {
      if (used.has(candidate.item.id)) continue;
      used.add(candidate.item.id);
      chosen.push({ ...candidate, requirement: entry.requirement, requirementIndex: entry.requirementIndex });
      assign(position + 1, used, chosen, fulfilled + 1, sum + candidate.similarity);
      chosen.pop();
      used.delete(candidate.item.id);
    }
    assign(position + 1, used, chosen, fulfilled, sum);
  };
  assign(0, new Set(), [], 0, 0);
  const matches = Array(look.pieces.length).fill(null);
  best.matches.forEach(match => { matches[match.requirementIndex] = match; });
  const fulfilled = matches.filter(Boolean).length;
  const similarity = fulfilled ? matches.filter(Boolean).reduce((sum, match) => sum + match.similarity, 0) / fulfilled : 0;
  const colorSimilarity = fulfilled ? matches.filter(Boolean).reduce((sum, match) => sum + match.detail.color, 0) / fulfilled : 0;
  const weatherHits = look.weather.filter(tag => todayWeather.tags.includes(tag)).length;
  return {
    matches,
    pieces: matches.filter(Boolean).map(match => match.item),
    fulfilled,
    total: look.pieces.length,
    similarity,
    colorSimilarity,
    score: Math.round(Math.min(98, similarity * 88 + weatherHits * 3)),
    weatherHits,
    analyzed: true
  };
}

function bodySimilarity(look, measurements = avatarMeasurements || readMeasurements()) {
  if (measurements?.photoBased) return look.bodyShapes.includes(measurements.body_shape) ? 96 : 72;
  const height = Number(measurements?.height || 170);
  const weight = Number(measurements?.weight || 65);
  const bmi = weight / ((height / 100) ** 2);
  const rangeDistance = (value, range) => value < range[0] ? range[0] - value : value > range[1] ? value - range[1] : 0;
  const heightPenalty = Math.min(45, rangeDistance(height, look.heightRange) * 2.2);
  const bmiPenalty = Math.min(35, rangeDistance(bmi, look.bmiRange) * 7);
  const shapeBonus = look.bodyShapes.includes(measurements?.body_shape) ? 10 : 0;
  return Math.max(20, Math.round(90 - heightPenalty - bmiPenalty + shapeBonus));
}

function availableInfluencerMatches({ selectedOnly = false } = {}) {
  return influencerLooks
    .filter(look => look.gender === (selectedGender || "women"))
    .filter(look => look.analysisReady)
    .filter(look => !selectedOnly || selectedInfluencerLookIds.has(look.id))
    .map(look => ({ look, match: buildInfluencerMatch(look), bodyScore: bodySimilarity(look) }))
    .filter(result => result.match.fulfilled === result.match.total && result.match.colorSimilarity >= .62 && result.match.similarity >= .61)
    .sort((a, b) => (b.bodyScore + b.match.score) - (a.bodyScore + a.match.score));
}

function buildPersonalizedLooks() {
  const candidates = availableInfluencerMatches({ selectedOnly: selectedInfluencerLookIds.size > 0 });
  if (!candidates.length) return lookSets[selectedGender || "women"];
  return candidates.map(({ look, match }) => ({
    title: `${look.creator}의 ${look.mood}`,
    mood: look.mood,
    score: match.score,
    pieces: match.pieces.map(item => wardrobe.findIndex(candidate => candidate.id === item.id)),
    reasons: [`오늘 ${todayWeather.condition}에 어울려요`, `${look.creator} 룩과 ${Math.round(match.similarity * 100)}% 유사`, "내 옷장에 있는 옷만 사용"]
  }));
}

function renderItemMatches() {
  const item = selectedWardrobeItem;
  if (!item) return;
  $("#selectedItemName").textContent = item.name;
  $("#selectedItemHint").textContent = `${genderLabel()} · ${[...selectedStyles][0] || "내 취향"} · 오늘 서울 날씨를 함께 반영했어요.`;
  $("#selectedItemPanel").innerHTML = `
    <div class="selected-item-image"><img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.name)}" /><span class="selected-badge">선택한 옷</span></div>
    <div class="selected-item-copy"><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.color)} · ${escapeHtml(item.category)}</span></div>`;
  const analysis = item.analysis || window.WearwellVLM.seedGarmentAnalysis(item);
  const analysisRows = [
    ["소재·질감", `${analysis.material} · ${analysis.texture}`],
    ["핏·실루엣", `${analysis.fit} · ${analysis.silhouette}`],
    ["주름", analysis.wrinkle],
    ["마감", analysis.finish],
    ["디테일", (analysis.construction || []).join(" · ")]
  ];
  $("#selectedItemAnalysis").innerHTML = `
    <div class="garment-analysis-head"><strong>옷 사진 텍스트 분석</strong><span>${analysis.engine === VLM_ANALYSIS_ENGINE ? "Qwen3-VL" : "기본 분석"}</span></div>
    <p>${escapeHtml(analysis.summary)}</p>
    <dl>${analysisRows.map(([label, value]) => `<div><dt>${label}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>
    <button class="analysis-rerun" id="analyzeSelectedItem">Qwen으로 사진 정밀 분석</button>`;
  const matches = [buildMatch(item, matchVariation), buildMatch(item, matchVariation + 1)];
  $("#bestMatchList").innerHTML = matches.map((match, index) => `
    <article class="match-option ${index === 0 ? "recommended" : ""}">
      <div class="match-option-top"><div class="match-option-title">${index === 0 ? '<span class="best-badge">BEST</span>' : ""}<b>${match.title}</b></div><span class="match-option-score">${match.score}% 잘 맞아요</span></div>
      <div class="match-pieces-grid">${match.pieces.map(piece => `
        <div class="match-piece ${piece.id === item.id ? "anchor" : ""}"><div class="match-piece-image"><img src="${escapeHtml(piece.image)}" alt="${escapeHtml(piece.name)}" /></div><span>${escapeHtml(piece.name)}</span></div>`).join("")}</div>
      <div class="match-reason">${match.reasons.map(reason => `<span>✓ ${escapeHtml(reason)}</span>`).join("")}</div>
      <div class="match-option-actions"><button class="use-match-button" data-use-match="${index}">이 코디로 입기</button></div>
    </article>`).join("");
  $$('[data-use-match]').forEach(button => button.addEventListener("click", () => {
    const match = matches[Number(button.dataset.useMatch)];
    $("#itemMatchDialog").close();
    tryOnItems(match.pieces);
  }));
  $("#analyzeSelectedItem").addEventListener("click", () => analyzeGarment(item, true));
}

async function persistGarment(item) {
  try {
    await window.WearwellDB.putGarment({
      id: item.id, image: item.userAdded ? item.image : undefined, userAdded: Boolean(item.userAdded),
      gender: item.gender, category: item.category, name: item.name, color: item.color,
      analysis: item.analysis, updatedAt: new Date().toISOString()
    });
  } catch (error) {
    console.warn("옷 분석 저장 실패", error);
  }
}

async function analyzeGarment(item, reopen = false) {
  const button = reopen ? $("#analyzeSelectedItem") : null;
  if (button) { button.disabled = true; button.textContent = "스타일 분석 준비 중…"; }
  try {
    item.analysis = await window.WearwellVLM.analyzeImage(item.image, item);
    await persistGarment(item);
    if (reopen) renderItemMatches();
    renderWardrobe(); renderDiscover();
    showToast("주름·핏·마감·소재 분석을 저장했어요");
  } catch (error) {
    showToast(error.message || "Qwen 분석을 시작하지 못했어요");
    if (button) { button.disabled = false; button.textContent = "Qwen으로 사진 정밀 분석"; }
  }
}

function applyLookAnalysis(look, result) {
  const detectedPieces = Array.isArray(result?.pieces) ? result.pieces : [];
  const validCategories = new Set(["상의", "하의", "아우터", "원피스", "신발", "가방", "액세서리"]);
  const arrayValue = value => (Array.isArray(value) ? value : [value]).map(entry => String(entry || "").trim()).filter(Boolean);
  const normalizedPieces = detectedPieces.map((piece, index) => ({
    pieceId: String(piece.pieceId || `piece-${index + 1}`),
    label: String(piece.label || piece.subcategory || piece.category || `의류 ${index + 1}`),
    layer: String(piece.layer || "단독"),
    category: String(piece.category || "").trim(),
    bbox: Array.isArray(piece.bbox) && piece.bbox.length === 4 ? piece.bbox.map(value => Math.max(0, Math.min(1000, Number(value) || 0))) : null,
    colors: arrayValue(piece.colors || piece.color),
    materials: arrayValue(piece.materials || piece.material),
    fits: arrayValue(piece.fits || piece.fit),
    details: arrayValue(piece.details),
    confidence: Math.max(0, Math.min(1, Number(piece.confidence ?? .8)))
  })).filter(piece => validCategories.has(piece.category) && piece.colors.length && piece.confidence >= .5);

  if (!normalizedPieces.length) throw new Error("룩북에서 개별 의류를 찾지 못했어요.");
  look.summary = result?.summary || look.summary;
  // 검출된 옷 배열을 그대로 사용한다. 같은 카테고리의 이너와 아우터를 첫 항목 하나로 합치지 않는다.
  look.pieces = normalizedPieces;
  look.vlmAnalysis = result;
  look.analysisReady = true;
  look.analysisState = "ready";
  look.analysisError = null;
}

function pieceObjectPosition(piece) {
  if (!Array.isArray(piece?.bbox) || piece.bbox.length !== 4) return "50% 50%";
  const [left, top, right, bottom] = piece.bbox;
  return `${Math.max(0, Math.min(100, (left + right) / 20))}% ${Math.max(0, Math.min(100, (top + bottom) / 20))}%`;
}

async function analyzeLookbooks(lookbooks) {
  for (const look of lookbooks) {
    try {
      const stored = await window.WearwellDB.getLook(look.id);
      look.analysisState = "analyzing";
      let result = stored?.analysisEngine === VLM_ANALYSIS_ENGINE && stored?.analysisVersion === LOOK_ANALYSIS_VERSION ? stored.analysis : null;
      if (!result) result = await window.WearwellVLM.analyzeLookImage(look.image, look);
      applyLookAnalysis(look, result);
      await window.WearwellDB.putLook({
        id: look.id, creator: look.creator, sourceUrl: look.sourceUrl, image: look.image,
        summary: look.summary, pieces: look.pieces, styles: look.styles, weather: look.weather,
        analysis: result, analysisEngine: VLM_ANALYSIS_ENGINE, analysisVersion: LOOK_ANALYSIS_VERSION, updatedAt: new Date().toISOString()
      });
      looks = buildPersonalizedLooks(); currentLook = 0;
      renderCurrentLook(); renderRotation(); renderDiscover();
      if ($('.preference-step[data-step="3"]').classList.contains("active")) renderPreferenceChoices();
    } catch (error) {
      look.analysisState = "failed";
      look.analysisError = error.message || "분석 실패";
      console.warn("룩북 Qwen 분석 실패", look.id, error);
      if ($('.preference-step[data-step="3"]').classList.contains("active")) renderPreferenceChoices();
    }
  }
}

function analyzeSelectedLookbooks() {
  return analyzeLookbooks(influencerLooks.filter(candidate => selectedInfluencerLookIds.has(candidate.id) && !candidate.vlmAnalysis && !lookbookAnalysisQueued.has(candidate.id)));
}

function queueVisibleLookbookAnalysis(looksToAnalyze) {
  const pending = looksToAnalyze.filter(look => !look.vlmAnalysis && !lookbookAnalysisQueued.has(look.id));
  pending.forEach(look => { lookbookAnalysisQueued.add(look.id); look.analysisState = "analyzing"; });
  if (!pending.length) return;
  const start = () => analyzeLookbooks(pending);
  if ("requestIdleCallback" in window) requestIdleCallback(start, { timeout: 1200 });
  else setTimeout(start, 150);
}

function openItemMatches(itemId) {
  selectedWardrobeItem = wardrobe.find(item => item.id === itemId);
  matchVariation = 0;
  renderItemMatches();
  openDialog($("#itemMatchDialog"));
}

function filteredWardrobe() {
  const query = $("#wardrobeSearch").value.trim().toLowerCase();
  return wardrobe.filter(item =>
    itemMatchesGender(item) &&
    (activeCategory === "전체" || item.category === activeCategory) &&
    `${item.name} ${item.color} ${item.category} ${item.analysis?.material || ""} ${item.analysis?.fit || ""} ${item.analysis?.finish || ""}`.toLowerCase().includes(query)
  );
}

function renderWardrobe() {
  const filtered = filteredWardrobe();
  $("#wardrobeGrid").innerHTML = filtered.slice(0, visibleWardrobe).map(item => `
    <article class="wardrobe-item" data-item-id="${escapeHtml(item.id)}" role="button" tabindex="0" aria-label="${escapeHtml(item.name)} 활용 코디 보기">
      <div class="wardrobe-image"><img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.color)} ${escapeHtml(item.name)}" loading="lazy" />${item.userAdded ? '<span class="wardrobe-status">내 사진</span>' : ""}<span class="analysis-badge ${item.analysis?.engine === VLM_ANALYSIS_ENGINE ? "qwen" : ""}">${item.analysis?.engine === VLM_ANALYSIS_ENGINE ? "Qwen 분석" : "특징 저장됨"}</span><button class="item-menu" aria-label="옷 정보 더 보기">···</button><button class="garment-select-button ${selectedGarmentIds.has(item.id) ? "selected" : ""}" data-select-garment="${escapeHtml(item.id)}">${selectedGarmentIds.has(item.id) ? "✓ 선택됨" : "+ 아바타에 입기"}</button></div>
      <div class="wardrobe-info"><strong>${item.brand ? `${escapeHtml(item.brand)} · ` : ""}${escapeHtml(item.name)}</strong><span>${item.sourceRank ? `랭킹 ${escapeHtml(item.sourceRank)}위 · ` : ""}${escapeHtml(item.color)} · ${escapeHtml(item.category)}${item.worn ? ` · ${escapeHtml(item.worn)}번 입음` : ""}</span></div>
    </article>`).join("");
  $("#loadMore").hidden = visibleWardrobe >= filtered.length;
  $("#wardrobeCount").textContent = filtered.length;
  $$('[data-item-id]').forEach(card => {
    card.addEventListener("click", event => { if (!event.target.closest(".item-menu") && !event.target.closest(".garment-select-button")) openItemMatches(card.dataset.itemId); });
    card.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") openItemMatches(card.dataset.itemId); });
  });
  $$('.item-menu').forEach(button => button.addEventListener("click", event => { event.stopPropagation(); showToast("옷 정보를 수정할 수 있어요"); }));
  $$('[data-select-garment]').forEach(button => button.addEventListener("click", event => { event.stopPropagation(); toggleGarmentSelection(button.dataset.selectGarment); }));
}

function toggleGarmentSelection(itemId) {
  if (selectedGarmentIds.has(itemId)) selectedGarmentIds.delete(itemId);
  else if (selectedGarmentIds.size >= 4) return showToast("한 번에 최대 4개까지 입혀볼 수 있어요");
  else selectedGarmentIds.add(itemId);
  updateSelectionTray(); renderWardrobe();
}

function updateSelectionTray() {
  const items = [...selectedGarmentIds].map(id => wardrobe.find(item => item.id === id)).filter(Boolean);
  $("#selectionTray").classList.toggle("show", items.length > 0);
  $("#selectionTrayCount").textContent = `${items.length}/4개 선택`;
  $("#selectionThumbs").innerHTML = items.map(item => `<img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.name)}" title="${escapeHtml(item.name)}" />`).join("");
}

function renderDiscover() {
  const results = availableInfluencerMatches({ selectedOnly: selectedInfluencerLookIds.size > 0 });
  const moods = ["전체", ...new Set(results.map(result => result.look.mood))];
  $("#moodFilters").innerHTML = moods.map(mood => `<button class="mood-chip ${mood === activeMood ? "active" : ""}" data-mood="${mood}">${mood}</button>`).join("");
  $$('[data-mood]').forEach(button => button.addEventListener("click", () => { activeMood = button.dataset.mood; renderDiscover(); }));
  const cards = results.filter(result => activeMood === "전체" || result.look.mood === activeMood);
  $("#discoverGrid").innerHTML = cards.length ? cards.map(({ look, match, bodyScore }) => `
    <article class="discover-card">
      <div class="discover-image"><img data-trend-hero="${escapeHtml(look.id)}" src="${escapeHtml(look.image)}" alt="${escapeHtml(look.sourceTitle)}" loading="lazy" /><span class="trend-reference-badge">사진 룩북 · ${escapeHtml(look.creator)}</span></div>
      <button class="save-discover ${savedLooks.has(`d-${selectedGender}-${look.id}`) ? "saved" : ""}" data-save-discover="${escapeHtml(look.id)}" aria-label="코디 저장">${savedLooks.has(`d-${selectedGender}-${look.id}`) ? "♥" : "♡"}</button>
      <div class="discover-info"><div><strong>${escapeHtml(look.mood)}</strong><p><a href="${escapeHtml(look.sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(look.credit || look.creator)} · 원본 사진</a></p></div><span class="closet-match">체형 ${bodyScore}%</span></div>
      <p class="look-analysis-copy">${escapeHtml(look.summary)}</p>
      <div class="look-feature-chips">${look.styles.map(style => `<span>#${escapeHtml(style)}</span>`).join("")}</div>
      <div class="discover-closet-match">
        <div class="discover-match-head"><strong>이 옷들을 입어서 비슷하게 연출해봐요</strong><span>색상 ${Math.round(match.colorSimilarity * 100)}% · 전체 ${Math.round(match.similarity * 100)}%</span></div>
        <div class="matched-items">${match.matches.map(entry => `<button class="matched-thumb" data-match-item="${escapeHtml(entry.item.id)}" aria-label="룩북 ${escapeHtml(entry.requirement.label || entry.requirement.category)}와 내 옷 ${escapeHtml(entry.item.name)} 비교"><span class="match-pair-images"><span><img src="${escapeHtml(look.image)}" alt="룩북 ${escapeHtml(entry.requirement.label || entry.requirement.category)}" style="object-position:${pieceObjectPosition(entry.requirement)}" /><em>룩북</em></span><span><img src="${escapeHtml(entry.item.image)}" alt="내 옷 ${escapeHtml(entry.item.name)}" /><em>내 옷</em></span></span><small><b>${escapeHtml(entry.requirement.label || entry.requirement.category)}</b><span>→ ${escapeHtml(entry.item.name)}</span></small><i>${Math.round(entry.similarity * 100)}%</i></button>`).join("")}</div>
        <ol class="matched-instructions">${match.matches.map(entry => `<li><b>룩북 ${escapeHtml(entry.requirement.label || entry.requirement.category)}</b><span>${escapeHtml((entry.requirement.colors || []).join("·"))} → 내 옷 ${escapeHtml(entry.item.name)}</span></li>`).join("")}</ol>
        <button class="trend-avatar-button" data-try-trend="${escapeHtml(look.id)}">룩북 원본과 내 아바타 비교하기 ✦</button>
      </div>
    </article>`).join("") : '<div class="discover-empty"><strong>개별 옷이 모두 맞는 룩이 아직 없어요.</strong><p>Qwen 분석이 끝난 뒤, 룩북 속 각 옷에 대응하는 내 옷이 전부 있을 때만 추천해요.</p></div>';
  $$('[data-save-discover]').forEach(button => button.addEventListener("click", () => {
    const id = `d-${selectedGender}-${button.dataset.saveDiscover}`;
    savedLooks.has(id) ? savedLooks.delete(id) : savedLooks.add(id);
    persistSaves(); renderDiscover();
  }));
  $$('[data-match-item]').forEach(button => button.addEventListener("click", () => openItemMatches(button.dataset.matchItem)));
  $$('[data-try-trend]').forEach(button => button.addEventListener("click", () => {
    const result = cards.find(card => card.look.id === button.dataset.tryTrend);
    tryOnItems(result.match.pieces, { image: result.look.image, title: `${result.look.creator} · ${result.look.mood}` });
  }));
  $("#analysisPieces").textContent = filteredWardrobe().length;
  $("#analysisLooks").textContent = cards.length;
  updateSavedCount();
}

function persistSaves() { localStorage.setItem("오늘옷-saved", JSON.stringify([...savedLooks])); updateSavedCount(); }
function updateSavedCount() { $("#savedCount").textContent = savedLooks.size; }

function setView(view) {
  $$('.nav-link').forEach(button => button.classList.toggle("active", button.dataset.view === view));
  $$('.view').forEach(section => section.classList.toggle("active", section.id === `${view}View`));
  if (view === "wardrobe") renderWardrobe();
  if (view === "discover") renderDiscover();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderGenderChoices() {
  const choices = [
    ["women", "여성", "여성 데일리룩과 옷장으로 시작해요", "♀"],
    ["men", "남성", "남성 데일리룩과 옷장으로 시작해요", "♂"]
  ];
  $("#genderChoices").innerHTML = choices.map(([value, title, detail, icon]) => `
    <button class="gender-choice ${selectedGender === value ? "selected" : ""}" data-gender="${value}"><b>${selectedGender === value ? "✓" : icon}</b><strong>${title}</strong><span>${detail}</span></button>`).join("");
  $$('[data-gender]').forEach(button => button.addEventListener("click", () => {
    if (selectedGender !== button.dataset.gender) {
      selectedStyles.clear(); selectedInfluencerLookIds.clear(); avatarMeasurements = null; avatarImage = null; fullBodyPhoto = null;
      localStorage.removeItem("오늘옷-avatar");
      $("#avatarPreview").innerHTML = "<span>내 아바타가<br />여기에 만들어져요</span>";
      $("#profileAvatar").classList.remove("has-image"); $("#profileAvatar").style.backgroundImage = "";
    }
    selectedGender = button.dataset.gender;
    $("#nextGender").disabled = false;
    renderGenderChoices();
  }));
}

function renderPreferenceChoices() {
  const readyOptions = availableInfluencerMatches().slice(0, 6);
  const bodyCandidates = influencerLooks
    .filter(look => look.gender === (selectedGender || "women"))
    .sort((left, right) => bodySimilarity(right) - bodySimilarity(left));
  const pendingLooks = bodyCandidates
    .filter(look => !look.analysisReady && look.analysisState !== "failed")
    .slice(0, Math.max(0, 6 - readyOptions.length));
  const options = [...readyOptions, ...pendingLooks.map(look => ({ look, match: null, bodyScore: bodySimilarity(look) }))];
  $("#styleChoices").innerHTML = options.length ? options.map(({ look, match, bodyScore }) => `
    <button class="influencer-choice ${selectedInfluencerLookIds.has(look.id) ? "selected" : ""} ${match ? "" : "analyzing"}" data-influencer-look="${escapeHtml(look.id)}">
      <span class="influencer-image"><img src="${escapeHtml(look.image)}" alt="${escapeHtml(look.sourceTitle)}" /><i>✓</i></span>
      <span class="influencer-copy"><b>${escapeHtml(look.creator)}</b><strong>${escapeHtml(look.mood)}</strong><small>${escapeHtml(look.publicSpec)}</small><em>${match ? `개별 옷 ${match.fulfilled}/${match.total} 매칭 · 색상 ${Math.round(match.colorSimilarity * 100)}%` : "취향 선택 가능 · 룩북 속 옷을 하나씩 분석 중…"}</em></span>
    </button>`).join("") : '<div class="influencer-empty">개별 의류 분석을 통과하고 내 옷장과 모두 매칭되는 룩을 찾지 못했어요.</div>';
  $$('[data-influencer-look]').forEach(button => button.addEventListener("click", () => {
    const id = button.dataset.influencerLook;
    selectedInfluencerLookIds.has(id) ? selectedInfluencerLookIds.delete(id) : selectedInfluencerLookIds.add(id);
    selectedStyles = new Set(influencerLooks.filter(look => selectedInfluencerLookIds.has(look.id)).flatMap(look => look.styles));
    renderPreferenceChoices(); updatePreferenceCount();
  }));
  if (selectedGender && $('.preference-step[data-step="3"]').classList.contains("active")) queueVisibleLookbookAnalysis(pendingLooks);
}

function updatePreferenceCount() {
  $("#selectionCount").textContent = `${selectedInfluencerLookIds.size}개 선택`;
  $("#nextPreferences").disabled = selectedInfluencerLookIds.size < 2;
}

function renderPriorityChoices() {
  $("#priorityChoices").innerHTML = priorityOptions.map(([icon, title, detail]) => `<button class="priority-choice ${selectedPriorities.has(title) ? "selected" : ""}" data-priority="${title}"><b>${icon}</b><span><strong>${title}</strong><small>${detail}</small></span></button>`).join("");
  $$('[data-priority]').forEach(button => button.addEventListener("click", () => {
    const priority = button.dataset.priority;
    selectedPriorities.has(priority) ? selectedPriorities.delete(priority) : selectedPriorities.add(priority);
    renderPriorityChoices();
  }));
}

function showPreferenceStep(step) {
  $$('.preference-step').forEach(section => section.classList.toggle("active", Number(section.dataset.step) === step));
  $("#progressOne").classList.toggle("active", step >= 1);
  $("#progressTwo").classList.toggle("active", step >= 2);
  $("#progressThree").classList.toggle("active", step >= 3);
  $("#progressFour").classList.toggle("active", step >= 4);
}

function savePreferences() {
  avatarMeasurements = readMeasurements();
  const profile = { gender: selectedGender, styles: [...selectedStyles], influencerLooks: [...selectedInfluencerLookIds], priorities: [...selectedPriorities], measurements: avatarMeasurements, avatar: avatarImage };
  localStorage.setItem("오늘옷-profile", JSON.stringify(profile));
  applyProfile(profile);
  closeDialogs();
  showToast(`${genderLabel()} 맞춤 옷장이 준비됐어요`);
  analyzeSelectedLookbooks();
}

function applyProfile(profile) {
  if (!profile?.gender || !profile?.styles?.length) return;
  selectedGender = profile.gender;
  selectedStyles = new Set(profile.styles);
  selectedInfluencerLookIds = new Set(profile.influencerLooks || []);
  selectedPriorities = new Set(profile.priorities || []);
  avatarMeasurements = profile.measurements || avatarMeasurements;
  if (avatarMeasurements?.photoBased) {
    fullBodyPhoto = profile.avatar || avatarImage;
    setBodyInputMethod("photo");
    if (fullBodyPhoto) $("#fullBodyPreview").innerHTML = `<img src="${escapeHtml(fullBodyPhoto)}" alt="저장한 전신사진" /><span><strong>전신사진 저장됨</strong><small>다시 누르면 사진을 바꿀 수 있어요</small></span>`;
  } else setBodyInputMethod("measurements");
  if (!selectedInfluencerLookIds.size) availableInfluencerMatches().slice(0, 3).forEach(result => selectedInfluencerLookIds.add(result.look.id));
  looks = buildPersonalizedLooks();
  currentLook = 0;
  activeMood = "전체";
  $("#signalTags").innerHTML = profile.styles.slice(0, 3).map(style => `<span>${style}</span>`).join("");
  $("#tasteSummary").textContent = profile.styles[0];
  $("#tasteDetail").textContent = `${genderLabel()} · ${profile.priorities[0] || "내 취향 중심"}으로 추천 중`;
  if (avatarMeasurements) applyMeasurementValues(avatarMeasurements);
  if (profile.avatar || avatarImage) showAvatar(profile.avatar || avatarImage, "saved");
  renderCurrentLook(); renderRotation(); renderWardrobe(); renderDiscover();
}

function handleUploads(files) {
  uploadFiles = [...files].filter(file => file.type.startsWith("image/")).slice(0, 8);
  $("#uploadPreview").innerHTML = uploadFiles.map(file => `<div class="upload-thumb"><img src="${URL.createObjectURL(file)}" alt="올릴 사진 미리보기" /></div>`).join("");
  $("#addUploads").disabled = uploadFiles.length === 0;
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

async function addUploadsToWardrobe() {
  $("#addUploads").disabled = true;
  const added = [];
  for (const [index, file] of uploadFiles.entries()) {
    const lower = file.name.toLowerCase();
    let category = "상의";
    if (/coat|jacket|blazer|코트|재킷/.test(lower)) category = "아우터";
    else if (/pant|jean|skirt|short|바지|치마/.test(lower)) category = "하의";
    else if (/dress|원피스/.test(lower)) category = "원피스";
    else if (/shoe|boot|sneaker|loafer|신발/.test(lower)) category = "신발";
    else if (/bag|tote|가방/.test(lower)) category = "가방";
    const item = { id: `upload-${Date.now()}-${index}`, image: await fileToDataUrl(file), gender: selectedGender, category, name: file.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ") || "새로 추가한 옷", color: "색상 미분류", worn: 0, userAdded: true };
    item.analysis = window.WearwellVLM.seedGarmentAnalysis(item);
    wardrobe.unshift(item);
    added.push(item);
    await persistGarment(item);
  }
  uploadFiles = [];
  closeDialogs(); activeCategory = "전체";
  renderCategoryFilters(); renderWardrobe();
  showToast("내 옷장과 분석 데이터베이스에 저장했어요");
  (async () => {
    for (const item of added) await analyzeGarment(item);
  })();
}

async function restoreWardrobeDatabase() {
  try {
    const records = await window.WearwellDB.getAllGarments();
    records.forEach(record => {
      const existing = wardrobe.find(item => item.id === record.id);
      if (existing) existing.analysis = record.analysis || existing.analysis;
      else if (record.userAdded && record.image) wardrobe.unshift({ ...record, worn: 0 });
    });
    renderWardrobe(); renderDiscover();
    rankingWardrobe.forEach(item => {
      const current = wardrobe.find(candidate => candidate.id === item.id);
      if (current) persistGarment(current);
    });
    for (const look of influencerLooks) {
      const stored = await window.WearwellDB.getLook(look.id);
      if (stored?.analysisEngine === VLM_ANALYSIS_ENGINE && stored?.analysisVersion === LOOK_ANALYSIS_VERSION && stored.analysis) {
        applyLookAnalysis(look, stored.analysis);
        continue;
      }
      await window.WearwellDB.putLook({
        id: look.id, creator: look.creator, sourceUrl: look.sourceUrl, image: look.image,
        summary: look.summary, pieces: look.pieces, styles: look.styles, weather: look.weather,
        analysisEngine: "curated-qwen-schema", updatedAt: new Date().toISOString()
      });
    }
    looks = buildPersonalizedLooks();
    renderPreferenceChoices(); renderDiscover();
    queueVisibleLookbookAnalysis(influencerLooks.filter(look => selectedInfluencerLookIds.has(look.id) && !look.analysisReady));
  } catch (error) {
    console.warn("옷장 데이터베이스 복원 실패", error);
  }
}

async function analyzeVisibleWardrobe() {
  const button = $("#analyzeVisibleWardrobe");
  const items = filteredWardrobe().slice(0, visibleWardrobe);
  button.disabled = true;
  try {
    for (let index = 0; index < items.length; index++) {
      $("#analysisStatus").textContent = `Qwen3-VL로 ${index + 1}/${items.length}번째 옷의 주름·핏·마감을 읽고 있어요.`;
      await analyzeGarment(items[index]);
    }
  } finally {
    button.disabled = false;
    $("#analysisStatus").textContent = "주름·핏·마감·소재·봉제 디테일을 옷별로 저장하고 비교해요.";
  }
}

function initEvents() {
  $$('.nav-link').forEach(button => button.addEventListener("click", () => setView(button.dataset.view)));
  $$('[data-go]').forEach(button => button.addEventListener("click", () => setView(button.dataset.go)));
  $$('[data-close-dialog]').forEach(button => button.addEventListener("click", closeDialogs));
  $("#refreshLook").addEventListener("click", nextLook);
  $("#skipLook").addEventListener("click", nextLook);
  $("#wearLook").addEventListener("click", () => showToast("오늘 입을 코디로 정했어요. 좋은 하루 보내세요!"));
  $("#saveLook").addEventListener("click", () => {
    const id = `${selectedGender}-${currentLook}`;
    savedLooks.has(id) ? savedLooks.delete(id) : savedLooks.add(id);
    persistSaves(); renderCurrentLook(); showToast(savedLooks.has(id) ? "코디를 저장했어요" : "저장을 취소했어요");
  });
  $("#openPreferences").addEventListener("click", () => {
    showPreferenceStep(1); renderGenderChoices(); renderPreferenceChoices(); updatePreferenceCount();
    $("#nextGender").disabled = !selectedGender; openDialog($("#preferenceDialog"));
  });
  $("#profileAvatar").addEventListener("click", () => { showPreferenceStep(2); openDialog($("#preferenceDialog")); });
  $("#tuneTaste").addEventListener("click", () => $("#openPreferences").click());
  $("#nextGender").addEventListener("click", () => {
    if (!avatarMeasurements) applyMeasurementValues({ gender: selectedGender, height: selectedGender === "men" ? 175 : 165, weight: selectedGender === "men" ? 70 : 55, body_shape: "보통" });
    showPreferenceStep(2);
  });
  $("#backToGender").addEventListener("click", () => showPreferenceStep(1));
  $$("[data-body-method]").forEach(button => button.addEventListener("click", () => {
    if (bodyInputMethod !== button.dataset.bodyMethod) {
      avatarImage = null; avatarMeasurements = null;
      $("#avatarPreview").innerHTML = "<span>내 아바타가<br />여기에 만들어져요</span>";
    }
    setBodyInputMethod(button.dataset.bodyMethod);
  }));
  $("#fullBodyInput").addEventListener("change", async event => {
    const file = event.target.files?.[0];
    if (!file) return;
    fullBodyPhoto = await resizeBodyPhoto(file);
    avatarImage = null; avatarMeasurements = null;
    $("#fullBodyPreview").innerHTML = `<img src="${fullBodyPhoto}" alt="선택한 전신사진" /><span><strong>전신사진 선택 완료</strong><small>다시 누르면 사진을 바꿀 수 있어요</small></span>`;
    $("#avatarEngineStatus").textContent = "이 사진을 아바타 기준으로 사용할 수 있어요";
  });
  $("#generateAvatar").addEventListener("click", generateAvatar);
  $("#nextBody").addEventListener("click", async () => { if (!avatarImage) await generateAvatar(); renderPreferenceChoices(); updatePreferenceCount(); showPreferenceStep(3); });
  $("#backGender").addEventListener("click", () => showPreferenceStep(2));
  $("#nextPreferences").addEventListener("click", () => showPreferenceStep(4));
  $("#backPreferences").addEventListener("click", () => showPreferenceStep(3));
  $("#finishPreferences").addEventListener("click", savePreferences);
  $("#uploadButton").addEventListener("click", () => openDialog($("#uploadDialog")));
  $("#fileInput").addEventListener("change", event => handleUploads(event.target.files));
  $("#addUploads").addEventListener("click", addUploadsToWardrobe);
  $("#analyzeVisibleWardrobe").addEventListener("click", analyzeVisibleWardrobe);
  $("#trySelectedGarments").addEventListener("click", () => tryOnItems([...selectedGarmentIds].map(id => wardrobe.find(item => item.id === id)).filter(Boolean)));
  $("#clearGarmentSelection").addEventListener("click", () => { selectedGarmentIds.clear(); updateSelectionTray(); renderWardrobe(); });
  $("#regenerateMatches").addEventListener("click", () => { matchVariation = (matchVariation + 1) % 4; renderItemMatches(); showToast("내 옷장에서 다른 조합을 찾았어요"); });
  $("#loadMore").addEventListener("click", () => { visibleWardrobe += 24; renderWardrobe(); });
  $("#wardrobeSearch").addEventListener("input", () => { visibleWardrobe = 24; renderWardrobe(); });
  $("#weatherButton").addEventListener("click", () => openDialog($("#weatherDialog")));
  const dropZone = $("#dropZone");
  ["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add("drag"); }));
  ["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.remove("drag"); }));
  dropZone.addEventListener("drop", event => handleUploads(event.dataTransfer.files));
  $$('dialog').forEach(dialog => dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }));
}

function init() {
  renderCurrentLook(); renderRotation(); renderCategoryFilters(); renderWardrobe(); renderDiscover();
  renderGenderChoices(); renderPreferenceChoices(); renderPriorityChoices(); updatePreferenceCount(); updateSavedCount(); initEvents();
  const storedProfile = JSON.parse(localStorage.getItem("오늘옷-profile") || localStorage.getItem("morrow-profile") || "null");
  if (storedProfile?.gender) applyProfile(storedProfile);
  else setTimeout(() => openDialog($("#preferenceDialog")), 350);
  restoreWardrobeDatabase();
  window.addEventListener("wearwell:vlm-status", event => {
    const { state, detail } = event.detail;
    $("#analysisTitle").textContent = state === "loading" ? "Qwen3-VL을 준비하고 있어요" : state === "analyzing" ? "옷 사진을 정밀 분석 중이에요" : state === "error" ? "기본 분석으로 계속 추천할게요" : "Qwen3-VL 옷 분석 준비 완료";
    $("#analysisStatus").textContent = detail;
  });
}

init();
