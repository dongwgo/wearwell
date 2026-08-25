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
let wardrobe = rankingWardrobe.map(item => ({ ...item }));
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
let selectedPriorities = new Set(["날씨에 잘 맞기"]);
let uploadFiles = [];
let savedLooks = new Set(JSON.parse(localStorage.getItem("오늘옷-saved") || "[]"));
let selectedWardrobeItem = null;
let matchVariation = 0;
let avatarImage = localStorage.getItem("오늘옷-avatar") || null;
let avatarMeasurements = null;
const selectedGarmentIds = new Set();
const tryonCache = new Map();
const API_BASE = window.resolveWearwellApiBase();
const API_TOKEN = window.resolveWearwellApiToken();
const API_HEADERS = { "Content-Type": "application/json", ...(API_TOKEN ? { Authorization: `Bearer ${API_TOKEN}` } : {}) };

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
  localStorage.setItem("오늘옷-avatar", image);
  $("#avatarPreview").innerHTML = `<img src="${image}" alt="내 체형 아바타" />`;
  $("#profileAvatar").classList.add("has-image");
  $("#profileAvatar").style.backgroundImage = `url(${image})`;
  if (engine) $("#avatarEngineStatus").textContent = engine.includes("cuda") ? "내 체형 아바타가 완성됐어요" : "기본 아바타 미리보기";
}

async function generateAvatar() {
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

async function imageToDataUrl(source) {
  if (source.startsWith("data:")) return source;
  const response = await fetch(source);
  const blob = await response.blob();
  return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(reader.result); reader.onerror = reject; reader.readAsDataURL(blob); });
}

function garmentType(category) {
  return ({ 상의: "upper", 하의: "lower", 원피스: "overall", 아우터: "outer", 신발: "shoes", 가방: "bag", 액세서리: "accessory" })[category] || null;
}

async function tryOnItems(items, targetImage = null) {
  if (!items.length) return showToast("입혀볼 옷을 먼저 골라주세요");
  if (!avatarImage) await generateAvatar();
  const avatarKey = avatarMeasurements ? JSON.stringify(avatarMeasurements) : avatarImage.slice(-64);
  const cacheKey = `${items.map(item => item.id).join("-")}-${avatarKey}`;
  $("#tryonGarments").innerHTML = items.map(item => `<div><img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.name)}" /><span>${escapeHtml(item.name)}</span></div>`).join("");
  $("#tryonStage").innerHTML = '<div class="generation-loader"><span></span><strong>내 아바타에 입혀보는 중</strong><small>선택한 옷을 한 번에 조합하고 있어요</small></div>';
  $("#tryonStatusText").textContent = "선택한 옷의 색과 형태를 살려 조합하고 있어요.";
  openDialog($("#avatarTryonDialog"));
  if (tryonCache.has(cacheKey)) {
    const cached = tryonCache.get(cacheKey); $("#tryonStage").innerHTML = `<img src="${cached}" alt="AI 가상 착장 결과" />`; return;
  }
  try {
    const wearableItems = items.filter(item => garmentType(item.category)).slice(0, 4);
    if (!wearableItems.length) throw new Error("No supported garments");
    const garments = await Promise.all(wearableItems.map(async item => ({ image: await imageToDataUrl(item.image), category: garmentType(item.category), name: item.name })));
    const response = await fetch(`${API_BASE}/api/tryon`, { method: "POST", headers: API_HEADERS, body: JSON.stringify({ avatar: avatarImage, garments, seed: 42 }) });
    if (!response.ok) throw new Error("Try-on backend unavailable");
    const result = await response.json();
    tryonCache.set(cacheKey, result.image);
    $("#tryonStage").innerHTML = `<img src="${result.image}" alt="AI 가상 착장 결과" />`;
    $("#tryonStatusText").textContent = "완성됐어요. 옷 사진의 디테일을 아바타 체형에 맞춰 표현했어요.";
    $("#tryonEngineLabel").textContent = "AI 착장 결과";
    if (targetImage) targetImage.src = result.image;
  } catch {
    $("#tryonStage").innerHTML = `<img src="${avatarImage}" alt="아바타 미리보기" />`;
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

function renderItemMatches() {
  const item = selectedWardrobeItem;
  if (!item) return;
  $("#selectedItemName").textContent = item.name;
  $("#selectedItemHint").textContent = `${genderLabel()} · ${[...selectedStyles][0] || "내 취향"} · 오늘 서울 날씨를 함께 반영했어요.`;
  $("#selectedItemPanel").innerHTML = `
    <div class="selected-item-image"><img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.name)}" /><span class="selected-badge">선택한 옷</span></div>
    <div class="selected-item-copy"><strong>${escapeHtml(item.name)}</strong><span>${escapeHtml(item.color)} · ${escapeHtml(item.category)}</span></div>`;
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
    `${item.name} ${item.color} ${item.category}`.toLowerCase().includes(query)
  );
}

function renderWardrobe() {
  const filtered = filteredWardrobe();
  $("#wardrobeGrid").innerHTML = filtered.slice(0, visibleWardrobe).map(item => `
    <article class="wardrobe-item" data-item-id="${escapeHtml(item.id)}" role="button" tabindex="0" aria-label="${escapeHtml(item.name)} 활용 코디 보기">
      <div class="wardrobe-image"><img src="${escapeHtml(item.image)}" alt="${escapeHtml(item.color)} ${escapeHtml(item.name)}" loading="lazy" />${item.userAdded ? '<span class="wardrobe-status">방금 추가</span>' : ""}<button class="item-menu" aria-label="옷 정보 더 보기">···</button><button class="garment-select-button ${selectedGarmentIds.has(item.id) ? "selected" : ""}" data-select-garment="${escapeHtml(item.id)}">${selectedGarmentIds.has(item.id) ? "✓ 선택됨" : "+ 아바타에 입기"}</button></div>
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
  const trends = trendSets[selectedGender || "women"];
  const moods = ["전체", ...new Set(trends.map(trend => trend.mood))];
  $("#moodFilters").innerHTML = moods.map(mood => `<button class="mood-chip ${mood === activeMood ? "active" : ""}" data-mood="${mood}">${mood}</button>`).join("");
  $$('[data-mood]').forEach(button => button.addEventListener("click", () => { activeMood = button.dataset.mood; renderDiscover(); }));
  const cards = Array.from({ length: 24 }, (_, index) => {
    const trend = trends[index % trends.length];
    const match = buildTrendMatch(trend, Math.floor(index / trends.length));
    return { ...trend, id: index, image: photo(trend.image + Math.floor(index / trends.length)), match };
  }).filter(trend => activeMood === "전체" || trend.mood === activeMood);
  $("#discoverGrid").innerHTML = cards.map(trend => `
    <article class="discover-card">
      <div class="discover-image"><img data-trend-hero="${trend.id}" src="${trend.image}" alt="${trend.title}" loading="lazy" /><span class="trend-reference-badge">최신 룩 레퍼런스</span></div>
      <button class="save-discover ${savedLooks.has(`d-${selectedGender}-${trend.id}`) ? "saved" : ""}" data-save-discover="${trend.id}" aria-label="코디 저장">${savedLooks.has(`d-${selectedGender}-${trend.id}`) ? "♥" : "♡"}</button>
      <div class="discover-info"><div><strong>${trend.title}</strong><p><a href="${trend.sourceUrl}" target="_blank" rel="noreferrer">${trend.source}</a> · 최신 트렌드</p></div><span class="closet-match">2026.08</span></div>
      <div class="discover-closet-match">
        <div class="discover-match-head"><strong>내 옷으로 이렇게 입어요</strong><span>${trend.match.fulfilled}/${trend.match.total}개 조건 매칭</span></div>
        <div class="matched-items">${trend.match.pieces.map(piece => `<button class="matched-thumb" data-match-item="${piece.id}" aria-label="${piece.name} 활용 코디 보기"><img src="${piece.image}" alt="${piece.name}" /><small>${piece.name}</small></button>`).join("")}</div>
        <button class="trend-avatar-button" data-try-trend="${trend.id}">내 아바타에 이 조합 입혀보기 ✦</button>
      </div>
    </article>`).join("");
  $$('[data-save-discover]').forEach(button => button.addEventListener("click", () => {
    const id = `d-${selectedGender}-${button.dataset.saveDiscover}`;
    savedLooks.has(id) ? savedLooks.delete(id) : savedLooks.add(id);
    persistSaves(); renderDiscover();
  }));
  $$('[data-match-item]').forEach(button => button.addEventListener("click", () => openItemMatches(button.dataset.matchItem)));
  $$('[data-try-trend]').forEach(button => button.addEventListener("click", () => {
    const trend = cards.find(card => String(card.id) === button.dataset.tryTrend);
    const target = $(`[data-trend-hero="${trend.id}"]`);
    tryOnItems(trend.match.pieces, target);
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
      selectedStyles.clear(); avatarMeasurements = null; avatarImage = null;
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
  const options = styleOptions[selectedGender || "women"];
  $("#styleChoices").innerHTML = options.map(option => `<button class="style-choice ${selectedStyles.has(option.name) ? "selected" : ""}" data-style="${option.name}"><img src="${photo(option.image)}" alt="${option.name}" /><span>${option.name}</span><i>✓</i></button>`).join("");
  $$('[data-style]').forEach(button => button.addEventListener("click", () => {
    const style = button.dataset.style;
    selectedStyles.has(style) ? selectedStyles.delete(style) : selectedStyles.add(style);
    renderPreferenceChoices(); updatePreferenceCount();
  }));
}

function updatePreferenceCount() {
  $("#selectionCount").textContent = `${selectedStyles.size}개 선택`;
  $("#nextPreferences").disabled = selectedStyles.size < 3;
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
  const profile = { gender: selectedGender, styles: [...selectedStyles], priorities: [...selectedPriorities], measurements: avatarMeasurements, avatar: avatarImage };
  localStorage.setItem("오늘옷-profile", JSON.stringify(profile));
  applyProfile(profile);
  closeDialogs();
  showToast(`${genderLabel()} 맞춤 옷장이 준비됐어요`);
}

function applyProfile(profile) {
  if (!profile?.gender || !profile?.styles?.length) return;
  selectedGender = profile.gender;
  selectedStyles = new Set(profile.styles);
  selectedPriorities = new Set(profile.priorities || []);
  avatarMeasurements = profile.measurements || avatarMeasurements;
  looks = lookSets[selectedGender];
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

function addUploadsToWardrobe() {
  uploadFiles.forEach((file, index) => {
    const lower = file.name.toLowerCase();
    let category = "상의";
    if (/coat|jacket|blazer|코트|재킷/.test(lower)) category = "아우터";
    else if (/pant|jean|skirt|short|바지|치마/.test(lower)) category = "하의";
    else if (/dress|원피스/.test(lower)) category = "원피스";
    else if (/shoe|boot|sneaker|loafer|신발/.test(lower)) category = "신발";
    else if (/bag|tote|가방/.test(lower)) category = "가방";
    wardrobe.unshift({ id: `upload-${Date.now()}-${index}`, image: URL.createObjectURL(file), gender: selectedGender, category, name: file.name.replace(/\.[^.]+$/, "").replace(/[-_]/g, " ") || "새로 추가한 옷", color: "색상 미분류", worn: 0, userAdded: true });
  });
  uploadFiles = [];
  closeDialogs(); activeCategory = "전체";
  renderCategoryFilters(); renderWardrobe();
  showToast("내 옷장에 추가했어요. 이제 코디에 활용할게요");
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
  $("#generateAvatar").addEventListener("click", generateAvatar);
  $("#nextBody").addEventListener("click", async () => { if (!avatarImage) await generateAvatar(); renderPreferenceChoices(); updatePreferenceCount(); showPreferenceStep(3); });
  $("#backGender").addEventListener("click", () => showPreferenceStep(2));
  $("#nextPreferences").addEventListener("click", () => showPreferenceStep(4));
  $("#backPreferences").addEventListener("click", () => showPreferenceStep(3));
  $("#finishPreferences").addEventListener("click", savePreferences);
  $("#uploadButton").addEventListener("click", () => openDialog($("#uploadDialog")));
  $("#fileInput").addEventListener("change", event => handleUploads(event.target.files));
  $("#addUploads").addEventListener("click", addUploadsToWardrobe);
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
}

init();
