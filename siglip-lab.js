/**
 * 개발용 SigLIP 탭 — 옷 사진 한 장을 내 옷장 전체와 견줘 닮은 순으로 세운다.
 *
 * Seg Lab은 "옷을 어디까지 잡았나", Refine Lab은 "그 조각을 어떻게 그림으로 되돌리나",
 * Qwen Lab은 "사진을 무슨 말로 옮기나"를 본다. 이 탭이 보는 건 앱에서 **말을 거치지
 * 않는** 한 칸이다: 사진을 벡터 하나로 옮겨 코사인 하나로 견주는 일. 옷장 추천이
 * 룩북의 옷과 내 옷을 이어 붙일 때 실제로 쓰는 신호가 이것이라, 여기서 보는 순위가
 * 곧 추천이 무엇을 "닮았다"고 부르는지다.
 *
 * 다른 랩과 같이 app.js의 전역에 기대지 않는다. 옷장은 두 출처에서 직접 모은다 —
 * 저장소에 함께 배포되는 기본 옷장(window.MUSINSA_RANKING)과, 사용자가 직접 올려
 * IndexedDB에 들어 있는 옷(WearwellDB.getAllGarments).
 */
(() => {
  const ENDPOINT_KEY = "wearwell-sigliplab-endpoint";
  const SCOPE_KEY = "wearwell-sigliplab-scope";
  // SigLIP 입력은 224px 정사각이다. 원본을 그대로 실어 보내면 요청만 무거워진다.
  const MAX_UPLOAD_PX = 512;
  const DEFAULT_BATCH = 32;
  // 앱이 실제로 쓰는 두 선. garmentSimilarityDetail()의 구제 기준과 같은 값이다.
  const CATEGORY_RESCUE = 0.95;
  const TYPE_RESCUE = 0.92;

  const $ = selector => document.querySelector(selector);
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);

  const devMode = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)
    || new URLSearchParams(location.search).get("dev") === "1";
  if (!devMode) return;

  const config = window.WEARWELL_CONFIG || {};
  const readStorage = key => { try { return localStorage.getItem(key) || ""; } catch { return ""; } };
  const writeStorage = (key, value) => { try { localStorage.setItem(key, value); } catch { /* 시크릿 창 등 */ } };

  const cleanUrl = value => String(value || "").trim().replace(/\/+$/, "");
  const colabBase = cleanUrl(config.API_BASE);
  const localBase = cleanUrl(config.LOCAL_API_BASE) || "http://127.0.0.1:8787";
  const defaultEndpoint = colabBase || localBase;
  const labelFor = url => url === colabBase ? `Colab GPU (${url})` : url === localBase ? `로컬 백엔드 (${url})` : url;

  function tokenFor(url) {
    if (colabBase && url === colabBase) return String(config.API_TOKEN || "");
    if (url === localBase) return String(config.LOCAL_API_TOKEN || "");
    // 직접 입력한 임의 주소로 저장된 토큰을 보내면 토큰이 외부 서버로 새어 나간다.
    return "";
  }
  const authHeaders = () => {
    const token = tokenFor(endpoint);
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  const CATEGORIES = ["아우터", "상의", "하의", "신발", "가방", "액세서리"];

  const nav = $("#siglipLabNavLink");
  const healthBox = $("#siglipLabHealth");
  const endpointInput = $("#siglipLabEndpoint");
  const presetBox = $("#siglipLabPresets");
  const categoryBox = $("#siglipLabCategories");
  const onlyMineInput = $("#siglipOnlyMine");
  const limitInput = $("#siglipLimit");
  const topInput = $("#siglipTop");
  const scopeNote = $("#siglipLabScope");
  const cacheNote = $("#siglipLabCacheNote");
  const clearCacheButton = $("#siglipLabClearCache");
  const dropZone = $("#siglipLabDropZone");
  const fileInput = $("#siglipLabFileInput");
  const sourceFigure = $("#siglipLabSource");
  const runButton = $("#siglipLabRun");
  const statusBox = $("#siglipLabStatus");
  const resultBox = $("#siglipLabResult");
  if (!nav || !categoryBox) return;
  nav.hidden = false;

  let endpoint = readStorage(ENDPOINT_KEY) || defaultEndpoint;
  let health = null;
  let model = "";
  let batchSize = DEFAULT_BATCH;
  let batchSupported = true;
  let closet = [];
  let closetLoaded = false;
  let selectedCategories = new Set();
  let sourceFile = null;
  let sourceDataUrl = null;
  let running = false;
  let ticker = null;
  /** id → { model, hash, vector }. IndexedDB 캐시를 세션 동안 들고 있는 사본. */
  const vectors = new Map();

  try {
    const stored = JSON.parse(readStorage(SCOPE_KEY) || "{}");
    if (Array.isArray(stored.categories)) selectedCategories = new Set(stored.categories.filter(name => CATEGORIES.includes(name)));
    if (typeof stored.onlyMine === "boolean") onlyMineInput.checked = stored.onlyMine;
    if (Number.isFinite(stored.limit)) limitInput.value = String(stored.limit);
    if (Number.isFinite(stored.top)) topInput.value = String(stored.top);
  } catch { /* 저장된 설정이 깨졌으면 기본값으로 */ }

  const saveScope = () => writeStorage(SCOPE_KEY, JSON.stringify({
    categories: [...selectedCategories],
    onlyMine: onlyMineInput.checked,
    limit: Number(limitInput.value),
    top: Number(topInput.value),
  }));

  function setHealth(state, message) {
    healthBox.className = `seg-lab-health ${state}`;
    healthBox.innerHTML = `<span class="seg-dot"></span><small>${escapeHtml(message)}</small>`;
  }

  function setStatus(message, tone = "info") {
    statusBox.hidden = !message;
    statusBox.className = `seg-lab-status ${tone}`;
    statusBox.textContent = message || "";
  }

  /** 사진 문자열 자체의 지문. 같은 id라도 사진이 바뀌면 캐시된 벡터를 버려야 한다. */
  function fingerprint(value) {
    let hash = 0x811c9dc5;
    const text = String(value || "");
    for (let index = 0; index < text.length; index++) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193);
    }
    return `${text.length.toString(36)}-${(hash >>> 0).toString(36)}`;
  }

  function cosineSimilarity(left, right) {
    if (!Array.isArray(left) || !Array.isArray(right) || left.length !== right.length || !left.length) return null;
    let dot = 0, leftNorm = 0, rightNorm = 0;
    for (let index = 0; index < left.length; index++) {
      dot += left[index] * right[index];
      leftNorm += left[index] ** 2;
      rightNorm += right[index] ** 2;
    }
    return dot / Math.max(1e-12, Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
  }

  /**
   * 내 옷장 = 저장소가 함께 배포하는 기본 옷장 + 사용자가 직접 올린 옷.
   * 지운 옷은 app.js와 같은 localStorage 키로 걸러 낸다.
   */
  async function loadCloset() {
    let deleted = new Set();
    try { deleted = new Set(JSON.parse(readStorage("오늘옷-deleted-garments") || "[]")); } catch { /* 무시 */ }

    const byId = new Map();
    for (const item of window.MUSINSA_RANKING || []) {
      if (deleted.has(item.id)) continue;
      byId.set(item.id, {
        id: item.id, image: item.image, name: item.name, brand: item.brand || "",
        category: item.category || "미분류", color: item.color || "", userAdded: false,
      });
    }
    try {
      for (const record of await window.WearwellDB.getAllGarments()) {
        if (!record.userAdded || !record.image || deleted.has(record.id)) continue;
        byId.set(record.id, {
          id: record.id, image: record.image, name: record.name || "직접 올린 옷", brand: "내가 추가",
          category: record.analysis?.category || record.category || "미분류",
          color: record.analysis?.primaryColor || record.color || "", userAdded: true,
          // 옷장에 넣을 때 이미 SigLIP을 돌린 옷이 있다. 같은 모델이면 그 벡터를 그대로 쓴다.
          storedVector: Array.isArray(record.analysis?.visualEmbedding) ? record.analysis.visualEmbedding : null,
          storedEngine: String(record.analysis?.visualEmbeddingEngine || ""),
        });
      }
    } catch (error) {
      console.warn("옷장 데이터베이스를 읽지 못했어요", error);
    }

    closet = [...byId.values()];
    closetLoaded = true;

    try {
      for (const record of await window.WearwellDB.getAllEmbeddings()) vectors.set(record.id, record);
    } catch (error) {
      console.warn("벡터 캐시를 읽지 못했어요", error);
    }
    renderScope();
    updateRunButton();
  }

  const inScope = () => closet.filter(item =>
    (!onlyMineInput.checked || item.userAdded)
    && (!selectedCategories.size || selectedCategories.has(item.category)));

  /** 지금 설정으로 실제로 비교될 옷들. 슬라이더의 최대 벌수까지만 자른다. */
  const targets = () => inScope().slice(0, Number(limitInput.value));

  const cachedVector = item => {
    const hit = vectors.get(item.id);
    if (hit && hit.model === model && hit.hash === fingerprint(item.image)) return hit.vector;
    // 앱이 옷장에 넣으며 만들어 둔 벡터. 엔진 문자열이 같은 모델일 때만 믿는다.
    if (item.storedVector?.length && item.storedEngine === `siglip:${model}`) return item.storedVector;
    return null;
  };

  function renderCategories() {
    categoryBox.innerHTML = `<small>카테고리</small>` + CATEGORIES.map(name => `
      <button type="button" class="refine-chip${selectedCategories.has(name) ? " on" : ""}" data-category="${escapeHtml(name)}">${escapeHtml(name)}</button>`).join("");
    categoryBox.querySelectorAll("[data-category]").forEach(button => button.addEventListener("click", () => {
      const name = button.dataset.category;
      if (selectedCategories.has(name)) selectedCategories.delete(name);
      else selectedCategories.add(name);
      saveScope();
      renderCategories();
      renderScope();
    }));
  }

  function renderScope() {
    if (!closetLoaded) return;
    const scoped = inScope();
    const chosen = targets();
    const ready = model ? chosen.filter(item => cachedVector(item)).length : 0;
    const mine = closet.filter(item => item.userAdded).length;
    scopeNote.textContent = scoped.length
      ? `옷장 ${closet.length}벌 (직접 올린 옷 ${mine}벌) 중 조건에 맞는 ${scoped.length}벌 · 이번에 비교할 ${chosen.length}벌`
        + (model ? ` · 벡터 준비됨 ${ready}벌 / 새로 인코딩할 ${chosen.length - ready}벌` : "")
      : "조건에 맞는 옷이 없어요. 카테고리를 풀거나 '내가 추가한 옷만'을 꺼 보세요.";
    const cached = model ? closet.filter(item => cachedVector(item)).length : vectors.size;
    cacheNote.textContent = model
      ? `${model} 벡터 캐시 ${cached}벌 · 새로고침해도 남아요`
      : `벡터 캐시 ${vectors.size}벌 · 백엔드에 붙으면 모델이 맞는지 확인해요`;
    clearCacheButton.disabled = !vectors.size;
  }

  function updateRunButton() {
    runButton.disabled = running || !sourceDataUrl || !model || !closetLoaded || !targets().length;
    runButton.textContent = running ? "옷장을 훑는 중…" : "옷장에서 닮은 옷 찾기";
  }

  function renderPresets() {
    const presets = [
      colabBase && { url: colabBase, label: "Colab GPU", hint: "local-config.js의 API_BASE" },
      { url: localBase, label: "로컬 백엔드", hint: "옵션 · backend/app.py" },
    ].filter(Boolean);
    presetBox.innerHTML = presets.map(preset => `
      <button type="button" class="refine-preset${preset.url === endpoint ? " on" : ""}" data-url="${escapeHtml(preset.url)}">
        ${escapeHtml(preset.label)}<small>${escapeHtml(preset.hint)}</small>
      </button>`).join("");
    presetBox.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
      if (button.dataset.url !== endpoint) setEndpoint(button.dataset.url);
    }));
  }

  function setEndpoint(url) {
    endpoint = cleanUrl(url) || defaultEndpoint;
    endpointInput.value = endpoint;
    writeStorage(ENDPOINT_KEY, endpoint);
    renderPresets();
    checkHealth();
  }

  async function checkHealth() {
    setHealth("checking", `${labelFor(endpoint)} 확인 중…`);
    model = "";
    try {
      const response = await fetch(`${endpoint}/api/health`);
      if (!response.ok) throw new Error(`헬스 응답이 HTTP ${response.status}`);
      health = await response.json();
    } catch (error) {
      setHealth("down", `${labelFor(endpoint)} · 연결 실패 (${error.message}) — 백엔드가 떠 있는지, 터널 주소가 최신인지 확인하세요`);
      renderScope();
      updateRunButton();
      return;
    }
    model = String(health.embeddingModel || "");
    // 예전 백엔드에는 묶음 경로가 없다. 그때는 한 벌씩 부르되 분당 제한을 넘기지 않게 줄인다.
    batchSupported = Number.isFinite(health.maxEmbeddingBatch);
    batchSize = batchSupported ? Math.max(1, Number(health.maxEmbeddingBatch)) : 1;
    $("#siglipFlowModel").textContent = model || "이미지 → 벡터";
    if (!model) {
      setHealth("down", `${labelFor(endpoint)} · SigLIP 모델이 없는 백엔드예요`);
    } else if (!batchSupported) {
      setHealth("warn", `${labelFor(endpoint)} · ${model} · 묶음 경로(/api/embeddings)가 없는 예전 백엔드라 한 벌씩 보내요 — 분당 ${health.rateLimitPerMinute || 60}회 제한에 걸릴 수 있어요`);
    } else {
      const loaded = health.embeddingLoaded ? "적재됨" : "첫 요청 때 적재";
      const device = health.embeddingDevice || (health.cuda ? "cuda" : "cpu");
      setHealth(health.cuda ? "up" : "warn", `${labelFor(endpoint)} · ${model} · ${device} · ${loaded} · 한 요청에 ${batchSize}벌`);
    }
    renderScope();
    updateRunButton();
  }

  async function toDataUrl(source, maxPx = MAX_UPLOAD_PX) {
    const image = new Image();
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = source; });
    const scale = Math.min(1, maxPx / Math.max(image.width, image.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.width * scale));
    canvas.height = Math.max(1, Math.round(image.height * scale));
    const context = canvas.getContext("2d");
    // 투명 PNG(분할해 저장한 옷)를 그대로 JPEG로 만들면 배경이 검게 깔린다.
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return { dataUrl: canvas.toDataURL("image/jpeg", 0.92), width: image.width, height: image.height };
  }

  async function fileToDataUrl(file) {
    const source = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    return toDataUrl(source);
  }

  async function imageSourceToDataUrl(source) {
    if (String(source || "").startsWith("data:")) return (await toDataUrl(source)).dataUrl;
    const response = await fetch(source);
    if (!response.ok) throw new Error(`사진을 읽지 못했어요 (${response.status})`);
    const blob = await response.blob();
    const encoded = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
    return (await toDataUrl(encoded)).dataUrl;
  }

  async function embedBatch(images) {
    if (!batchSupported) {
      const vectorList = [];
      for (const image of images) {
        const response = await fetch(`${endpoint}/api/embedding`, {
          method: "POST",
          headers: { "Content-Type": "application/json", ...authHeaders() },
          body: JSON.stringify({ image }),
        });
        const result = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
        vectorList.push(result.vector);
      }
      return vectorList;
    }
    const response = await fetch(`${endpoint}/api/embeddings`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ images }),
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
    if (!Array.isArray(result.vectors) || result.vectors.length !== images.length) throw new Error("벡터 개수가 보낸 사진 수와 달라요");
    return result.vectors;
  }

  /** 캐시에 없는 옷만 묶어서 인코딩한다. 만든 벡터는 IndexedDB에 남겨 다음 실행에서 건너뛴다. */
  async function ensureClosetVectors(items, onProgress) {
    const missing = items.filter(item => !cachedVector(item));
    let done = 0;
    for (let start = 0; start < missing.length; start += batchSize) {
      const chunk = missing.slice(start, start + batchSize);
      const images = [];
      const encoded = [];
      for (const item of chunk) {
        try {
          images.push(await imageSourceToDataUrl(item.image));
          encoded.push(item);
        } catch (error) {
          console.warn("옷 사진을 읽지 못해 건너뜁니다", item.id, error);
        }
      }
      if (!images.length) continue;
      const results = await embedBatch(images);
      await Promise.all(encoded.map(async (item, index) => {
        const record = { id: item.id, model, hash: fingerprint(item.image), vector: results[index], updatedAt: new Date().toISOString() };
        vectors.set(item.id, record);
        try { await window.WearwellDB.putEmbedding(record); } catch { /* 캐시는 없어도 동작한다 */ }
      }));
      done += encoded.length;
      onProgress(done, missing.length);
    }
    // 사진을 읽지 못해 건너뛴 옷이 있을 수 있으므로 실제로 만든 수를 돌려준다.
    return done;
  }

  const scoreBar = visual => `<span class="siglip-bar"><i style="width:${(visual * 100).toFixed(1)}%"></i></span>`;

  function renderRanking({ ranked, elapsed, encoded, reused, queryDims }) {
    const top = ranked.slice(0, Number(topInput.value));
    const best = ranked[0];
    resultBox.innerHTML = `
      <div class="siglip-summary">
        <figure class="siglip-query">
          <img src="${sourceDataUrl}" alt="올린 옷" />
          <figcaption>${escapeHtml(sourceFile?.name || "올린 옷")}</figcaption>
        </figure>
        <dl class="refine-readout">
          <div><dt>모델</dt><dd><code>${escapeHtml(model)}</code></dd></div>
          <div><dt>벡터 차원</dt><dd>${queryDims}</dd></div>
          <div><dt>비교한 옷</dt><dd>${ranked.length}벌</dd></div>
          <div><dt>이번에 인코딩</dt><dd>${encoded}벌 <i>· 캐시 재사용 ${reused}벌</i></dd></div>
          <div><dt>왕복</dt><dd>${elapsed}s</dd></div>
          <div class="${best.visual >= CATEGORY_RESCUE ? "ok" : ""}"><dt>가장 닮은 옷</dt><dd>cos ${best.cosine.toFixed(4)} · ${best.visual.toFixed(3)}</dd></div>
        </dl>
      </div>
      <ol class="siglip-ranking">${top.map((entry, index) => `
        <li class="${entry.visual >= CATEGORY_RESCUE ? "same" : entry.visual >= TYPE_RESCUE ? "close" : ""}">
          <b>${index + 1}</b>
          <img src="${escapeHtml(entry.item.image)}" alt="${escapeHtml(entry.item.name)}" loading="lazy" />
          <div>
            <strong>${escapeHtml(entry.item.name)}</strong>
            <small>${escapeHtml([entry.item.brand, entry.item.category, entry.item.color].filter(Boolean).join(" · "))}</small>
            ${scoreBar(entry.visual)}
          </div>
          <em>
            <span class="siglip-cos">${entry.cosine.toFixed(4)}</span>
            <span class="siglip-visual">${entry.visual.toFixed(3)}</span>
            ${entry.visual >= CATEGORY_RESCUE ? "<span class=\"siglip-flag\">같은 옷 취급</span>"
              : entry.visual >= TYPE_RESCUE ? "<span class=\"siglip-flag soft\">유형 오판 구제선</span>" : ""}
          </em>
        </li>`).join("")}</ol>
      <p class="qwen-note">왼쪽 큰 숫자가 코사인(−1~1), 오른쪽 작은 숫자가 앱이 쓰는 <code>(cos+1)/2</code> 값이에요.
        ${top.length < ranked.length ? `상위 ${top.length}벌만 그렸어요 — 나머지 ${ranked.length - top.length}벌은 이보다 낮아요.` : ""}</p>`;
  }

  async function pickFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    sourceFile = file;
    setStatus("");
    const { dataUrl, width, height } = await fileToDataUrl(file);
    sourceDataUrl = dataUrl;
    sourceFigure.hidden = false;
    sourceFigure.querySelector("img").src = sourceDataUrl;
    sourceFigure.querySelector("figcaption").textContent = `${file.name} · 원본 ${width}×${height} · ${MAX_UPLOAD_PX}px로 줄여 전송`;
    $("#siglipFlowQuery").textContent = `${width}×${height}`;
    updateRunButton();
  }

  /** 진행 문구는 배치마다 바뀌지만 경과 시간은 한 번의 실행 전체를 센다. */
  function startTicker(getMessage) {
    const started = performance.now();
    clearInterval(ticker);
    const paint = () => setStatus(`${getMessage()} · ${((performance.now() - started) / 1000).toFixed(0)}s 경과`, "info");
    ticker = setInterval(paint, 1000);
    setStatus(getMessage(), "info");
    return started;
  }

  async function run() {
    const items = targets();
    if (running || !sourceDataUrl || !model || !items.length) return;
    running = true;
    updateRunButton();
    const reused = items.filter(item => cachedVector(item)).length;
    let label = "올린 옷을 벡터로 만드는 중";
    const started = startTicker(() => label);
    try {
      const [queryVector] = await embedBatch([sourceDataUrl]);
      label = `옷장 ${items.length}벌 중 ${items.length - reused}벌을 인코딩하는 중 · 캐시 재사용 ${reused}벌`;
      const encoded = await ensureClosetVectors(items, (done, total) => {
        label = `옷장 벡터 ${done}/${total}벌 · 캐시 재사용 ${reused}벌`;
      });
      clearInterval(ticker);

      const ranked = items.map(item => {
        const vector = cachedVector(item);
        const cosine = cosineSimilarity(queryVector, vector);
        return cosine === null ? null : { item, cosine, visual: Math.max(0, Math.min(1, (cosine + 1) / 2)) };
      }).filter(Boolean).sort((left, right) => right.cosine - left.cosine);

      if (!ranked.length) throw new Error("비교할 벡터를 하나도 만들지 못했어요");
      const elapsed = ((performance.now() - started) / 1000).toFixed(1);
      renderRanking({ ranked, elapsed, encoded, reused: ranked.length - encoded, queryDims: queryVector.length });
      setStatus(`${ranked.length}벌과 견줬어요 · 1위 cos ${ranked[0].cosine.toFixed(4)} · 새로 인코딩 ${encoded}벌 · 왕복 ${elapsed}s`, "ok");
      renderScope();
    } catch (error) {
      clearInterval(ticker);
      // 429는 예전 백엔드에서 한 벌씩 보낼 때 나온다 — 비교 벌수를 줄이면 지나간다.
      setStatus(`실패: ${error.message}`, "error");
    } finally {
      running = false;
      updateRunButton();
    }
  }

  endpointInput.value = endpoint;
  renderPresets();
  renderCategories();
  endpointInput.addEventListener("change", () => setEndpoint(endpointInput.value));

  const syncSliders = () => {
    $("#siglipLimitValue").textContent = `${limitInput.value}벌`;
    $("#siglipTopValue").textContent = `${topInput.value}벌`;
  };
  syncSliders();
  [limitInput, topInput].forEach(input => input.addEventListener("input", () => {
    syncSliders();
    saveScope();
    renderScope();
    updateRunButton();
  }));
  onlyMineInput.addEventListener("change", () => {
    saveScope();
    renderScope();
    updateRunButton();
  });
  clearCacheButton.addEventListener("click", async () => {
    vectors.clear();
    try { await window.WearwellDB.clearEmbeddings(); } catch { /* 무시 */ }
    renderScope();
    setStatus("벡터 캐시를 비웠어요. 다음 실행에서 옷장을 다시 인코딩해요.", "info");
  });

  nav.addEventListener("click", () => document.querySelectorAll("dialog[open]").forEach(dialog => dialog.close()));

  fileInput.addEventListener("change", event => pickFile(event.target.files[0]));
  ["dragenter", "dragover"].forEach(type => dropZone.addEventListener(type, event => {
    event.preventDefault();
    dropZone.classList.add("drag");
  }));
  ["dragleave", "drop"].forEach(type => dropZone.addEventListener(type, event => {
    event.preventDefault();
    dropZone.classList.remove("drag");
  }));
  dropZone.addEventListener("drop", event => pickFile(event.dataTransfer.files[0]));
  window.WearwellClipboard?.register({
    element: dropZone,
    isActive: () => $("#siglipLabView").classList.contains("active"),
    onFiles: files => pickFile(files[0]),
  });
  runButton.addEventListener("click", run);

  loadCloset();
  checkHealth();
})();
