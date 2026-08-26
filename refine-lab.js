/**
 * 개발용 파이프라인 탭 — 전신샷 → 세그멘테이션 → 마스크 보수 → FLUX 재생성.
 *
 * Seg Lab이 "모델이 옷을 어디까지 잡았나"를 본다면 이 탭은 그 다음 질문을 본다:
 * 잘라낸 조각을 옷장에 넣을 수 있는 그림으로 어떻게 바꾸는가. 백엔드의
 * /api/closet/refine 하나를 부르고 단계별 중간 산출물을 나란히 늘어놓는다.
 *
 * 쓰는 경로는 전부 공개 경로(/api/closet/*)다 — Colab 공개 터널은 dev 도구를 끄고
 * 뜨므로 /api/dev/*에 기대면 이 탭은 GPU가 있는 곳에서 아예 열리지 않는다.
 *
 * seg-lab.js와 마찬가지로 app.js의 전역에 의존하지 않는다 — 앱 코드를 건드려도
 * 이 도구가 같이 깨지지 않게.
 */
(() => {
  // Seg Lab과 저장 키를 나눈다. Seg Lab은 세그멘테이션만 보므로 로컬이 기본이지만
  // 이 탭은 FLUX까지 돌리므로 GPU가 있는 Colab 백엔드가 기본이다.
  const ENDPOINT_KEY = "wearwell-refinelab-endpoint";
  const OPTIONS_KEY = "wearwell-refinelab-options";
  const MAX_UPLOAD_PX = 1400;

  const $ = selector => document.querySelector(selector);
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  const percent = value => `${(value * 100).toFixed(value < 0.1 ? 2 : 1)}%`;
  const px = value => `${Number(value || 0).toLocaleString("ko-KR")}px`;
  const signed = value => `${value > 0 ? "+" : ""}${Number(value || 0).toLocaleString("ko-KR")}px`;

  const devMode = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)
    || new URLSearchParams(location.search).get("dev") === "1";
  if (!devMode) return;

  const config = window.WEARWELL_CONFIG || {};
  const readStorage = key => { try { return localStorage.getItem(key) || ""; } catch { return ""; } };
  const writeStorage = (key, value) => { try { localStorage.setItem(key, value); } catch { /* 시크릿 창 등 */ } };

  const cleanUrl = value => String(value || "").trim().replace(/\/+$/, "");
  const colabBase = cleanUrl(config.API_BASE);
  const localBase = cleanUrl(config.LOCAL_API_BASE) || "http://127.0.0.1:8787";
  // 5단계(FLUX)가 실제로 도는 곳이 기본값이어야 한다. local-config.js에 Colab 주소가
  // 있으면 그쪽이 GPU를 들고 있으므로 먼저 쓰고, 없을 때만 로컬로 떨어진다.
  const defaultEndpoint = colabBase || localBase;
  const isLoopback = url => /^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(:|\/|$)/.test(url);

  const labelFor = url => url === colabBase ? `Colab GPU (${url})` : url === localBase ? `로컬 백엔드 (${url})` : url;

  /** 주소마다 토큰이 다르다. Colab 세션 토큰을 로컬에 보내면(또는 그 반대면) 401이다. */
  function tokenFor(url) {
    if (colabBase && url === colabBase) return String(config.API_TOKEN || "");
    if (url === localBase) return String(config.LOCAL_API_TOKEN || "");
    return String((isLoopback(url) ? config.LOCAL_API_TOKEN || config.API_TOKEN : config.API_TOKEN || config.LOCAL_API_TOKEN) || "");
  }

  let endpoint = readStorage(ENDPOINT_KEY) || defaultEndpoint;
  let registry = null;
  let health = null;
  let refinePath = "";
  let categories = new Set();
  let sourceFile = null;
  let sourceDataUrl = null;
  let running = false;
  let ticker = null;

  const nav = $("#refineLabNavLink");
  const healthBox = $("#refineLabHealth");
  const endpointInput = $("#refineLabEndpoint");
  const presetBox = $("#refineLabPresets");
  const modelSelect = $("#refineLabModel");
  const categoryBox = $("#refineCategories");
  const dropZone = $("#refineLabDropZone");
  const fileInput = $("#refineLabFileInput");
  const sourceFigure = $("#refineLabSource");
  const runButton = $("#refineLabRun");
  const statusBox = $("#refineLabStatus");
  const summaryBox = $("#refineLabSummary");
  const resultsBox = $("#refineLabResults");
  if (!nav || !modelSelect) return;
  nav.hidden = false;

  const controls = {
    close: $("#refineOptClose"),
    fillHoles: $("#refineOptHoles"),
    fillOccluded: $("#refineOptOccluded"),
    dropStrays: $("#refineOptStrays"),
    smooth: $("#refineOptSmooth"),
    generate: $("#refineOptGenerate"),
    closeScale: $("#refineCloseScale"),
    strayRatio: $("#refineStrayRatio"),
    occlusionEnclosure: $("#refineEnclosure"),
    seed: $("#refineSeed"),
    steps: $("#refineSteps"),
  };

  // 가려진 자리가 옷의 어디쯤인지. 백엔드는 프롬프트에도 쓰는 영어 코드로 준다.
  const WHERE_KO = {
    upper: "위", middle: "가운데", lower: "아래", left: "왼쪽", center: "가운데", right: "오른쪽",
  };
  // "lower-left" → "왼쪽 아래". 한국어는 가로 위치를 먼저 읽어야 자연스럽다.
  const whereLabel = code => !code ? "" : code === "center" ? "가운데"
    : code.split("-").map(part => WHERE_KO[part] || part).reverse().join(" ");

  function authHeaders() {
    const token = tokenFor(endpoint);
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  /** 어느 백엔드를 쓰는지 한 번에 바꾸는 버튼. 설정된 것만 보여준다. */
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
      if (button.dataset.url === endpoint) return;
      setEndpoint(button.dataset.url);
    }));
  }

  function setEndpoint(url) {
    endpoint = cleanUrl(url) || defaultEndpoint;
    endpointInput.value = endpoint;
    writeStorage(ENDPOINT_KEY, endpoint);
    renderPresets();
    loadRegistry();
  }

  function setHealth(state, message) {
    healthBox.className = `seg-lab-health ${state}`;
    healthBox.innerHTML = `<span class="seg-dot"></span><small>${escapeHtml(message)}</small>`;
  }

  function setStatus(message, tone = "info") {
    statusBox.hidden = !message;
    statusBox.className = `seg-lab-status ${tone}`;
    statusBox.textContent = message || "";
  }

  function readOptions() {
    return {
      model: modelSelect.value || "",
      close: controls.close.checked,
      fillHoles: controls.fillHoles.checked,
      fillOccluded: controls.fillOccluded.checked,
      dropStrays: controls.dropStrays.checked,
      smooth: controls.smooth.checked,
      generate: controls.generate.checked,
      closeScale: Number(controls.closeScale.value),
      strayRatio: Number(controls.strayRatio.value),
      occlusionEnclosure: Number(controls.occlusionEnclosure.value),
      seed: Number(controls.seed.value) || 0,
      steps: controls.steps.value ? Number(controls.steps.value) : null,
      categories: [...categories],
    };
  }

  function syncOutputs() {
    $("#refineCloseValue").textContent = `${Number(controls.closeScale.value).toFixed(1)}%`;
    $("#refineStrayValue").textContent = `${controls.strayRatio.value}%`;
    $("#refineEnclosureValue").textContent = `${controls.occlusionEnclosure.value}%`;
    controls.closeScale.disabled = !controls.close.checked;
    controls.strayRatio.disabled = !controls.dropStrays.checked;
    controls.occlusionEnclosure.disabled = !controls.fillOccluded.checked;
    controls.seed.disabled = controls.steps.disabled = !controls.generate.checked;
    writeStorage(OPTIONS_KEY, JSON.stringify(readOptions()));
  }

  function restoreOptions() {
    let stored = {};
    try { stored = JSON.parse(readStorage(OPTIONS_KEY) || "{}"); } catch { stored = {}; }
    for (const key of ["close", "fillHoles", "fillOccluded", "dropStrays", "smooth", "generate"]) {
      if (typeof stored[key] === "boolean") controls[key].checked = stored[key];
    }
    for (const key of ["closeScale", "strayRatio", "occlusionEnclosure", "seed"]) {
      if (Number.isFinite(stored[key])) controls[key].value = stored[key];
    }
    if (Number.isFinite(stored.steps)) controls.steps.value = stored.steps;
    if (Array.isArray(stored.categories)) categories = new Set(stored.categories);
    return stored.model || "";
  }

  function updateRunButton() {
    runButton.disabled = running || !sourceDataUrl || !registry || !refinePath;
    runButton.textContent = running ? "실행 중…" : "파이프라인 실행";
  }

  /** 원본 그대로 보내면 요청이 수 MB가 되므로 긴 변 기준으로만 줄인다. */
  async function toDataUrl(file) {
    const source = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    const image = new Image();
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = reject; image.src = source; });
    const scale = Math.min(1, MAX_UPLOAD_PX / Math.max(image.width, image.height));
    if (scale === 1 && file.type === "image/jpeg") return source;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(image.width * scale);
    canvas.height = Math.round(image.height * scale);
    canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.92);
  }

  function renderCategories() {
    categoryBox.innerHTML = Object.entries(registry.categoryColors).map(([name, color]) => `
      <button type="button" class="refine-chip${categories.has(name) ? " on" : ""}" data-category="${escapeHtml(name)}" style="--chip:${color}">
        ${escapeHtml(name)}
      </button>`).join("") + `<small>고르지 않으면 검출된 옷 전부</small>`;
    categoryBox.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
      const name = button.dataset.category;
      if (categories.has(name)) categories.delete(name); else categories.add(name);
      button.classList.toggle("on", categories.has(name));
      syncOutputs();
    }));
  }

  function failRegistry(message) {
    setHealth("down", `${labelFor(endpoint)} · ${message}`);
    modelSelect.innerHTML = "";
    categoryBox.innerHTML = "";
    registry = null;
    updateRunButton();
  }

  const STALE_BACKEND = "Colab 노트북 셀 2~3을 다시 실행해 최신 코드로 띄우세요";
  const BAD_TOKEN = "토큰이 맞지 않아요 — local-config.js의 토큰과 백엔드의 WEARWELL_API_TOKEN을 맞추세요";

  /** 모델 목록. 공개 경로가 먼저고, 그게 없는 예전 백엔드에서만 dev 경로로 떨어진다. */
  async function fetchRegistry() {
    const response = await fetch(`${endpoint}/api/closet/models`, { headers: authHeaders() });
    if (response.ok) return response.json();
    if (response.status !== 404) throw new Error(`모델 목록이 HTTP ${response.status}예요`);
    if (health?.devTools === false) throw new Error(`이 백엔드에는 /api/closet/models가 없어요 — ${STALE_BACKEND}`);
    const legacy = await fetch(`${endpoint}/api/dev/segment/models`, { headers: authHeaders() });
    if (!legacy.ok) throw new Error(`이 백엔드에는 모델 목록 경로가 없어요 — ${STALE_BACKEND}`);
    return legacy.json();
  }

  /**
   * 사진을 올리기 전에 refine 경로가 있는지 확인해 둔다. 몇 MB를 올린 뒤에야
   * 404를 보는 것보다 낫다. 본문이 빈 POST는 pydantic 검증에서 422로 끊기므로
   * GPU를 건드리지 않는다 — 즉 404(없음)와 422(있음)를 공짜로 구분할 수 있다.
   */
  async function detectRefinePath() {
    for (const path of ["/api/closet/refine", "/api/dev/closet/refine"]) {
      const response = await fetch(`${endpoint}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: "{}",
      });
      if (response.status === 401) throw new Error(BAD_TOKEN);
      if (response.status !== 404) return path;
    }
    throw new Error(`백엔드에 refine 경로가 없어요 — ${STALE_BACKEND}`);
  }

  async function loadRegistry() {
    setHealth("checking", `${labelFor(endpoint)} 확인 중…`);
    registry = null;
    health = null;
    refinePath = "";
    const storedModel = restoreOptions();

    // 목록보다 헬스를 먼저 본다. 라우트의 404만 보고는 "서버가 없다", "코드가 예전이다",
    // "dev 도구가 꺼졌다"를 구분할 수 없어서 추측한 문구를 띄우게 된다.
    try {
      const response = await fetch(`${endpoint}/api/health`);
      if (!response.ok) return failRegistry(`헬스 응답이 HTTP ${response.status}예요 — 주소를 확인하세요`);
      health = await response.json();
    } catch (error) {
      return failRegistry(`연결 실패 (${error.message}) — 백엔드가 떠 있는지, 터널 주소가 최신인지 확인하세요`);
    }

    try {
      registry = await fetchRegistry();
      refinePath = await detectRefinePath();
    } catch (error) {
      return failRegistry(error.message);
    }

    // 백엔드가 이미 올려 둔 모델이 기본값이다. 다른 모델을 고르면 그 요청에서
    // 콜드 적재가 한 번 일어난다 — 어느 쪽인지 목록에 그대로 적어 준다.
    const loaded = new Set(health.segmentationLoadedModels || (health.segmentationModelKey ? [health.segmentationModelKey] : []));
    const preferred = storedModel || health.segmentationModelKey || registry.production;
    modelSelect.innerHTML = registry.models.map(model => `
      <option value="${escapeHtml(model.key)}"${model.key === preferred ? " selected" : ""}>
        ${escapeHtml(model.title)}${model.key === registry.production ? " (프로덕션)" : ""} · ${escapeHtml(model.taxonomy)}${loaded.has(model.key) ? " · 적재됨" : ""}
      </option>`).join("");
    renderCategories();

    // 세그멘테이션과 FLUX는 따로 논다. 5단계가 실제로 돌지, 아니면 정규화본이
    // 그대로 나올지를 미리 알려준다.
    const where = labelFor(endpoint);
    const seg = `세그 ${health.segmentationModel || "?"}${health.segmentationLoaded ? "" : " (첫 요청 때 적재)"}`;
    if (health.cuda) {
      setHealth("up", `${where} · ${health.gpu} · ${seg} · FLUX ${health.model}${health.modelLoaded ? " (적재됨)" : " (첫 생성 때 적재)"}`);
    } else {
      setHealth("warn", `${where} · ${seg} · GPU가 없어 5단계는 정규화본을 그대로 내보내요 (1~4단계는 정상)`);
    }
    updateRunButton();
  }

  async function pickFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    sourceFile = file;
    setStatus("");
    summaryBox.innerHTML = "";
    resultsBox.innerHTML = "";
    sourceDataUrl = await toDataUrl(file);
    const image = new Image();
    await new Promise(resolve => { image.onload = resolve; image.src = sourceDataUrl; });
    sourceFigure.hidden = false;
    sourceFigure.querySelector("img").src = sourceDataUrl;
    sourceFigure.querySelector("figcaption").textContent = `${file.name} · ${image.width}×${image.height} 전송`;
    updateRunButton();
  }

  function colorFor(category) {
    return (registry?.categoryColors || {})[category] || "#8b8f96";
  }

  function stageFigure({ index, title, image, caption, body, tone = "" }) {
    const picture = image
      ? `<img src="${image}" alt="${escapeHtml(title)}" loading="lazy" />`
      : `<div class="refine-stage-empty">${escapeHtml(caption || "결과 없음")}</div>`;
    return `<figure class="refine-stage ${tone}">
      <figcaption><b>${index}</b>${escapeHtml(title)}</figcaption>
      <div class="refine-stage-image">${picture}</div>
      <div class="refine-stage-body">${body}</div>
    </figure>`;
  }

  function readout(rows) {
    return `<dl class="refine-readout">${rows.map(([label, value, tone = ""]) =>
      `<div class="${tone}"><dt>${escapeHtml(label)}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
  }

  function segmentStage(item) {
    return stageFigure({
      index: 1,
      title: "세그멘테이션 크롭",
      image: item.stages.crop,
      body: readout([
        ["면적", percent(item.areaRatio), item.areaRatio < item.thresholds.minArea ? "fail" : ""],
        ["채움", percent(item.fillRatio), item.fillRatio < item.thresholds.minFill ? "fail" : ""],
        ["확신도", item.confidence.toFixed(2), item.confidence < item.thresholds.minConfidence ? "fail" : ""],
        ["원본 라벨", `<code>${escapeHtml(item.label)}</code>`],
      ]),
    });
  }

  /** 손상 정도 한 숫자 — 구멍과 가림 중 큰 쪽. 둘 다 "옷이 온전했을 넓이" 대비다. */
  const damageOf = item => Math.max(item.diagnosis.holeRatio || 0, item.diagnosis.occludedRatio || 0);

  function defectStage(item) {
    const { diagnosis } = item;
    const occluded = diagnosis.occludedRatio || 0;
    const intact = diagnosis.holeCount === 0 && diagnosis.componentCount <= 1 && !occluded;
    // 가려짐은 라벨맵이 있어야 판정된다. 예전 백엔드면 "구멍 0개"가 곧 "멀쩡함"이
    // 아니라는 걸 알려줘야 한다 — 이 사진의 진짜 결함이 거기 있을 수 있다.
    const byLabel = (diagnosis.occludedBy || [])[0];
    const occludedValue = !diagnosis.occlusionAvailable
      ? `<i>라벨맵 없음</i>`
      : occluded
        ? `${percent(occluded)}${byLabel ? ` <i>${escapeHtml(byLabel.label)}</i>` : ""}`
        : "없음";
    return stageFigure({
      index: 2,
      title: "결함 진단",
      image: item.stages.defects,
      tone: intact ? "" : "warn",
      body: `${readout([
        ["조각", `${diagnosis.componentCount}개${diagnosis.strayCount ? ` <i>(부스러기 ${diagnosis.strayCount})</i>` : ""}`,
          diagnosis.componentCount > 1 ? "warn" : ""],
        ["구멍", `${diagnosis.holeCount}개 · ${percent(diagnosis.holeRatio)}`, diagnosis.holeRatio > 0.05 ? "warn" : ""],
        ["가려짐", occludedValue, occluded > 0.02 ? "warn" : ""],
        ["가려진 자리", occluded ? (whereLabel(diagnosis.occludedWhere) || "여러 곳") : "—"],
        // 볼록껍질을 얼마나 채우는가. 구멍이 없어도 파먹혔으면 여기서 떨어진다.
        // 예전 백엔드는 이 값을 주지 않는다 — 0%로 그리면 멀쩡한 옷이 빨갛게 보인다.
        ["볼록채움도", diagnosis.solidity == null ? "—" : percent(diagnosis.solidity),
          diagnosis.solidity != null && diagnosis.solidity < 0.85 ? "warn" : ""],
      ])}
      <p class="refine-legend"><span class="hole"></span>구멍 <span class="occluded"></span>가려짐 <span class="stray"></span>버릴 조각</p>`,
    });
  }

  const STEP_KO = {
    close: "닫기", fillHoles: "구멍 메우기", fillOccluded: "가림 메우기",
    smooth: "경계 다듬기", dropStrays: "조각 정리",
  };

  function repairStage(item) {
    const { repair, diagnosis } = item;
    const steps = repair.steps.length
      ? repair.steps.map(step => `<li><b>${escapeHtml(STEP_KO[step.step] || step.step)}</b><span>${escapeHtml(step.detail)}</span><em>${signed(step.delta)}</em></li>`).join("")
      : `<li class="none">보수 단계를 모두 껐어요</li>`;
    // 후보였지만 안전장치에 걸린 조각. 왜 안 메웠는지 여기 없으면 기준을 고칠 근거가 없다.
    const occlusion = repair.occlusion || {};
    const held = (occlusion.openCount || 0) + (occlusion.oversizedCount || 0);
    return stageFigure({
      index: 3,
      title: "마스크 보수",
      image: item.stages.repaired,
      body: `<ul class="refine-steps">${steps}</ul>
      ${readout([
        ["남은 구멍", repair.holesAfter ? px(repair.holesAfter) : "없음", repair.holesAfter ? "warn" : "ok"],
        ["남은 조각", `${repair.componentsAfter}개`],
        ["메운 자리", repair.patchedPixels
          ? `${px(repair.patchedPixels)} <i>${percent(repair.patchedPixels / Math.max(1, repair.pixelsAfter))}</i>`
          : "없음"],
        ["가림 후보", occlusion.candidatePixels
          ? `${occlusion.acceptedCount || 0}조각 적용${held ? ` <i>· ${held}조각 보류</i>` : ""}${
              occlusion.trimmedPixels ? ` <i>· 오목한 자리 ${px(occlusion.trimmedPixels)} 제외</i>` : ""}`
          : diagnosis.occlusionAvailable ? "없음" : "—"],
        ["마스크 변화", `${px(repair.pixelsBefore)} → ${px(repair.pixelsAfter)} <i>(${signed(repair.pixelsAfter - repair.pixelsBefore)})</i>`],
      ])}`,
    });
  }

  const FILL_KO = {
    propagate: "경계 색을 안쪽으로 전파",
    median: "옷 대표색으로 평평하게",
    none: "없음",
  };

  function normalizedStage(item) {
    const { repair } = item;
    const swatch = repair.patchColor
      ? `<span class="refine-swatch" style="--swatch:${escapeHtml(repair.patchColor)}"></span><code>${escapeHtml(repair.patchColor)}</code>`
      : "—";
    return stageFigure({
      index: 4,
      title: "정규화 (생성 입력)",
      image: item.stages.normalized,
      body: readout([
        ["규격", "768px 정사각 · 흰 배경 · 여백 6%"],
        ["메운 방식", FILL_KO[repair.patchFill] || repair.patchFill || "없음"],
        ["옷 대표색", swatch],
        // 알파를 그대로 얹으면 흰 배경 위에 계단이 남고, 그냥 흐리면 살색 후광이 생긴다.
        ["경계", "옷 색을 2px 번지게 한 뒤 알파 0.8px 페더"],
        ["남은 몫", "질감·주름·무늬"],
      ]),
    });
  }

  function closetStage(item) {
    if (item.generationError) {
      return stageFigure({
        index: 5, title: "FLUX 재생성", image: null, tone: "fail",
        caption: "생성 실패",
        body: `<p class="refine-error">${escapeHtml(item.generationError)}</p>`,
      });
    }
    if (!item.stages.closet) {
      return stageFigure({
        index: 5, title: "FLUX 재생성", image: null,
        caption: "생성을 껐어요",
        body: `<p class="refine-note">4단계 정규화본이 그대로 옷장 이미지가 됩니다.</p>`,
      });
    }
    const passthrough = item.generation?.engine?.includes("fallback");
    return stageFigure({
      index: 5,
      title: "FLUX 재생성",
      image: item.stages.closet,
      tone: passthrough ? "warn" : "ok",
      body: `${readout([
        ["엔진", `<code>${escapeHtml(item.generation.engine)}</code>`, passthrough ? "warn" : ""],
        ["소요", `${item.generation.seconds}s`],
        ["seed · steps", `${item.generation.seed} · ${item.generation.steps}`],
      ])}
      ${passthrough ? `<p class="refine-note">GPU가 없어 정규화본을 그대로 돌려줬어요.</p>` : ""}`,
    });
  }

  function renderItem(item) {
    return `<article class="refine-item">
      <header>
        <span class="seg-chip" style="--chip:${colorFor(item.category)}">${escapeHtml(item.category)}</span>
        <span class="seg-verdict">${item.accepted ? "품질 필터 통과" : "걸러진 후보"}</span>
        ${item.rejectReason ? `<span class="refine-reject">${escapeHtml(item.rejectReason)}</span>` : ""}
        <span class="refine-timing">보수 ${item.repairSeconds}s${item.generation ? ` · 생성 ${item.generation.seconds}s` : ""}</span>
      </header>
      <div class="refine-stages">
        ${segmentStage(item)}${defectStage(item)}${repairStage(item)}${normalizedStage(item)}${closetStage(item)}
      </div>
    </article>`;
  }

  function renderSummary(payload) {
    const worst = payload.items.reduce((max, item) => Math.max(max, damageOf(item)), 0);
    const occludedItems = payload.items.filter(item => item.diagnosis.occludedRatio > 0).length;
    summaryBox.innerHTML = `
      <figure class="refine-overlay">
        <img src="${payload.overlay}" alt="세그멘테이션 오버레이" loading="lazy" />
        <figcaption>${escapeHtml(payload.model.title)} · ${payload.imageSize.width}×${payload.imageSize.height}</figcaption>
      </figure>
      <div class="refine-summary-stats">
        ${readout([
          ["검출", `${payload.detectedCount}벌 중 ${payload.items.length}벌 처리${payload.skippedCount ? ` <i>(${payload.skippedCount}벌 생략)</i>` : ""}`],
          ["세그멘테이션", `${payload.segmentSeconds}s · ${escapeHtml(payload.device)}${payload.loadSeconds > 0 ? ` · 콜드 적재 ${payload.loadSeconds}s` : ""}`],
          ["가장 심한 손상", worst ? `넓이의 ${percent(worst)}가 비어 있었음` : "구멍·가림 없음", worst > 0.05 ? "warn" : ""],
          ["가려진 옷", occludedItems ? `${occludedItems}벌 (팔·가방 등에 덮임)` : "없음"],
          ["전체", `${payload.totalSeconds}s`],
        ])}
      </div>`;

    $("#refineFlowSource").textContent = `${payload.imageSize.width}×${payload.imageSize.height}`;
    $("#refineFlowSegment").textContent = `${payload.detectedCount}벌 · ${payload.segmentSeconds}s`;
    $("#refineFlowDiagnose").textContent = worst ? `최대 ${percent(worst)} 파먹힘` : "결함 없음";
    $("#refineFlowRepair").textContent = `가림 ${occludedItems}벌 · 남은 구멍 ${payload.items.reduce((sum, item) => sum + (item.repair.holesAfter ? 1 : 0), 0)}벌`;
    const generated = payload.items.filter(item => item.stages.closet).length;
    $("#refineFlowGenerate").textContent = generated ? `${generated}벌 생성` : "생성 안 함";
  }

  function startTicker(message) {
    const started = performance.now();
    clearInterval(ticker);
    ticker = setInterval(() => {
      setStatus(`${message} · ${((performance.now() - started) / 1000).toFixed(0)}s 경과`, "info");
    }, 1000);
    setStatus(message, "info");
    return started;
  }

  async function run() {
    if (running || !sourceDataUrl) return;
    running = true;
    updateRunButton();
    summaryBox.innerHTML = "";
    resultsBox.innerHTML = "";
    const options = readOptions();
    const started = startTicker(options.generate
      ? "분리하고 보수한 뒤 옷마다 FLUX를 돌리고 있어요. 첫 실행은 가중치 적재로 몇 분 걸릴 수 있어요"
      : "분리하고 마스크를 보수하고 있어요");
    try {
      const response = await fetch(`${endpoint}${refinePath}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          image: sourceDataUrl,
          name: sourceFile?.name?.replace(/\.[^.]+$/, "") || "전신샷",
          model: options.model || null,
          categories: options.categories,
          includeRejected: false,
          repair: {
            close: options.close,
            fillHoles: options.fillHoles,
            fillOccluded: options.fillOccluded,
            dropStrays: options.dropStrays,
            smooth: options.smooth,
            closeScale: options.closeScale / 100,
            strayRatio: options.strayRatio / 100,
            occlusionEnclosure: options.occlusionEnclosure / 100,
          },
          generate: options.generate,
          seed: options.seed,
          steps: options.steps,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);

      renderSummary(payload);
      resultsBox.innerHTML = payload.items.map(renderItem).join("")
        || `<p class="seg-empty">처리할 옷이 없어요. 카테고리 선택을 지우거나 다른 사진을 넣어보세요.</p>`;
      const failures = payload.items.filter(item => item.generationError).length;
      clearInterval(ticker);
      setStatus(
        `${payload.items.length}벌 완료 · 전체 ${((performance.now() - started) / 1000).toFixed(1)}s${failures ? ` · ${failures}벌 생성 실패` : ""}`,
        failures ? "warn" : "ok"
      );
    } catch (error) {
      clearInterval(ticker);
      setStatus(`파이프라인 실패: ${error.message}`, "error");
    } finally {
      running = false;
      updateRunButton();
    }
  }

  endpointInput.value = endpoint;
  renderPresets();
  endpointInput.addEventListener("change", () => setEndpoint(endpointInput.value));
  modelSelect.addEventListener("change", syncOutputs);
  Object.values(controls).forEach(control => control.addEventListener("input", syncOutputs));

  // 첫 방문이면 앱의 온보딩 모달이 떠서 이 탭을 덮는다. 개발 탭으로 넘어올 때는 닫는다.
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
  runButton.addEventListener("click", run);

  loadRegistry().then(syncOutputs);
})();
