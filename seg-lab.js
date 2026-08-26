/**
 * 개발용 세그멘테이션 비교 탭.
 *
 * 사진 한 장을 백엔드의 /api/dev/segment/compare에 보내 모델별 결과를 나란히 놓는다.
 * app.js의 전역에 의존하지 않는다 — 앱 코드를 건드려도 이 도구가 같이 깨지지 않게.
 */
(() => {
  const ENDPOINT_KEY = "wearwell-seglab-endpoint";
  const MODELS_KEY = "wearwell-seglab-models";
  const MAX_UPLOAD_PX = 1400;

  const $ = selector => document.querySelector(selector);
  const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
  // 작은 값은 자리를 하나 더 준다 — 1.16%와 기준 1.20%가 둘 다 "1.2%"로 보이면
  // 왜 걸러졌는지 화면에서 알 수가 없다.
  const percent = value => `${(value * 100).toFixed(value < 0.1 ? 2 : 1)}%`;

  // 개발 탭은 로컬에서 열었을 때나 ?dev=1을 붙였을 때만 보인다.
  const devMode = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname)
    || new URLSearchParams(location.search).get("dev") === "1";
  if (!devMode) return;

  const config = window.WEARWELL_CONFIG || {};
  const readStorage = key => { try { return localStorage.getItem(key) || ""; } catch { return ""; } };
  const writeStorage = (key, value) => { try { localStorage.setItem(key, value); } catch { /* 시크릿 창 등 */ } };

  const defaultEndpoint = String(config.LOCAL_API_BASE || "http://127.0.0.1:8787").replace(/\/+$/, "");
  let endpoint = readStorage(ENDPOINT_KEY) || defaultEndpoint;
  let registry = null;
  let selected = new Set();
  let sourceFile = null;
  let sourceDataUrl = null;
  let running = false;

  const nav = $("#segLabNavLink");
  const healthBox = $("#segLabHealth");
  const endpointInput = $("#segLabEndpoint");
  const picker = $("#segLabModelPicker");
  const dropZone = $("#segLabDropZone");
  const fileInput = $("#segLabFileInput");
  const sourceFigure = $("#segLabSource");
  const runButton = $("#segLabRun");
  const statusBox = $("#segLabStatus");
  const resultsBox = $("#segLabResults");
  const agreementBox = $("#segLabAgreement");
  if (!nav || !picker) return;
  nav.hidden = false;

  function authHeaders() {
    const token = String(config.LOCAL_API_TOKEN || config.API_TOKEN || "");
    return token ? { Authorization: `Bearer ${token}` } : {};
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

  function updateRunButton() {
    runButton.disabled = running || !sourceDataUrl || selected.size === 0 || !registry;
    runButton.textContent = running ? "실행 중…" : `모델 ${selected.size}개 비교 실행`;
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

  async function loadRegistry() {
    setHealth("checking", `${endpoint} 확인 중…`);
    picker.innerHTML = "";
    registry = null;
    try {
      const response = await fetch(`${endpoint}/api/dev/segment/models`, { headers: authHeaders() });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      registry = await response.json();
    } catch (error) {
      setHealth("down", `백엔드에 연결할 수 없어요 (${error.message}). backend/app.py를 실행하세요.`);
      picker.innerHTML = `<p class="seg-empty">모델 목록을 불러오지 못했어요.</p>`;
      updateRunButton();
      return;
    }

    // 비교 실행 후 "적재됨" 배지를 새로 그리려고 다시 부른다. 그때 사용자가 고른
    // 조합을 기본값으로 되돌리면 안 되므로, 이미 고른 게 있으면 그대로 둔다.
    const known = key => registry.models.some(model => model.key === key);
    const stored = readStorage(MODELS_KEY).split(",").filter(known);
    if (!selected.size) selected = new Set(stored.length ? stored : registry.default);
    else selected = new Set([...selected].filter(known));
    setHealth("up", `${endpoint} · 적재됨 ${registry.loaded.length ? registry.loaded.join(", ") : "없음"}`);
    renderPicker();
    updateRunButton();
  }

  /** 기본 기준을 덮어쓴 모델임을 표시. 같은 사진에서 기준선이 달라 보이는 이유다. */
  function renderOverrides(overrides) {
    const entries = Object.entries(overrides || {});
    if (!entries.length) return "";
    const text = entries.map(([category, values]) =>
      `${category} ${Object.entries(values).map(([key, value]) => `${key} ${value}`).join(", ")}`).join(" · ");
    return `<span class="seg-model-override">기준 보정: ${escapeHtml(text)}</span>`;
  }

  function renderPicker() {
    picker.innerHTML = registry.models.map(model => `
      <label class="seg-model-option${selected.has(model.key) ? " on" : ""}">
        <input type="checkbox" value="${escapeHtml(model.key)}"${selected.has(model.key) ? " checked" : ""} />
        <span class="seg-model-name">
          ${escapeHtml(model.title)}
          ${model.key === registry.production ? '<em class="seg-tag">프로덕션</em>' : ""}
          ${registry.loaded.includes(model.key) ? '<em class="seg-tag loaded">적재됨</em>' : ""}
        </span>
        <span class="seg-model-meta">${escapeHtml(model.taxonomy)} · ${model.weightsMb}MB · ${escapeHtml(model.modelId)}</span>
        <span class="seg-model-summary">${escapeHtml(model.summary)}</span>
        <span class="seg-model-watch">주의: ${escapeHtml(model.watchFor)}</span>
        ${renderOverrides(model.thresholdOverrides)}
      </label>`).join("");

    picker.querySelectorAll("input").forEach(input => input.addEventListener("change", () => {
      if (input.checked) selected.add(input.value); else selected.delete(input.value);
      input.closest(".seg-model-option").classList.toggle("on", input.checked);
      writeStorage(MODELS_KEY, [...selected].join(","));
      updateRunButton();
    }));
  }

  async function pickFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    sourceFile = file;
    setStatus("");
    resultsBox.innerHTML = "";
    agreementBox.innerHTML = "";
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

  /** 지표 하나를 기준선과 함께. 기준 미달이면 빨갛게 — 왜 걸러졌는지 눈으로 찾게. */
  function metric(label, value, threshold, formatter) {
    const failed = value < threshold;
    return `<div class="seg-metric${failed ? " fail" : ""}">
      <span>${escapeHtml(label)}</span>
      <b>${formatter(value)}</b>
      <small>기준 ${formatter(threshold)}</small>
    </div>`;
  }

  function renderItem(item) {
    const color = colorFor(item.category);
    return `<article class="seg-item${item.accepted ? "" : " rejected"}">
      <div class="seg-item-head">
        <span class="seg-chip" style="--chip:${color}">${escapeHtml(item.category)}</span>
        <span class="seg-verdict">${item.accepted ? "통과" : "걸러짐"}</span>
      </div>
      <div class="seg-item-body">
        <img src="${item.image}" alt="${escapeHtml(item.category)} 분리 결과" loading="lazy" />
        <div class="seg-metrics">
          ${metric("면적", item.areaRatio, item.thresholds.minArea, percent)}
          ${metric("채움", item.fillRatio, item.thresholds.minFill, percent)}
          ${metric("확신도", item.confidence, item.thresholds.minConfidence, value => value.toFixed(2))}
        </div>
      </div>
      <p class="seg-item-label">${escapeHtml(item.label)}</p>
      ${item.rejectReason ? `<p class="seg-reject">${escapeHtml(item.rejectReason)}</p>` : ""}
    </article>`;
  }

  function renderRawLabels(rawLabels) {
    if (!rawLabels.length) return "";
    const chips = rawLabels.map(row => {
      const kind = row.isPart ? "part" : row.category ? "mapped" : "unmapped";
      const style = row.category ? ` style="--chip:${colorFor(row.category)}"` : "";
      return `<span class="seg-raw ${kind}"${style}>${escapeHtml(row.label)}<b>${percent(row.pixelRatio)}</b></span>`;
    }).join("");
    const parts = rawLabels.filter(row => row.isPart);
    const note = parts.length
      ? `<small class="seg-raw-note">부속 라벨 ${parts.length}개가 픽셀 ${percent(parts.reduce((sum, row) => sum + row.pixelRatio, 0))}를 가져갔어요 — 그만큼 본체 마스크에 구멍이 납니다.</small>`
      : "";
    return `<details class="seg-raw-labels"><summary>모델이 예측한 원본 라벨 ${rawLabels.length}개</summary><div>${chips}</div>${note}</details>`;
  }

  function renderResult(result) {
    if (result.error) {
      return `<section class="seg-column failed">
        <header><h3>${escapeHtml(result.model.title)}</h3><code>${escapeHtml(result.model.modelId)}</code></header>
        <p class="seg-error">${escapeHtml(result.error)}</p>
      </section>`;
    }
    const rejected = result.items.length - result.acceptedCount;
    return `<section class="seg-column">
      <header>
        <h3>${escapeHtml(result.model.title)}</h3>
        <code>${escapeHtml(result.model.modelId)}</code>
        <div class="seg-runstats">
          <span><b>${result.inferenceSeconds}s</b> 추론</span>
          <span><b>${result.device}</b></span>
          <span><b>${result.acceptedCount}</b> 통과${rejected ? ` / ${rejected} 걸러짐` : ""}</span>
          ${result.loadSeconds > 0 ? `<span class="cold"><b>${result.loadSeconds}s</b> 콜드 적재</span>` : ""}
        </div>
      </header>
      <img class="seg-overlay" src="${result.overlay}" alt="${escapeHtml(result.model.title)} 세그멘테이션 오버레이" loading="lazy" />
      <div class="seg-items">${result.items.map(renderItem).join("") || '<p class="seg-empty">검출된 옷이 없어요.</p>'}</div>
      ${renderRawLabels(result.rawLabels)}
    </section>`;
  }

  function titleOf(key) {
    return registry.models.find(model => model.key === key)?.title || key;
  }

  /** 모델 쌍별 카테고리 IoU. 어느 옷에서 의견이 갈리는지가 비교의 핵심이다. */
  function renderAgreement(rows) {
    if (!rows.length) return "";
    const body = rows.map(row => {
      const cells = Object.entries(row.categories).map(([category, score]) => {
        if (typeof score === "object") {
          return `<div class="seg-iou only">
            <span class="seg-chip" style="--chip:${colorFor(category)}">${escapeHtml(category)}</span>
            <b>${escapeHtml(titleOf(score.onlyIn))}만 검출</b></div>`;
        }
        return `<div class="seg-iou">
          <span class="seg-chip" style="--chip:${colorFor(category)}">${escapeHtml(category)}</span>
          <b>${score.toFixed(3)}</b>
          <i style="width:${Math.round(score * 100)}%"></i></div>`;
      }).join("");
      return `<div class="seg-pair"><h4>${escapeHtml(titleOf(row.left))} <span>vs</span> ${escapeHtml(titleOf(row.right))}</h4><div class="seg-ious">${cells}</div></div>`;
    }).join("");
    return `<h2 class="seg-section-title">카테고리별 IoU</h2>
      <p class="seg-section-lede">두 모델이 같은 옷을 같은 픽셀로 봤는지. 1에 가까울수록 같은 판단이고, 낮으면 경계나 분류가 갈린 겁니다.</p>
      <div class="seg-pairs">${body}</div>`;
  }

  async function run() {
    if (running || !sourceDataUrl) return;
    running = true;
    updateRunButton();
    resultsBox.innerHTML = "";
    agreementBox.innerHTML = "";
    setStatus(`모델 ${selected.size}개를 순서대로 실행 중이에요. 처음 쓰는 모델은 가중치를 내려받느라 오래 걸릴 수 있어요.`, "info");
    const started = performance.now();
    try {
      const response = await fetch(`${endpoint}/api/dev/segment/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify({
          image: sourceDataUrl,
          name: sourceFile?.name?.replace(/\.[^.]+$/, "") || "전신샷",
          models: [...selected]
        })
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);

      resultsBox.innerHTML = payload.results.map(renderResult).join("");
      agreementBox.innerHTML = renderAgreement(payload.agreement || []);
      const failures = payload.results.filter(result => result.error).length;
      setStatus(
        `${payload.results.length}개 모델 완료 · 전체 ${((performance.now() - started) / 1000).toFixed(1)}s${failures ? ` · ${failures}개 실패` : ""}`,
        failures ? "warn" : "ok"
      );
      loadRegistry();
    } catch (error) {
      setStatus(`비교 실패: ${error.message}`, "error");
    } finally {
      running = false;
      updateRunButton();
    }
  }

  endpointInput.value = endpoint;
  endpointInput.addEventListener("change", () => {
    endpoint = endpointInput.value.trim().replace(/\/+$/, "") || defaultEndpoint;
    endpointInput.value = endpoint;
    writeStorage(ENDPOINT_KEY, endpoint);
    loadRegistry();
  });

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

  loadRegistry();
})();
