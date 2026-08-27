/**
 * 개발용 Qwen3-VL 탭 — 사진 한 장에 프롬프트를 붙여 보내고, 돌아온 것만 본다.
 *
 * Seg Lab은 "옷을 어디까지 잡았나", Refine Lab은 "그 조각을 어떻게 그림으로 되돌리나"를
 * 본다. 이 탭은 파이프라인에서 그림을 전혀 그리지 않는 한 칸을 본다: 픽셀을 읽어
 * 앱이 쓸 수 있는 스키마의 한국어 JSON으로 바꾸는 일. 그래서 화면도 두 덩이뿐이다 —
 * 보낸 것(이미지 + 프롬프트)과 받은 것(생성 원문 + 파싱된 필드).
 *
 * 프롬프트는 백엔드에서 받아 온다(GET /api/vlm/prompts). 여기 복사해 두면 백엔드
 * 프롬프트를 고칠 때마다 이 탭이 조용히 거짓말을 하게 된다.
 *
 * refine-lab.js와 마찬가지로 app.js의 전역에 기대지 않는다.
 */
(() => {
  const ENDPOINT_KEY = "wearwell-qwenlab-endpoint";
  const TASK_KEY = "wearwell-qwenlab-task";
  const MAX_UPLOAD_PX = 1400;

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
  // Qwen도 GPU에서 돈다. Refine Lab과 같은 이유로 Colab이 있으면 그쪽이 기본이다.
  const defaultEndpoint = colabBase || localBase;
  const labelFor = url => url === colabBase ? `Colab GPU (${url})` : url === localBase ? `로컬 백엔드 (${url})` : url;

  function tokenFor(url) {
    if (colabBase && url === colabBase) return String(config.API_TOKEN || "");
    if (url === localBase) return String(config.LOCAL_API_TOKEN || "");
    // 직접 입력한 임의 주소로 저장된 API 토큰을 보내면 토큰이 외부 서버에 유출될 수 있다.
    // 인증이 필요한 새 주소는 local-config.js에 먼저 등록해 신뢰할 주소임을 명시한다.
    return "";
  }
  const authHeaders = () => {
    const token = tokenFor(endpoint);
    return token ? { Authorization: `Bearer ${token}` } : {};
  };

  // 태스크마다 프롬프트에 함께 실리는 값이 다르다. 백엔드가 프롬프트 뒤에 붙이는
  // 것과 같은 값이라, 여기서 뭘 바꾸면 결과가 어떻게 달라지는지가 이 탭의 실험이다.
  const TASK_INPUTS = {
    garment: [
      { key: "name", label: "name", type: "text", value: "화이트 옥스퍼드 셔츠", hint: "옷장에 저장된 이름. 사진과 어긋나게 넣어 보면 모델이 어느 쪽을 믿는지 보인다" },
      { key: "category", label: "category", type: "text", value: "상의", hint: "상의/하의/아우터/원피스/신발/가방/액세서리" },
    ],
    lookbook: [],
    body: [
      { key: "gender", label: "gender", type: "select", value: "women", options: ["women", "men", ""], hint: "빈 값이면 '미지정'으로 나간다" },
    ],
    "tryon-judge": [
      { key: "manifest", label: "manifest", type: "textarea", value: "상의: 화이트 옥스퍼드 셔츠\n하의: 인디고 데님 팬츠\n신발: 블랙 더비", hint: "입히기로 한 목록. 심판은 이 목록과 사진만 보고 채점한다" },
    ],
  };

  // 파싱된 JSON에서 이건 모델이 답한 내용이 아니라 백엔드가 붙인 메타다.
  const META_KEYS = new Set([
    "engine", "model", "quantization", "seconds", "loadSeconds", "rawText",
    "outputTokens", "truncated", "imageSize", "prompt", "maxNewTokens",
  ]);

  const nav = $("#qwenLabNavLink");
  const healthBox = $("#qwenLabHealth");
  const endpointInput = $("#qwenLabEndpoint");
  const presetBox = $("#qwenLabPresets");
  const taskBox = $("#qwenLabTasks");
  const inputBox = $("#qwenLabInputs");
  const dropZone = $("#qwenLabDropZone");
  const fileInput = $("#qwenLabFileInput");
  const sourceFigure = $("#qwenLabSource");
  const runButton = $("#qwenLabRun");
  const statusBox = $("#qwenLabStatus");
  const runsBox = $("#qwenLabRuns");
  if (!nav || !taskBox) return;
  nav.hidden = false;

  let endpoint = readStorage(ENDPOINT_KEY) || defaultEndpoint;
  let catalog = null;
  let taskKey = readStorage(TASK_KEY) || "garment";
  let inputs = {};
  let sourceFile = null;
  let sourceDataUrl = null;
  let running = false;
  let ticker = null;
  let runSeq = 0;

  const task = () => (catalog?.tasks || []).find(item => item.key === taskKey) || null;

  function setHealth(state, message) {
    healthBox.className = `seg-lab-health ${state}`;
    healthBox.innerHTML = `<span class="seg-dot"></span><small>${escapeHtml(message)}</small>`;
  }

  function setStatus(message, tone = "info") {
    statusBox.hidden = !message;
    statusBox.className = `seg-lab-status ${tone}`;
    statusBox.textContent = message || "";
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
    loadCatalog();
  }

  function updateRunButton() {
    runButton.disabled = running || !sourceDataUrl || !catalog || !task();
    runButton.textContent = running ? "Qwen이 읽는 중…" : "Qwen에게 물어보기";
  }

  /** 태스크 카드. 각 카드에 '무슨 일을 맡았는지' 한 문장을 붙여 둔다 — 이 탭의 본론이다. */
  function renderTasks() {
    taskBox.innerHTML = (catalog?.tasks || []).map(item => `
      <button type="button" class="qwen-task${item.key === taskKey ? " on" : ""}" data-task="${escapeHtml(item.key)}">
        <b>${escapeHtml(item.title)}</b>
        <small>${escapeHtml(item.role)}</small>
        <code>${escapeHtml(item.path)} · ${escapeHtml(item.maxNewTokens)} tok</code>
      </button>`).join("");
    taskBox.querySelectorAll("button").forEach(button => button.addEventListener("click", () => {
      taskKey = button.dataset.task;
      writeStorage(TASK_KEY, taskKey);
      renderTasks();
      renderInputs();
      renderFlow();
      updateRunButton();
    }));
  }

  /** 프롬프트 뒤에 붙는 값들. 태스크마다 다르고, 없는 태스크는 사진만 보낸다. */
  function renderInputs() {
    const fields = TASK_INPUTS[taskKey] || [];
    inputs = Object.fromEntries(fields.map(field => [field.key, inputs[field.key] ?? field.value]));
    if (!fields.length) {
      inputBox.innerHTML = `<p class="qwen-empty">이 일은 사진만 보내요. 프롬프트에 더 실리는 값이 없어요.</p>`;
      return;
    }
    inputBox.innerHTML = fields.map(field => {
      const value = escapeHtml(inputs[field.key]);
      const control = field.type === "textarea"
        ? `<textarea data-key="${field.key}" rows="4">${value}</textarea>`
        : field.type === "select"
          ? `<select data-key="${field.key}">${field.options.map(option =>
              `<option value="${escapeHtml(option)}"${option === inputs[field.key] ? " selected" : ""}>${escapeHtml(option || "(미지정)")}</option>`).join("")}</select>`
          : `<input data-key="${field.key}" type="text" value="${value}" />`;
      return `<label class="qwen-input"><b>${escapeHtml(field.label)}</b>${control}<small>${escapeHtml(field.hint)}</small></label>`;
    }).join("");
    inputBox.querySelectorAll("[data-key]").forEach(control => control.addEventListener("input", () => {
      inputs[control.dataset.key] = control.value;
    }));
  }

  function renderFlow() {
    const current = task();
    $("#qwenFlowPrompt").textContent = current?.prompt
      ? `${current.prompt.length.toLocaleString("ko-KR")}자 · 스키마 고정`
      : "스키마를 문장으로 못 박음";
    $("#qwenFlowGenerate").textContent = current ? `greedy · 최대 ${current.maxNewTokens} tok` : "greedy · 샘플링 없음";
    $("#qwenFlowUse").textContent = current ? current.title : "옷장 · 추천 · 채점";
  }

  const STALE_BACKEND = "Colab 노트북 셀 2~3을 다시 실행해 최신 코드로 띄우세요";

  /**
   * /api/vlm/prompts가 없는 예전 백엔드용 목록. 프롬프트 본문은 **일부러 비운다** —
   * 여기 복사해 두면 백엔드 프롬프트를 고칠 때마다 이 탭이 조용히 거짓말을 한다.
   *
   * 이 상태로도 네 태스크를 돌려 파싱된 필드는 볼 수 있다. 이 탭의 본론인
   * 프롬프트와 생성 원문만 백엔드를 올려야 나온다 — 헬스 줄에 그렇게 적어 둔다.
   */
  const FALLBACK_TASKS = [
    { key: "garment", title: "옷 한 벌 분석", role: "옷장에 넣을 사진에서 카테고리·색·소재·핏·디테일을 뽑는다.", maxNewTokens: 640, path: "/api/vlm/garment", prompt: "" },
    { key: "lookbook", title: "룩북 분해", role: "한 사람이 입은 착장을 옷 단위로 쪼개고 bbox까지 준다.", maxNewTokens: 900, path: "/api/vlm/lookbook", prompt: "" },
    { key: "body", title: "체형 특징", role: "전신사진에서 보이는 체형·비율·어깨선만 말한다.", maxNewTokens: 320, path: "/api/vlm/body", prompt: "" },
    { key: "tryon-judge", title: "착장 결과 심판", role: "합성된 착장 사진의 레이어 순서·누락·정체성을 판정한다.", maxNewTokens: 700, path: "/api/vlm/tryon-judge", prompt: "" },
  ];

  function failCatalog(message) {
    setHealth("down", `${labelFor(endpoint)} · ${message}`);
    catalog = null;
    taskBox.innerHTML = "";
    updateRunButton();
  }

  async function loadCatalog() {
    setHealth("checking", `${labelFor(endpoint)} 확인 중…`);
    catalog = null;
    let health = null;
    try {
      const response = await fetch(`${endpoint}/api/health`);
      if (!response.ok) return failCatalog(`헬스 응답이 HTTP ${response.status}예요 — 주소를 확인하세요`);
      health = await response.json();
    } catch (error) {
      return failCatalog(`연결 실패 (${error.message}) — 백엔드가 떠 있는지, 터널 주소가 최신인지 확인하세요`);
    }

    // 프롬프트 목록이 없다고 탭을 닫아 버리면, GPU가 있는 백엔드가 예전 코드일 때
    // Qwen Lab만 통째로 못 쓰게 된다. 목록이 없으면 태스크만 들고 계속 간다.
    let stale = "";
    try {
      const response = await fetch(`${endpoint}/api/vlm/prompts`, { headers: authHeaders() });
      if (response.ok) {
        catalog = await response.json();
      } else if (response.status === 404) {
        stale = `프롬프트 목록이 없는 예전 백엔드예요 — ${STALE_BACKEND}`;
      } else {
        return failCatalog(`프롬프트 목록이 HTTP ${response.status}예요`);
      }
    } catch (error) {
      return failCatalog(`프롬프트 목록을 읽지 못했어요 (${error.message})`);
    }
    if (!catalog) {
      catalog = {
        model: health.vlmModel || "Qwen/Qwen3-VL-8B-Instruct",
        quantization: health.vlmQuantization || "nf4",
        tasks: FALLBACK_TASKS,
        stale: true,
      };
    }

    if (!task()) taskKey = catalog.tasks[0]?.key || "garment";
    renderTasks();
    renderInputs();
    renderFlow();

    // Qwen은 4bit로도 8B다. 콜드 적재가 첫 요청에 통째로 붙는다는 걸 미리 말해 준다.
    const where = labelFor(endpoint);
    const quant = catalog.quantization === "none" ? "양자화 없음" : catalog.quantization;
    const loaded = health.vlmLoaded ? "적재됨" : "첫 요청 때 적재 (1~2분)";
    if (stale) {
      // GPU도 Qwen도 멀쩡하지만 이 탭이 보여줄 것의 절반이 없다. 붙었다고만 하면 안 된다.
      setHealth("warn", `${where} · ${health.gpu || "GPU"} · 돌릴 수는 있지만 프롬프트·생성 원문은 안 보여요 — ${stale}`);
    } else if (health.cuda) {
      setHealth("up", `${where} · ${health.gpu} · ${catalog.model} · ${quant} · ${loaded}`);
    } else {
      setHealth("warn", `${where} · GPU가 없어요 — Qwen은 CPU에서 사실상 돌지 않아요. Colab 백엔드를 쓰세요`);
    }
    updateRunButton();
  }

  /** 원본 그대로 보내면 요청이 수 MB가 된다. 백엔드도 max_pixels로 한 번 더 줄인다. */
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

  async function pickFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    sourceFile = file;
    setStatus("");
    sourceDataUrl = await toDataUrl(file);
    const image = new Image();
    await new Promise(resolve => { image.onload = resolve; image.src = sourceDataUrl; });
    sourceFigure.hidden = false;
    sourceFigure.querySelector("img").src = sourceDataUrl;
    sourceFigure.querySelector("figcaption").textContent = `${file.name} · ${image.width}×${image.height} 전송`;
    $("#qwenFlowImage").textContent = `${image.width}×${image.height}`;
    updateRunButton();
  }

  function readout(rows) {
    return `<dl class="refine-readout">${rows.map(([label, value, tone = ""]) =>
      `<div class="${tone}"><dt>${escapeHtml(label)}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
  }

  /**
   * 프롬프트를 그대로 띄우되, 요청값이 붙은 꼬리만 갈라 보여준다. 스키마(고정)와
   * 이번 요청에만 실린 값(가변)을 눈으로 못 가르면 "왜 이 답이 나왔나"를 못 짚는다.
   */
  function promptPanel(prompt, base) {
    if (!prompt) return `<p class="qwen-empty">이 백엔드는 프롬프트를 돌려주지 않아요 — ${escapeHtml(STALE_BACKEND)}.</p>`;
    const tail = base && prompt.startsWith(base) ? prompt.slice(base.length) : "";
    const head = tail ? prompt.slice(0, prompt.length - tail.length) : prompt;
    return `<pre class="qwen-prompt">${escapeHtml(head)}${tail ? `<mark>${escapeHtml(tail)}</mark>` : ""}</pre>`;
  }

  const formatValue = value => {
    if (Array.isArray(value)) return value.length ? value.map(item => typeof item === "object" ? JSON.stringify(item) : String(item)).join(", ") : "—";
    if (value && typeof value === "object") return JSON.stringify(value);
    if (value === "" || value == null) return "—";
    return String(value);
  };

  /** 모델이 채운 필드만. 확신이 없을 때 "확인 어려움"으로 답하는지도 여기서 보인다. */
  function fieldsPanel(result) {
    const entries = Object.entries(result).filter(([key]) => !META_KEYS.has(key) && key !== "pieces");
    if (!entries.length) return `<p class="qwen-empty">평면 필드가 없어요.</p>`;
    return `<dl class="qwen-fields">${entries.map(([key, value]) => {
      const text = formatValue(value);
      const unsure = /확인 어려움|미분류/.test(text);
      return `<div${unsure ? ' class="unsure"' : ""}><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(text)}</dd></div>`;
    }).join("")}</dl>`;
  }

  /**
   * 룩북만 bbox를 준다. 사진 위에 그대로 얹어야 "옷을 실제 경계대로 쪼갰나"가 보인다.
   * 좌표계는 왼쪽 위 0,0 / 오른쪽 아래 1000,1000이다.
   */
  function piecesPanel(result, image) {
    const pieces = Array.isArray(result.pieces) ? result.pieces : [];
    if (!pieces.length) return "";
    const boxes = pieces.map((piece, index) => {
      const box = Array.isArray(piece.bbox) && piece.bbox.length === 4 ? piece.bbox : null;
      if (!box) return "";
      const [x1, y1, x2, y2] = box.map(Number);
      const style = `left:${x1 / 10}%;top:${y1 / 10}%;width:${(x2 - x1) / 10}%;height:${(y2 - y1) / 10}%`;
      return `<span class="qwen-box" style="${style}" data-index="${index + 1}"><i>${index + 1}</i></span>`;
    }).join("");
    return `<div class="qwen-pieces">
      <figure class="qwen-boxes"><img src="${image}" alt="bbox 오버레이" /><div>${boxes}</div></figure>
      <ol class="qwen-piece-list">${pieces.map((piece, index) => `
        <li><b>${index + 1}</b>
          <div>
            <strong>${escapeHtml(piece.label || piece.subcategory || "이름 없음")}</strong>
            <small>${escapeHtml([piece.layer, piece.category, piece.subcategory].filter(Boolean).join(" · "))}</small>
            <small>${escapeHtml([...(piece.colors || []), ...(piece.materials || []), ...(piece.fits || [])].join(", "))}</small>
          </div>
          <em>${piece.confidence == null ? "" : Number(piece.confidence).toFixed(2)}</em>
        </li>`).join("")}</ol>
    </div>`;
  }

  function renderRun({ index, taskMeta, result, image, elapsed }) {
    const budget = result.maxNewTokens || taskMeta.maxNewTokens;
    const article = document.createElement("article");
    article.className = "qwen-run";
    article.innerHTML = `
      <header>
        <span class="seg-chip">${escapeHtml(taskMeta.title)}</span>
        <span class="qwen-role">${escapeHtml(taskMeta.role)}</span>
        <span class="refine-timing">#${index} · 왕복 ${elapsed}s</span>
      </header>
      ${readout([
        ["모델", `<code>${escapeHtml(result.model || catalog.model)}</code> <i>${escapeHtml(result.quantization || "")}</i>`],
        ["백엔드가 본 이미지", result.imageSize ? `${escapeHtml(result.imageSize.width)}×${escapeHtml(result.imageSize.height)}` : "—"],
        ["생성", `${escapeHtml(result.seconds ?? "?")}s${result.loadSeconds ? ` <i>· 콜드 적재 ${escapeHtml(result.loadSeconds)}s</i>` : ""}`],
        ["출력 토큰", `${escapeHtml(result.outputTokens ?? "?")} / ${escapeHtml(budget)}`, result.truncated ? "warn" : ""],
        ["파싱", result.truncated ? "예산에서 잘렸어요 — 필드가 비었을 수 있어요" : "코드펜스를 걷고 객체만 취함", result.truncated ? "warn" : "ok"],
      ])}
      <div class="qwen-panes">
        <section>
          <h4>보낸 것 · 프롬프트</h4>
          ${promptPanel(result.prompt || "", taskMeta.prompt)}
          <p class="qwen-note"><mark>표시</mark>는 이번 요청에만 붙은 값이에요. 나머지는 백엔드에 고정된 스키마예요.</p>
        </section>
        <section>
          <h4>받은 것 · 생성 원문</h4>
          <pre class="qwen-raw">${escapeHtml(result.rawText || "(원문 없음 — 예전 백엔드예요)")}</pre>
          <h4>파싱된 필드</h4>
          ${fieldsPanel(result)}
        </section>
      </div>
      ${piecesPanel(result, image)}`;
    runsBox.prepend(article);
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

  function requestBody() {
    const body = { image: sourceDataUrl, debug: true };
    if (taskKey === "tryon-judge") return { image: sourceDataUrl, debug: true, manifest: inputs.manifest || "상의: 확인 필요" };
    body.name = inputs.name || sourceFile?.name?.replace(/\.[^.]+$/, "") || "사진";
    if (taskKey === "garment") body.category = inputs.category || null;
    if (taskKey === "body") body.gender = inputs.gender || null;
    return body;
  }

  async function run() {
    const taskMeta = task();
    if (running || !sourceDataUrl || !taskMeta) return;
    running = true;
    updateRunButton();
    const started = startTicker(`Qwen3-VL이 사진을 읽고 있어요 — ${taskMeta.title}. 첫 요청은 가중치 적재로 몇 분 걸릴 수 있어요`);
    try {
      const response = await fetch(`${endpoint}${taskMeta.path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify(requestBody()),
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(result.detail || `HTTP ${response.status}`);
      const elapsed = ((performance.now() - started) / 1000).toFixed(1);
      renderRun({ index: ++runSeq, taskMeta, result, image: sourceDataUrl, elapsed });
      clearInterval(ticker);
      setStatus(
        `${taskMeta.title} 완료 · 생성 ${result.seconds ?? "?"}s · 왕복 ${elapsed}s${result.truncated ? " · 출력이 토큰 예산에서 잘렸어요" : ""}`,
        result.truncated ? "warn" : "ok"
      );
    } catch (error) {
      clearInterval(ticker);
      // 422는 대개 "JSON이 아닌 답을 했다"이다 — 프롬프트 문제지 서버 문제가 아니다.
      setStatus(`실패: ${error.message}`, "error");
    } finally {
      running = false;
      updateRunButton();
    }
  }

  endpointInput.value = endpoint;
  renderPresets();
  renderInputs();
  endpointInput.addEventListener("change", () => setEndpoint(endpointInput.value));

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
    isActive: () => $("#qwenlabView").classList.contains("active"),
    onFiles: files => pickFile(files[0]),
  });
  runButton.addEventListener("click", run);

  loadCatalog();
})();
