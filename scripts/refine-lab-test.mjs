/**
 * Refine Lab 탭 E2E 스모크 테스트.
 *
 * 실행 전 두 서버가 떠 있어야 한다.
 *   1) 정적 프론트: python -m http.server 8000 --bind 127.0.0.1
 *   2) 백엔드:      cd backend && WEARWELL_API_TOKEN=... uvicorn app:app --port 8787
 *
 * file:// 로 열면 백엔드 CORS(http://127.0.0.1 만 허용)에 막히므로 반드시 8000으로 연다.
 *   node scripts/refine-lab-test.mjs [사진경로]
 *
 * 기본값은 FLUX 재생성을 끄고 1~4단계(분리·진단·보수·정규화)만 검사한다 — GPU 없는
 * 기계에서도 돌아야 하기 때문이다. 5단계까지 보려면 REFINE_LAB_GENERATE=1을 준다
 * (가중치 콜드 적재 때문에 몇 분 걸릴 수 있다).
 */
import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const pageUrl = process.env.REFINE_LAB_URL || "http://127.0.0.1:8000/?dev=1";
const endpoint = process.env.REFINE_LAB_ENDPOINT || "";
const generate = process.env.REFINE_LAB_GENERATE === "1";
const photo = path.resolve(process.argv[2] || "scripts/.musinsa-snaps/snap-004.jpg");
const profile = await mkdtemp(path.join(os.tmpdir(), "wearwell-refinelab-"));
const chromeProcess = spawn(chrome, [
  "--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1600,1400",
  "--remote-debugging-port=9225", `--user-data-dir=${profile}`, pageUrl
], { windowsHide: true, stdio: "ignore" });

let tabs;
for (let attempt = 0; attempt < 40; attempt++) {
  try {
    tabs = await fetch("http://127.0.0.1:9225/json").then(response => response.json());
    if (tabs.some(tab => tab.type === "page")) break;
  } catch { /* 아직 안 떴다 */ }
  await new Promise(resolve => setTimeout(resolve, 200));
}
const tab = tabs?.find(item => item.type === "page");
if (!tab) throw new Error("Chrome에 연결하지 못했습니다");

const socket = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener("open", resolve, { once: true }));
let messageId = 0;
const pending = new Map();
const consoleErrors = [];
socket.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
  if (message.method === "Runtime.exceptionThrown") {
    consoleErrors.push(message.params.exceptionDetails?.exception?.description || "unknown exception");
  }
});
const send = (method, params = {}) => new Promise(resolve => {
  const id = ++messageId;
  pending.set(id, resolve);
  socket.send(JSON.stringify({ id, method, params }));
});
const evaluate = async expression => {
  const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
  if (result.result?.exceptionDetails) throw new Error(result.result.exceptionDetails.exception?.description || "evaluate failed");
  return result.result?.result?.value;
};
/** 조건이 참이 될 때까지 폴링. 모델 적재는 콜드일 때 수십 초가 걸린다. */
const waitFor = async (expression, timeoutMs, label) => {
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    if (await evaluate(expression)) return;
    if (Date.now() > deadline) throw new Error(`시간 초과: ${label}`);
    await new Promise(resolve => setTimeout(resolve, 400));
  }
};

await send("Runtime.enable");
await send("DOM.enable");
await evaluate("localStorage.removeItem('wearwell-refinelab-options'), true");
if (endpoint) {
  await evaluate(`localStorage.setItem('wearwell-refinelab-endpoint', ${JSON.stringify(endpoint)}), true`);
  await send("Page.reload");
  await new Promise(resolve => setTimeout(resolve, 800));
}

await waitFor("!document.querySelector('#refineLabNavLink').hidden", 8000, "Refine Lab 탭 노출");
await waitFor("document.querySelectorAll('#refineLabModel option').length > 0", 15000, "모델 목록 로딩");
const setup = await evaluate(`({
  health: document.querySelector('#refineLabHealth').className,
  healthText: document.querySelector('#refineLabHealth').textContent.trim(),
  models: document.querySelectorAll('#refineLabModel option').length,
  selected: document.querySelector('#refineLabModel').value,
  chips: document.querySelectorAll('.refine-chip').length,
  repairControls: document.querySelectorAll('#refineOptClose, #refineOptHoles, #refineOptOccluded, #refineOptStrays, #refineOptSmooth, #refineEnclosure').length,
  presets: document.querySelectorAll('.refine-preset').length,
  activePreset: document.querySelector('.refine-preset.on')?.textContent.trim() || "",
  endpoint: document.querySelector('#refineLabEndpoint').value,
  flowSteps: document.querySelectorAll('.refine-flow li').length
})`);

await evaluate("document.querySelector('[data-view=\"refinelab\"]').click(), true");
await waitFor("document.querySelector('#refinelabView').classList.contains('active')", 3000, "탭 전환");
// 온보딩 모달은 옷장 복원이 끝난 뒤에 열리기도 한다. 탭 클릭보다 늦게 뜨면 화면을 덮는다.
await evaluate("document.querySelectorAll('dialog[open]').forEach(dialog => dialog.close()), true");
if (!generate) {
  await evaluate(`(() => {
    const box = document.querySelector('#refineOptGenerate');
    box.checked = false;
    box.dispatchEvent(new Event('input', { bubbles: true }));
    return true;
  })()`);
}

// 파일 입력에 실제 사진을 넣는다. setFileInputFiles가 change 이벤트까지 발생시킨다.
const { result: doc } = await send("DOM.getDocument");
const { result: node } = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#refineLabFileInput" });
await send("DOM.setFileInputFiles", { files: [photo], nodeId: node.nodeId });
await waitFor("!document.querySelector('#refineLabRun').disabled", 10000, "사진 준비");

await evaluate("document.querySelector('#refineLabRun').click(), true");
await waitFor(
  "['ok', 'warn', 'error'].some(tone => document.querySelector('#refineLabStatus').classList.contains(tone))",
  generate ? 900000 : 300000,
  "파이프라인 실행"
);

// 스테이지 이미지는 loading="lazy"라 화면 밖에 있으면 naturalWidth가 0이다.
// 깨진 이미지와 구분하려면 전부 강제로 받아온 뒤에 재야 한다.
await evaluate(`(async () => {
  const images = [...document.querySelectorAll('.refine-stage img, .refine-overlay img')];
  images.forEach(image => { image.loading = "eager"; });
  await Promise.all(images.map(image => image.complete ? null : new Promise(resolve => {
    image.addEventListener("load", resolve, { once: true });
    image.addEventListener("error", resolve, { once: true });
  })));
  return true;
})()`);

const value = await evaluate(`({
  status: document.querySelector('#refineLabStatus').textContent,
  statusTone: document.querySelector('#refineLabStatus').className,
  items: document.querySelectorAll('.refine-item').length,
  stages: document.querySelectorAll('.refine-stage').length,
  stagesPerItem: [...document.querySelectorAll('.refine-item')].map(item => item.querySelectorAll('.refine-stage').length),
  imagesLoaded: [...document.querySelectorAll('.refine-stage img, .refine-overlay img')].filter(img => img.naturalWidth > 0).length,
  imagesTotal: document.querySelectorAll('.refine-stage img, .refine-overlay img').length,
  repairSteps: document.querySelectorAll('.refine-steps li').length,
  diagnosed: [...document.querySelectorAll('.refine-item')].filter(item => /구멍/.test(item.textContent)).length,
  occlusionDiagnosed: [...document.querySelectorAll('.refine-item')].filter(item => /가려짐/.test(item.textContent)).length,
  legendColors: document.querySelectorAll('.refine-legend span').length,
  summaryFilled: document.querySelector('#refineLabSummary').children.length,
  flowFilled: document.querySelector('#refineFlowSegment').textContent,
  generatedItems: [...document.querySelectorAll('.refine-item')].filter(item => item.querySelectorAll('.refine-stage')[4]?.querySelector('img')).length,
  failedStages: document.querySelectorAll('.refine-stage.fail').length,
  horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
})`);

if (process.env.SCREENSHOT) {
  await evaluate("document.querySelectorAll('dialog[open]').forEach(dialog => dialog.close()), true");
  await new Promise(resolve => setTimeout(resolve, 700));
  const capture = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  await writeFile(path.resolve(process.env.SCREENSHOT), Buffer.from(capture.result.data, "base64"));
}
await send("Browser.close");
chromeProcess.kill();

const problems = [];
if (setup.health.includes("down")) problems.push(`백엔드 연결 실패 (${setup.healthText})`);
if (setup.models < 4) problems.push(`모델 목록 ${setup.models}개`);
if (!setup.selected) problems.push("기본 모델이 선택되지 않음");
if (setup.chips !== 7) problems.push(`카테고리 칩 ${setup.chips}개`);
if (!setup.presets) problems.push("백엔드 프리셋 버튼 없음");
if (endpoint && setup.endpoint !== endpoint) problems.push(`주소가 ${setup.endpoint}로 잡힘`);
if (setup.flowSteps !== 5) problems.push(`파이프라인 단계 ${setup.flowSteps}개`);
if (setup.repairControls !== 6) problems.push(`보수 옵션 ${setup.repairControls}개`);
if (!value.items) problems.push("처리된 옷이 없음");
if (value.stagesPerItem.some(count => count !== 5)) problems.push(`옷별 단계 ${value.stagesPerItem.join("/")}`);
if (value.imagesLoaded !== value.imagesTotal || !value.imagesTotal) problems.push(`이미지 로딩 ${value.imagesLoaded}/${value.imagesTotal}`);
if (!value.repairSteps) problems.push("보수 단계 내역이 렌더되지 않음");
if (value.diagnosed !== value.items) problems.push(`결함 진단 ${value.diagnosed}/${value.items}`);
if (value.occlusionDiagnosed !== value.items) problems.push(`가림 진단 ${value.occlusionDiagnosed}/${value.items}`);
if (value.legendColors !== value.items * 3) problems.push(`결함 범례 ${value.legendColors}개 (옷당 3색이어야 함)`);
if (!value.summaryFilled) problems.push("요약 카드가 비어 있음");
if (!/\d/.test(value.flowFilled)) problems.push("파이프라인 다이어그램에 실행 수치가 반영되지 않음");
if (value.failedStages) problems.push(`실패한 단계 ${value.failedStages}개`);
if (generate && value.generatedItems !== value.items) problems.push(`생성 결과 ${value.generatedItems}/${value.items}벌`);
if (value.horizontalOverflow) problems.push("가로 스크롤 발생");
if (consoleErrors.length) problems.push(`콘솔 예외: ${consoleErrors.join(" | ")}`);

if (problems.length) throw new Error(`Refine Lab 테스트 실패:\n  - ${problems.join("\n  - ")}\n  ${JSON.stringify(value)}`);
console.log(`Refine Lab 테스트 통과: 옷 ${value.items}벌 · 단계 ${value.stages}개 · 이미지 ${value.imagesLoaded}장${generate ? " · 생성 포함" : " · 생성 생략"} · ${value.status}`);
