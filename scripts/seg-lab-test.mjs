/**
 * Seg Lab 탭 E2E 스모크 테스트.
 *
 * 실행 전 두 서버가 떠 있어야 한다.
 *   1) 정적 프론트: python -m http.server 8000 --bind 127.0.0.1
 *   2) 백엔드:      cd backend && WEARWELL_API_TOKEN=... uvicorn app:app --port 8787
 *
 * file:// 로 열면 백엔드 CORS(http://127.0.0.1 만 허용)에 막히므로 반드시 8000으로 연다.
 *   node scripts/seg-lab-test.mjs [사진경로]
 */
import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const pageUrl = process.env.SEG_LAB_URL || "http://127.0.0.1:8000/?dev=1";
const photo = path.resolve(process.argv[2] || "scripts/.musinsa-snaps/snap-004.jpg");
const profile = await mkdtemp(path.join(os.tmpdir(), "wearwell-seglab-"));
const chromeProcess = spawn(chrome, [
  "--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1500,1200",
  "--remote-debugging-port=9224", `--user-data-dir=${profile}`, pageUrl
], { windowsHide: true, stdio: "ignore" });

let tabs;
for (let attempt = 0; attempt < 40; attempt++) {
  try {
    tabs = await fetch("http://127.0.0.1:9224/json").then(response => response.json());
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
await evaluate("localStorage.removeItem('wearwell-seglab-models'), localStorage.removeItem('wearwell-seglab-endpoint'), true");

// 개발 탭이 노출되고 백엔드에서 모델 목록을 받아왔는지.
await waitFor("!document.querySelector('#segLabNavLink').hidden", 8000, "Seg Lab 탭 노출");
await waitFor("document.querySelectorAll('#segLabModelPicker .seg-model-option').length > 0", 15000, "모델 목록 로딩");
const setup = await evaluate(`({
  health: document.querySelector('#segLabHealth').className,
  models: [...document.querySelectorAll('#segLabModelPicker .seg-model-option')].length,
  checked: [...document.querySelectorAll('#segLabModelPicker input:checked')].map(node => node.value),
  productionTagged: Boolean(document.querySelector('.seg-model-option .seg-tag'))
})`);

await evaluate("document.querySelector('[data-view=\"seglab\"]').click(), true");
await waitFor("document.querySelector('#seglabView').classList.contains('active')", 3000, "탭 전환");

// 파일 입력에 실제 사진을 넣는다. setFileInputFiles가 change 이벤트까지 발생시킨다.
const { result: doc } = await send("DOM.getDocument");
const { result: node } = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#segLabFileInput" });
await send("DOM.setFileInputFiles", { files: [photo], nodeId: node.nodeId });
await waitFor("!document.querySelector('#segLabRun').disabled", 10000, "사진 준비");

await evaluate("document.querySelector('#segLabRun').click(), true");
await waitFor("document.querySelector('#segLabStatus').classList.contains('ok') || document.querySelector('#segLabStatus').classList.contains('error') || document.querySelector('#segLabStatus').classList.contains('warn')", 600000, "비교 실행");

const value = await evaluate(`({
  status: document.querySelector('#segLabStatus').textContent,
  statusTone: document.querySelector('#segLabStatus').className,
  columns: document.querySelectorAll('.seg-column').length,
  failedColumns: document.querySelectorAll('.seg-column.failed').length,
  overlays: document.querySelectorAll('.seg-overlay').length,
  overlaysLoaded: [...document.querySelectorAll('.seg-overlay')].filter(img => img.naturalWidth > 0).length,
  items: document.querySelectorAll('.seg-item').length,
  rejected: document.querySelectorAll('.seg-item.rejected').length,
  cropsLoaded: [...document.querySelectorAll('.seg-item img')].filter(img => img.naturalWidth > 0).length,
  cropsTotal: document.querySelectorAll('.seg-item img').length,
  failedMetrics: document.querySelectorAll('.seg-metric.fail').length,
  rawLabelBlocks: document.querySelectorAll('.seg-raw-labels').length,
  partLabels: document.querySelectorAll('.seg-raw.part').length,
  pairs: document.querySelectorAll('.seg-pair').length,
  iouCells: document.querySelectorAll('.seg-iou').length,
  outerFound: [...document.querySelectorAll('.seg-chip')].some(node => node.textContent.trim() === '아우터'),
  horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1
})`);

if (process.env.SCREENSHOT) {
  await new Promise(resolve => setTimeout(resolve, 700));
  const capture = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true });
  await writeFile(path.resolve(process.env.SCREENSHOT), Buffer.from(capture.result.data, "base64"));
}
await send("Browser.close");
chromeProcess.kill();

const problems = [];
if (!setup.health.includes("up")) problems.push(`백엔드 연결 실패 (${setup.health})`);
if (setup.models < 4) problems.push(`모델 목록 ${setup.models}개`);
if (setup.checked.length !== 3) problems.push(`기본 선택 ${setup.checked.length}개`);
if (!setup.productionTagged) problems.push("프로덕션 배지 없음");
if (value.columns !== 3) problems.push(`결과 컬럼 ${value.columns}개`);
if (value.failedColumns) problems.push(`실패한 모델 ${value.failedColumns}개`);
if (value.overlaysLoaded !== 3) problems.push(`오버레이 로딩 ${value.overlaysLoaded}/3`);
if (value.cropsLoaded !== value.cropsTotal || !value.cropsTotal) problems.push(`크롭 로딩 ${value.cropsLoaded}/${value.cropsTotal}`);
if (!value.rejected) problems.push("걸러진 후보가 하나도 렌더되지 않음 (필터 진단이 안 보임)");
if (!value.failedMetrics) problems.push("기준 미달 지표 강조 없음");
if (value.rawLabelBlocks !== 3) problems.push(`원본 라벨 블록 ${value.rawLabelBlocks}개`);
if (!value.partLabels) problems.push("Fashionpedia 부속 라벨이 표시되지 않음");
if (value.pairs !== 3) problems.push(`IoU 쌍 ${value.pairs}개`);
if (!value.iouCells) problems.push("IoU 셀 없음");
if (!value.outerFound) problems.push("아우터 카테고리 미검출 (b3_fashion 매핑 확인)");
if (value.horizontalOverflow) problems.push("가로 스크롤 발생");
if (consoleErrors.length) problems.push(`콘솔 예외: ${consoleErrors.join(" | ")}`);

if (problems.length) throw new Error(`Seg Lab 테스트 실패:\n  - ${problems.join("\n  - ")}\n  ${JSON.stringify(value)}`);
console.log(`Seg Lab 테스트 통과: 모델 ${value.columns}개 · 후보 ${value.items}개(걸러짐 ${value.rejected}) · IoU 쌍 ${value.pairs} · ${value.status}`);
