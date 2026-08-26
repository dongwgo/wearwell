/**
 * 아바타 뷰어 회귀 테스트 (headless Chrome).
 *
 * 두 가지를 지킨다.
 *
 * 1. 전신 이미지가 잘리지 않는다 — object-fit: cover 였을 때 768x1152 이미지를
 *    390x510 상자에 넣으면 폭에 맞춰 확대되면서 머리와 발이 잘려나갔다.
 *    이건 CSS 계산값으로 확인해야 한다. 눈으로 보고 넘기면 다시 돌아온다.
 * 2. 시점이 여러 개일 때 가로 드래그로 정면 -> 측면 -> 후면이 돌아간다.
 *
 * 실행: CHROME_PATH=/path/to/chrome node scripts/avatar-view-test.mjs
 */
import { spawn } from "node:child_process";
import { mkdtemp } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const chrome = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const profile = await mkdtemp(path.join(os.tmpdir(), "wearwell-views-"));
const pageUrl = pathToFileURL(path.resolve("index.html")).href;
const port = 9224;
const chromeProcess = spawn(
  chrome,
  ["--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1440,1000",
   `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, pageUrl],
  { windowsHide: true, stdio: "ignore" },
);

let tabs;
for (let attempt = 0; attempt < 20; attempt++) {
  try {
    tabs = await fetch(`http://127.0.0.1:${port}/json`).then(response => response.json());
    if (tabs.some(tab => tab.type === "page")) break;
  } catch {}
  await new Promise(resolve => setTimeout(resolve, 150));
}
const tab = tabs?.find(item => item.type === "page");
if (!tab) throw new Error("Could not connect to the demo page");

const socket = new WebSocket(tab.webSocketDebuggerUrl);
await new Promise(resolve => socket.addEventListener("open", resolve, { once: true }));
let messageId = 0;
const pending = new Map();
socket.addEventListener("message", event => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    pending.get(message.id)(message);
    pending.delete(message.id);
  }
});
const send = (method, params = {}) => new Promise(resolve => {
  const id = ++messageId;
  pending.set(id, resolve);
  socket.send(JSON.stringify({ id, method, params }));
});

await new Promise(resolve => setTimeout(resolve, 1200));

const expression = `
(() => {
  // 실제 생성 없이 시점 3개를 가진 아바타를 흉내 낸다. 색만 다른 1x1 PNG면
  // 어느 시점이 화면에 걸려 있는지 구분하기에 충분하다.
  const pixel = tag =>
    "data:image/svg+xml;base64," + btoa('<svg xmlns="http://www.w3.org/2000/svg" width="768" height="1152"><rect width="768" height="1152" fill="#ccc"/><text>' + tag + '</text></svg>');
  const views = { front: pixel("front"), side: pixel("side"), back: pixel("back") };
  showAvatar(views.front, "flux2-klein-4b-cuda-bfloat16+silhouette-fallback", views);

  const preview = document.querySelector("#avatarPreview");
  const currentView = () => {
    const alt = preview.querySelector("img")?.getAttribute("alt") || "";
    return alt.includes("정면") ? "front" : alt.includes("측면") ? "side" : alt.includes("후면") ? "back" : null;
  };

  const drag = pixels => {
    preview.dispatchEvent(new PointerEvent("pointerdown", { clientX: 0, pointerId: 1, bubbles: true }));
    preview.dispatchEvent(new PointerEvent("pointermove", { clientX: pixels, pointerId: 1, bubbles: true }));
    preview.dispatchEvent(new PointerEvent("pointerup", { clientX: pixels, pointerId: 1, bubbles: true }));
    return currentView();
  };

  const results = {
    rotatable: preview.classList.contains("rotatable"),
    dots: preview.querySelectorAll(".avatar-view-dots i").length,
    startsOnFront: currentView() === "front",
    afterDragRight: drag(60),
    afterDragBack: drag(-60),
    // 왼쪽으로 더 밀면 한 바퀴 돌아 후면으로 간다.
    afterWrapAround: drag(-60),
  };

  // 이미지가 잘리지 않는지는 계산된 CSS로 확인한다.
  const avatarImg = preview.querySelector("img");
  results.avatarObjectFit = getComputedStyle(avatarImg).objectFit;

  const stage = document.querySelector("#tryonStage");
  stage.classList.remove("comparison");
  stage.innerHTML = '<img src="' + pixel("tryon") + '" />';
  results.tryonObjectFit = getComputedStyle(stage.querySelector("img")).objectFit;

  // contain이면 세로가 상자에 맞고 전체가 보인다. 실제 렌더 높이로 확인한다.
  const box = stage.getBoundingClientRect();
  const rendered = stage.querySelector("img").getBoundingClientRect();
  results.tryonFitsInsideBox = rendered.height <= box.height + 1;

  // 시점이 하나뿐이면 회전 UI가 붙지 않아야 한다.
  showAvatar(views.front, "flux2-klein-4b-cuda-bfloat16+silhouette-fallback");
  results.singleViewNotRotatable = !preview.classList.contains("rotatable");
  results.singleViewNoDots = preview.querySelectorAll(".avatar-view-dots i").length === 0;

  return results;
})()
`;

const result = await send("Runtime.evaluate", { expression, returnByValue: true });
const value = result.result?.result?.value;
// Browser.close는 응답을 돌려주기 전에 소켓이 끊길 수 있다. 응답을 기다리다
// 프로세스가 매달리지 않도록 짧은 타임아웃을 건다.
await Promise.race([send("Browser.close"), new Promise(resolve => setTimeout(resolve, 1000))]);
chromeProcess.kill();
socket.close();

const expected = {
  rotatable: true,
  dots: 3,
  startsOnFront: true,
  afterDragRight: "side",
  afterDragBack: "front",
  afterWrapAround: "back",
  avatarObjectFit: "contain",
  tryonObjectFit: "contain",
  tryonFitsInsideBox: true,
  singleViewNotRotatable: true,
  singleViewNoDots: true,
};

const failures = Object.entries(expected)
  .filter(([key, want]) => value?.[key] !== want)
  .map(([key, want]) => `${key}: expected ${JSON.stringify(want)}, got ${JSON.stringify(value?.[key])}`);

if (failures.length) {
  console.error("Avatar view test failed:\n  " + failures.join("\n  "));
  process.exit(1);
}
console.log("Avatar view test passed: 시점 3개 회전 + 전신 이미지 비잘림(object-fit: contain)");
