import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

const chrome = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const profile = await mkdtemp(path.join(os.tmpdir(), "oneulout-smoke-"));
const pageUrl = pathToFileURL(path.resolve("index.html")).href;
const chromeProcess = spawn(chrome, ["--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1440,1000", "--remote-debugging-port=9223", `--user-data-dir=${profile}`, pageUrl], { windowsHide: true, stdio: "ignore" });

let tabs;
for (let attempt = 0; attempt < 20; attempt++) {
  try {
    tabs = await fetch("http://127.0.0.1:9223/json").then(response => response.json());
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
  (async () => {
  document.querySelector('[data-gender="women"]').click();
  document.querySelector('#nextGender').click();
  await generateAvatar();
  showPreferenceStep(3);
  [...document.querySelectorAll('[data-style]')].slice(0, 3).forEach(button => button.click());
  document.querySelector('#nextPreferences').click();
  document.querySelector('#finishPreferences').click();
  document.querySelector('#wardrobeGrid [data-item-id]').click();
  const matchDialogOpened = document.querySelector('#itemMatchDialog').open;
  const matchOptions = document.querySelectorAll('.match-option').length;
  const matchedNames = [...document.querySelectorAll('.match-piece span')].map(node => node.textContent);
  document.querySelector('#itemMatchDialog').close();
  document.querySelector('[data-view="discover"]').click();
  return ({
    profile: JSON.parse(localStorage.getItem('오늘옷-profile')),
    look: document.querySelector('#lookTitle').textContent,
    wardrobeCount: document.querySelector('#wardrobeCount').textContent,
    dialogOpen: document.querySelector('#preferenceDialog').open,
    matchDialogOpened,
    matchOptions,
    matchedNames,
    trendCards: document.querySelectorAll('.discover-card').length,
    closetMatches: document.querySelectorAll('.discover-closet-match').length,
    avatarReady: Boolean(avatarImage)
  });
  })()
`;
const result = await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true });
const value = result.result?.result?.value;
if (process.env.SCREENSHOT) {
  await new Promise(resolve => setTimeout(resolve, 600));
  const capture = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  await writeFile(path.resolve(process.env.SCREENSHOT), Buffer.from(capture.result.data, "base64"));
}
await send("Browser.close");
chromeProcess.kill();

if (!value?.profile || value.profile.gender !== "women" || value.dialogOpen || value.wardrobeCount !== "100" || !value.matchDialogOpened || value.matchOptions !== 2 || value.trendCards !== 24 || value.closetMatches !== 24 || !value.avatarReady) {
  throw new Error(`Smoke test failed: ${JSON.stringify(value)}`);
}
console.log(`Smoke test passed: ${value.look} / ${value.matchedNames.slice(0, 4).join(" + ")} / 트렌드 추천 ${value.trendCards}개`);
