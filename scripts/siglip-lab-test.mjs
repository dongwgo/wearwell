/**
 * SigLIP Lab 탭 E2E 스모크 테스트.
 *
 *   node scripts/siglip-lab-test.mjs
 *
 * 백엔드는 **가짜로 세운다**. SigLIP 가중치가 없는 기계에서도 돌아야 하고, 이 탭에서
 * 검사할 것은 벡터의 품질이 아니라 탭의 계약이기 때문이다:
 *   1) 옷장을 한 벌씩이 아니라 묶음(/api/embeddings)으로 인코딩하는가
 *      — 한 벌씩 보내면 200벌짜리 옷장은 분당 요청 제한에 먼저 걸린다
 *   2) 코사인 내림차순으로 세우는가
 *   3) 올린 사진과 같은 사진이 옷장에 있으면 그게 1위인가 (cos ≈ 1)
 *   4) 한 번 만든 벡터를 다시 만들지 않는가 (IndexedDB 캐시)
 *
 * 가짜 백엔드는 같은 이미지 문자열에 늘 같은 단위 벡터를 돌려준다. 옷장 사진과 올린
 * 사진이 같은 축소·인코딩 경로를 지나므로, 같은 원본이면 문자열까지 같아진다.
 *
 * 정적 서버는 이 스크립트가 직접 띄운다(IndexedDB는 file://에서 막힌다).
 */
import { spawn } from "node:child_process";
import { mkdtemp, readFile } from "node:fs/promises";
import net from "node:net";
import os from "node:os";
import path from "node:path";

/** python -m http.server는 HTTP/1.0으로 답한다 — fetch로 찔러 보면 undici가 터진다. */
const waitForPort = async (target, attempts = 60) => {
  for (let attempt = 0; attempt < attempts; attempt++) {
    const open = await new Promise(resolve => {
      const socket = net.connect(target, "127.0.0.1");
      socket.once("connect", () => { socket.destroy(); resolve(true); });
      socket.once("error", () => resolve(false));
    });
    if (open) return true;
    await new Promise(resolve => setTimeout(resolve, 200));
  }
  return false;
};

const chrome = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const port = Number(process.env.SIGLIP_LAB_PORT || 8010);
const pageUrl = `http://127.0.0.1:${port}/?dev=1`;
const sampleAsset = "assets/lookbook/look-001.jpg";
const compareCount = 20;

const python = process.env.PYTHON || "python";
const server = spawn(python, ["-m", "http.server", String(port), "--bind", "127.0.0.1"], { windowsHide: true, stdio: "ignore" });
if (!await waitForPort(port)) throw new Error(`정적 서버(${port})가 뜨지 않았습니다`);

const profile = await mkdtemp(path.join(os.tmpdir(), "wearwell-sigliplab-"));
const chromeProcess = spawn(chrome, [
  "--headless=new", "--disable-gpu", "--no-sandbox", "--window-size=1500,1400",
  "--remote-debugging-port=9226", `--user-data-dir=${profile}`, "about:blank"
], { windowsHide: true, stdio: "ignore" });

let tabs;
for (let attempt = 0; attempt < 50; attempt++) {
  try {
    tabs = await fetch("http://127.0.0.1:9226/json").then(response => response.json());
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
async function evaluate(expression) {
  const response = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
  const details = response.result?.exceptionDetails;
  if (details) throw new Error(details.exception?.description || details.text);
  return response.result?.result?.value;
}

await send("Runtime.enable");
await send("Page.enable");

// 페이지 스크립트보다 먼저 실행되어야 fetch를 가로챌 수 있다.
const stub = `
  (() => {
    const realFetch = window.fetch.bind(window);
    const calls = [];
    window.__siglipCalls = calls;
    const unitVector = text => {
      const dims = 8;
      const vector = [];
      for (let slot = 0; slot < dims; slot++) {
        let hash = 0x811c9dc5 ^ (slot * 0x9e3779b9);
        for (let index = 0; index < text.length; index++) {
          hash ^= text.charCodeAt(index);
          hash = Math.imul(hash, 0x01000193);
        }
        vector.push(((hash >>> 0) / 0xffffffff) - 0.5);
      }
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
      return vector.map(value => value / norm);
    };
    window.fetch = async (input, init) => {
      const url = String(typeof input === "string" ? input : input?.url || "");
      if (url.includes("/api/health")) {
        calls.push({ path: "health", count: 0 });
        return new Response(JSON.stringify({
          ok: true, cuda: true, gpu: "fake", embeddingModel: "google/siglip-base-patch16-224",
          embeddingLoaded: true, embeddingDevice: "cuda", maxEmbeddingBatch: 32, rateLimitPerMinute: 60,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (url.includes("/api/embeddings")) {
        const images = JSON.parse(init.body).images;
        calls.push({ path: "embeddings", count: images.length });
        return new Response(JSON.stringify({ model: "google/siglip-base-patch16-224", vectors: images.map(unitVector) }),
          { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (url.includes("/api/embedding")) {
        const image = JSON.parse(init.body).image;
        calls.push({ path: "embedding", count: 1 });
        return new Response(JSON.stringify({ model: "google/siglip-base-patch16-224", vector: unitVector(image) }),
          { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return realFetch(input, init);
    };
  })();
`;
await send("Page.addScriptToEvaluateOnNewDocument", { source: stub });
await send("Page.navigate", { url: pageUrl });
await new Promise(resolve => setTimeout(resolve, 2500));

const sampleBytes = [...new Uint8Array(await readFile(path.resolve(sampleAsset)))];

/** 탭을 열고, 옷장에 있는 사진과 같은 파일을 올리고, 한 번 돌린다. */
const runOnce = `
  (async () => {
    document.querySelector('[data-view="siglipLab"]').click();
    document.querySelector('#siglipOnlyMine').checked = false;
    const limit = document.querySelector('#siglipLimit');
    limit.value = '${compareCount}';
    limit.dispatchEvent(new Event('input', { bubbles: true }));
    const bytes = new Uint8Array(${JSON.stringify(sampleBytes)});
    const file = new File([bytes], 'query.jpg', { type: 'image/jpeg' });
    const transfer = new DataTransfer();
    transfer.items.add(file);
    const input = document.querySelector('#siglipLabFileInput');
    input.files = transfer.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    for (let attempt = 0; attempt < 60 && document.querySelector('#siglipLabRun').disabled; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    window.__siglipCalls.length = 0;
    document.querySelector('#siglipLabRun').click();
    for (let attempt = 0; attempt < 300; attempt++) {
      if (document.querySelector('#siglipLabStatus').className.includes('ok')) break;
      if (document.querySelector('#siglipLabStatus').className.includes('error')) break;
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    const rows = [...document.querySelectorAll('#siglipLabResult .siglip-ranking li')];
    return {
      status: document.querySelector('#siglipLabStatus').textContent,
      tone: document.querySelector('#siglipLabStatus').className,
      health: document.querySelector('#siglipLabHealth').textContent,
      scope: document.querySelector('#siglipLabScope').textContent,
      rendered: rows.length,
      topRequested: Number(document.querySelector('#siglipTop').value),
      scores: rows.map(row => Number(row.querySelector('.siglip-cos').textContent)),
      topName: rows[0]?.querySelector('strong')?.textContent || '',
      topFlag: rows[0]?.className || '',
      calls: window.__siglipCalls.map(call => call.path + ':' + call.count),
    };
  })();
`;

const first = await evaluate(runOnce);

// 두 번째 실행은 캐시만으로 끝나야 한다. 새로고침 뒤에도 마찬가지다(IndexedDB).
await send("Page.navigate", { url: pageUrl });
await new Promise(resolve => setTimeout(resolve, 2500));
const second = await evaluate(runOnce);

const batchCalls = first.calls.filter(call => call.startsWith("embeddings:"));
const perItemCalls = first.calls.filter(call => call.startsWith("embedding:"));
const embeddedFirst = batchCalls.reduce((sum, call) => sum + Number(call.split(":")[1]), 0);
const embeddedSecond = second.calls.reduce((sum, call) => sum + Number(call.split(":")[1]), 0);
const sortedDescending = first.scores.every((score, index) => index === 0 || first.scores[index - 1] >= score);

const checks = [
  ["탭이 뜨고 실행이 성공한다", first.tone.includes("ok"), first.status],
  ["가짜 백엔드에 붙어 모델을 읽는다", first.health.includes("siglip-base-patch16-224"), first.health],
  ["코사인 내림차순으로 정렬된다", sortedDescending, first.scores.slice(0, 5).join(", ")],
  ["같은 사진이 1위이고 코사인이 1이다", first.scores[0] >= 0.9999, `1위 ${first.scores[0]} · ${first.topName}`],
  ["1위를 '같은 옷 취급'으로 표시한다", first.topFlag.includes("same"), first.topFlag],
  ["상위 N개만 그린다", first.rendered === Math.min(first.topRequested, compareCount), `${first.rendered}개 / 상위 ${first.topRequested}`],
  ["한 벌씩 보내지 않는다", perItemCalls.length === 0, first.calls.join(" ")],
  ["묶음 한 번에 32벌을 넘기지 않는다", batchCalls.every(call => Number(call.split(":")[1]) <= 32), batchCalls.join(" ")],
  ["요청 수가 옷 수보다 훨씬 적다", batchCalls.length <= Math.ceil(compareCount / 32) + 1, `${batchCalls.length}회 / ${compareCount}벌`],
  ["올린 사진 + 옷장을 모두 인코딩한다", embeddedFirst >= compareCount, `${embeddedFirst}장`],
  ["새로고침 뒤에도 캐시를 재사용한다", embeddedSecond <= 1, `두 번째 실행에서 ${embeddedSecond}장만 인코딩`],
  ["콘솔 예외가 없다", consoleErrors.length === 0, consoleErrors.join(" | ")],
];

console.log("");
for (const [label, passed, detail] of checks) {
  console.log(`${passed ? "PASS" : "FAIL"}  ${label}${detail ? `  — ${detail}` : ""}`);
}
console.log("");
console.log(`첫 실행 상태: ${first.status}`);
console.log(`옷장 범위:   ${first.scope}`);
console.log(`요청 내역:   ${first.calls.join(" ")}`);
console.log(`두 번째 실행: ${second.status}`);

socket.close();
chromeProcess.kill();
server.kill();
process.exit(checks.every(([, passed]) => passed) ? 0 : 1);
