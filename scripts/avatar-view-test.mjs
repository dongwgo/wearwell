/**
 * 아바타 뷰어 회귀 테스트 (headless Chrome).
 *
 * 두 가지를 지킨다.
 *
 * 1. 전신 이미지가 잘리지 않는다 — object-fit: cover 였을 때 768x1152 이미지를
 *    390x510 상자에 넣으면 폭에 맞춰 확대되면서 머리와 발이 잘려나갔다.
 *    이건 CSS 계산값으로 확인해야 한다. 눈으로 보고 넘기면 다시 돌아온다.
 * 2. 시점이 여러 개일 때 가로 드래그로 정면 -> 측면 -> 후면이 돌아간다.
 *    아바타 미리보기와 착장 결과 양쪽 다.
 *
 * 실행: CHROME_PATH=/path/to/chrome node scripts/avatar-view-test.mjs
 */
import { spawn } from "node:child_process";
import { mkdtemp, readFile } from "node:fs/promises";
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

// 착용 순서 규칙은 백엔드(tryon_prompt.py)와 프론트(app.js) 양쪽에 있다.
// 프론트가 6벌로 자르기 전에 무엇을 남길지 스스로 정해야 하기 때문이다.
// 같은 픽스처로 두 구현을 묶어 둔다 — 파이썬 쪽은 test_tryon_prompt.py가 본다.
const orderingCases = JSON.parse(await readFile("eval/ordering-cases.json", "utf8"));

const expression = `
(() => {
  // 실제 생성 없이 시점 3개를 가진 아바타를 흉내 낸다. 색만 다른 1x1 PNG면
  // 어느 시점이 화면에 걸려 있는지 구분하기에 충분하다.
  const pixel = tag =>
    "data:image/svg+xml;base64," + btoa('<svg xmlns="http://www.w3.org/2000/svg" width="768" height="1152"><rect width="768" height="1152" fill="#ccc"/><text>' + tag + '</text></svg>');
  const views = { front: pixel("front"), side: pixel("side"), back: pixel("back") };
  showAvatar(views.front, "flux2-klein-4b-cuda-bfloat16+silhouette-fallback", views);

  const preview = document.querySelector("#avatarPreview");
  const viewOf = element => {
    const alt = element.querySelector("img")?.getAttribute("alt") || "";
    return alt.includes("정면") ? "front" : alt.includes("측면") ? "side" : alt.includes("후면") ? "back" : null;
  };
  const currentView = () => viewOf(preview);

  const dragOn = (element, pixels) => {
    element.dispatchEvent(new PointerEvent("pointerdown", { clientX: 0, pointerId: 1, bubbles: true }));
    element.dispatchEvent(new PointerEvent("pointermove", { clientX: pixels, pointerId: 1, bubbles: true }));
    element.dispatchEvent(new PointerEvent("pointerup", { clientX: pixels, pointerId: 1, bubbles: true }));
    return viewOf(element);
  };
  const drag = pixels => dragOn(preview, pixels);

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

  // 착장 결과도 같은 뷰어를 쓴다. 아바타에 시점이 있어야 회전 버튼이 뜬다.
  // 대화상자를 열어야 스테이지에 실제 레이아웃이 생긴다 — 닫힌 채로 재면
  // 0x0이 나와서 크기 검증이 아무것도 확인하지 못한다.
  document.querySelector("#avatarTryonDialog").showModal();
  const stage = document.querySelector("#tryonStage");
  showAvatar(views.front, "flux2-klein-4b-cuda-bfloat16+silhouette-fallback", views);
  renderTryonStage(null, { front: pixel("t-front"), side: pixel("t-side"), back: pixel("t-back") });
  results.tryonRotatable = stage.classList.contains("rotatable");
  results.tryonDots = stage.querySelectorAll(".avatar-view-dots i").length;
  results.tryonStartsOnFront = viewOf(stage) === "front";
  results.tryonAfterDrag = dragOn(stage, 60);
  results.tryonObjectFit = getComputedStyle(stage.querySelector("img")).objectFit;

  // 상자 비율을 이미지 비율에 맞췄으므로 잘림도 레터박스도 없어야 한다.
  const box = stage.getBoundingClientRect();
  const rendered = stage.querySelector("img").getBoundingClientRect();
  results.tryonFitsInsideBox = rendered.height <= box.height + 1;
  results.stageHasLayout = box.width > 100 && box.height > 100;
  // 상자 비율을 이미지 비율(2:3)에 맞췄으므로 잘림도 레터박스도 없다.
  results.stageMatchesImageAspect = Math.abs(box.width / box.height - 2 / 3) < 0.02;
  results.imageFillsStage = Math.abs(rendered.width - box.width) < 2;

  // 비교 모드에서는 회전을 붙이지 않는다(좌우 두 칸을 쓰기 때문).
  renderTryonStage({ image: pixel("look"), title: "룩북" }, { front: pixel("t-front") });
  results.comparisonNotRotatable = !stage.classList.contains("rotatable");
  results.comparisonPanes = stage.querySelectorAll("figure").length;

  // 시점이 하나뿐이면 회전 UI가 붙지 않아야 한다.
  renderTryonStage(null, { front: pixel("t-front") });
  results.singleTryonNotRotatable = !stage.classList.contains("rotatable");

  showAvatar(views.front, "flux2-klein-4b-cuda-bfloat16+silhouette-fallback");
  results.singleViewNotRotatable = !preview.classList.contains("rotatable");
  results.singleViewNoDots = preview.querySelectorAll(".avatar-view-dots i").length === 0;

  // 연달아 두 번 입혀보면 나중 결과가 화면에 남아야 한다. 예전에는 먼저 보낸
  // 요청이 늦게 도착하면 나중 결과를 덮어써서 "고른 옷이 반영되지 않는" 것처럼
  // 보였다.
  results.race = null;

  // 착용 순서가 백엔드와 같은지 — 같은 픽스처로 두 구현을 묶는다.
  results.ordering = ${JSON.stringify(orderingCases)}.map(testCase => ({
    title: testCase.title,
    expected: testCase.expected,
    actual: orderGarmentsForTryon(
      testCase.items.map((item, index) => ({ ...item, id: "fixture-" + index, image: "" })),
    ).map(item => item.name),
  }));

  return results;
})()
`;

const raceExpression = `
(async () => {
  const pixel = tag =>
    "data:image/svg+xml;base64," + btoa('<svg xmlns="http://www.w3.org/2000/svg" width="768" height="1152"><rect fill="#ccc" width="768" height="1152"/><desc>' + tag + '</desc></svg>');

  avatarImage = pixel("avatar");
  avatarMeasurements = { gender: "men", height: 175, weight: 70 };

  // 먼저 보낸 요청이 더 늦게 끝나도록 만든다 — 정확히 실제로 문제가 됐던 순서.
  const delays = { A: 400, B: 60 };
  const original = fetchGpuJson;
  fetchGpuJson = async (path, payload) => {
    const tag = payload.garments[0].name;
    await new Promise(resolve => setTimeout(resolve, delays[tag]));
    return { image: pixel("result-" + tag), views: { front: pixel("result-" + tag) } };
  };

  const item = tag => [{ id: "item-" + tag, name: tag, category: "상의", image: pixel("garment-" + tag) }];
  const first = runTryOnItems(item("A"));
  await new Promise(resolve => setTimeout(resolve, 20));
  const second = runTryOnItems(item("B"));
  await Promise.all([first, second]);
  await new Promise(resolve => setTimeout(resolve, 500));

  fetchGpuJson = original;
  const shown = document.querySelector("#tryonStage img")?.getAttribute("src") || "";
  const race = atob(shown.split(",")[1] || "").includes("result-B") ? "B" : "A";

  // 결과 비율은 원본 사진을 따라간다. 3:4 휴대폰 사진을 2:3 상자에 넣으면
  // 사람과 배경이 세로로 늘어나 보인다 — 상자가 이미지 비율을 따라야 한다.
  const sized = (w, h) =>
    "data:image/svg+xml;base64," + btoa('<svg xmlns="http://www.w3.org/2000/svg" width="' + w + '" height="' + h + '"><rect fill="#ccc" width="' + w + '" height="' + h + '"/></svg>');
  const stage = document.querySelector("#tryonStage");
  renderTryonStage(null, { front: sized(1080, 1440) });   // 3:4
  await new Promise(resolve => setTimeout(resolve, 250));
  const box = stage.getBoundingClientRect();
  const stageAspect = box.width / box.height;
  const image = stage.querySelector("img").getBoundingClientRect();

  // "아바타로 보기"는 전신사진으로 시작한 경우에만 뜬다. 치수로 만든 아바타는
  // 이미 스튜디오 컷이라 바꿀 것이 없다.
  const modeButton = document.querySelector("#tryonAvatarButton");
  tryonContext = { payload: {}, cacheKey: "x", requestId: tryonRequestId };
  avatarMeasurements = { gender: "men", height: 175, weight: 70 };
  fullBodyPhoto = null;
  renderTryonStage(null, { front: sized(768, 1152) });
  const hiddenForMeasurements = modeButton.hidden;

  avatarMeasurements = { gender: "men", height: 175, weight: 70, photoBased: true };
  fullBodyPhoto = sized(1080, 1440);
  renderTryonStage(null, { front: sized(1080, 1440) });
  const shownForPhoto = !modeButton.hidden;
  const modeLabel = modeButton.textContent;

  return {
    race,
    hiddenForMeasurements,
    shownForPhoto,
    modeLabel,
    // 상자가 3:4(0.75)를 따라가야 한다.
    stageFollowsImage: Math.abs(stageAspect - 0.75) < 0.03,
    // 그래야 이미지가 상자를 꽉 채우고 레터박스가 생기지 않는다.
    noLetterbox: Math.abs(image.width - box.width) < 3 && Math.abs(image.height - box.height) < 3,
  };
})()
`;

const result = await send("Runtime.evaluate", { expression, returnByValue: true });
const value = result.result?.result?.value;

const raceResult = await send("Runtime.evaluate", {
  expression: raceExpression, returnByValue: true, awaitPromise: true,
});
const raceValue = raceResult.result?.result?.value ?? {};
if (value) {
  value.race = raceValue.race ?? null;
  value.stageFollowsImage = raceValue.stageFollowsImage ?? null;
  value.noLetterbox = raceValue.noLetterbox ?? null;
  value.hiddenForMeasurements = raceValue.hiddenForMeasurements ?? null;
  value.shownForPhoto = raceValue.shownForPhoto ?? null;
  value.modeLabel = raceValue.modeLabel ?? null;
}
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
  tryonRotatable: true,
  tryonDots: 3,
  tryonStartsOnFront: true,
  tryonAfterDrag: "side",
  tryonObjectFit: "contain",
  tryonFitsInsideBox: true,
  stageHasLayout: true,
  stageMatchesImageAspect: true,
  imageFillsStage: true,
  comparisonNotRotatable: true,
  comparisonPanes: 2,
  singleTryonNotRotatable: true,
  singleViewNotRotatable: true,
  singleViewNoDots: true,
  // 늦게 도착한 예전 요청이 최신 결과를 덮어쓰지 않는다.
  race: "B",
  // 3:4 사진을 넣으면 상자도 3:4가 된다 — 늘어남도 레터박스도 없다.
  stageFollowsImage: true,
  noLetterbox: true,
  // 아바타 모드 전환 버튼은 전신사진으로 시작한 경우에만.
  hiddenForMeasurements: true,
  shownForPhoto: true,
  modeLabel: "✦ 아바타로 보기",
};

const failures = Object.entries(expected)
  .filter(([key, want]) => value?.[key] !== want)
  .map(([key, want]) => `${key}: expected ${JSON.stringify(want)}, got ${JSON.stringify(value?.[key])}`);

for (const row of value?.ordering ?? []) {
  if (JSON.stringify(row.actual) !== JSON.stringify(row.expected)) {
    failures.push(`착용 순서 "${row.title}": expected ${JSON.stringify(row.expected)}, got ${JSON.stringify(row.actual)}`);
  }
}
if (!value?.ordering?.length) failures.push("착용 순서 픽스처가 실행되지 않았다");

if (failures.length) {
  console.error("Avatar view test failed:\n  " + failures.join("\n  "));
  process.exit(1);
}
console.log(`Avatar view test passed: 시점 3개 회전 + 전신 이미지 비잘림 + 착용 순서 ${value.ordering.length}케이스가 백엔드와 일치`);
