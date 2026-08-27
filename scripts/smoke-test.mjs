import { spawn } from "node:child_process";
import { mkdtemp, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Windows 기본 설치 경로. 리눅스/CI에서는 CHROME_PATH로 덮어쓴다.
const chrome = process.env.CHROME_PATH || "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
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
  document.querySelector('[data-gender="men"]').click();
  document.querySelector('#nextGender').click();
  await generateAvatar();
  WearwellVLM.analyzeLookImage = async (_image, look) => ({
    summary: look.summary,
    pieces: look.pieces.map((piece, index) => ({
      pieceId: 'detected-' + index,
      label: (piece.colors?.[0] || '') + ' ' + piece.category + ' ' + (index + 1),
      layer: piece.category === '아우터' ? '아우터' : piece.category === '상의' ? '이너' : piece.category,
      category: piece.category,
      colors: [piece.colors?.[0] || '블랙'],
      materials: [piece.materials?.[0] || '코튼'],
      fits: [piece.fits?.[0] || '레귤러'],
      details: [piece.details?.[0] || '기본 마감'],
      confidence: .99
    })),
    engine: 'test-qwen'
  });
  showPreferenceStep(3);
  for (const look of influencerLooks.filter(look => look.gender === 'men')) {
    await analyzeLookbooks([look]);
    if (availableInfluencerMatches().length >= 2) break;
  }
  renderPreferenceChoices();
  [...document.querySelectorAll('[data-influencer-look]')].slice(0, 2).forEach(button => button.click());
  document.querySelector('#nextPreferences').click();
  document.querySelector('#finishPreferences').click();
  const wardrobeCount = document.querySelector('#wardrobeCount').textContent;
  document.querySelector('#uploadButton').click();
  const clipboard = new DataTransfer();
  clipboard.items.add(new File([new Uint8Array([137, 80, 78, 71])], 'clipboard.png', { type: '' }));
  const paste = new Event('paste', { bubbles: true, cancelable: true });
  Object.defineProperty(paste, 'clipboardData', { value: clipboard });
  document.dispatchEvent(paste);
  const clipboardUploadReady = uploadFiles.length === 1 && document.querySelectorAll('#uploadPreview .upload-thumb').length === 1;
  handleUploads([]);
  Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
    read: async () => [{ types: ['image/png'], getType: async () => new Blob([new Uint8Array([137, 80, 78, 71])], { type: 'image/png' }) }]
  } });
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'v', ctrlKey: true, bubbles: true }));
  await new Promise(resolve => setTimeout(resolve, 180));
  const clipboardShortcutReady = uploadFiles.length === 1 && uploadFiles[0].type === 'image/png';
  document.querySelector('#uploadDialog').close();
  document.querySelector('#wardrobeGrid [data-item-id]').click();
  const matchDialogOpened = document.querySelector('#itemMatchDialog').open;
  const matchOptions = document.querySelectorAll('.match-option').length;
  const matchedNames = [...document.querySelectorAll('.match-piece span')].map(node => node.textContent);
  document.querySelector('#itemMatchDialog').close();
  window.__wearwellXss = false;
  const maliciousName = '\"><img src=x onerror="window.__wearwellXss=true">';
  wardrobe.unshift({ id: 'xss-test', image: 'assets/lookbook/look-001.jpg', gender: selectedGender, category: '상의', name: maliciousName, color: '검정', worn: 0, userAdded: true });
  renderWardrobe();
  await new Promise(resolve => setTimeout(resolve, 100));
  const maliciousCard = document.querySelector('[data-item-id="xss-test"]');
  const safeNameRendered = maliciousCard?.textContent.includes(maliciousName);
  document.querySelector('[data-view="discover"]').click();
  const feedbackButton = document.querySelector('[data-look-feedback="up"]');
  feedbackButton?.click();
  const feedbackStored = Object.values(JSON.parse(localStorage.getItem('오늘옷-look-feedback') || '{}')).includes('up');
  WearwellVLM.analyzeBodyImage = async () => ({ body_shape: '보통', proportion: '균형', shoulderLine: '보통', silhouette: '기본' });
  fullBodyPhoto = avatarImage; avatarImage = null; setBodyInputMethod('photo');
  await generateAvatar();
  const photoAvatarReady = Boolean(avatarImage && avatarMeasurements?.photoBased);
  renderTryonStage({ image: 'assets/influencers/look-051.jpg', title: '사진 룩북' }, avatarImage);
  const whiteRequirement = { category: '상의', colors: ['화이트'], materials: ['코튼'], fits: ['레귤러'], details: ['셔츠'] };
  const exactColor = garmentSimilarityDetail({ category: '상의', color: '화이트', name: '화이트 코튼 셔츠', analysis: { primaryColor: '화이트', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '셔츠', pattern: '무지', finish: '무광', construction: ['카라'] } }, whiteRequirement);
  const wrongColor = garmentSimilarityDetail({ category: '상의', color: '레드', name: '레드 코튼 셔츠', analysis: { primaryColor: '레드', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '셔츠', pattern: '무지', finish: '무광', construction: ['카라'] } }, whiteRequirement);
  const unknownColor = garmentSimilarityDetail({ category: '상의', color: '색상 다양', name: '코튼 셔츠 4 colors', analysis: { primaryColor: '색상 다양', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '셔츠', pattern: '무지', finish: '무광', construction: ['카라'] } }, whiteRequirement);
  const referencedRequirement = { ...whiteRequirement, visualEmbedding: [1, 0], visualEmbeddingEngine: 'siglip:test' };
  const exactReference = garmentSimilarityDetail({ category: '상의', color: '레드', name: '직접 크롭', userAdded: true, referenceLookIds: ['look-safe'], analysis: { primaryColor: '레드', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '셔츠', pattern: '무지', finish: '무광', construction: [], visualEmbedding: [0, 1], visualEmbeddingEngine: 'siglip:test' } }, referencedRequirement, 'look-safe');
  const wrongCategoryReference = garmentSimilarityDetail({ category: '하의', color: '화이트', name: '잘못 분류된 크롭', userAdded: true, referenceLookIds: ['look-safe'], analysis: { primaryColor: '화이트', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '팬츠', pattern: '무지', finish: '무광', construction: [], visualEmbedding: [0, 1], visualEmbeddingEngine: 'siglip:test' } }, referencedRequirement, 'look-safe');
  const mixedEngine = garmentSimilarityDetail({ category: '상의', color: '화이트', name: '혼합 엔진', analysis: { primaryColor: '화이트', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '셔츠', pattern: '무지', finish: '무광', construction: [], visualEmbedding: [1, 0], visualEmbeddingEngine: 'local-fingerprint-v1' } }, referencedRequirement);
  const shortSleeveRequirement = { ...whiteRequirement, subcategory: '티셔츠', sleeveLength: '반팔', details: ['티셔츠'] };
  const shortSleeveMatch = garmentSimilarityDetail({ category: '상의', color: '화이트', name: '화이트 반팔 티셔츠', analysis: { primaryColor: '화이트', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '티셔츠', sleeveLength: '반팔', pattern: '무지', finish: '무광', construction: [] } }, shortSleeveRequirement);
  const longSleeveConflict = garmentSimilarityDetail({ category: '상의', color: '화이트', name: '화이트 긴팔 티셔츠', analysis: { primaryColor: '화이트', secondaryColors: [], material: '코튼', texture: '평직', fit: '레귤러', silhouette: '기본', subcategory: '티셔츠', sleeveLength: '긴팔', pattern: '무지', finish: '무광', construction: [] } }, shortSleeveRequirement);
  const liveScores = availableInfluencerMatches().map(entry => Math.round(entry.match.similarity * 1000));
  const firstAnalyzed = availableInfluencerMatches()[0];
  const oneToOne = firstAnalyzed?.match;
  const duplicateTops = wardrobe.filter(item => item.category === '상의').slice(0, 2);
  const duplicateCategoryLook = {
    analysisReady: true,
    weather: [],
    pieces: duplicateTops.map((item, index) => ({ pieceId: 'top-' + index, label: item.name, category: item.category, colors: [item.color], materials: [], fits: [], details: [] }))
  };
  const duplicateCategoryMatch = buildInfluencerMatch(duplicateCategoryLook);
  const unanalyzedMatch = buildInfluencerMatch({ analysisReady: false, weather: [], pieces: duplicateCategoryLook.pieces });
  segmentBatches = [{ image: 'data:image/jpeg;base64,AA==', name: '테스트 전신샷', items: [
    { id: 'seg-upper', image: 'assets/lookbook/look-001.jpg', name: '상의', category: '상의', sourceCategory: '상의', confidence: .9, refine: true },
    { id: 'seg-lower', image: 'assets/lookbook/look-002.jpg', name: '하의', category: '하의', sourceCategory: '하의', confidence: .8, refine: false }
  ] }];
  renderSegmentChoices();
  const segmentCards = document.querySelectorAll('[data-segment-choice]');
  segmentCards[0].querySelector('[data-segment-refine]').click();
  const categorySelect = segmentCards[0].querySelector('[data-segment-category]');
  categorySelect.value = '아우터';
  categorySelect.dispatchEvent(new Event('change', { bubbles: true }));
  return ({
    profile: JSON.parse(localStorage.getItem('오늘옷-profile')),
    look: document.querySelector('#lookTitle').textContent,
    wardrobeCount,
    clipboardUploadReady,
    clipboardShortcutReady,
    dialogOpen: document.querySelector('#preferenceDialog').open,
    matchDialogOpened,
    matchOptions,
    matchedNames,
    trendCards: document.querySelectorAll('.discover-card').length,
    closetMatches: document.querySelectorAll('.discover-closet-match').length,
    avatarReady: Boolean(avatarImage),
    xssTriggered: window.__wearwellXss,
    safeNameRendered,
    photoMethodAvailable: Boolean(document.querySelector('[data-body-method="photo"]') && document.querySelector('#fullBodyInput')),
    comparisonPanes: document.querySelectorAll('#tryonStage.comparison figure').length,
    feedPhotos: window.WEARWELL_INFLUENCER_LOOKS.filter(look => look.image.startsWith('assets/influencers/')).length,
    similarityProbe: { exact: exactColor.total, wrong: wrongColor.total, unknown: unknownColor.total },
    referenceProbe: { exact: exactReference.total, wrongCategory: wrongCategoryReference.total, mixedVisual: mixedEngine.visual },
    garmentFormProbe: { shortSleeve: shortSleeveMatch.total, longSleeve: longSleeveConflict.total },
    feedbackStored,
    lookbookUploadReady: Boolean(document.querySelector('#lookbookUploadDialog') && WearwellDB.getAllLooks && WearwellVLM.embedImage),
    distinctSimilarityScores: new Set(liveScores).size,
    oneToOnePieces: oneToOne?.total || 0,
    oneToOneMatches: oneToOne?.matches.length || 0,
    oneToOneUniqueItems: new Set(oneToOne?.matches.map(entry => entry.item.id) || []).size,
    duplicateCategoryMatches: duplicateCategoryMatch.fulfilled,
    duplicateCategoryUniqueItems: new Set(duplicateCategoryMatch.matches.filter(Boolean).map(entry => entry.item.id)).size,
    unanalyzedTotal: unanalyzedMatch.total,
    photoAvatarReady,
    segmentChoices: segmentCards.length,
    segmentRefineToggle: segmentBatches[0].items[0].refine,
    segmentCategoryChange: segmentBatches[0].items[0].category
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

if (!value?.profile || value.profile.gender !== "men" || value.profile.influencerLooks?.length !== 2 || value.dialogOpen || Number(value.wardrobeCount) < 1 || !value.clipboardUploadReady || !value.clipboardShortcutReady || !value.matchDialogOpened || value.matchOptions !== 2 || value.trendCards < 1 || value.closetMatches !== value.trendCards || !value.avatarReady || value.xssTriggered || !value.safeNameRendered || !value.photoMethodAvailable || value.comparisonPanes !== 2 || value.feedPhotos !== 100 || !value.photoAvatarReady || value.similarityProbe.exact <= value.similarityProbe.wrong || value.similarityProbe.exact <= value.similarityProbe.unknown || value.similarityProbe.unknown >= .56 || value.referenceProbe.exact < .98 || value.referenceProbe.wrongCategory !== 0 || value.referenceProbe.mixedVisual !== .5 || value.garmentFormProbe.shortSleeve <= 0 || value.garmentFormProbe.longSleeve !== 0 || !value.feedbackStored || !value.lookbookUploadReady || value.distinctSimilarityScores < 2 || value.oneToOnePieces < 2 || value.oneToOneMatches !== value.oneToOnePieces || value.oneToOneUniqueItems !== value.oneToOnePieces || value.duplicateCategoryMatches < 1 || value.duplicateCategoryUniqueItems !== value.duplicateCategoryMatches || value.unanalyzedTotal !== 0 || value.segmentChoices !== 2 || value.segmentRefineToggle !== false || value.segmentCategoryChange !== "아우터") {
  throw new Error(`Smoke test failed: ${JSON.stringify(value)}`);
}
console.log(`Smoke test passed: 룩북 ${value.oneToOnePieces}벌 ↔ 내 옷 ${value.oneToOneUniqueItems}벌 1:1 / 동일 카테고리 매칭 ${value.duplicateCategoryUniqueItems}벌도 중복 없음 / 미분석 추천 ${value.unanalyzedTotal}벌`);
