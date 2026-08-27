import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";

const ROOT = path.resolve();
const OUTPUT = path.join(ROOT, "assets", "lookbook-flux-next");
const MANIFEST = path.join(OUTPUT, "manifest.json");
await mkdir(OUTPUT, { recursive: true });

function browserData(text, key) {
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(text, context);
  return context.window[key];
}

const config = browserData(await readFile(path.join(ROOT, "local-config.js"), "utf8"), "WEARWELL_CONFIG");
if (!config?.API_BASE || !config?.API_TOKEN) throw new Error("local-config.js에 원격 GPU API 설정이 필요합니다.");
const wardrobe = browserData(await readFile(path.join(ROOT, "assets", "lookbook-data.js"), "utf8"), "MUSINSA_RANKING");
const sources = JSON.parse(await readFile(path.join(ROOT, "assets", "pinterest-sources.json"), "utf8")).images;
const sourceByUrl = new Map(sources.filter(item => item.group === "closet").map(item => [item.sourceUrl, item]));
let manifest = { model: "black-forest-labs/FLUX.2-klein-4B", startedAt: new Date().toISOString(), completed: {} };
try { manifest = JSON.parse(await readFile(MANIFEST, "utf8")); } catch {}

const groups = new Map();
for (const item of wardrobe) {
  const source = sourceByUrl.get(item.sourceUrl);
  if (!source) throw new Error(`원본을 찾을 수 없습니다: ${item.id}`);
  const list = groups.get(item.sourceUrl) || { source, records: [] };
  list.records.push(item);
  groups.set(item.sourceUrl, list);
}

function mimeType(filename) {
  const extension = path.extname(filename).toLowerCase();
  return extension === ".png" ? "image/png" : extension === ".webp" ? "image/webp" : "image/jpeg";
}

function decodeDataUrl(value) {
  const match = /^data:(image\/[\w.+-]+);base64,(.+)$/s.exec(value || "");
  if (!match) throw new Error("FLUX 결과가 data URL이 아닙니다.");
  return { mime: match[1], bytes: Buffer.from(match[2], "base64") };
}

async function requestJson(payload, attempt = 1) {
  const response = await fetch(`${config.API_BASE}/api/closet/refine`, {
    method: "POST",
    headers: { authorization: `Bearer ${config.API_TOKEN}`, "content-type": "application/json" },
    body: JSON.stringify(payload), signal: AbortSignal.timeout(600_000)
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (attempt < 4 && [429, 502, 503, 504].includes(response.status)) {
      await new Promise(resolve => setTimeout(resolve, attempt * 5000));
      return requestJson(payload, attempt + 1);
    }
    throw new Error(`HTTP ${response.status}: ${body.detail || "refine 실패"}`);
  }
  return body;
}

let completed = Object.keys(manifest.completed).length;
const total = wardrobe.length;
const jobs = [];
for (const { source, records } of groups.values()) {
  const pending = records.filter(record => !manifest.completed[record.id]);
  for (let start = 0; start < pending.length; start += 4) jobs.push({ source, records: pending.slice(start, start + 4) });
}

async function processJob({ source, records }) {
  const sourcePath = path.join(ROOT, "assets", "pinterest-refine-sources", `${path.parse(source.file).name}.jpg`);
  const bytes = await readFile(sourcePath);
  const result = await requestJson({
    image: `data:image/jpeg;base64,${bytes.toString("base64")}`,
    name: `Pinterest ${source.id}`, model: "b3_clothes", categories: records.map(record => record.category),
    includeRejected: false, gender: "men", generate: true, seed: Number(source.id.slice(-8)) || 42, steps: 4
  });
  const byCategory = new Map(result.items.map(item => [item.category, item]));
  for (const record of records) {
    const generated = byCategory.get(record.category);
    if (!generated?.stages?.closet || generated.generationError) throw new Error(`${record.id}: ${generated?.generationError || "FLUX 결과 없음"}`);
    const image = decodeDataUrl(generated.stages.closet);
    const extension = image.mime.includes("png") ? "png" : "jpg";
    const filename = `look-${String(record.rank).padStart(3, "0")}.${extension}`;
    await writeFile(path.join(OUTPUT, filename), image.bytes);
    manifest.completed[record.id] = { filename, sourceFile: source.file, category: record.category, generation: generated.generation };
    completed++;
  }
  manifest.updatedAt = new Date().toISOString();
  await writeFile(MANIFEST, JSON.stringify(manifest, null, 2));
  process.stdout.write(`\rFLUX 재생성 ${completed}/${total}`);
}

for (let start = 0; start < jobs.length; start += 2) {
  const results = await Promise.allSettled(jobs.slice(start, start + 2).map(processJob));
  const failure = results.find(result => result.status === "rejected");
  if (failure) throw failure.reason;
}

for (const record of wardrobe) {
  const generated = manifest.completed[record.id];
  if (!generated) throw new Error(`미완료 항목: ${record.id}`);
  record.image = `assets/lookbook/${generated.filename}`;
  record.generation = generated.generation;
}
await writeFile(path.join(OUTPUT, "lookbook-data.js"), `window.MUSINSA_RANKING = ${JSON.stringify(wardrobe, null, 2)};\n`);
manifest.completedAt = new Date().toISOString();
await writeFile(MANIFEST, JSON.stringify(manifest, null, 2));
console.log(`\n완료: ${total}개 FLUX 상품컷 생성`);
