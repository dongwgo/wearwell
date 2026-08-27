import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const query = "남성패션";
const total = 200;
const headers = {
  "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
  accept: "application/json, text/javascript, */*, q=0.01",
  "accept-language": "ko-KR,ko;q=0.9,en;q=0.7",
  "x-requested-with": "XMLHttpRequest",
  "x-pinterest-appstate": "active",
  "x-pinterest-pws-handler": "www/search/[scope].js"
};
const sourceUrl = `/search/pins/?q=${encodeURIComponent(query)}&rs=srs`;
const staging = path.resolve("assets/pinterest-next");
const closetDir = path.join(staging, "closet-sources");
const trendsDir = path.join(staging, "trends");
await rm(staging, { recursive: true, force: true });
await mkdir(closetDir, { recursive: true });
await mkdir(trendsDir, { recursive: true });

function preferredImage(pin) {
  const images = pin.images || {};
  return images.orig?.url || images["1200x"]?.url || images["736x"]?.url || images["564x"]?.url;
}

const pins = [];
const seenPins = new Set();
const seenImages = new Set();
let bookmark;
for (let page = 0; pins.length < total && page < 20; page++) {
  const options = { query, scope: "pins", page_size: 50, rs: "typed", source_url: sourceUrl };
  if (bookmark) options.bookmarks = [bookmark];
  const data = { options, context: {} };
  const endpoint = `https://www.pinterest.com/resource/BaseSearchResource/get/?source_url=${encodeURIComponent(sourceUrl)}&data=${encodeURIComponent(JSON.stringify(data))}`;
  const response = await fetch(endpoint, { headers });
  if (!response.ok) throw new Error(`Pinterest search failed: HTTP ${response.status}`);
  const payload = await response.json();
  const resource = payload.resource_response || {};
  for (const entry of resource.data?.results || resource.data || []) {
    const pin = entry?.type === "pin" ? entry : entry?.pin || entry;
    const imageUrl = preferredImage(pin);
    if (!pin?.id || !imageUrl || seenPins.has(String(pin.id)) || seenImages.has(imageUrl)) continue;
    const width = Number(pin.images?.orig?.width || 0);
    const height = Number(pin.images?.orig?.height || 0);
    if (width && height && (width < 360 || height < 480 || height / width < 0.8)) continue;
    seenPins.add(String(pin.id));
    seenImages.add(imageUrl);
    pins.push({
      id: String(pin.id), imageUrl, width, height,
      title: pin.grid_title || pin.title || pin.description || `${query} Pin`,
      description: pin.description || "", sourceUrl: `https://kr.pinterest.com/pin/${pin.id}/`,
      link: pin.link || null, dominantColor: pin.dominant_color || null
    });
    if (pins.length === total) break;
  }
  bookmark = resource.bookmark;
  if (!bookmark || bookmark === "-end-") break;
  process.stdout.write(`\r검색 ${pins.length}/${total}`);
}
if (pins.length < total) throw new Error(`Pinterest에서 고유 이미지 ${pins.length}/${total}개만 찾았습니다.`);

async function download(pin, index) {
  const group = index < 100 ? "closet" : "trend";
  const number = String((index % 100) + 1).padStart(3, "0");
  const response = await fetch(pin.imageUrl, { headers: { "user-agent": headers["user-agent"], referer: pin.sourceUrl } });
  if (!response.ok) throw new Error(`${pin.id}: image HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < 10_000) throw new Error(`${pin.id}: image is too small`);
  const type = response.headers.get("content-type") || "image/jpeg";
  const extension = type.includes("png") ? "png" : type.includes("webp") ? "webp" : "jpg";
  const filename = `${group}-${number}.${extension}`;
  await writeFile(path.join(group === "closet" ? closetDir : trendsDir, filename), bytes);
  return { ...pin, file: filename, group, contentType: type, bytes: bytes.length };
}

const records = [];
for (let start = 0; start < pins.length; start += 8) {
  const batch = await Promise.all(pins.slice(start, start + 8).map((pin, offset) => download(pin, start + offset)));
  records.push(...batch);
  process.stdout.write(`\r다운로드 ${records.length}/${total}`);
}
await writeFile(path.join(staging, "sources.json"), JSON.stringify({
  query, searchUrl: `https://kr.pinterest.com${sourceUrl}`, retrievedAt: new Date().toISOString(),
  notice: "Pinterest 공개 검색 결과에서 수집. 각 항목의 Pin URL과 원본 링크를 보존함.", images: records
}, null, 2));
console.log(`\n스테이징 완료: 옷장 원본 100개, Trends 100개 (${staging})`);
