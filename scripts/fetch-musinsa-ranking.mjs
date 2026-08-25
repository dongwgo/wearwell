import { mkdir, mkdtemp, readFile, rename, rm, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const assetsRoot = path.join(projectRoot, "assets");
const targetRoot = path.join(assetsRoot, "lookbook");
const metadataPath = path.join(assetsRoot, "lookbook-sources.json");
const rankingPage = "https://www.musinsa.com/main/musinsa/ranking";
const rankingApi = "https://api.musinsa.com/api2/hm/web/v5/pans/ranking/sections/199";
const rankingSources = [
  { code: "001001", label: "상의", subcategory: "티셔츠", quota: 12 },
  { code: "001002", label: "상의", subcategory: "셔츠", quota: 6 },
  { code: "001006", label: "상의", subcategory: "니트·이너", quota: 8 },
  { code: "001004", label: "상의", subcategory: "후드", quota: 3 },
  { code: "001005", label: "상의", subcategory: "스웨트셔츠", quota: 3 },
  { code: "002000", label: "아우터", subcategory: "아우터", quota: 18 },
  { code: "003000", label: "하의", subcategory: "바지", quota: 22 },
  { code: "103000", label: "신발", subcategory: "신발", quota: 12 },
  { code: "004000", label: "가방", subcategory: "가방", quota: 8 },
  { code: "101000", label: "액세서리", subcategory: "액세서리", quota: 8 }
];
const TARGET_TOTAL = 200;
const excludedProducts = /상품권|세럼|토너|크림|마스크팩|클렌징|선크림|향수|드링크|음료|딸기맛|초코맛|캐리어|수화물|여행용|머리부/i;

function collectProductNodes(value, products = []) {
  if (!value || typeof value !== "object") return products;
  if (value.type === "PRODUCT_COLUMN" && value.id && value.image?.url) products.push(value);
  for (const child of Array.isArray(value) ? value : Object.values(value)) collectProductNodes(child, products);
  return products;
}

async function fetchRanking(source) {
  const endpoint = new URL(rankingApi);
  endpoint.search = new URLSearchParams({
    storeCode: "musinsa", categoryCode: source.code, contentsId: "", subPan: "product", gf: "M", ageBand: "AGE_BAND_ALL"
  });
  const response = await fetch(endpoint, {
    headers: { accept: "application/json", referer: rankingPage, "user-agent": "Mozilla/5.0" }
  });
  if (!response.ok) throw new Error(`Musinsa ${source.label} ranking failed: HTTP ${response.status}`);
  return collectProductNodes(await response.json()).map(product => ({ product, source }));
}

function inferCategory(name, sourceLabel) {
  if (sourceLabel !== "전체") return sourceLabel;
  const value = name.toLowerCase();
  if (/백팩|백\b|bag\b|가방|크로스백|토트백|슬링백|브리프케이스/.test(value)) return "가방";
  if (/스니커|슈즈|구두|더비|로퍼|부츠|워커|샌들|슬리퍼|뮬\b|클로그|운동화|에어 포스|젤-\d|엔지니어|nbpf|nbpd|u509|w480|아디스타|\bmano\b|\bshoes?\b|\bboots?\b/.test(value)) return "신발";
  if (/삭스|양말|벨트|팔찌|목걸이|반지|메탈밴드|스카프|두건|볼 캡|볼캡|\bcap\b|비니|선글라스|아이웨어|포러너|시계/.test(value)) return "액세서리";
  if (/팬츠|바지|슬랙스|데님|쇼츠|반바지|트라우저|드로즈|\bpants?\b|\bshorts?\b|\btrousers?\b|\bdenim\b/.test(value)) return "하의";
  if (/자켓|재킷|점퍼|파카|패딩|다운|윈드쉘|윈드브레이커|집업|집 후드|후디|트랙탑|블루종|코트|가디건|jacket|\bparka\b|\bwindbreaker\b|\bhoodie\b/.test(value)) return "아우터";
  if (/티셔츠|반팔|긴팔|셔츠|니트|스웨트 셔츠|저지|폴로|티_|\btee\b|\bt-shirt\b|\bshirt\b|\bknit\b|\bjersey\b|\bsweatshirt\b/.test(value)) return "상의";
  return null;
}

function inferColor(name) {
  const value = name.toLowerCase();
  const detected = [];
  const colors = [
    ["블랙", /블랙|black|카본|charcoal/], ["화이트", /화이트|white|snow/],
    ["아이보리", /아이보리|ivory|cream|크림/], ["그레이", /그레이|grey|gray|ash|애쉬/],
    ["네이비", /네이비|navy/], ["블루", /블루|blue|indigo|인디고/],
    ["브라운", /브라운|brown|whiskey|카카오|camel|카멜/], ["베이지", /베이지|beige|sand|샌드|oatmeal|오트밀/],
    ["카키", /카키|khaki|olive|올리브|sage|세이지/], ["그린", /그린|green|habitat/],
    ["레드", /레드|red|burgundy|버건디|pink|핑크/], ["실버", /실버|silver|metal|메탈/]
  ];
  for (const [label, pattern] of colors) if (pattern.test(value)) detected.push(label);
  if (detected.length) return detected.slice(0, 3).join("·");
  if (/\d+\s*colors?|\d+컬러|\d+색|color\b/.test(value)) return "색상 다양";
  return "색상 미분류";
}

function findAnalytics(value, productId) {
  if (!value || typeof value !== "object") return null;
  if (String(value.product_id || value.item_id || "") === String(productId) && (value.reviewCount != null || value.original_price != null)) return value;
  for (const child of Array.isArray(value) ? value : Object.values(value)) {
    const match = findAnalytics(child, productId);
    if (match) return match;
  }
  return null;
}

function normalizeProduct({ product, source }, datasetRank) {
  const name = product.info.productName.trim();
  const analytics = findAnalytics(product, product.id) || {};
  return {
    datasetRank, sourceRank: Number(product.image.rank), sourceCategory: source.label,
    id: String(product.id), gender: "men", category: inferCategory(name, source.label), subcategory: source.subcategory, name,
    brand: product.info.brandName.trim(), color: inferColor(name),
    price: Number(product.info.finalPrice || analytics.price || 0),
    originalPrice: Number(analytics.original_price || product.info.finalPrice || 0),
    discountRatio: Number(product.info.discountRatio || analytics.discount_rate || 0),
    reviewCount: Number(analytics.reviewCount || 0), reviewScore: Number(analytics.reviewScore || 0),
    rankingLabels: (product.image.labels || []).map(label => label.text),
    productUrl: product.onClick?.url || `https://www.musinsa.com/products/${product.id}`,
    imageSourceUrl: product.image.url
  };
}

const existingMetadata = JSON.parse(await readFile(metadataPath, "utf8"));
const existingRecords = Array.isArray(existingMetadata.images) ? existingMetadata.images : [];
if (existingRecords.length > TARGET_TOTAL) throw new Error(`Existing dataset already exceeds ${TARGET_TOTAL} products`);
const additionsNeeded = TARGET_TOTAL - existingRecords.length;
const fetched = await Promise.all(rankingSources.map(fetchRanking));
const selected = [];
const seen = new Set(existingRecords.map(record => String(record.id)));
function addEntry(entry) {
  if (!entry || seen.has(entry.product.id) || excludedProducts.test(entry.product.info.productName)) return;
  if (!inferCategory(entry.product.info.productName, entry.source.label)) return;
  seen.add(entry.product.id);
  selected.push(entry);
}
for (const categoryProducts of fetched) {
  const quota = Math.min(categoryProducts[0]?.source.quota || 0, additionsNeeded - selected.length);
  const before = selected.length;
  for (const entry of categoryProducts.sort((a, b) => a.product.image.rank - b.product.image.rank)) {
    addEntry(entry);
    if (selected.length - before === quota) break;
  }
}
if (selected.length !== additionsNeeded) throw new Error(`Only found ${selected.length}/${additionsNeeded} new wearable ranking products`);
const newRecords = selected.map((entry, index) => normalizeProduct(entry, existingRecords.length + index + 1));
const records = [...existingRecords, ...newRecords];

if (process.argv.includes("--inspect") || process.argv.includes("--dry-run")) {
  const counts = Object.groupBy(records, record => record.category);
  console.log(`Ready: ${existingRecords.length} existing + ${newRecords.length} new = ${records.length} products`);
  console.log(Object.fromEntries(Object.entries(counts).map(([key, values]) => [key, values.length])));
  for (const record of newRecords) console.log(`${record.datasetRank}\t${record.category}\t${record.subcategory}\t${record.brand}\t${record.name}`);
  process.exit(0);
}

await mkdir(targetRoot, { recursive: true });
const stagingRoot = await mkdtemp(path.join(assetsRoot, ".lookbook-musinsa-"));

async function downloadRecord(record) {
  const response = await fetch(record.imageSourceUrl, {
    headers: { accept: "image/avif,image/webp,image/apng,image/*,*/*;q=0.8", referer: rankingPage, "user-agent": "Mozilla/5.0" }
  });
  if (!response.ok) throw new Error(`${record.id} image failed: HTTP ${response.status}`);
  const buffer = Buffer.from(await response.arrayBuffer());
  if (buffer.length < 5000) throw new Error(`${record.id} image response is too small`);
  const contentType = response.headers.get("content-type") || "";
  const extension = contentType.includes("png") || /\.png(?:\?|$)/i.test(record.imageSourceUrl) ? "png" : contentType.includes("webp") ? "webp" : "jpg";
  record.file = `look-${String(record.datasetRank).padStart(3, "0")}.${extension}`;
  await writeFile(path.join(stagingRoot, record.file), buffer);
}

try {
  let completed = 0;
  for (let start = 0; start < newRecords.length; start += 6) {
    const batch = newRecords.slice(start, start + 6);
    await Promise.all(batch.map(downloadRecord));
    completed += batch.length;
    process.stdout.write(`\rDownloaded ${completed}/${newRecords.length}`);
  }

  for (const record of newRecords) {
    const destination = path.join(targetRoot, record.file);
    await unlink(destination).catch(error => { if (error.code !== "ENOENT") throw error; });
    await rename(path.join(stagingRoot, record.file), destination);
  }
  await rm(stagingRoot, { recursive: true, force: true });

  const metadata = {
    notice: "오늘옷 데모용으로 수집한 무신사 남성 실시간 상품 랭킹 이미지입니다. 상품 이미지의 권리는 각 브랜드와 무신사에 있으며, 재배포·상업적 이용 전 별도 허가가 필요합니다.",
    methodology: "기존 남성 랭킹 상품 100개를 보존하고, 티셔츠·셔츠·니트/이너·후드/스웨트·아우터·바지·신발·가방·액세서리 랭킹에서 중복 상품을 제외한 100개를 균형 있게 추가했습니다.",
    rankingPage, genderFilter: "M", retrievedAt: new Date().toISOString(), targetTotal: TARGET_TOTAL, images: records
  };
  await writeFile(metadataPath, JSON.stringify(metadata, null, 2), "utf8");
  const appData = records.map(record => ({
    id: `musinsa-${record.id}`, image: `assets/lookbook/${record.file}`, gender: "all", rankingGender: record.gender,
    category: record.category, subcategory: record.subcategory || null, name: record.name, brand: record.brand, color: record.color,
    rank: record.datasetRank, sourceRank: record.sourceRank, price: record.price, sourceUrl: record.productUrl,
    worn: 0, userAdded: false
  }));
  await writeFile(path.join(assetsRoot, "lookbook-data.js"), `window.MUSINSA_RANKING = ${JSON.stringify(appData, null, 2)};\n`, "utf8");
  console.log(`\nAdded ${newRecords.length} images; dataset now has ${records.length} labeled Musinsa ranking products.`);
} catch (error) {
  await rm(stagingRoot, { recursive: true, force: true });
  throw error;
}
