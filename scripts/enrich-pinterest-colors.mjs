import { readFile, writeFile } from "node:fs/promises";
import vm from "node:vm";

function loadBrowserData(file, key) {
  const context = { window: {} };
  vm.createContext(context);
  vm.runInContext(file, context);
  return context.window[key];
}

function colorName(hex) {
  const match = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!match) return "멀티컬러";
  const value = Number.parseInt(match[1], 16);
  const r = (value >> 16) & 255, g = (value >> 8) & 255, b = value & 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b), light = (max + min) / 510;
  if (max - min < 22) return light < .18 ? "블랙" : light > .82 ? "화이트" : light > .58 ? "라이트 그레이" : "그레이";
  const delta = max - min;
  let hue = max === r ? ((g - b) / delta) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
  hue = (hue * 60 + 360) % 360;
  if (light < .24) return hue >= 190 && hue < 260 ? "네이비" : "블랙";
  if (hue < 15 || hue >= 345) return "레드";
  if (hue < 45) return light < .45 ? "브라운" : "베이지";
  if (hue < 75) return "베이지";
  if (hue < 165) return "그린";
  if (hue < 195) return "민트";
  if (hue < 255) return light < .42 ? "네이비" : "블루";
  if (hue < 300) return "퍼플";
  return "핑크";
}

const sources = JSON.parse(await readFile("assets/pinterest-sources.json", "utf8")).images;
const colorByUrl = new Map(sources.map(source => [source.sourceUrl, colorName(source.dominantColor)]));
const wardrobeText = await readFile("assets/lookbook-data.js", "utf8");
const wardrobe = loadBrowserData(wardrobeText, "MUSINSA_RANKING");
for (const item of wardrobe) {
  item.color = colorByUrl.get(item.sourceUrl) || "멀티컬러";
  const type = item.category === "상의" ? (item.color === "화이트" ? "셔츠" : "티셔츠")
    : item.category === "하의" ? "팬츠"
    : item.category === "신발" ? "슈즈"
    : item.category === "가방" ? "가방"
    : item.category === "액세서리" ? "액세서리" : item.category;
  item.name = `Pinterest 남성패션 · ${item.color} ${type} ${String(item.rank).padStart(3, "0")}`;
}
await writeFile("assets/lookbook-data.js", `window.MUSINSA_RANKING = ${JSON.stringify(wardrobe, null, 2)};\n`);

const trendsText = await readFile("assets/influencer-data.js", "utf8");
const trends = loadBrowserData(trendsText, "WEARWELL_INFLUENCER_LOOKS");
for (const look of trends) {
  const color = colorByUrl.get(look.sourceUrl) || "멀티컬러";
  for (const piece of look.pieces || []) piece.colors = [color];
}
await writeFile("assets/influencer-data.js", `(function () {\n  window.WEARWELL_INFLUENCER_LOOKS = ${JSON.stringify(trends, null, 2)};\n})();\n`);
console.log(`대표 색상 반영: 옷장 ${wardrobe.length}개, Trends ${trends.length}개`);
