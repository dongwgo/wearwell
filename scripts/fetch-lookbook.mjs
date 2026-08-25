import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const target = path.resolve("assets/lookbook");
await mkdir(target, { recursive: true });

const candidates = [];
const seen = new Set();

const groups = [
  { gender: "women", queries: ["Korean woman fashion", "Seoul woman street style", "Asian woman casual outfit", "Korean female outfit"] },
  { gender: "men", queries: ["Korean man fashion", "Seoul man street style", "Asian man casual outfit", "Korean male outfit"] }
];

for (const group of groups) {
  const groupResults = [];
  for (const query of group.queries) {
    for (let page = 1; page <= 3 && groupResults.length < 50; page++) {
      const endpoint = `https://unsplash.com/napi/search/photos?query=${encodeURIComponent(query)}&per_page=20&page=${page}`;
      const response = await fetch(endpoint);
      if (!response.ok) throw new Error(`Unsplash search failed: HTTP ${response.status}`);
      const payload = await response.json();
      for (const result of payload.results || []) {
        if (seen.has(result.id)) continue;
        seen.add(result.id);
        groupResults.push({
          id: result.id,
          gender: group.gender,
          query,
          source: result.links.html,
          photographer: result.user?.name || "Unknown photographer",
          photographerUrl: result.user?.links?.html || "https://unsplash.com",
          download: `${result.urls.raw}&w=720&h=900&fit=crop&crop=faces,edges&fm=jpg&q=82`
        });
        if (groupResults.length === 50) break;
      }
    }
  }
  if (groupResults.length < 50) throw new Error(`Only found ${groupResults.length} unique ${group.gender} images`);
  candidates.push(...groupResults);
}

if (candidates.length < 100) throw new Error(`Only found ${candidates.length} unique images`);

const records = [];
let completed = 0;

async function download(record, index) {
  const number = String(index + 1).padStart(3, "0");
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      const response = await fetch(record.download, { redirect: "follow" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const buffer = Buffer.from(await response.arrayBuffer());
      if (buffer.length < 5000) throw new Error("response too small");
      await writeFile(path.join(target, `look-${number}.jpg`), buffer);
      records[index] = { file: `look-${number}.jpg`, id: record.id, gender: record.gender, source: record.source, photographer: record.photographer, photographerUrl: record.photographerUrl, query: record.query };
      completed++;
      process.stdout.write(`\rFetched ${completed}/100`);
      return;
    } catch (error) {
      if (attempt === 3) throw new Error(`look-${number}: ${error.message}`);
    }
  }
}

for (let start = 0; start < 100; start += 8) {
  await Promise.all(candidates.slice(start, start + 8).map((record, offset) => download(record, start + offset)));
}

await writeFile(path.resolve("assets/lookbook-sources.json"), JSON.stringify({
  notice: "Demo imagery sourced from Korean and Seoul style searches on Unsplash. Images 1-50 support the women's flow and 51-100 support the men's flow. Photographer and original photo links are retained below.",
  retrievedAt: new Date().toISOString(),
  images: records
}, null, 2));

console.log("\nSaved 100 images and source metadata.");
