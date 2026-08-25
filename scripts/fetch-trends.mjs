import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const trends = {
  women: [
    "woman full body white shirt cuffed jeans street style",
    "woman full body wide leg pants tank top street style",
    "woman full body midi skirt flats outfit",
    "woman full body cream blazer street style",
    "woman full body tonal outfit street style",
    "woman full body colorful accessories street style"
  ],
  men: [
    "man full body lightweight tailoring street style",
    "man full body bermuda shorts knit outfit",
    "man full body white shirt dark jeans outfit",
    "man full body all black street style",
    "man full body layered shirt wide pants",
    "man full body preppy street style"
  ]
};

const target = path.resolve("assets/trends");
await mkdir(target, { recursive: true });
const records = [];
const seen = new Set();

for (const [gender, queries] of Object.entries(trends)) {
  for (let index = 0; index < queries.length; index++) {
    const query = queries[index];
    const response = await fetch(`https://unsplash.com/napi/search/photos?query=${encodeURIComponent(query)}&per_page=20&page=1&orientation=portrait`);
    const payload = await response.json();
    const result = (payload.results || []).find(photo => !seen.has(photo.id));
    if (!result) throw new Error(`No unique trend image for ${query}`);
    seen.add(result.id);
    const imageResponse = await fetch(`${result.urls.raw}&w=760&h=980&fit=crop&crop=faces,edges&fm=jpg&q=84`);
    const file = `trend-${gender}-${index + 1}.jpg`;
    await writeFile(path.join(target, file), Buffer.from(await imageResponse.arrayBuffer()));
    records.push({ file, gender, query, source: result.links.html, photographer: result.user?.name, photographerUrl: result.user?.links?.html });
    console.log(`Fetched ${gender} trend ${index + 1}/6`);
  }
}

await writeFile(path.join(target, "sources.json"), JSON.stringify({ retrievedAt: new Date().toISOString(), images: records }, null, 2));
