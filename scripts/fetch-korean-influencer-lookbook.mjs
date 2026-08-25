import { createHash } from "node:crypto";
import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";

const FINAL_OUTPUT_DIR = path.resolve("assets/influencers");
const OUTPUT_DIR = path.resolve("assets/influencers-next");
const DATA_FILE = path.resolve("assets/influencer-data.js");
const PER_PERSON = 10;
const USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36";

const influencerCatalog = [
  { name: "김나영", handle: "@nayoungkeem", gender: "women", bodyLabel: "마른·보통 체형", bmiRange: [17, 24], bodyShapes: ["마른 체형", "보통"], styles: ["도시적인 캐주얼", "빈티지 믹스", "컬러 포인트"] },
  { name: "차정원", handle: "@ch_amii", gender: "women", bodyLabel: "마른·보통 체형", bmiRange: [17, 23], bodyShapes: ["마른 체형", "보통"], styles: ["미니멀", "프렌치 캐주얼", "뉴트럴 톤"] },
  { name: "기은세", handle: "@kieunse", gender: "women", bodyLabel: "보통·균형 체형", bmiRange: [18, 25], bodyShapes: ["보통", "하체가 발달한 체형"], styles: ["클래식", "페미닌", "테일러드"] },
  { name: "아이린 김", handle: "@ireneisgood", gender: "women", bodyLabel: "마른·큰 키 체형", bmiRange: [17, 23], bodyShapes: ["마른 체형", "보통"], styles: ["스트리트", "컬러 블록", "Y2K"] },
  { name: "정려원", handle: "@yoanaloves", gender: "women", bodyLabel: "마른 체형", bmiRange: [16, 22], bodyShapes: ["마른 체형", "보통"], styles: ["빈티지", "보헤미안", "레이어드"] },
  { name: "강민경", handle: "@iammingki", gender: "women", bodyLabel: "마른·균형 체형", bmiRange: [17, 23], bodyShapes: ["마른 체형", "보통"], styles: ["오피스 캐주얼", "미니멀", "톤온톤"] },
  { name: "최수영", handle: "@sooyoungchoi", gender: "women", bodyLabel: "마른·큰 키 체형", bmiRange: [17, 23], bodyShapes: ["마른 체형", "보통"], styles: ["모던", "테일러드", "데님 캐주얼"] },
  { name: "공효진", handle: "@rovvxhyo", gender: "women", bodyLabel: "마른·보통 체형", bmiRange: [17, 24], bodyShapes: ["마른 체형", "보통"], styles: ["내추럴", "빈티지", "에코 캐주얼"] },
  { name: "이호정", handle: "@holly608", gender: "women", bodyLabel: "마른·큰 키 체형", bmiRange: [17, 23], bodyShapes: ["마른 체형", "보통"], styles: ["모델 오프듀티", "스트리트", "미니멀"] },
  { name: "한혜연", handle: "@hhy6588", gender: "women", bodyLabel: "통통·균형 체형", bmiRange: [23, 35], bodyShapes: ["통통한 체형", "상체가 발달한 체형", "하체가 발달한 체형"], styles: ["맥시멀", "컬러 포인트", "스트리트"] },
  { name: "이동휘", handle: "@dlehdgnl", gender: "men", bodyLabel: "마른·보통 체형", bmiRange: [18, 25], bodyShapes: ["마른 체형", "보통"], styles: ["빈티지", "아메카지", "스트리트"] },
  { name: "주우재", handle: "@ophen28", gender: "men", bodyLabel: "마른·큰 키 체형", bmiRange: [17, 23], bodyShapes: ["마른 체형", "보통"], styles: ["미니멀", "와이드 실루엣", "모델 오프듀티"] },
  { name: "류준열", handle: "@ryusdb", gender: "men", bodyLabel: "보통·균형 체형", bmiRange: [19, 26], bodyShapes: ["보통", "마른 체형"], styles: ["고프코어", "빈티지", "스포츠 믹스"] },
  { name: "코드 쿤스트", handle: "@code_kunst", gender: "men", bodyLabel: "마른 체형", bmiRange: [16, 22], bodyShapes: ["마른 체형", "보통"], styles: ["힙합", "오버핏", "스트리트"] },
  { name: "봉태규", handle: "@taegyu_bong", gender: "men", bodyLabel: "마른·보통 체형", bmiRange: [18, 25], bodyShapes: ["마른 체형", "보통"], styles: ["젠더리스", "클래식", "빈티지"] },
  { name: "김원중", handle: "@keemwj", gender: "men", bodyLabel: "마른·큰 키 체형", bmiRange: [18, 24], bodyShapes: ["마른 체형", "보통"], styles: ["모델 오프듀티", "테일러드", "스트리트"] },
  { name: "박성진", handle: "@teriyakipapi", gender: "men", bodyLabel: "마른·큰 키 체형", bmiRange: [18, 24], bodyShapes: ["마른 체형", "보통"], styles: ["아방가르드", "스트리트", "레이어드"] },
  { name: "장기용", handle: "@juanxkui", gender: "men", bodyLabel: "보통·큰 키 체형", bmiRange: [19, 26], bodyShapes: ["보통", "상체가 발달한 체형"], styles: ["미니멀", "댄디", "모델 오프듀티"] },
  { name: "변요한", handle: "@byunyohan_official", gender: "men", bodyLabel: "보통·균형 체형", bmiRange: [20, 27], bodyShapes: ["보통", "상체가 발달한 체형"], styles: ["클래식", "워크웨어", "캐주얼"] },
  { name: "오정규", handle: "@jung9_6", gender: "men", bodyLabel: "보통·큰 키 체형", bmiRange: [19, 27], bodyShapes: ["보통", "마른 체형"], styles: ["남친룩", "미니멀", "시티보이"] }
];

// 이름·계정명으로 공개 사진을 충분히 교차 확인할 수 있는 인물만 데모에 사용한다.
const selectedHandles = new Set([
  "@nayoungkeem", "@ch_amii", "@kieunse", "@ireneisgood", "@yoanaloves", "@iammingki", "@sooyoungchoi", "@rovvxhyo", "@holly608", "@hhy6588",
  "@dlehdgnl", "@ophen28", "@ryusdb", "@code_kunst", "@taegyu_bong", "@teriyakipapi", "@juanxkui", "@byunyohan_official"
]);
const influencers = influencerCatalog.filter(person => selectedHandles.has(person.handle));

const trustedSourcePatterns = [
  /(^|\.)instagram\.com$/i,
  /\.co\.kr$/i, /\.kr$/i,
  /(^|\.)naver\.com$/i, /(^|\.)daum\.net$/i, /(^|\.)nate\.com$/i,
  /(^|\.)sbs\.co\.kr$/i, /(^|\.)kbs\.co\.kr$/i, /(^|\.)mbc\.co\.kr$/i,
  /(^|\.)hankyung\.com$/i, /(^|\.)mk\.co\.kr$/i, /(^|\.)mksports\.co\.kr$/i,
  /(^|\.)hankooki\.com$/i, /(^|\.)sports\.hankooki\.com$/i,
  /(^|\.)xportsnews\.com$/i, /(^|\.)spotvnews\.co\.kr$/i, /(^|\.)topstarnews\.net$/i,
  /(^|\.)tenasia\.co\.kr$/i, /(^|\.)newsen\.com$/i, /(^|\.)news1\.kr$/i,
  /(^|\.)donga\.com$/i, /(^|\.)sports\.donga\.com$/i, /(^|\.)mt\.co\.kr$/i,
  /(^|\.)fnnews\.com$/i, /(^|\.)ytn\.co\.kr$/i, /(^|\.)dispatch\.co\.kr$/i,
  /(^|\.)fashionn\.com$/i, /(^|\.)vogue\./i, /(^|\.)elle\./i,
  /(^|\.)wkorea\.com$/i, /(^|\.)marieclairekorea\.com$/i, /(^|\.)allurekorea\.com$/i,
  /(^|\.)harpersbazaar\.co\.kr$/i, /(^|\.)hypebeast\./i, /(^|\.)gqkorea\.co\.kr$/i,
  /(^|\.)starnewskorea\.com$/i, /(^|\.)osen\.co\.kr$/i, /(^|\.)heraldcorp\.com$/i
];

const templates = {
  women: [
    { mood: "셔츠와 데님", summary: "여유 있는 셔츠를 데님과 단정하게 연결한 데일리 룩", pieces: [["상의", ["화이트", "아이보리", "블루"], ["코튼"], ["레귤러", "오버"], ["셔츠", "카라", "버튼"]], ["하의", ["블루", "네이비"], ["데님"], ["스트레이트", "와이드"], ["청바지"]], ["신발", ["화이트", "블랙"], ["가죽", "메시"], ["로우"], ["스니커즈", "로퍼"]]] },
    { mood: "뉴트럴 레이어드", summary: "베이지와 크림 계열을 겹쳐 입어 부드러운 깊이를 만든 룩", pieces: [["아우터", ["베이지", "브라운", "아이보리"], ["코튼", "울"], ["레귤러", "오버"], ["재킷", "코트"]], ["상의", ["화이트", "아이보리"], ["코튼", "니트"], ["슬림", "레귤러"], ["티셔츠", "니트"]], ["하의", ["베이지", "브라운", "그레이"], ["코튼", "울"], ["와이드", "스트레이트"], ["슬랙스"]]] },
    { mood: "모던 테일러링", summary: "재킷과 곧은 실루엣의 하의로 비율을 정리한 도시적인 룩", pieces: [["아우터", ["블랙", "그레이", "네이비"], ["울", "폴리에스터"], ["레귤러", "오버"], ["블레이저", "라펠"]], ["상의", ["화이트", "블랙", "그레이"], ["코튼", "니트"], ["슬림", "레귤러"], ["티셔츠", "블라우스"]], ["하의", ["블랙", "그레이", "네이비"], ["울", "폴리에스터"], ["와이드", "스트레이트"], ["슬랙스"]]] },
    { mood: "빈티지 포인트", summary: "패턴이나 컬러 한 가지를 중심으로 힘을 준 빈티지 캐주얼 룩", pieces: [["상의", ["레드", "블루", "브라운", "멀티"], ["코튼", "니트"], ["레귤러", "오버"], ["패턴", "프린트"]], ["하의", ["블루", "브라운", "베이지"], ["데님", "코튼"], ["와이드", "스트레이트"], ["빈티지 워싱"]], ["가방", ["브라운", "블랙"], ["가죽", "캔버스"], ["미디엄"], ["숄더백"]]] },
    { mood: "편안한 원마일", summary: "부드러운 상의와 여유 있는 하의에 운동화를 더한 편안한 룩", pieces: [["상의", ["그레이", "화이트", "네이비"], ["코튼", "니트"], ["오버", "레귤러"], ["스웨트", "티셔츠"]], ["하의", ["그레이", "블랙", "네이비"], ["코튼", "나일론"], ["와이드", "조거"], ["밴딩"]], ["신발", ["화이트", "실버", "블랙"], ["메시", "가죽"], ["로우"], ["스니커즈"]]] }
  ],
  men: [
    { mood: "시티보이 셔츠", summary: "넉넉한 셔츠와 와이드 팬츠로 실루엣을 만든 도심 캐주얼 룩", pieces: [["상의", ["화이트", "블루", "스트라이프"], ["코튼"], ["오버", "레귤러"], ["셔츠", "카라"]], ["하의", ["네이비", "그레이", "베이지"], ["코튼", "울"], ["와이드", "스트레이트"], ["슬랙스", "치노"]], ["신발", ["화이트", "블랙"], ["가죽", "메시"], ["로우"], ["스니커즈", "로퍼"]]] },
    { mood: "빈티지 워크웨어", summary: "워싱된 아우터와 데님을 질감 차이로 쌓은 빈티지 룩", pieces: [["아우터", ["브라운", "카키", "네이비"], ["코튼", "데님"], ["오버", "레귤러"], ["워크 재킷", "워싱"]], ["상의", ["화이트", "그레이"], ["코튼"], ["레귤러"], ["티셔츠"]], ["하의", ["블루", "네이비", "브라운"], ["데님", "코튼"], ["스트레이트", "와이드"], ["워싱", "카고"]]] },
    { mood: "미니멀 테일러드", summary: "절제된 색과 반듯한 재킷·슬랙스로 완성한 깔끔한 룩", pieces: [["아우터", ["블랙", "그레이", "네이비"], ["울", "폴리에스터"], ["레귤러", "오버"], ["블레이저", "라펠"]], ["상의", ["화이트", "블랙", "아이보리"], ["코튼", "니트"], ["레귤러", "슬림"], ["티셔츠", "니트"]], ["하의", ["블랙", "그레이"], ["울", "폴리에스터"], ["와이드", "스트레이트"], ["슬랙스"]]] },
    { mood: "고프코어 믹스", summary: "기능성 아우터와 편한 팬츠에 러닝화를 더한 실용적인 룩", pieces: [["아우터", ["블랙", "카키", "그레이"], ["나일론", "폴리에스터"], ["오버", "레귤러"], ["바람막이", "포켓"]], ["하의", ["블랙", "카키", "그레이"], ["나일론", "코튼"], ["와이드", "조거"], ["카고", "밴딩"]], ["신발", ["그레이", "블랙", "화이트"], ["메시", "고무"], ["로우"], ["러닝화"]]] },
    { mood: "프레피 레이어드", summary: "셔츠와 니트, 단정한 팬츠를 겹쳐 입은 친근한 프레피 룩", pieces: [["상의", ["화이트", "네이비", "그레이"], ["코튼", "니트"], ["레귤러", "오버"], ["셔츠", "니트", "카라"]], ["하의", ["베이지", "네이비", "그레이"], ["코튼", "울"], ["스트레이트", "와이드"], ["치노", "슬랙스"]], ["신발", ["브라운", "블랙", "화이트"], ["가죽"], ["로우"], ["로퍼", "스니커즈"]]] }
  ]
};

function decodeEntities(value) {
  return value.replaceAll("&quot;", "\"").replaceAll("&amp;", "&").replaceAll("&#39;", "'").replaceAll("&lt;", "<").replaceAll("&gt;", ">");
}

function candidatesFromBing(html) {
  const found = [];
  for (const match of html.matchAll(/\sm="([^"]+)"/g)) {
    if (!match[1].includes("murl")) continue;
    try {
      const item = JSON.parse(decodeEntities(match[1]));
      if (item.murl && item.purl) found.push(item);
    } catch {}
  }
  return found;
}

async function searchImages(query, first = 1) {
  const url = `https://www.bing.com/images/search?q=${encodeURIComponent(query)}&form=HDRSC2&first=${first}`;
  const response = await fetch(url, { headers: { "user-agent": USER_AGENT, "accept-language": "ko-KR,ko;q=0.9,en;q=0.7" } });
  if (!response.ok) throw new Error(`Bing search failed: ${response.status}`);
  return candidatesFromBing(await response.text());
}

function imageExtension(contentType) {
  if (contentType.includes("png")) return "png";
  if (contentType.includes("webp")) return "webp";
  if (contentType.includes("avif")) return "avif";
  return "jpg";
}

function imageDimensions(bytes, contentType) {
  if (contentType.includes("png") && bytes.length >= 24) {
    return { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) };
  }
  if (contentType.includes("jpeg") || contentType.includes("jpg")) {
    let offset = 2;
    while (offset + 9 < bytes.length) {
      if (bytes[offset] !== 0xff) { offset++; continue; }
      const marker = bytes[offset + 1];
      if ([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf].includes(marker)) {
        return { height: bytes.readUInt16BE(offset + 5), width: bytes.readUInt16BE(offset + 7) };
      }
      if (marker === 0xd8 || marker === 0xd9) { offset += 2; continue; }
      const length = bytes.readUInt16BE(offset + 2);
      if (!length) break;
      offset += 2 + length;
    }
  }
  if (contentType.includes("webp") && bytes.toString("ascii", 12, 16) === "VP8X") {
    return {
      width: 1 + bytes.readUIntLE(24, 3),
      height: 1 + bytes.readUIntLE(27, 3)
    };
  }
  return { width: 0, height: 0 };
}

async function downloadImage(url, minimumBytes = 18_000) {
  const response = await fetch(url, { redirect: "follow", signal: AbortSignal.timeout(18000), headers: { "user-agent": USER_AGENT, accept: "image/avif,image/webp,image/png,image/jpeg,*/*" } });
  if (!response.ok) throw new Error(`image ${response.status}`);
  const contentType = (response.headers.get("content-type") || "").toLowerCase();
  if (!contentType.startsWith("image/") || contentType.includes("svg") || contentType.includes("gif")) throw new Error(`unsupported ${contentType}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (bytes.length < minimumBytes || bytes.length > 15_000_000) throw new Error(`invalid size ${bytes.length}`);
  return { bytes, contentType };
}

function safeUrl(url) {
  try {
    const parsed = new URL(url);
    return ["http:", "https:"].includes(parsed.protocol) ? parsed.href : "";
  } catch { return ""; }
}

function sourceDomain(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return "출처 미상"; }
}

function isTrustedSource(url) {
  const domain = sourceDomain(url);
  return trustedSourcePatterns.some(pattern => pattern.test(domain));
}

function fashionRelevance(candidate) {
  const title = String(candidate.t || candidate.desc || "");
  const positive = [["사복", 40], ["데일리룩", 35], ["코디", 30], ["패션", 24], ["착장", 22], ["전신", 18], ["스타일", 14], ["옷", 10], ["outfit", 24], ["fashion", 18]];
  const negative = [["프로필", -24], ["인터뷰", -18], ["제작발표회", -20], ["시사회", -16], ["포토월", -14]];
  const source = String(candidate.purl || "");
  const sourceBonus = /instagram\.com/i.test(source) ? 120 : /blog\.naver\.com/i.test(source) ? 55 : 0;
  return sourceBonus + [...positive, ...negative].reduce((score, [token, weight]) => title.toLowerCase().includes(token.toLowerCase()) ? score + weight : score, 0);
}

function piece([category, colors, materials, fits, details]) {
  return { category, colors, materials, fits, details };
}

function makeLook(record, influencer, localIndex) {
  const template = templates[influencer.gender][localIndex % templates[influencer.gender].length];
  return {
    id: record.id,
    gender: influencer.gender,
    creator: influencer.name,
    creatorHandle: influencer.handle,
    creatorUrl: `https://www.instagram.com/${influencer.handle.slice(1)}/`,
    credit: `${influencer.name} 공개 룩북 · ${record.sourceDomain}`,
    sourceTitle: `${influencer.name} · ${template.mood}`,
    sourceUrl: record.sourcePageUrl,
    image: record.file,
    published: "공개 사진 룩북",
    publicSpec: `사진 기반 · ${influencer.bodyLabel}`,
    bodyLabel: influencer.bodyLabel,
    bmiRange: influencer.bmiRange,
    bodyShapes: influencer.bodyShapes,
    heightRange: [150, 200],
    weather: ["맑음", "간절기", "약한 비"],
    mood: template.mood,
    styles: influencer.styles,
    summary: template.summary,
    pieces: template.pieces.map(piece)
  };
}

await rm(OUTPUT_DIR, { recursive: true, force: true });
await mkdir(OUTPUT_DIR, { recursive: true });

const records = [];
const looks = [];
const seenHashes = new Set();
const seenUrls = new Set();
const genderCounts = { women: 0, men: 0 };

for (const influencer of influencers) {
  if (genderCounts[influencer.gender] >= 50) continue;
  const handle = influencer.handle.slice(1);
  const queries = [
    `site:instagram.com/${handle} ${handle} outfit`,
    `"${influencer.name}" 패션 사복 전신`,
    `"${influencer.name}" 인스타 패션 코디`,
    `"${influencer.name}" 데일리룩 스타일`,
    `"${influencer.name}" 옷 잘 입는 패션`,
    `"${influencer.name}" 공항패션 전신`,
    `"${influencer.name}" 스트리트 패션 착장`
  ];
  const candidates = [];
  for (const query of queries) {
    for (const first of [1, 36, 71, 106, 141]) {
      try {
        const results = await searchImages(query, first);
        results.forEach(result => candidates.push({ ...result, query }));
      } catch (error) {
        console.warn(`Search retry skipped: ${query} (${error.message})`);
      }
    }
  }
  candidates.sort((left, right) => fashionRelevance(right) - fashionRelevance(left));

  let accepted = 0;
  for (const candidate of candidates) {
    if (accepted >= PER_PERSON || genderCounts[influencer.gender] >= 50) break;
    const originalUrl = safeUrl(candidate.murl);
    const thumbnailUrl = safeUrl(candidate.turl);
    const pageUrl = safeUrl(candidate.purl);
    const resultTitle = String(candidate.t || candidate.desc || "");
    const normalizedTitle = resultTitle.replaceAll(" ", "").toLowerCase();
    const normalizedName = influencer.name.replaceAll(" ", "").toLowerCase();
    if (!normalizedTitle.includes(normalizedName) && !normalizedTitle.includes(handle.toLowerCase())) continue;
    if (!originalUrl || !pageUrl || seenUrls.has(originalUrl) || !isTrustedSource(pageUrl)) continue;
    const width = Number(candidate.mw || candidate.w || 0);
    const height = Number(candidate.mh || candidate.h || 0);
    if (width && height && (width < 300 || height < 430 || height / width < 0.9)) continue;
    if (/youtube\.com|youtu\.be|tiktok\.com|ytimg\.com/i.test(`${pageUrl} ${originalUrl}`)) continue;

    let downloaded;
    let downloadedFrom = originalUrl;
    try {
      downloaded = await downloadImage(originalUrl);
    } catch {
      if (!thumbnailUrl) continue;
      try {
        downloaded = await downloadImage(thumbnailUrl, 3_000);
        downloadedFrom = thumbnailUrl;
      } catch { continue; }
    }
    const hash = createHash("sha256").update(downloaded.bytes).digest("hex");
    if (seenHashes.has(hash)) continue;
    const dimensions = imageDimensions(downloaded.bytes, downloaded.contentType);
    if (dimensions.width && dimensions.height && (dimensions.width < 250 || dimensions.height < 330 || dimensions.height / dimensions.width < 1.08)) continue;

    const number = records.length + 1;
    const ext = imageExtension(downloaded.contentType);
    const filename = `look-${String(number).padStart(3, "0")}.${ext}`;
    const id = `kr-look-${String(number).padStart(3, "0")}`;
    await writeFile(path.join(OUTPUT_DIR, filename), downloaded.bytes);
    const record = {
      id,
      file: `assets/influencers/${filename}`,
      influencer: influencer.name,
      handle: influencer.handle,
      gender: influencer.gender,
      query: candidate.query,
      imageSourceUrl: originalUrl,
      downloadedFrom,
      sourcePageUrl: pageUrl,
      sourceDomain: sourceDomain(pageUrl),
      sourceTitle: resultTitle,
      width: dimensions.width || width || null,
      height: dimensions.height || height || null,
      sha256: hash,
      retrievedAt: new Date().toISOString()
    };
    records.push(record);
    looks.push(makeLook(record, influencer, accepted));
    seenHashes.add(hash);
    seenUrls.add(originalUrl);
    accepted++;
    genderCounts[influencer.gender]++;
    console.log(`${String(records.length).padStart(3, "0")}/100 ${influencer.name} ${accepted}/${PER_PERSON} · ${record.sourceDomain}`);
  }
  if (accepted < 3) console.warn(`${influencer.name}: 제목·출처·세로 비율 검수를 통과한 사진이 ${accepted}장뿐이라 다음 인물로 넘어갑니다.`);
}

if (records.length !== 100 || genderCounts.women !== 50 || genderCounts.men !== 50) {
  throw new Error(`Expected women 50 + men 50, got women ${genderCounts.women} + men ${genderCounts.men}`);
}

await writeFile(path.join(OUTPUT_DIR, "sources.json"), JSON.stringify({
  retrievedAt: new Date().toISOString(),
  count: records.length,
  notice: "프로젝트 데모용 공개 검색 이미지입니다. 각 이미지의 권리는 원저작자에게 있으며 재배포·상업 사용 전 별도 허락이 필요합니다.",
  verification: "인물명·공개 계정명 검색 결과를 기반으로 수집했으며, 앱에서는 원문 페이지와 출처 도메인을 함께 표시합니다.",
  images: records
}, null, 2));

await rm(FINAL_OUTPUT_DIR, { recursive: true, force: true });
await rename(OUTPUT_DIR, FINAL_OUTPUT_DIR);

const js = `(function () {\n  window.WEARWELL_INFLUENCER_LOOKS = ${JSON.stringify(looks, null, 2)};\n})();\n`;
await writeFile(DATA_FILE, js);
console.log(`완료: 한국 패션 인물 ${new Set(records.map(record => record.influencer)).size}명 · 여성 50장 + 남성 50장 = ${records.length}장`);
