import { createHash } from "node:crypto";
import { readFile, mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { MODERN_HEAD_DATASET } from "./config";
import type { BenchmarkItem } from "../types";

interface HuggingFaceTree {
  siblings: Array<{ rfilename: string }>;
}

interface OpenImagesRow {
  imageId: string;
  originalUrl: string;
  landingUrl: string;
  license: string;
  authorProfileUrl: string;
  author: string;
  title: string;
  rotation: string;
}

interface Candidate {
  name: string;
  priority: string;
}

type DatasetSplit = "train" | "validation" | "test";

const outputDirectory = path.resolve(MODERN_HEAD_DATASET.outputDirectory);
const concurrency = 8;

function sha256(value: Uint8Array | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchWithRetry(url: string, attempts = 8): Promise<Response> {
  let lastStatus = 0;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(60_000) });
      if (response.ok) return response;
      lastStatus = response.status;
      if (response.status !== 429 && response.status < 500) return response;
    } catch (error) {
      if (attempt + 1 === attempts) throw error;
    }
    await delay(Math.min(750 * 2 ** attempt, 20_000));
  }
  throw new Error(`Request failed after ${attempts} attempts (last HTTP ${lastStatus})`);
}

function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index]!;
    if (quoted) {
      if (character === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function value(row: string[], headers: Map<string, number>, name: string): string {
  const index = headers.get(name);
  if (index === undefined) throw new Error(`Open Images metadata is missing ${name}`);
  return row[index] ?? "";
}

async function readAuditExclusions(): Promise<{ names: Set<string>; hashes: Set<string> }> {
  const manifest = await readFile(path.resolve(MODERN_HEAD_DATASET.auditExclusionsManifest), "utf8").catch(() => "");
  const items = manifest.trim().split("\n").filter(Boolean).map((line) => JSON.parse(line) as BenchmarkItem);
  return {
    names: new Set(items.map((item) => path.basename(item.path))),
    hashes: new Set(items.map((item) => item.imageSha256)),
  };
}

async function selectQwen(excludedNames: Set<string>): Promise<Array<Candidate & { source: string; split: DatasetSplit }>> {
  const config = MODERN_HEAD_DATASET.qwenImageBench;
  const apiUrl = `https://huggingface.co/api/datasets/${config.dataset}/revision/${config.revision}`;
  const response = await fetchWithRetry(apiUrl);
  if (!response.ok) throw new Error(`Qwen Image Bench manifest failed with HTTP ${response.status}`);
  const tree = await response.json() as HuggingFaceTree;
  const selected: Array<Candidate & { source: string; split: DatasetSplit }> = [];
  const groups: Array<{ split: DatasetSplit; target: number; offset: number; sources: readonly string[] }> = [
    { split: "train", target: config.trainPerSource, offset: 0, sources: config.trainSources },
    { split: "validation", target: config.validationPerSource, offset: 0, sources: config.validationSources },
    { split: "test", target: config.testPerSource, offset: config.validationPerSource, sources: config.validationSources },
  ];
  for (const group of groups) {
    for (const source of group.sources) {
      const prefix = `images/${source}/`;
      const candidates = tree.siblings
        .map((entry) => entry.rfilename)
        .filter((name) => name.startsWith(prefix) && /\.(?:jpe?g|png)$/i.test(name) && !excludedNames.has(path.basename(name)))
        .map((name) => ({ name, priority: sha256(`${config.revision}:${name}`) }))
        .sort((left, right) => left.priority.localeCompare(right.priority));
      if (candidates.length < group.offset + group.target) {
        throw new Error(`${source} has only ${candidates.length} eligible images; expected ${group.offset + group.target}`);
      }
      selected.push(...candidates.slice(group.offset, group.offset + group.target).map((candidate) => ({ ...candidate, source, split: group.split })));
    }
  }
  return selected;
}

async function selectOpenImages(): Promise<Array<OpenImagesRow & Candidate & { split: DatasetSplit }>> {
  const config = MODERN_HEAD_DATASET.openImages;
  const response = await fetchWithRetry(config.metadataUrl);
  if (!response.ok) throw new Error(`Open Images metadata failed with HTTP ${response.status}`);
  const rows = parseCsv(await response.text());
  const header = rows.shift();
  if (!header) throw new Error("Open Images metadata is empty");
  const headers = new Map(header.map((name, index) => [name, index]));
  const eligible = rows.map((row): OpenImagesRow & Candidate => {
    const imageId = value(row, headers, "ImageID");
    return {
      imageId,
      originalUrl: value(row, headers, "OriginalURL"),
      landingUrl: value(row, headers, "OriginalLandingURL"),
      license: value(row, headers, "License"),
      authorProfileUrl: value(row, headers, "AuthorProfileURL"),
      author: value(row, headers, "Author"),
      title: value(row, headers, "Title"),
      rotation: value(row, headers, "Rotation"),
      name: `${imageId}.jpg`,
      priority: sha256(`${config.revision}:${imageId}`),
    };
  }).filter((row) => row.imageId && row.license.includes("creativecommons.org/licenses/by/2.0"))
    .sort((left, right) => left.priority.localeCompare(right.priority));
  const total = config.trainCount + config.validationCount + config.testCount;
  if (eligible.length < total) throw new Error(`Open Images has only ${eligible.length} eligible rows; expected ${total}`);
  return eligible.slice(0, total).map((row, index) => ({
    ...row,
    split: index < config.trainCount
      ? "train"
      : index < config.trainCount + config.validationCount
      ? "validation"
      : "test",
  }));
}

async function mapConcurrent<T, R>(items: readonly T[], worker: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: concurrency }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await worker(items[index]!, index);
    }
  }));
  return results;
}

async function downloadQwen(
  candidate: Candidate & { source: string; split: DatasetSplit },
  index: number,
  total: number,
): Promise<BenchmarkItem> {
  const config = MODERN_HEAD_DATASET.qwenImageBench;
  const url = `https://huggingface.co/datasets/${config.dataset}/resolve/${config.revision}/${candidate.name}?download=true`;
  const relativePath = `${candidate.split}/synthetic/${candidate.source}/${path.basename(candidate.name)}`;
  const absolutePath = path.join(outputDirectory, relativePath);
  const bytes = await readFile(absolutePath).catch(async () => {
    const response = await fetchWithRetry(url);
    if (!response.ok) throw new Error(`Qwen image ${candidate.name} failed with HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  });
  await mkdir(path.join(outputDirectory, path.dirname(relativePath)), { recursive: true });
  await writeFile(absolutePath, bytes);
  if ((index + 1) % 50 === 0 || index + 1 === total) console.log(`Downloaded Qwen ${index + 1}/${total}`);
  return {
    id: `qwen-image-bench:${config.revision}:${candidate.source}:${path.basename(candidate.name)}`,
    dataset: config.dataset,
    datasetRevision: config.revision,
    split: candidate.split,
    rowIndex: index,
    path: relativePath,
    imageSha256: sha256(bytes),
    label: 1,
    source: candidate.source,
  };
}

async function downloadOpenImage(
  candidate: OpenImagesRow & Candidate & { split: DatasetSplit },
  index: number,
  total: number,
): Promise<BenchmarkItem> {
  const config = MODERN_HEAD_DATASET.openImages;
  const relativePath = `${candidate.split}/real/open-images/${candidate.name}`;
  const absolutePath = path.join(outputDirectory, relativePath);
  const bytes = await readFile(absolutePath).catch(async () => {
    const response = await fetchWithRetry(`${config.imageBaseUrl}/${candidate.name}`);
    if (!response.ok) throw new Error(`Open Images image ${candidate.imageId} failed with HTTP ${response.status}`);
    return Buffer.from(await response.arrayBuffer());
  });
  await mkdir(path.join(outputDirectory, path.dirname(relativePath)), { recursive: true });
  await writeFile(absolutePath, bytes);
  if ((index + 1) % 50 === 0 || index + 1 === total) console.log(`Downloaded Open Images ${index + 1}/${total}`);
  return {
    id: `open-images:${config.revision}:validation:${candidate.imageId}`,
    dataset: config.dataset,
    datasetRevision: config.revision,
    split: candidate.split,
    rowIndex: index,
    path: relativePath,
    imageSha256: sha256(bytes),
    label: 0,
    source: "open-images",
  };
}

async function main(): Promise<void> {
  await mkdir(outputDirectory, { recursive: true });
  const exclusions = await readAuditExclusions();
  const [qwen, openImages] = await Promise.all([
    selectQwen(exclusions.names),
    selectOpenImages(),
  ]);
  const qwenItems = await mapConcurrent(qwen, (candidate, index) => downloadQwen(candidate, index, qwen.length));
  const openImageItems = await mapConcurrent(openImages, (candidate, index) => downloadOpenImage(candidate, index, openImages.length));
  for (const item of [...qwenItems, ...openImageItems]) {
    if (exclusions.hashes.has(item.imageSha256)) throw new Error(`Audit image leaked into training data: ${item.id}`);
  }
  const items = [...qwenItems, ...openImageItems].sort((left, right) => left.id.localeCompare(right.id));
  for (const split of ["train", "validation", "test"] as const) {
    const splitItems = items.filter((item) => item.split === split);
    await writeFile(path.join(outputDirectory, `${split}-manifest.jsonl`), `${splitItems.map((item) => JSON.stringify(item)).join("\n")}\n`);
  }
  const attribution = openImages.map((row) => ({
    imageId: row.imageId,
    split: row.split,
    license: row.license,
    author: row.author,
    authorProfileUrl: row.authorProfileUrl,
    title: row.title,
    originalUrl: row.originalUrl,
    landingUrl: row.landingUrl,
    rotation: row.rotation,
  }));
  await writeFile(path.join(outputDirectory, "open-images-attribution.json"), `${JSON.stringify(attribution, null, 2)}\n`);
  await writeFile(path.join(outputDirectory, "selection.json"), `${JSON.stringify({
    ...MODERN_HEAD_DATASET,
    strategy: "Lowest SHA-256 priority within each pinned source; validation holds out entire generator families; exposed frontier audit names and hashes excluded",
    counts: {
      train: items.filter((item) => item.split === "train").length,
      validation: items.filter((item) => item.split === "validation").length,
      test: items.filter((item) => item.split === "test").length,
      synthetic: qwenItems.length,
      real: openImageItems.length,
    },
  }, null, 2)}\n`);
  console.log(`Prepared ${items.length} images in ${outputDirectory}`);
}

await main();
