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
  groupId?: string;
}

interface LegacyEvaluationExclusions {
  sourceCommit: string;
  qwenPromptGroups: string[];
  openImageIds: string[];
  evaluationIds: string[];
  evaluationImageSha256: string[];
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

async function selectQwen(
  legacy: LegacyEvaluationExclusions,
): Promise<Array<Candidate & { source: string; split: DatasetSplit }>> {
  const config = MODERN_HEAD_DATASET.qwenImageBench;
  const apiUrl = `https://huggingface.co/api/datasets/${config.dataset}/revision/${config.revision}`;
  const response = await fetchWithRetry(apiUrl);
  if (!response.ok) throw new Error(`Qwen Image Bench manifest failed with HTTP ${response.status}`);
  const tree = await response.json() as HuggingFaceTree;
  const allSources = [...config.trainSources, ...config.validationSources, ...config.testSources];
  const candidatesBySource = new Map<string, Map<string, Candidate>>();
  for (const source of allSources) {
      const prefix = `images/${source}/`;
      const candidates = tree.siblings
        .map((entry) => entry.rfilename)
        .filter((name) => name.startsWith(prefix) && /\.(?:jpe?g|png)$/i.test(name))
        .map((name) => {
          const match = /^(\d+)_/u.exec(path.basename(name));
          if (!match) throw new Error(`Qwen image has no stable prompt group: ${name}`);
          return { name, groupId: match[1]!, priority: sha256(`${config.revision}:${name}`) };
        })
        .sort((left, right) => left.priority.localeCompare(right.priority));
      const byGroup = new Map<string, Candidate>();
      for (const candidate of candidates) {
        if (!byGroup.has(candidate.groupId)) byGroup.set(candidate.groupId, candidate);
      }
      candidatesBySource.set(source, byGroup);
  }
  const commonGroups = [...(candidatesBySource.get(allSources[0]!)?.keys() ?? [])]
    .filter((groupId) => allSources.every((source) => candidatesBySource.get(source)?.has(groupId)))
    .sort((left, right) => sha256(`${config.revision}:prompt:${left}`).localeCompare(sha256(`${config.revision}:prompt:${right}`)));
  const requiredGroups = config.trainPerSource + config.validationPerSource + config.testPerSource;
  if (commonGroups.length < requiredGroups) {
    throw new Error(`Qwen sources share only ${commonGroups.length} prompt groups; expected ${requiredGroups}`);
  }
  const legacyPromptGroups = new Set(legacy.qwenPromptGroups);
  const heldOutGroups = commonGroups.filter((groupId) => !legacyPromptGroups.has(groupId)).slice(
    0,
    config.validationPerSource + config.testPerSource,
  );
  if (heldOutGroups.length !== config.validationPerSource + config.testPerSource) {
    throw new Error("Not enough prompt groups remain after excluding the legacy evaluation");
  }
  const heldOutSet = new Set(heldOutGroups);
  const trainingGroups = commonGroups
    .filter((groupId) => !legacyPromptGroups.has(groupId) && !heldOutSet.has(groupId))
    .slice(0, config.trainPerSource);
  if (trainingGroups.length !== config.trainPerSource) {
    throw new Error("Not enough prompt groups remain for training after all evaluation exclusions");
  }
  const splitGroups: Array<{ split: DatasetSplit; sources: readonly string[]; groupIds: string[] }> = [
    { split: "train", sources: config.trainSources, groupIds: trainingGroups },
    {
      split: "validation",
      sources: config.validationSources,
      groupIds: heldOutGroups.slice(0, config.validationPerSource),
    },
    {
      split: "test",
      sources: config.testSources,
      groupIds: heldOutGroups.slice(config.validationPerSource),
    },
  ];
  const selected: Array<Candidate & { source: string; split: DatasetSplit }> = [];
  for (const group of splitGroups) {
    for (const source of group.sources) {
      for (const groupId of group.groupIds) {
        const candidate = candidatesBySource.get(source)?.get(groupId);
        if (!candidate) throw new Error(`Qwen source ${source} is missing prompt group ${groupId}`);
        selected.push({ ...candidate, source, split: group.split });
      }
    }
  }
  return selected;
}

async function selectOpenImages(
  legacy: LegacyEvaluationExclusions,
): Promise<Array<OpenImagesRow & Candidate & { split: DatasetSplit }>> {
  const config = MODERN_HEAD_DATASET.openImages;
  const response = await fetchWithRetry(config.metadataUrl);
  if (!response.ok) throw new Error(`Open Images metadata failed with HTTP ${response.status}`);
  const rows = parseCsv(await response.text());
  const header = rows.shift();
  if (!header) throw new Error("Open Images metadata is empty");
  const headers = new Map(header.map((name, index) => [name, index]));
  const legacyOpenImageIds = new Set(legacy.openImageIds);
  const dataQualityExcludedIds = new Set<string>(config.excludedImageIds);
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
      priority: sha256(`${MODERN_HEAD_DATASET.seed}:v2:${config.revision}:${imageId}`),
    };
  }).filter((row) => row.imageId && !legacyOpenImageIds.has(row.imageId) &&
      !dataQualityExcludedIds.has(row.imageId) &&
      row.license.includes("creativecommons.org/licenses/by/2.0"))
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
    groupId: candidate.groupId,
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
    groupId: candidate.imageId,
  };
}

async function main(): Promise<void> {
  await mkdir(outputDirectory, { recursive: true });
  const legacyBytes = await readFile(path.resolve(MODERN_HEAD_DATASET.legacyEvaluationExclusions));
  const legacy = JSON.parse(legacyBytes.toString("utf8")) as LegacyEvaluationExclusions;
  const [qwen, openImages] = await Promise.all([
    selectQwen(legacy),
    selectOpenImages(legacy),
  ]);
  const qwenItems = await mapConcurrent(qwen, (candidate, index) => downloadQwen(candidate, index, qwen.length));
  const openImageItems = await mapConcurrent(openImages, (candidate, index) => downloadOpenImage(candidate, index, openImages.length));
  const hashOwners = new Map<string, string>();
  for (const item of [...qwenItems, ...openImageItems]) {
    const owner = hashOwners.get(item.imageSha256);
    if (owner) throw new Error(`Duplicate image bytes across selected splits: ${owner} and ${item.id}`);
    hashOwners.set(item.imageSha256, item.id);
  }
  const legacyIds = new Set(legacy.evaluationIds);
  const legacyHashes = new Set(legacy.evaluationImageSha256);
  const legacyOverlap = [...qwenItems, ...openImageItems].find(
    (item) => legacyIds.has(item.id) || legacyHashes.has(item.imageSha256),
  );
  if (legacyOverlap) throw new Error(`New benchmark overlaps the legacy evaluation: ${legacyOverlap.id}`);
  const items = [...qwenItems, ...openImageItems].sort((left, right) => left.id.localeCompare(right.id));
  for (const split of ["train", "validation"] as const) {
    const splitItems = items.filter((item) => item.split === split);
    await writeFile(path.join(outputDirectory, `${split}-manifest.jsonl`), `${splitItems.map((item) => JSON.stringify(item)).join("\n")}\n`);
  }
  const syntheticTestItems = qwenItems.filter((item) => item.split === "test");
  await writeFile(
    path.join(outputDirectory, "test-synthetic-manifest.jsonl"),
    `${syntheticTestItems.map((item) => JSON.stringify(item)).join("\n")}\n`,
  );
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
    strategy: "Pinned prompt groups and generator families are disjoint across train, validation, and the synthetic confirmatory test; exact duplicate bytes across selected splits are rejected",
    legacyEvaluationExclusions: {
      path: MODERN_HEAD_DATASET.legacyEvaluationExclusions,
      sha256: sha256(legacyBytes),
      sourceCommit: legacy.sourceCommit,
      overlapIds: 0,
      overlapImageSha256: 0,
    },
    dataQualityExclusions: {
      openImageIds: [...MODERN_HEAD_DATASET.openImages.excludedImageIds],
      rationale: "Near-black or otherwise non-informative visual frames are excluded before model training or validation",
    },
    counts: {
      train: items.filter((item) => item.split === "train").length,
      validation: items.filter((item) => item.split === "validation").length,
      testSynthetic: syntheticTestItems.length,
      synthetic: qwenItems.length,
      real: openImageItems.length,
    },
  }, null, 2)}\n`);
  console.log(`Prepared ${items.length} images in ${outputDirectory}`);
}

await main();
