import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";


const CURRENT_M2_START = "<!-- PROOFLENS_CURRENT_M2_START -->";
const CURRENT_M2_END = "<!-- PROOFLENS_CURRENT_M2_END -->";
const CURRENT_M4_START = "<!-- PROOFLENS_CURRENT_M4_START -->";
const CURRENT_M4_END = "<!-- PROOFLENS_CURRENT_M4_END -->";
const HISTORICAL_M2_START = "<!-- PROOFLENS_HISTORICAL_M2_START -->";
const HISTORICAL_M2_END = "<!-- PROOFLENS_HISTORICAL_M2_END -->";
const VARIANTS = ["original", "screenshot", "social-q75", "social-heavy"];

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function replaceExact(value, before, after, label) {
  requireCondition(value.split(before).length === 2, `${label} source text changed`);
  return value.replace(before, after);
}

function replaceHeadingSection(value, startHeading, endHeading, replacement, label) {
  const start = value.indexOf(startHeading);
  const end = value.indexOf(endHeading, start + startHeading.length);
  requireCondition(start >= 0 && end > start && value.indexOf(startHeading, start + 1) < 0,
    `${label} heading boundary changed`);
  return value.slice(0, start) + replacement.trimEnd() + "\n\n" + value.slice(end);
}

function replaceCurrentM2(value, currentM4, historicalHeading, label) {
  const start = value.indexOf(CURRENT_M2_START);
  const end = value.indexOf(CURRENT_M2_END);
  requireCondition(start >= 0 && end > start && value.indexOf(CURRENT_M2_START, start + 1) < 0 &&
    value.indexOf(CURRENT_M2_END, end + 1) < 0, `${label} current-M2 markers changed`);
  const oldBody = value.slice(start + CURRENT_M2_START.length, end).trim();
  const historical = oldBody.replace(/^## Current M2[^\n]*/u, historicalHeading) +
    "\n\nM2 remains immutable development history. The subsequent M3 experiment failed its fresh selector and published no model. V1 and replacement-v2 remain consumed and acceptance-ineligible.";
  const replacement = [
    CURRENT_M4_START,
    currentM4,
    CURRENT_M4_END,
    "",
    HISTORICAL_M2_START,
    historical,
    HISTORICAL_M2_END,
  ].join("\n");
  return value.slice(0, start) + replacement + value.slice(end + CURRENT_M2_END.length);
}

function percent(value) {
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function metricTable(metrics) {
  const labels = {
    original: "Original",
    screenshot: "Screenshot",
    "social-q75": "JPEG 75",
    "social-heavy": "Heavy double-JPEG",
  };
  const rows = VARIANTS.map((variant) => {
    const row = metrics[variant];
    const family = Math.min(...Object.values(row.syntheticRecallBySource));
    const realSource = Math.min(...Object.values(row.realRecallBySource));
    return `| ${labels[variant]} | ${percent(row.balancedAccuracy)} | ${percent(row.realRecall)} | ` +
      `${percent(row.syntheticRecall)} | ${percent(family)} | ${percent(realSource)} |`;
  });
  return [
    "| View | Balanced accuracy | Non-AI recall | Synthetic recall | Worst synthetic source | Worst required real source |",
    "|---|---:|---:|---:|---:|---:|",
    ...rows,
  ].join("\n");
}

function currentSections(summary, modelSha256, modelBytes) {
  const selected = summary.selectedCandidate;
  const firstRegression = summary.regressions[0];
  const secondRegression = summary.regressions[1];
  requireCondition(summary.status === "accepted-development-candidate" &&
    firstRegression.name === "m3-selector-regression" && secondRegression.name === "m2-development-regression" &&
    firstRegression.passed === true && secondRegression.passed === true && summary.h3HoldoutScored === false,
  "M4 documentation input is not an accepted development candidate");
  const readme = `## Current M4 model

The shipped model keeps the Community Forensics ViT-S/16 backbone and the M2 classifier frozen. M4 inserts a 49,600-parameter residual adapter at the 384-dimensional feature tensor. The upstream backbone was trained on 5.4 million paired real/synthetic examples spanning 4,803 generators.

M4 training used **112,698 public images** and 150,792 feature views: the complete 108,378-image M3 training pool plus 2,400 British Library plate images and 1,920 Rapidata synthetic images across DALL-E 3, FLUX, Midjourney, and Stable Diffusion family labels. The source cards report Public Domain Mark/no known restrictions for the British Library corpus and CDLA-Permissive-2.0 for Rapidata. Rapidata does not supply exact generator revisions or seeds, so its rows are development-only evidence.

The fresh 600-image British Library/Rapidata selector alone chose candidate \`${selected.candidateId}\` and its raw threshold. The consumed M3 and M2 development packets then passed once in that order as terminal regressions; neither could trigger reselection. These are development results, not an untouched generalization estimate or a bounty score. H3 remains unscored and unread.

The packaged model is ${modelBytes.toLocaleString("en-US")} bytes with SHA-256 \`${modelSha256}\`. Independent reconstruction proves that all M2 backbone and classifier initializers remain byte-identical; M4 adds exactly six adapter/normalization initializers and seven graph nodes. The pixel-free training receipt, source packet, candidate grid, calibration, and adapter comparison are under \`benchmark/evidence/m4/\`.`;
  const modelCard = `## Current M4 adapter-training data

The shipped artifact is ${modelBytes.toLocaleString("en-US")} bytes with SHA-256 \`${modelSha256}\`. It preserves every M2 backbone and classifier initializer byte-for-byte and inserts one residual adapter: normalize the 384-dimensional feature, apply 384→64 ReLU→384, scale by the training standard deviation, add the residual, then run the frozen classifier.

M4 trained on 112,698 unique public images and 150,792 feature views. It retains the full M3 training pool and adds 2,400 book-disjoint British Library plate images plus 1,920 Rapidata synthetic images, 480 for each of four publisher-reported model families. Selector rows never enter gradients. Exact IDs, revisions, byte hashes, source groups, dHashes, attribution, and deterministic selection evidence are committed without source pixels.

The fresh selector contains 300 different British Library books and 300 Rapidata images from 75 prompt groups. It alone selected candidate \`${selected.candidateId}\` and the raw threshold. Candidate tensors were sealed before selector evaluation; the winner and threshold were sealed before the ordered M3 and M2 regressions. Both regressions passed without reselection. H3 remains untouched.

The British Library source describes the collection as public-domain/no-known-restrictions material, but the plates configuration is an algorithmic page-layout class rather than a guarantee about depicted rights or visual type. Rapidata reports CDLA-Permissive-2.0 and model-family provenance but omits exact generator revisions and seeds. These sources improve development coverage; they do not establish ordinary-web generalization or acceptance.`;
  const benchmark = `## Current M4 model-selection boundary

The Community Forensics ViT-S/16 backbone and M2 classifier are frozen. M4 trains only a 49,600-parameter residual feature adapter. Training uses 112,698 images and 150,792 views. Its two additions are 2,400 British Library plate images and 1,920 Rapidata synthetic images across four publisher-reported model families.

Candidate tensors for the fixed 12-member grid were sealed before selector evaluation. Only the fresh 600-image British Library/Rapidata selector could choose the candidate and threshold. The selected candidate \`${selected.candidateId}\` then ran the consumed M3 and M2 packets once, in that order, as terminal regressions. A regression failure could not select another candidate, alter the threshold, or change a gate. H3 remains unscored and unread.

### Fresh M4 selector

${metricTable(selected.selectorMetrics)}

### Consumed M3 terminal regression

${metricTable(firstRegression.metrics)}

### Consumed M2 terminal regression

${metricTable(secondRegression.metrics)}

All three tables are development evidence. They are not an untouched generalization estimate, an acceptance result, or a bounty score.`;
  return { readme, modelCard, benchmark };
}

export function renderM4PublicDocuments({ readme, modelCard, benchmark, summary, modelSha256, modelBytes }) {
  const sections = currentSections(summary, modelSha256, modelBytes);
  let nextReadme = replaceCurrentM2(readme, sections.readme, "## Historical M2 model", "README");
  nextReadme = replaceExact(nextReadme,
    "npm run verify:static      # current M2 model, evidence, package, and lineage checks",
    "npm run verify:static      # current M4 model, evidence, package, and lineage checks",
    "README verification command");
  nextReadme = replaceExact(nextReadme,
    "Keep `.verify-venv` active for `verify:static`. The command detects the repository stage and, on the M2 publication, requires the exact public lineage, training packet, model lock, classifier-only comparison, deterministic package, and current documentation. GitHub Actions runs that pixel-free contract and the portable forced-WASM browser path. WebGPU remains a separate fixed-head local gate because hosted-runner GPU availability is not stable. The older `verify:release` script replays the consumed replacement-v2/M1 packet from its historical checkout; it is not an M2 acceptance test.",
    "Keep `.verify-venv` active for `verify:static`. The command detects the repository stage and, on the M4 publication, requires the exact protocol, source, output-lock, and final-publication lineage; fresh-feature packet; selector-only winner; ordered terminal regressions; model lock; adapter reconstruction; deterministic package; documentation; and QA fixtures. GitHub Actions runs that pixel-free contract and the portable forced-WASM browser path. WebGPU remains a separate fixed-head local gate because hosted-runner GPU availability is not stable. The older `verify:release` script replays the consumed replacement-v2/M1 packet from its historical checkout; it is not an M4 acceptance test.",
    "README verification explanation");
  nextReadme = replaceExact(nextReadme,
    "- Bytes: `87,442,080`", `- Bytes: \`${modelBytes.toLocaleString("en-US")}\``, "README model bytes");
  nextReadme = replaceExact(nextReadme,
    "- SHA-256: `a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47`",
    `- SHA-256: \`${modelSha256}\``, "README model hash");
  nextReadme = replaceExact(nextReadme,
    "The complete machine-readable contract is [model-lock.json](model-lock.json). Training replaces only the frozen backbone’s 384-to-1 classifier head; it does not add a second model or heuristic score.",
    "The complete machine-readable contract is [model-lock.json](model-lock.json). M4 preserves the frozen backbone and classifier and inserts one local residual feature adapter; it does not add a second model, remote call, or heuristic score.",
    "README model contract");

  let nextModelCard = replaceCurrentM2(modelCard, sections.modelCard,
    "## Historical M2 head-training data", "MODEL_CARD");
  nextModelCard = replaceExact(nextModelCard,
    "ProofLens freezes that backbone and trains only `classifier.weight [1,384]` and `classifier.bias [1]`.",
    "ProofLens freezes that backbone and the current M2 classifier. M4 trains one residual 384→64→384 feature adapter and leaves the classifier tensors unchanged.",
    "MODEL_CARD architecture");
  const evaluation = `## Evaluation boundary

The fresh M4 selector contains 300 British Library non-AI images from different books and 300 Rapidata synthetic images across four publisher-reported families. It alone selected the M4 candidate and raw threshold.

${metricTable(summary.selectedCandidate.selectorMetrics)}

The consumed M3 selector and M2 development packets then ran once as ordered terminal regressions. Their metrics are recorded in [BENCHMARK.md](BENCHMARK.md). They could reject the already-selected candidate but could not change the candidate, threshold, gates, training, or ranking.

These are development results. The separately reserved H3 packet remains unscored and unread.`;
  nextModelCard = replaceHeadingSection(nextModelCard, "## Evaluation boundary", "## Intended use",
    evaluation, "MODEL_CARD evaluation");

  let nextBenchmark = replaceCurrentM2(benchmark, sections.benchmark,
    "## Historical M2 model-selection boundary", "BENCHMARK");
  nextBenchmark = replaceExact(nextBenchmark,
    "M2 responds only to that observed false-positive category. It keeps the backbone, preprocessing, fixed 65/100 display threshold, candidate grid, and external acceptance gates unchanged. Consumed v1/v2 rows are excluded from M2 gradients and development metrics.",
    "M4 is a bounded development response to the Met-versus-FLUX conflict recorded by the failed M3 experiment. It changes candidate generation by adding a residual feature adapter while preserving the backbone, M2 classifier, preprocessing, fixed 65/100 display threshold, and future H3 boundary.",
    "BENCHMARK M4 purpose");
  nextBenchmark = replaceExact(nextBenchmark,
    "## Current M2 training and development splits",
    `## Current M4 training and development splits

| Split | Images | Role | SHA-256 |
|---|---:|---|---|
| M4 training manifest | 112,698 | residual-adapter training only | \`${summary.trainManifestSha256}\` |
| M4 fresh selector | 600 | candidate and threshold selection only | \`${summary.selectorManifestSha256}\` |
| Consumed M3 packet | 600 | first terminal post-selection regression | \`${summary.m3RegressionManifestSha256}\` |
| Consumed M2 packet | 900 | second terminal post-selection regression | \`${summary.m2RegressionManifestSha256}\` |

## Historical M2 training and development splits`,
    "BENCHMARK split heading");
  const currentResults = `## Current results

M4 adapter training, fresh-selector selection, and both terminal development regressions are complete. The current M4 section records their metrics. They are development evidence, not an untouched generalization estimate and not an acceptance result.

The reserved H3 packet remains unscored and unread. Public repository evidence does not establish the bounty maintainer's private score, acceptance decision, or payment.`;
  nextBenchmark = replaceHeadingSection(nextBenchmark, "## Current results", "## Browser evidence",
    currentResults, "BENCHMARK current results");

  for (const [label, value] of [["README", nextReadme], ["MODEL_CARD", nextModelCard], ["BENCHMARK", nextBenchmark]]) {
    requireCondition(!value.includes(CURRENT_M2_START) && !value.includes(CURRENT_M2_END),
      `${label} retains current-M2 markers after rendering`);
    requireCondition(value.split(CURRENT_M4_START).length === 2 && value.split(CURRENT_M4_END).length === 2,
      `${label} current-M4 marker count changed`);
    requireCondition(value.endsWith("\n"), `${label} must retain a final newline`);
  }
  return { README: nextReadme, MODEL_CARD: nextModelCard, BENCHMARK: nextBenchmark };
}

function parseArguments(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    requireCondition(argv[index]?.startsWith("--") && argv[index + 1], "Renderer arguments must be flag/value pairs");
    values.set(argv[index], argv[index + 1]);
  }
  for (const name of ["--summary", "--model", "--output-dir"]) {
    requireCondition(values.has(name), `Missing ${name}`);
  }
  return values;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const arguments_ = parseArguments(process.argv.slice(2));
  const [summaryBytes, modelBytes, readme, modelCard, benchmark] = await Promise.all([
    readFile(arguments_.get("--summary")),
    readFile(arguments_.get("--model")),
    readFile("README.md", "utf8"),
    readFile("MODEL_CARD.md", "utf8"),
    readFile("BENCHMARK.md", "utf8"),
  ]);
  const summary = JSON.parse(summaryBytes);
  const modelSha256 = createHash("sha256").update(modelBytes).digest("hex");
  const rendered = renderM4PublicDocuments({
    readme, modelCard, benchmark, summary, modelSha256, modelBytes: modelBytes.length,
  });
  const outputDir = arguments_.get("--output-dir");
  await mkdir(outputDir, { recursive: true });
  await Promise.all(Object.entries(rendered).map(([name, value]) =>
    writeFile(path.join(outputDir, `${name}.md`), value)));
  console.log(JSON.stringify({ modelSha256, modelBytes: modelBytes.length, outputDir, policy: "pass" }));
}
