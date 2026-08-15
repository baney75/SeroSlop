import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { expectedM3CurrentSections, M3_VARIANTS } from "./m3-training-contract.mjs";


const CURRENT_M2_START = "<!-- PROOFLENS_CURRENT_M2_START -->";
const CURRENT_M2_END = "<!-- PROOFLENS_CURRENT_M2_END -->";
const CURRENT_M3_START = "<!-- PROOFLENS_CURRENT_M3_START -->";
const CURRENT_M3_END = "<!-- PROOFLENS_CURRENT_M3_END -->";
const HISTORICAL_M2_START = "<!-- PROOFLENS_HISTORICAL_M2_START -->";
const HISTORICAL_M2_END = "<!-- PROOFLENS_HISTORICAL_M2_END -->";

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

function replaceCurrentM2(value, currentM3, historicalHeading, label) {
  const start = value.indexOf(CURRENT_M2_START);
  const end = value.indexOf(CURRENT_M2_END);
  requireCondition(start >= 0 && end > start && value.indexOf(CURRENT_M2_START, start + 1) < 0 &&
    value.indexOf(CURRENT_M2_END, end + 1) < 0, `${label} current-M2 markers changed`);
  const oldBody = value.slice(start + CURRENT_M2_START.length, end).trim();
  const historical = oldBody.replace(/^## Current M2[^\n]*/u, historicalHeading) +
    "\n\nM2 remains immutable development history. The v1 and replacement-v2 evaluation records are consumed and carry `acceptanceEligible: false`; M2 has no untouched acceptance result.";
  const replacement = [
    CURRENT_M3_START,
    currentM3,
    CURRENT_M3_END,
    "",
    HISTORICAL_M2_START,
    historical,
    HISTORICAL_M2_END,
  ].join("\n");
  return value.slice(0, start) + replacement + value.slice(end + CURRENT_M2_END.length);
}

function metricTable(summary, key) {
  const labels = {
    original: "Original",
    screenshot: "Screenshot",
    "social-q75": "JPEG 75",
    "social-heavy": "Heavy double-JPEG",
  };
  const rows = M3_VARIANTS.map((variant) => {
    const metrics = summary[key].variants[variant];
    const family = Math.min(...Object.values(metrics.syntheticRecallBySource));
    const realSource = Math.min(...Object.values(metrics.realRecallBySource));
    return `| ${labels[variant]} | ${(metrics.balancedAccuracy * 100).toFixed(2)}% | ` +
      `${(metrics.realRecall * 100).toFixed(2)}% | ${(metrics.syntheticRecall * 100).toFixed(2)}% | ` +
      `${(family * 100).toFixed(2)}% | ${(realSource * 100).toFixed(2)}% |`;
  });
  return [
    "| View | Balanced accuracy | Non-AI recall | Synthetic recall | Worst synthetic source | Worst required real source |",
    "|---|---:|---:|---:|---:|---:|",
    ...rows,
  ].join("\n");
}

export function renderM3PublicDocuments({ readme, modelCard, benchmark, summary, modelSha256, modelBytes }) {
  const sections = expectedM3CurrentSections({ summary, modelSha256 });
  let nextReadme = replaceCurrentM2(readme, sections.README, "## Historical M2 model", "README");
  nextReadme = replaceExact(
    nextReadme,
    "npm run verify:static      # current M2 model, evidence, package, and lineage checks",
    "npm run verify:static      # current M3 model, evidence, package, and lineage checks",
    "README verification command",
  );
  nextReadme = replaceExact(
    nextReadme,
    "Keep `.verify-venv` active for `verify:static`. The command detects the repository stage and, on the M2 publication, requires the exact public lineage, training packet, model lock, classifier-only comparison, deterministic package, and current documentation. GitHub Actions runs that pixel-free contract and the portable forced-WASM browser path. WebGPU remains a separate fixed-head local gate because hosted-runner GPU availability is not stable. The older `verify:release` script replays the consumed replacement-v2/M1 packet from its historical checkout; it is not an M2 acceptance test.",
    "Keep `.verify-venv` active for `verify:static`. The command detects the repository stage and, on the M3 publication, requires the exact source, output-lock, and final-publication lineage; training packet; post-selection regression; model lock; classifier-only comparison; deterministic package; documentation; and browser fixtures. GitHub Actions runs that pixel-free contract and the portable forced-WASM browser path. WebGPU remains a separate fixed-head local gate because hosted-runner GPU availability is not stable. The older `verify:release` script replays the consumed replacement-v2/M1 packet from its historical checkout; it is not an M3 acceptance test.",
    "README verification explanation",
  );
  nextReadme = replaceExact(
    nextReadme,
    "- SHA-256: `a994b1bd4d0323909b2b308db848bf668fd00e2f02c8973ec546c400efe2dc47`",
    `- SHA-256: \`${modelSha256}\``,
    "README model lock",
  );
  nextReadme = replaceExact(
    nextReadme,
    "- Bytes: `87,442,080`",
    `- Bytes: \`${modelBytes.toLocaleString("en-US")}\``,
    "README model bytes",
  );

  let nextModelCard = replaceCurrentM2(
    modelCard, sections.MODEL_CARD, "## Historical M2 head-training data", "MODEL_CARD");
  const evaluationBoundary = `## Evaluation boundary

The 600-image fresh selector contains 300 Met Open Access non-AI images and 300 FLUX.1-dev synthetic images. It alone selected the M3 candidate and raw threshold.

${metricTable(summary, "selector")}

After selection was frozen, the consumed 900-image M2 development packet ran once as a post-selection regression gate:

${metricTable(summary, "regression")}

Both tables are development evidence. They are not an untouched generalization estimate and not an acceptance result. The separately reserved H3 packet remains unscored.`;
  nextModelCard = replaceHeadingSection(
    nextModelCard, "## Evaluation boundary", "## Intended use", evaluationBoundary, "MODEL_CARD evaluation");
  nextModelCard = replaceExact(
    nextModelCard,
    "- Open Images, DOCCI, Library of Congress, and StockImages photographs do not span every real-web visual type.",
    "- Open Images, DOCCI, Library of Congress, StockImages, and Met Open Access images do not span every real-web visual type.",
    "MODEL_CARD limitation",
  );

  let nextBenchmark = replaceCurrentM2(
    benchmark, sections.BENCHMARK, "## Historical M2 model-selection boundary", "BENCHMARK");
  nextBenchmark = replaceExact(
    nextBenchmark,
    "M2 responds only to that observed false-positive category. It keeps the backbone, preprocessing, fixed 65/100 display threshold, candidate grid, and external acceptance gates unchanged. Consumed v1/v2 rows are excluded from M2 gradients and development metrics.",
    "M3 responds only to the museum-image false-positive category measured by a consumed 100-item Met development probe. It preserves the frozen backbone, preprocessing, fixed 65/100 display threshold, 25-pair head grid, consumed-evaluation exclusions, and future H3 boundary.",
    "BENCHMARK M3 purpose",
  );
  nextBenchmark = replaceExact(
    nextBenchmark,
    "## Current M2 training and development splits",
    `## Current M3 training and development splits

| Split | Images | Role | SHA-256 |
|---|---:|---|---|
| M3 training manifest | 108,378 | classifier-head training only | \`${summary.trainManifestSha256}\` |
| M3 fresh selector | 600 | candidate and threshold selection only | \`${summary.validationManifestSha256}\` |
| M2 regression packet | 900 | post-selection failure gate only | \`${summary.regressionManifestSha256}\` |

## Historical M2 training and development splits`,
    "BENCHMARK split heading",
  );
  const currentResults = `## Current results

M3 head training, selector evaluation, and the post-selection M2 regression gate are complete. The selector results are recorded in the current M3 section above. They are development-selection evidence, not an untouched generalization estimate and not an acceptance result.

The reserved H3 packet remains unscored. Public repository evidence does not establish the bounty maintainer's private score, acceptance decision, or payment.`;
  nextBenchmark = replaceHeadingSection(
    nextBenchmark, "## Current results", "## Browser evidence", currentResults, "BENCHMARK current results");
  nextBenchmark = replaceExact(
    nextBenchmark,
    "Dataset pixels are excluded from Git. Public IDs, revisions, byte hashes, source-reported licenses, attribution, selection code, predictions, and aggregate evidence are committed. Full reconstruction requires about 106 GB plus source archives. DiffusionDB supplies scale but overrepresents an older Stable Diffusion era. Validation covers two modern generator families, and replacement confirmation covers one unseen family; neither represents every future generator. StockImages is one real-photo corpus, not the full ordinary web. Illustrations, CGI, charts, memes, screenshots, scans, and edited photographs remain important false-positive risks.",
    "Dataset pixels are excluded from Git. Public IDs, revisions, byte hashes, source-reported licenses, attribution, selection code, and aggregate evidence are committed. Full reconstruction requires the local source corpus and archives. DiffusionDB supplies scale but overrepresents an older Stable Diffusion era. The M3 selector covers one FLUX dataset and one Met source; the consumed regression covers earlier development sources. None represents every future generator or ordinary-web visual type. Illustrations, CGI, charts, memes, screenshots, scans, and edited photographs remain important false-positive risks.",
    "BENCHMARK reproduction limits",
  );

  for (const [label, value] of [["README", nextReadme], ["MODEL_CARD", nextModelCard], ["BENCHMARK", nextBenchmark]]) {
    requireCondition(!value.includes(CURRENT_M2_START) && !value.includes(CURRENT_M2_END),
      `${label} retains current-M2 markers after rendering`);
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
  const rendered = renderM3PublicDocuments({
    readme, modelCard, benchmark, summary, modelSha256, modelBytes: modelBytes.length,
  });
  const outputDir = arguments_.get("--output-dir");
  await mkdir(outputDir, { recursive: true });
  await Promise.all(Object.entries(rendered).map(([name, value]) =>
    writeFile(path.join(outputDir, `${name}.md`), value)));
  console.log(JSON.stringify({ modelSha256, modelBytes: modelBytes.length, outputDir, policy: "pass" }));
}
