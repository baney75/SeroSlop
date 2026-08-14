import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import { inspectOnnxStructure } from "./onnx-structure.mjs";


const [modelPath, expectedSha256, outputPath] = process.argv.slice(2);
if (!modelPath || !expectedSha256 || !outputPath) {
  throw new Error("usage: node scripts/extract-upstream-model-structure.mjs MODEL EXPECTED_SHA256 OUTPUT");
}
const bytes = await readFile(modelPath);
const sha256 = createHash("sha256").update(bytes).digest("hex");
if (sha256 !== expectedSha256) throw new Error(`Unexpected upstream model SHA-256: ${sha256}`);
const evidence = {
  schemaVersion: 1,
  model: { sha256, bytes: bytes.length },
  protobufReader: "onnxruntime-web@1.22.0 generated ONNX schema",
  ...inspectOnnxStructure(bytes),
};
await writeFile(outputPath, `${JSON.stringify(evidence, null, 2)}\n`);
console.log(JSON.stringify({ outputPath, sha256, initializers: evidence.initializers.length }));
