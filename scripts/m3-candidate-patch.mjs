import { createRequire } from "node:module";
import path from "node:path";
import { digest, jsonEqual, requireCondition } from "./m3-training-contract.mjs";


const require = createRequire(import.meta.url);
const { onnx } = require(path.resolve("node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js"));
const CLASSIFIER_NAMES = ["classifier.bias", "classifier.weight"];


export function inspectM3ClassifierInitializers(modelBytes) {
  const bytes = Buffer.from(modelBytes);
  const model = onnx.ModelProto.decode(bytes);
  requireCondition(model.graph, "M3 base ONNX has no graph");
  const initializers = new Map(model.graph.initializer.map((value) => [value.name, value]));
  return CLASSIFIER_NAMES.map((name) => {
    const value = initializers.get(name);
    requireCondition(value, `M3 base ONNX lacks ${name}`);
    const rawData = Buffer.from(value.rawData ?? []);
    requireCondition(rawData.length > 0, `M3 ${name} is not stored as raw tensor data`);
    const offset = bytes.indexOf(rawData);
    requireCondition(offset >= 0 && bytes.indexOf(rawData, offset + 1) < 0,
      `M3 ${name} raw bytes are not unique in the base model`);
    return {
      name,
      dimensions: value.dims.map((dimension) => Number(dimension)),
      offset,
      bytes: rawData.length,
      rawData,
      sha256: digest(rawData),
    };
  });
}


export function reconstructM3CandidateModel({ baseBytes, patch }) {
  const base = Buffer.from(baseBytes);
  requireCondition(patch?.schemaVersion === 1 && patch.baseSha256 === digest(base) &&
    patch.candidateBytes === base.length && /^[a-f0-9]{64}$/u.test(patch.candidateSha256 ?? "") &&
    Array.isArray(patch.replacements) && patch.replacements.length === CLASSIFIER_NAMES.length,
  "M3 classifier patch header changed");
  const baseInitializers = new Map(inspectM3ClassifierInitializers(base).map((row) => [row.name, row]));
  requireCondition(jsonEqual(patch.replacements.map((row) => row.name).sort(), CLASSIFIER_NAMES),
    "M3 classifier patch initializer set changed");
  const output = Buffer.from(base);
  const ranges = [];
  for (const row of patch.replacements) {
    const before = baseInitializers.get(row.name);
    requireCondition(before && jsonEqual(row.dimensions, before.dimensions) && row.offset === before.offset &&
      row.bytes === before.bytes && row.beforeSha256 === before.sha256 &&
      Number.isInteger(row.offset) && row.offset >= 0 && Number.isInteger(row.bytes) && row.bytes > 0,
    `M3 ${row.name} patch location changed`);
    const after = Buffer.from(String(row.afterBase64 ?? ""), "base64");
    requireCondition(after.length === row.bytes && after.toString("base64") === row.afterBase64 &&
      digest(after) === row.afterSha256 && row.afterSha256 !== row.beforeSha256,
    `M3 ${row.name} patch bytes changed`);
    const end = row.offset + row.bytes;
    requireCondition(ranges.every(([start, stop]) => end <= start || row.offset >= stop),
      "M3 classifier patch ranges overlap");
    ranges.push([row.offset, end]);
    after.copy(output, row.offset);
  }
  requireCondition(digest(output) === patch.candidateSha256,
    "M3 classifier patch does not reconstruct the locked candidate");
  return output;
}
