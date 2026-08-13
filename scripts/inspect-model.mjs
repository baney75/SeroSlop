import { readFile } from "node:fs/promises";
import * as ort from "onnxruntime-web";

const modelPath = process.argv[2];
if (!modelPath) throw new Error("Usage: node scripts/inspect-model.mjs <model.onnx>");

ort.env.wasm.numThreads = 1;
const bytes = await readFile(modelPath);
const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
console.log(JSON.stringify({ inputNames: session.inputNames, outputNames: session.outputNames }, null, 2));
await session.release();
