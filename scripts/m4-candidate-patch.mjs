import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import path from "node:path";


const require = createRequire(import.meta.url);
const { onnx } = require(path.resolve("node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js"));
const ADDED_INITIALIZERS = new Map([
  ["m4.feature_mean", [384]],
  ["m4.feature_std", [384]],
  ["m4.adapter_in.weight", [64, 384]],
  ["m4.adapter_in.bias", [64]],
  ["m4.adapter_out.weight", [384, 64]],
  ["m4.adapter_out.bias", [384]],
]);
const ADDED_NODES = [
  ["m4_sub_mean", "Sub", ["/Gather_output_0", "m4.feature_mean"], ["m4.centered"], {}],
  ["m4_div_std", "Div", ["m4.centered", "m4.feature_std"], ["m4.normalized"], {}],
  ["m4_adapter_in", "Gemm", ["m4.normalized", "m4.adapter_in.weight", "m4.adapter_in.bias"], ["m4.hidden_pre"], { alpha: 1, beta: 1, transB: 1 }],
  ["m4_relu", "Relu", ["m4.hidden_pre"], ["m4.hidden"], {}],
  ["m4_adapter_out", "Gemm", ["m4.hidden", "m4.adapter_out.weight", "m4.adapter_out.bias"], ["m4.residual_normalized"], { alpha: 1, beta: 1, transB: 1 }],
  ["m4_scale_residual", "Mul", ["m4.residual_normalized", "m4.feature_std"], ["m4.residual"], {}],
  ["m4_add_residual", "Add", ["/Gather_output_0", "m4.residual"], ["m4.adapted"], {}],
];

export function digest(value) {
  return createHash("sha256").update(value).digest("hex");
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function encode(type, value) {
  return Buffer.from(type.encode(value).finish());
}

function clone(type, value) {
  return type.decode(type.encode(value).finish());
}

function attributes(node) {
  const output = {};
  for (const attribute of node.attribute ?? []) {
    if (attribute.type === 1) output[attribute.name] = Number(attribute.f);
    else if (attribute.type === 2) output[attribute.name] = Number(attribute.i);
    else throw new Error(`M4 unexpected node attribute type: ${node.name}/${attribute.name}`);
  }
  return output;
}

function jsonEqual(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

function exactKeys(value, keys, label) {
  requireCondition(value !== null && typeof value === "object" && !Array.isArray(value) &&
    jsonEqual(Object.keys(value).sort(), [...keys].sort()), `${label} keys changed`);
}

function initializerMap(model, label) {
  const values = new Map();
  for (const value of model.graph.initializer) {
    requireCondition(!values.has(value.name), `${label} contains duplicate initializer ${value.name}`);
    values.set(value.name, value);
  }
  return values;
}

function readVarint(bytes, start) {
  let value = 0n;
  let shift = 0n;
  let offset = start;
  while (offset < bytes.length && shift <= 63n) {
    const byte = BigInt(bytes[offset]);
    value |= (byte & 0x7fn) << shift;
    offset += 1;
    if ((byte & 0x80n) === 0n) {
      requireCondition(value <= BigInt(Number.MAX_SAFE_INTEGER), "M4 protobuf varint exceeds safe integer range");
      return [Number(value), offset];
    }
    shift += 7n;
  }
  throw new Error("M4 protobuf contains a malformed varint");
}

function encodeVarint(value) {
  let remaining = BigInt(value);
  requireCondition(remaining >= 0n, "M4 protobuf varint is negative");
  const output = [];
  do {
    let byte = Number(remaining & 0x7fn);
    remaining >>= 7n;
    if (remaining !== 0n) byte |= 0x80;
    output.push(byte);
  } while (remaining !== 0n);
  return Buffer.from(output);
}

function protobufFields(value) {
  const bytes = Buffer.from(value);
  const output = [];
  let offset = 0;
  while (offset < bytes.length) {
    const start = offset;
    const [tag, afterTag] = readVarint(bytes, offset);
    const field = Math.floor(tag / 8);
    const wire = tag & 7;
    offset = afterTag;
    let payloadStart = offset;
    let payloadEnd;
    if (wire === 0) {
      [, offset] = readVarint(bytes, offset);
      payloadEnd = offset;
    } else if (wire === 1) {
      offset += 8;
      payloadEnd = offset;
    } else if (wire === 2) {
      const [length, afterLength] = readVarint(bytes, offset);
      payloadStart = afterLength;
      payloadEnd = payloadStart + length;
      offset = payloadEnd;
    } else if (wire === 5) {
      offset += 4;
      payloadEnd = offset;
    } else {
      throw new Error(`M4 protobuf wire type is unsupported: ${wire}`);
    }
    requireCondition(offset <= bytes.length && field > 0, "M4 protobuf field exceeds its message");
    output.push({ field, wire, start, end: offset, payloadStart, payloadEnd, raw: bytes.subarray(start, offset) });
  }
  return output;
}

function lengthDelimited(field, payload) {
  const body = Buffer.from(payload);
  return Buffer.concat([encodeVarint((field << 3) | 2), encodeVarint(body.length), body]);
}

function canonicalBase64(value, label) {
  requireCondition(typeof value === "string", `${label} is missing`);
  const bytes = Buffer.from(value, "base64");
  requireCondition(bytes.toString("base64") === value, `${label} is not canonical base64`);
  return bytes;
}

export function reconstructM4Candidate({ baseBytes, adapterPatch }) {
  exactKeys(adapterPatch, [
    "schemaVersion", "baseSha256", "candidateSha256", "candidateBytes", "featureTensor",
    "classifierNodeName", "classifierInputBefore", "classifierInputAfter", "addedInitializers", "addedNodes",
    "classifierNodeProtoSha256", "classifierNodeProtoBase64", "reconstructedCandidateSha256",
  ], "M4 adapter patch");
  const baseRaw = Buffer.from(baseBytes);
  requireCondition(adapterPatch.schemaVersion === 1 && adapterPatch.baseSha256 === digest(baseRaw) &&
    adapterPatch.featureTensor === "/Gather_output_0" && adapterPatch.classifierNodeName === "/classifier/Gemm" &&
    adapterPatch.classifierInputBefore === "/Gather_output_0" && adapterPatch.classifierInputAfter === "m4.adapted",
  "M4 adapter patch boundary changed");

  requireCondition(Array.isArray(adapterPatch.addedInitializers) &&
    adapterPatch.addedInitializers.length === ADDED_INITIALIZERS.size,
  "M4 adapter patch initializer count changed");
  const initializerProtos = [];
  for (let index = 0; index < ADDED_INITIALIZERS.size; index += 1) {
    const [name, dimensions] = [...ADDED_INITIALIZERS][index];
    const row = adapterPatch.addedInitializers[index];
    exactKeys(row, ["name", "dimensions", "dataType", "tensorProtoSha256", "tensorProtoBase64",
      "rawDataSha256", "rawDataBase64"], `M4 adapter initializer ${name}`);
    const raw = canonicalBase64(row.rawDataBase64, `M4 ${name} raw data`);
    const proto = canonicalBase64(row.tensorProtoBase64, `M4 ${name} tensor proto`);
    const decoded = onnx.TensorProto.decode(proto);
    requireCondition(row.name === name && decoded.name === name && jsonEqual(row.dimensions, dimensions) &&
      jsonEqual(decoded.dims.map(Number), dimensions) && row.dataType === "FLOAT" && decoded.dataType === 1 &&
      digest(proto) === row.tensorProtoSha256 && digest(raw) === row.rawDataSha256 &&
      Buffer.from(decoded.rawData ?? []).equals(raw), `M4 adapter initializer proto changed: ${name}`);
    initializerProtos.push(proto);
  }

  requireCondition(Array.isArray(adapterPatch.addedNodes) && adapterPatch.addedNodes.length === ADDED_NODES.length,
    "M4 adapter patch node count changed");
  const nodeProtos = [];
  for (let index = 0; index < ADDED_NODES.length; index += 1) {
    const [name, opType, inputs, outputs, expectedAttributes] = ADDED_NODES[index];
    const row = adapterPatch.addedNodes[index];
    exactKeys(row, ["name", "opType", "inputs", "outputs", "attributes", "nodeProtoSha256", "nodeProtoBase64"],
      `M4 adapter node ${name}`);
    const proto = canonicalBase64(row.nodeProtoBase64, `M4 ${name} node proto`);
    const decoded = onnx.NodeProto.decode(proto);
    requireCondition(row.name === name && decoded.name === name && row.opType === opType && decoded.opType === opType &&
      jsonEqual(row.inputs, inputs) && jsonEqual(decoded.input, inputs) && jsonEqual(row.outputs, outputs) &&
      jsonEqual(decoded.output, outputs) && jsonEqual(row.attributes, expectedAttributes) &&
      jsonEqual(attributes(decoded), expectedAttributes) && digest(proto) === row.nodeProtoSha256,
    `M4 adapter node proto changed: ${name}`);
    nodeProtos.push(proto);
  }
  const classifierProto = canonicalBase64(adapterPatch.classifierNodeProtoBase64,
    "M4 rewritten classifier node proto");
  const classifier = onnx.NodeProto.decode(classifierProto);
  requireCondition(digest(classifierProto) === adapterPatch.classifierNodeProtoSha256 &&
    classifier.name === adapterPatch.classifierNodeName &&
    jsonEqual(classifier.input, [adapterPatch.classifierInputAfter, "classifier.weight", "classifier.bias"]),
  "M4 rewritten classifier node changed");

  const modelFields = protobufFields(baseRaw);
  const graphFields = modelFields.filter((field) => field.field === 7 && field.wire === 2);
  requireCondition(graphFields.length === 1, "M4 base model graph field changed");
  const graph = graphFields[0];
  const graphRaw = baseRaw.subarray(graph.payloadStart, graph.payloadEnd);
  const fields = protobufFields(graphRaw);
  const initializerIndexes = fields.map((field, index) => field.field === 5 && field.wire === 2 ? index : -1)
    .filter((index) => index >= 0);
  requireCondition(initializerIndexes.length > 0, "M4 base graph has no initializer fields");
  const lastInitializer = initializerIndexes.at(-1);
  let classifierSeen = false;
  const rebuiltGraph = [];
  for (let index = 0; index < fields.length; index += 1) {
    const field = fields[index];
    if (field.field === 1 && field.wire === 2) {
      const node = onnx.NodeProto.decode(graphRaw.subarray(field.payloadStart, field.payloadEnd));
      if (node.name === adapterPatch.classifierNodeName) {
        requireCondition(!classifierSeen && jsonEqual(node.input,
          [adapterPatch.classifierInputBefore, "classifier.weight", "classifier.bias"]),
        "M4 base classifier node changed");
        rebuiltGraph.push(...nodeProtos.map((proto) => lengthDelimited(1, proto)), lengthDelimited(1, classifierProto));
        classifierSeen = true;
        continue;
      }
    }
    rebuiltGraph.push(field.raw);
    if (index === lastInitializer) {
      rebuiltGraph.push(...initializerProtos.map((proto) => lengthDelimited(5, proto)));
    }
  }
  requireCondition(classifierSeen, "M4 base classifier node was not found");
  const graphCandidate = Buffer.concat(rebuiltGraph);
  const modelCandidate = Buffer.concat(modelFields.map((field) =>
    field === graph ? lengthDelimited(7, graphCandidate) : field.raw));
  requireCondition(modelCandidate.length === adapterPatch.candidateBytes &&
    digest(modelCandidate) === adapterPatch.candidateSha256 &&
    adapterPatch.reconstructedCandidateSha256 === adapterPatch.candidateSha256,
  "M4 adapter patch did not reconstruct the locked candidate bytes");
  return modelCandidate;
}

export function validateM4AdapterModel({ baseBytes, candidateBytes, comparison }) {
  const baseRaw = Buffer.from(baseBytes);
  const candidateRaw = Buffer.from(candidateBytes);
  const base = onnx.ModelProto.decode(baseRaw);
  const candidate = onnx.ModelProto.decode(candidateRaw);
  requireCondition(base.graph && candidate.graph, "M4 ONNX graph is missing");
  exactKeys(comparison, ["schemaVersion", "profile", "base", "candidate", "unchangedBaseInitializerCount",
    "unchangedBaseNodeCount", "addedInitializers", "addedNodes", "classifierInputBefore",
    "classifierInputAfter", "classifierNodeProtoSha256", "classifierNodeProtoBase64",
    "reconstructedCandidateSha256", "reconstructedCandidateBytes",
    "backboneAndClassifierInitializersByteIdentical"], "M4 model comparison");
  exactKeys(comparison.base, ["path", "sha256", "bytes"], "M4 comparison base");
  exactKeys(comparison.candidate, ["path", "sha256", "bytes"], "M4 comparison candidate");
  requireCondition(comparison?.schemaVersion === 1 && comparison.profile === "m4",
    "M4 comparison header changed");
  requireCondition(comparison.base?.sha256 === digest(baseRaw) && comparison.base?.bytes === baseRaw.length &&
    comparison.candidate?.sha256 === digest(candidateRaw) && comparison.candidate?.bytes === candidateRaw.length,
  "M4 comparison model binding changed");
  for (const [field, type] of [["input", onnx.ValueInfoProto], ["output", onnx.ValueInfoProto]]) {
    const before = base.graph[field].map((value) => digest(encode(type, value)));
    const after = candidate.graph[field].map((value) => digest(encode(type, value)));
    requireCondition(jsonEqual(before, after), `M4 graph ${field} contract changed`);
  }
  requireCondition(jsonEqual(
    base.opsetImport.map((value) => digest(encode(onnx.OperatorSetIdProto, value))),
    candidate.opsetImport.map((value) => digest(encode(onnx.OperatorSetIdProto, value))),
  ), "M4 opset contract changed");

  const baseInitializers = initializerMap(base, "M4 base model");
  const candidateInitializers = initializerMap(candidate, "M4 candidate model");
  requireCondition(candidateInitializers.size === baseInitializers.size + ADDED_INITIALIZERS.size,
    "M4 initializer count changed");
  for (const [name, before] of baseInitializers) {
    const after = candidateInitializers.get(name);
    requireCondition(after && encode(onnx.TensorProto, before).equals(encode(onnx.TensorProto, after)),
      `M4 changed frozen initializer ${name}`);
  }
  const comparisonInitializers = new Map((comparison.addedInitializers ?? []).map((row) => [row.name, row]));
  requireCondition(comparisonInitializers.size === ADDED_INITIALIZERS.size,
    "M4 comparison added-initializer set changed");
  for (const [name, dimensions] of ADDED_INITIALIZERS) {
    const value = candidateInitializers.get(name);
    const row = comparisonInitializers.get(name);
    const rawData = Buffer.from(value?.rawData ?? []);
    requireCondition(value && row && jsonEqual(value.dims.map(Number), dimensions) && value.dataType === 1 &&
      rawData.length > 0 && row.dataType === "FLOAT" && jsonEqual(row.dimensions, dimensions) &&
      row.rawDataSha256 === digest(rawData) && row.rawDataBase64 === rawData.toString("base64"),
    `M4 added initializer changed: ${name}`);
  }

  const baseNodes = base.graph.node;
  const candidateNodes = candidate.graph.node;
  const baseClassifier = baseNodes.findIndex((node) => node.name === "/classifier/Gemm");
  const candidateClassifier = candidateNodes.findIndex((node) => node.name === "/classifier/Gemm");
  requireCondition(baseClassifier >= 0 && candidateClassifier === baseClassifier + ADDED_NODES.length &&
    candidateNodes.length === baseNodes.length + ADDED_NODES.length,
  "M4 classifier position or node count changed");
  for (let index = 0; index < baseNodes.length; index += 1) {
    const candidateIndex = index < baseClassifier ? index : index + ADDED_NODES.length;
    const expected = clone(onnx.NodeProto, baseNodes[index]);
    if (expected.name === "/classifier/Gemm") {
      requireCondition(jsonEqual(expected.input, ["/Gather_output_0", "classifier.weight", "classifier.bias"]),
        "M4 base classifier input changed");
      expected.input[0] = "m4.adapted";
    }
    requireCondition(encode(onnx.NodeProto, expected).equals(encode(onnx.NodeProto, candidateNodes[candidateIndex])),
      `M4 changed frozen graph node ${expected.name}`);
  }
  const comparisonNodes = new Map((comparison.addedNodes ?? []).map((row) => [row.name, row]));
  requireCondition(comparisonNodes.size === ADDED_NODES.length, "M4 comparison added-node set changed");
  for (let offset = 0; offset < ADDED_NODES.length; offset += 1) {
    const [name, opType, inputs, outputs, expectedAttributes] = ADDED_NODES[offset];
    const node = candidateNodes[baseClassifier + offset];
    const row = comparisonNodes.get(name);
    requireCondition(node.name === name && node.opType === opType && jsonEqual(node.input, inputs) &&
      jsonEqual(node.output, outputs) && jsonEqual(attributes(node), expectedAttributes) && row &&
      row.opType === opType && jsonEqual(row.inputs, inputs) && jsonEqual(row.outputs, outputs) &&
      jsonEqual(row.attributes, expectedAttributes),
    `M4 added node changed: ${name}`);
  }
  requireCondition(comparison.classifierInputBefore === "/Gather_output_0" &&
    comparison.classifierInputAfter === "m4.adapted" &&
    typeof comparison.classifierNodeProtoSha256 === "string" &&
    typeof comparison.classifierNodeProtoBase64 === "string" &&
    comparison.backboneAndClassifierInitializersByteIdentical === true &&
    comparison.reconstructedCandidateSha256 === digest(candidateRaw) &&
    comparison.reconstructedCandidateBytes === candidateRaw.length &&
    comparison.unchangedBaseInitializerCount === baseInitializers.size &&
    comparison.unchangedBaseNodeCount === baseNodes.length,
  "M4 comparison reconstruction claim changed");
  const patch = {
    schemaVersion: 1,
    baseSha256: digest(baseRaw),
    candidateSha256: digest(candidateRaw),
    candidateBytes: candidateRaw.length,
    featureTensor: "/Gather_output_0",
    classifierNodeName: "/classifier/Gemm",
    classifierInputBefore: "/Gather_output_0",
    classifierInputAfter: "m4.adapted",
    addedInitializers: comparison.addedInitializers,
    addedNodes: comparison.addedNodes,
    classifierNodeProtoSha256: comparison.classifierNodeProtoSha256,
    classifierNodeProtoBase64: comparison.classifierNodeProtoBase64,
    reconstructedCandidateSha256: comparison.reconstructedCandidateSha256,
  };
  requireCondition(reconstructM4Candidate({ baseBytes: baseRaw, adapterPatch: patch }).equals(candidateRaw),
    "M4 independent adapter reconstruction changed candidate bytes");
  return {
    baseSha256: digest(baseRaw),
    candidateSha256: digest(candidateRaw),
    baseInitializerCount: baseInitializers.size,
    addedInitializerCount: ADDED_INITIALIZERS.size,
    baseNodeCount: baseNodes.length,
    addedNodeCount: ADDED_NODES.length,
    classifierInput: "m4.adapted",
  };
}

export { ADDED_INITIALIZERS, ADDED_NODES };
