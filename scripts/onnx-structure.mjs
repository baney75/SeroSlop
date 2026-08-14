import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import path from "node:path";


const require = createRequire(import.meta.url);
const { onnx } = require(path.resolve("node_modules/onnxruntime-web/lib/onnxjs/ort-schema/protobuf/onnx.js"));


function digestBytes(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}


function sequenceDigest(values, type) {
  const hash = createHash("sha256");
  for (const value of values) {
    const encoded = Buffer.from(type.encode(value).finish());
    const length = Buffer.alloc(8);
    length.writeBigUInt64BE(BigInt(encoded.length));
    hash.update(length);
    hash.update(encoded);
  }
  return hash.digest("hex");
}


export function inspectOnnxStructure(bytes) {
  const model = onnx.ModelProto.decode(bytes);
  if (!model.graph) throw new Error("ONNX model has no graph");
  const initializers = model.graph.initializer.map((value) => ({
    name: value.name,
    dimensions: value.dims.map((dimension) => Number(dimension)),
    sha256: digestBytes(Buffer.from(onnx.TensorProto.encode(value).finish())),
  })).sort((left, right) => left.name.localeCompare(right.name));
  if (new Set(initializers.map((value) => value.name)).size !== initializers.length) {
    throw new Error("ONNX model contains duplicate initializer names");
  }
  return {
    graphNodesSha256: sequenceDigest(model.graph.node, onnx.NodeProto),
    graphInputsSha256: sequenceDigest(model.graph.input, onnx.ValueInfoProto),
    graphOutputsSha256: sequenceDigest(model.graph.output, onnx.ValueInfoProto),
    opsetsSha256: sequenceDigest(model.opsetImport, onnx.OperatorSetIdProto),
    initializers,
  };
}
