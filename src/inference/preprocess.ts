export function centerCropGeometry(
  width: number,
  height: number,
  resizeShortEdge = 440,
  cropSize = 384,
): { sourceX: number; sourceY: number; sourceSize: number; targetSize: number } {
  const shortEdge = Math.min(width, height);
  const cropInSourcePixels = (cropSize / resizeShortEdge) * shortEdge;
  return {
    sourceX: (width - cropInSourcePixels) / 2,
    sourceY: (height - cropInSourcePixels) / 2,
    sourceSize: cropInSourcePixels,
    targetSize: cropSize,
  };
}

export function resizeShortEdgeGeometry(
  width: number,
  height: number,
  shortEdge = 440,
): { width: number; height: number } {
  if (width <= 0 || height <= 0) throw new Error("Image dimensions must be positive");
  const scale = shortEdge / Math.min(width, height);
  return {
    width: Math.max(1, Math.round(width * scale)),
    height: Math.max(1, Math.round(height * scale)),
  };
}

export function imageDataToNormalizedChw(
  imageData: ImageData,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Float32Array {
  const { data, width, height } = imageData;
  return rgbBytesToNormalizedChw(data, width, height, 4, mean, std);
}

export function rgbBytesToNormalizedChw(
  data: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  channels: 3 | 4,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
): Float32Array {
  const pixels = width * height;
  if (data.length !== pixels * channels) throw new Error("Malformed RGB image data");
  const output = new Float32Array(3 * pixels);

  for (let pixel = 0; pixel < pixels; pixel += 1) {
    for (let channel = 0; channel < 3; channel += 1) {
      const value = data[pixel * channels + channel];
      if (value === undefined) throw new Error("Malformed RGB image data");
      output[channel * pixels + pixel] = (value / 255 - mean[channel]!) / std[channel]!;
    }
  }

  return output;
}

export function softmaxSynthetic(logits: readonly number[], syntheticIndex: number): number {
  if (logits.length < 2 || syntheticIndex < 0 || syntheticIndex >= logits.length) {
    throw new Error("Detector returned invalid logits");
  }
  const maximum = Math.max(...logits);
  const exponentials = logits.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  const synthetic = exponentials[syntheticIndex];
  if (synthetic === undefined || total === 0) throw new Error("Detector returned invalid probabilities");
  return synthetic / total;
}

export function sigmoidLogit(logit: number): number {
  if (!Number.isFinite(logit)) throw new Error("Detector returned an invalid logit");
  return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, logit))));
}
