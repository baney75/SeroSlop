export interface RasterDimensions {
  format: "png" | "jpeg" | "gif" | "webp" | "bmp" | "avif-heif";
  width: number;
  height: number;
}

export interface RasterLimits {
  maxEdge: number;
  maxPixels: number;
  maxAspectRatio: number;
}

function u16be(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] ?? 0) * 256 + (bytes[offset + 1] ?? 0);
}

function u24le(bytes: Uint8Array, offset: number): number {
  return (bytes[offset] ?? 0) + (bytes[offset + 1] ?? 0) * 256 + (bytes[offset + 2] ?? 0) * 65_536;
}

function u32be(bytes: Uint8Array, offset: number): number {
  return ((bytes[offset] ?? 0) * 16_777_216) + ((bytes[offset + 1] ?? 0) * 65_536) +
    ((bytes[offset + 2] ?? 0) * 256) + (bytes[offset + 3] ?? 0);
}

function ascii(bytes: Uint8Array, offset: number, length: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + length));
}

function dimensions(format: RasterDimensions["format"], width: number, height: number): RasterDimensions {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error("Image dimensions are invalid");
  }
  return { format, width, height };
}

function jpegDimensions(bytes: Uint8Array): RasterDimensions | undefined {
  if (bytes[0] !== 0xff || bytes[1] !== 0xd8) return undefined;
  const startOfFrame = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
  let offset = 2;
  while (offset + 3 < bytes.length) {
    while (bytes[offset] === 0xff) offset += 1;
    const marker = bytes[offset];
    offset += 1;
    if (marker === undefined || marker === 0xd9 || marker === 0xda) break;
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (offset + 2 > bytes.length) break;
    const length = u16be(bytes, offset);
    if (length < 2 || offset + length > bytes.length) throw new Error("JPEG segment is malformed");
    if (startOfFrame.has(marker)) {
      if (length < 7) throw new Error("JPEG frame header is malformed");
      return dimensions("jpeg", u16be(bytes, offset + 5), u16be(bytes, offset + 3));
    }
    offset += length;
  }
  throw new Error("JPEG dimensions are missing");
}

function webpDimensions(bytes: Uint8Array): RasterDimensions | undefined {
  if (bytes.length < 30 || ascii(bytes, 0, 4) !== "RIFF" || ascii(bytes, 8, 4) !== "WEBP") return undefined;
  const chunk = ascii(bytes, 12, 4);
  if (chunk === "VP8X") {
    return dimensions("webp", u24le(bytes, 24) + 1, u24le(bytes, 27) + 1);
  }
  if (chunk === "VP8L" && bytes[20] === 0x2f) {
    const width = 1 + ((bytes[21] ?? 0) | ((bytes[22] ?? 0) & 0x3f) << 8);
    const height = 1 + (((bytes[22] ?? 0) >> 6) | (bytes[23] ?? 0) << 2 | ((bytes[24] ?? 0) & 0x0f) << 10);
    return dimensions("webp", width, height);
  }
  if (chunk === "VP8 " && bytes[23] === 0x9d && bytes[24] === 0x01 && bytes[25] === 0x2a) {
    const width = ((bytes[26] ?? 0) | (bytes[27] ?? 0) << 8) & 0x3fff;
    const height = ((bytes[28] ?? 0) | (bytes[29] ?? 0) << 8) & 0x3fff;
    return dimensions("webp", width, height);
  }
  throw new Error("WebP dimensions are missing");
}

function avifHeifDimensions(bytes: Uint8Array): RasterDimensions | undefined {
  if (bytes.length < 24 || ascii(bytes, 4, 4) !== "ftyp") return undefined;
  const brands = ascii(bytes, 8, Math.min(bytes.length - 8, 32));
  if (!/(?:avif|avis|heic|heix|mif1)/u.test(brands)) return undefined;
  const searchLimit = Math.min(bytes.length - 16, 2 * 1024 * 1024);
  for (let offset = 4; offset <= searchLimit; offset += 1) {
    if (ascii(bytes, offset, 4) !== "ispe") continue;
    return dimensions("avif-heif", u32be(bytes, offset + 8), u32be(bytes, offset + 12));
  }
  throw new Error("AVIF/HEIF dimensions are missing");
}

export function inspectRasterDimensions(buffer: ArrayBuffer): RasterDimensions {
  const bytes = new Uint8Array(buffer);
  if (bytes.length >= 24 && bytes.slice(0, 8).every((value, index) => value === [137, 80, 78, 71, 13, 10, 26, 10][index])) {
    return dimensions("png", u32be(bytes, 16), u32be(bytes, 20));
  }
  if (bytes.length >= 10 && ["GIF87a", "GIF89a"].includes(ascii(bytes, 0, 6))) {
    return dimensions("gif", (bytes[6] ?? 0) | (bytes[7] ?? 0) << 8, (bytes[8] ?? 0) | (bytes[9] ?? 0) << 8);
  }
  const jpeg = jpegDimensions(bytes);
  if (jpeg) return jpeg;
  const webp = webpDimensions(bytes);
  if (webp) return webp;
  if (bytes.length >= 26 && ascii(bytes, 0, 2) === "BM") {
    const view = new DataView(buffer);
    return dimensions("bmp", Math.abs(view.getInt32(18, true)), Math.abs(view.getInt32(22, true)));
  }
  const avifHeif = avifHeifDimensions(bytes);
  if (avifHeif) return avifHeif;
  throw new Error("Image format does not expose safe raster dimensions");
}

export function assertRasterWithinLimits(value: RasterDimensions, limits: RasterLimits): void {
  const longest = Math.max(value.width, value.height);
  const shortest = Math.min(value.width, value.height);
  if (longest > limits.maxEdge || value.width * value.height > limits.maxPixels || longest / shortest > limits.maxAspectRatio) {
    throw new Error("Image dimensions exceed the safe decode budget");
  }
}
