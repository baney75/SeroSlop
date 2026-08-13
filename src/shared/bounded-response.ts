export const MAX_IMAGE_BYTES = 32 * 1024 * 1024;
const MAX_RESPONSE_CHUNKS = 4_096;

/** Read a response without ever retaining more than the declared byte budget. */
export async function readBoundedResponseBytes(response: Response, maximumBytes = MAX_IMAGE_BYTES): Promise<ArrayBuffer> {
  const advertised = response.headers.get("content-length");
  if (advertised !== null) {
    const advertisedLength = Number(advertised);
    if (!Number.isSafeInteger(advertisedLength) || advertisedLength < 0) {
      await response.body?.cancel("Image Content-Length is invalid").catch(() => undefined);
      throw new Error("Image Content-Length is invalid");
    }
    if (advertisedLength > maximumBytes) {
      await response.body?.cancel("Image Content-Length exceeds the bounded read budget").catch(() => undefined);
      throw new Error("Image is too large to analyze safely");
    }
  }
  if (!response.body) throw new Error("Image response body is missing");

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      total += value.byteLength;
      if (total > maximumBytes || chunks.length >= MAX_RESPONSE_CHUNKS) {
        await reader.cancel("Image response exceeded its bounded read budget").catch(() => undefined);
        throw new Error("Image is too large to analyze safely");
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  if (!total) throw new Error("Image byte size is invalid");

  const joined = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    joined.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return joined.buffer;
}
