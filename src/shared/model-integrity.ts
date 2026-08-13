import { sha256Hex } from "./hash";

export interface IntegrityRecord {
  modelId: string;
  sha256: string;
  bytes: ArrayBuffer;
}

export interface IntegrityLock {
  modelId: string;
  sha256: string;
  bytes: number;
}

export async function verifyModelRecord(record: IntegrityRecord | undefined, lock: IntegrityLock): Promise<ArrayBuffer> {
  if (!record || record.modelId !== lock.modelId || record.sha256 !== lock.sha256) {
    throw new Error("Model is not prepared");
  }
  if (record.bytes.byteLength !== lock.bytes) throw new Error("Stored model size check failed");
  const actualHash = await sha256Hex(record.bytes);
  if (actualHash !== lock.sha256) throw new Error("Stored model integrity check failed");
  return record.bytes;
}
