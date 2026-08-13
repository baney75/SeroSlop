import * as ort from "onnxruntime-web/webgpu";
import {
  classifyLikelihood,
  type InferenceResult,
  type InferenceSource,
  type ModelStatus,
  type RuntimeProvider,
} from "../shared/contracts";
import { readBoundedResponseBytes } from "../shared/bounded-response";
import { sha256Hex } from "../shared/hash";
import { assertRasterWithinLimits, inspectRasterDimensions } from "../shared/image-metadata";
import { verifyModelRecord } from "../shared/model-integrity";
import { MODEL_SPEC } from "../shared/model-spec";
import { deleteStoredModel, readStoredModel, storeModel } from "../shared/storage";
import { assertLocalImageUrl } from "../shared/url-policy";
import { calibrateAiLikelihood } from "./calibration";
import { imageDataToNormalizedChw, resizeShortEdgeGeometry, sigmoidLogit } from "./preprocess";

const MAX_IMAGE_SOURCE_CHARACTERS = 8 * 1024 * 1024;
const MAX_IMAGE_EDGE = 8_192;
const MAX_IMAGE_PIXELS = 25_000_000;
const MAX_IMAGE_ASPECT_RATIO = 16;
const MAX_RESULT_CACHE_ENTRIES = 256;

interface SetupProgress {
  completedBytes: number;
  totalBytes: number;
}

interface SessionState {
  session: ort.InferenceSession;
  provider: RuntimeProvider;
}

type ProgressCallback = (progress: SetupProgress) => void;

export class BrowserDetector {
  private sessionState: SessionState | undefined;
  private sessionLoading: Promise<SessionState> | undefined;
  private verifiedBytes: Promise<ArrayBuffer> | undefined;
  private webGpuDisabled = false;
  private readonly resultCache = new Map<string, InferenceResult>();
  private status: ModelStatus = {
    state: "not-installed",
    modelId: MODEL_SPEC.id,
    completedBytes: 0,
    totalBytes: MODEL_SPEC.weightsBytes,
  };

  constructor() {
    ort.env.wasm.wasmPaths = chrome.runtime.getURL("ort/");
    ort.env.wasm.numThreads = 1;
  }

  async getStatus(): Promise<ModelStatus> {
    try {
      const record = await readStoredModel();
      if (!record) return this.status;
      await this.getVerifiedBytes();
      this.status = {
        state: "ready",
        modelId: MODEL_SPEC.id,
        completedBytes: record.bytes.byteLength,
        totalBytes: MODEL_SPEC.weightsBytes,
      };
    } catch (error) {
      await deleteStoredModel().catch(() => undefined);
      this.verifiedBytes = undefined;
      this.status = {
        state: "error",
        modelId: MODEL_SPEC.id,
        completedBytes: 0,
        totalBytes: MODEL_SPEC.weightsBytes,
        error: error instanceof Error ? error.message : String(error),
      };
    }
    return this.status;
  }

  async prepare(onProgress: ProgressCallback): Promise<ModelStatus> {
    const existing = await this.getStatus();
    if (existing.state === "ready") return existing;

    this.status = {
      state: "preparing",
      modelId: MODEL_SPEC.id,
      completedBytes: 0,
      totalBytes: MODEL_SPEC.weightsBytes,
    };

    try {
      const response = await fetch(chrome.runtime.getURL(MODEL_SPEC.bundledWeightsPath), {
        cache: "no-store",
        credentials: "omit",
      });
      if (!response.ok || !response.body) throw new Error(`Bundled model load failed with HTTP ${response.status}`);
      const reader = response.body.getReader();
      const chunks: Uint8Array[] = [];
      let completedBytes = 0;
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        completedBytes += value.byteLength;
        if (completedBytes > MODEL_SPEC.weightsBytes) throw new Error("Bundled model is larger than its lock");
        onProgress({ completedBytes, totalBytes: MODEL_SPEC.weightsBytes });
      }
      if (completedBytes !== MODEL_SPEC.weightsBytes) {
        throw new Error(`Bundled model size mismatch: ${completedBytes}`);
      }
      const modelBytes = new Uint8Array(completedBytes);
      let offset = 0;
      for (const chunk of chunks) {
        modelBytes.set(chunk, offset);
        offset += chunk.byteLength;
      }
      const actualHash = await sha256Hex(modelBytes.buffer);
      if (actualHash !== MODEL_SPEC.weightsSha256) {
        throw new Error(`Model integrity check failed (received ${actualHash})`);
      }
      await storeModel({
        modelId: MODEL_SPEC.id,
        sha256: actualHash,
        bytes: modelBytes.buffer,
        installedAt: new Date().toISOString(),
      });
      this.verifiedBytes = Promise.resolve(modelBytes.buffer);
      this.sessionState = undefined;
      this.sessionLoading = undefined;
      await this.getSession();
      this.status = {
        state: "ready",
        modelId: MODEL_SPEC.id,
        completedBytes,
        totalBytes: MODEL_SPEC.weightsBytes,
      };
      return this.status;
    } catch (error) {
      await deleteStoredModel().catch(() => undefined);
      this.verifiedBytes = undefined;
      this.status = {
        state: "error",
        modelId: MODEL_SPEC.id,
        completedBytes: 0,
        totalBytes: MODEL_SPEC.weightsBytes,
        error: error instanceof Error ? error.message : String(error),
      };
      throw error;
    }
  }

  async infer(source: InferenceSource): Promise<InferenceResult> {
    const startedAt = performance.now();
    const loaded = await this.loadInferenceImage(source);
    const cached = this.resultCache.get(loaded.contentSha256);
    if (cached) {
      loaded.image.close();
      return { ...cached, durationMs: Math.round(performance.now() - startedAt) };
    }

    try {
      const resizedGeometry = resizeShortEdgeGeometry(
        loaded.image.width,
        loaded.image.height,
        MODEL_SPEC.resizeShortEdge,
      );
      const resizedCanvas = document.createElement("canvas");
      resizedCanvas.width = resizedGeometry.width;
      resizedCanvas.height = resizedGeometry.height;
      const resizedContext = resizedCanvas.getContext("2d", { willReadFrequently: false });
      if (!resizedContext) throw new Error("Canvas image processing is unavailable");
      resizedContext.imageSmoothingEnabled = true;
      resizedContext.imageSmoothingQuality = "high";
      resizedContext.drawImage(loaded.image, 0, 0, resizedCanvas.width, resizedCanvas.height);

      const inputCanvas = document.createElement("canvas");
      inputCanvas.width = MODEL_SPEC.inputSize;
      inputCanvas.height = MODEL_SPEC.inputSize;
      const inputContext = inputCanvas.getContext("2d", { willReadFrequently: true });
      if (!inputContext) throw new Error("Canvas image processing is unavailable");
      const left = Math.floor((resizedCanvas.width - MODEL_SPEC.inputSize) / 2);
      const top = Math.floor((resizedCanvas.height - MODEL_SPEC.inputSize) / 2);
      inputContext.drawImage(
        resizedCanvas,
        left,
        top,
        MODEL_SPEC.inputSize,
        MODEL_SPEC.inputSize,
        0,
        0,
        MODEL_SPEC.inputSize,
        MODEL_SPEC.inputSize,
      );
      const input = imageDataToNormalizedChw(
        inputContext.getImageData(0, 0, MODEL_SPEC.inputSize, MODEL_SPEC.inputSize),
        MODEL_SPEC.imageMean,
        MODEL_SPEC.imageStd,
      );
      const tensor = new ort.Tensor("float32", input, [1, 3, MODEL_SPEC.inputSize, MODEL_SPEC.inputSize]);
      const { outputs, provider } = await this.runWithFallback(tensor);
      const output = outputs[MODEL_SPEC.outputName];
      if (!output) throw new Error(`Model output ${MODEL_SPEC.outputName} is missing`);
      const logit = Number((output.data as Float32Array)[0]);
      const rawAiLikelihood = sigmoidLogit(logit);
      const aiLikelihood = calibrateAiLikelihood(rawAiLikelihood, MODEL_SPEC.calibration);
      const result: InferenceResult = {
        aiLikelihood,
        classification: classifyLikelihood(aiLikelihood),
        modelId: MODEL_SPEC.id,
        provider,
        contentSha256: loaded.contentSha256,
        durationMs: Math.round(performance.now() - startedAt),
      };
      this.remember(loaded.contentSha256, result);
      return result;
    } finally {
      loaded.image.close();
    }
  }

  private remember(contentSha256: string, result: InferenceResult): void {
    this.resultCache.delete(contentSha256);
    this.resultCache.set(contentSha256, result);
    if (this.resultCache.size > MAX_RESULT_CACHE_ENTRIES) {
      const oldest = this.resultCache.keys().next().value as string | undefined;
      if (oldest) this.resultCache.delete(oldest);
    }
  }

  private async getVerifiedBytes(): Promise<ArrayBuffer> {
    if (this.verifiedBytes) return this.verifiedBytes;
    this.verifiedBytes = (async () => {
      const stored = await readStoredModel();
      return verifyModelRecord(stored, {
        modelId: MODEL_SPEC.id,
        sha256: MODEL_SPEC.weightsSha256,
        bytes: MODEL_SPEC.weightsBytes,
      });
    })();
    try {
      return await this.verifiedBytes;
    } catch (error) {
      this.verifiedBytes = undefined;
      throw error;
    }
  }

  private async getSession(): Promise<SessionState> {
    if (this.sessionState) return this.sessionState;
    if (this.sessionLoading) return this.sessionLoading;
    this.sessionLoading = this.createSession();
    try {
      this.sessionState = await this.sessionLoading;
      return this.sessionState;
    } finally {
      this.sessionLoading = undefined;
    }
  }

  private async createSession(): Promise<SessionState> {
    const bytes = await this.getVerifiedBytes();
    if (!this.webGpuDisabled && "gpu" in navigator) {
      try {
        const session = await ort.InferenceSession.create(bytes.slice(0), {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all",
          logSeverityLevel: 3,
        });
        return { session, provider: "webgpu" };
      } catch {
        this.webGpuDisabled = true;
      }
    }
    const session = await ort.InferenceSession.create(bytes.slice(0), {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
      logSeverityLevel: 3,
    });
    return { session, provider: "wasm" };
  }

  private async runWithFallback(tensor: ort.Tensor): Promise<{ outputs: ort.InferenceSession.OnnxValueMapType; provider: RuntimeProvider }> {
    let state = await this.getSession();
    try {
      return {
        outputs: await state.session.run({ [MODEL_SPEC.inputName]: tensor }),
        provider: state.provider,
      };
    } catch (error) {
      if (state.provider !== "webgpu") throw error;
      this.webGpuDisabled = true;
      await state.session.release();
      this.sessionState = undefined;
      this.sessionLoading = undefined;
      state = await this.getSession();
      return {
        outputs: await state.session.run({ [MODEL_SPEC.inputName]: tensor }),
        provider: state.provider,
      };
    }
  }

  private async loadImage(value: string): Promise<{ image: ImageBitmap; contentSha256: string }> {
    if (value.length > MAX_IMAGE_SOURCE_CHARACTERS) throw new Error("Rendered image payload is too large");
    assertLocalImageUrl(value);
    const response = await fetch(value, {
      credentials: "omit",
      cache: "no-store",
      referrerPolicy: "no-referrer",
      redirect: "error",
    });
    if (!response.ok) throw new Error(`Image fetch failed with HTTP ${response.status}`);
    const bytes = await readBoundedResponseBytes(response);
    const raster = inspectRasterDimensions(bytes);
    assertRasterWithinLimits(raster, {
      maxEdge: MAX_IMAGE_EDGE,
      maxPixels: MAX_IMAGE_PIXELS,
      maxAspectRatio: MAX_IMAGE_ASPECT_RATIO,
    });
    const contentSha256 = await sha256Hex(bytes);
    const declaredType = response.headers.get("content-type")?.split(";", 1)[0]?.trim().toLowerCase();
    const inferredType = raster.format === "avif-heif" ? "image/avif" : `image/${raster.format}`;
    const blob = new Blob([bytes], { type: declaredType?.startsWith("image/") ? declaredType : inferredType });
    try {
      const image = await createImageBitmap(blob, { imageOrientation: "from-image" });
      assertRasterWithinLimits({ ...raster, width: image.width, height: image.height }, {
        maxEdge: MAX_IMAGE_EDGE,
        maxPixels: MAX_IMAGE_PIXELS,
        maxAspectRatio: MAX_IMAGE_ASPECT_RATIO,
      });
      return { image, contentSha256 };
    } catch {
      throw new Error("Image decode failed");
    }
  }

  private async loadInferenceImage(source: InferenceSource): Promise<{ image: ImageBitmap; contentSha256: string }> {
    const loaded = await this.loadImage(source.url);
    if (source.kind !== "captured-viewport") return loaded;
    const crop = source.crop;
    if (!crop || ![crop.left, crop.top, crop.width, crop.height, crop.viewportWidth, crop.viewportHeight].every(Number.isFinite) ||
      crop.width <= 0 || crop.height <= 0 || crop.viewportWidth <= 0 || crop.viewportHeight <= 0) {
      loaded.image.close();
      throw new Error("Viewport crop is invalid");
    }
    const left = Math.max(0, Math.floor((crop.left / crop.viewportWidth) * loaded.image.width));
    const top = Math.max(0, Math.floor((crop.top / crop.viewportHeight) * loaded.image.height));
    const right = Math.min(loaded.image.width, Math.ceil(((crop.left + crop.width) / crop.viewportWidth) * loaded.image.width));
    const bottom = Math.min(loaded.image.height, Math.ceil(((crop.top + crop.height) / crop.viewportHeight) * loaded.image.height));
    if (right <= left || bottom <= top) {
      loaded.image.close();
      throw new Error("Viewport crop is outside the captured page");
    }
    try {
      const image = await createImageBitmap(loaded.image, left, top, right - left, bottom - top);
      const descriptor = new TextEncoder().encode(`${loaded.contentSha256}:${left}:${top}:${right}:${bottom}`);
      return { image, contentSha256: await sha256Hex(descriptor.buffer) };
    } finally {
      loaded.image.close();
    }
  }
}
