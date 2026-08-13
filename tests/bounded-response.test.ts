import { describe, expect, it } from "vitest";
import { readBoundedResponseBytes } from "../src/shared/bounded-response";

describe("bounded local image reads", () => {
  it("joins a bounded stream", async () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1, 2]));
        controller.enqueue(Uint8Array.from([3, 4]));
        controller.close();
      },
    });
    const bytes = await readBoundedResponseBytes(new Response(body), 4);
    expect([...new Uint8Array(bytes)]).toEqual([1, 2, 3, 4]);
  });

  it("cancels as soon as a streamed body crosses the limit", async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(Uint8Array.from([1, 2, 3]));
        controller.enqueue(Uint8Array.from([4, 5, 6]));
      },
      cancel() {
        cancelled = true;
      },
    });
    await expect(readBoundedResponseBytes(new Response(body), 4)).rejects.toThrow("too large");
    expect(cancelled).toBe(true);
  });

  it("rejects an oversized declared body before reading it", async () => {
    let cancelled = false;
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        controller.enqueue(Uint8Array.from([1]));
      },
      cancel() {
        cancelled = true;
      },
    });
    await expect(readBoundedResponseBytes(new Response(body, { headers: { "content-length": "5" } }), 4))
      .rejects.toThrow("too large");
    expect(cancelled).toBe(true);
  });
});
