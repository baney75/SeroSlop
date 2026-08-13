type Clock = () => number;
type Sleep = (milliseconds: number) => Promise<void>;

const systemSleep: Sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

/** Serializes operations and keeps their start times at least `intervalMs` apart. */
export class MinimumIntervalGate {
  private tail: Promise<void> = Promise.resolve();
  private lastStartedAt = Number.NEGATIVE_INFINITY;

  constructor(
    private readonly intervalMs: number,
    private readonly now: Clock = Date.now,
    private readonly sleep: Sleep = systemSleep,
  ) {
    if (!Number.isFinite(intervalMs) || intervalMs < 0) throw new Error("intervalMs must be a non-negative number");
  }

  run<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.tail.then(async () => {
      const remaining = this.intervalMs - (this.now() - this.lastStartedAt);
      if (remaining > 0) await this.sleep(remaining);
      this.lastStartedAt = this.now();
      return operation();
    });
    this.tail = result.then(() => undefined, () => undefined);
    return result;
  }
}
