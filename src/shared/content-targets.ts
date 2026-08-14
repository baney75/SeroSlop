export interface TargetDescriptor {
  slot: string;
  kind: "image" | "background";
  source: string;
}

/** Parse the serialized CSSOM background-image value without evaluating CSS text. */
export function extractCssImageUrls(backgroundImage: string, baseUrl: string): string[] {
  const urls: string[] = [];
  const expression = /url\(\s*(?:"([^"]*)"|'([^']*)'|([^)]*?))\s*\)/giu;
  for (const match of backgroundImage.matchAll(expression)) {
    const value = (match[1] ?? match[2] ?? match[3] ?? "").trim();
    if (!value) continue;
    try {
      urls.push(new URL(value, baseUrl).href);
    } catch {
      // Invalid CSS URLs are deliberately ignored rather than guessed.
    }
  }
  return [...new Set(urls)];
}

export function formatAiScore(value: number): string {
  if (!Number.isFinite(value) || value < 0 || value > 1) throw new Error("Score input must be finite and bounded");
  return `${(value * 100).toFixed(1)}/100`;
}
