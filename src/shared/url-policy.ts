/** Offscreen inference accepts pixels only. It never follows a page-controlled network URL. */
export function assertLocalImageUrl(value: string): void {
  const url = new URL(value);
  if (url.protocol !== "data:" || !value.toLowerCase().startsWith("data:image/")) {
    throw new Error("Only locally rendered image pixels are accepted");
  }
}
