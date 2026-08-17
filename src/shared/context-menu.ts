export const IMAGE_CONTEXT_MENU_ID = "seroslop-choose-image";
export const IMAGE_CONTEXT_MENU_TITLE = "Analyze this image with SeroSlop";

export function supportedContextPage(url: string | undefined): boolean {
  if (!url) return false;
  try {
    const protocol = new URL(url).protocol;
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}
