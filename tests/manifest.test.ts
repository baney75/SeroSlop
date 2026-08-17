import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

interface Manifest {
  manifest_version: number;
  icons?: Record<string, string>;
  action?: { default_icon?: Record<string, string> };
  background?: { service_worker?: string };
  content_security_policy?: { extension_pages?: string };
  permissions?: string[];
}

describe("extension manifest", () => {
  it("is Manifest V3 with a packaged service worker and offscreen permission", async () => {
    const source = await readFile("src/static/manifest.json", "utf8");
    const manifest = JSON.parse(source) as Manifest;
    expect(manifest.manifest_version).toBe(3);
    expect(manifest.background?.service_worker).toBe("background.js");
    expect(manifest.permissions).toContain("offscreen");
    expect(manifest.permissions).toContain("contextMenus");
  });

  it("uses the selected SeroSlop mark for the extension and toolbar action", async () => {
    const source = await readFile("src/static/manifest.json", "utf8");
    const manifest = JSON.parse(source) as Manifest;
    const expected = {
      "16": "icons/seroslop-16.png",
      "32": "icons/seroslop-32.png",
      "48": "icons/seroslop-48.png",
      "128": "icons/seroslop-128.png",
    };
    expect(manifest.icons).toEqual(expected);
    expect(manifest.action?.default_icon).toEqual(expected);
    for (const path of Object.values(expected)) {
      expect((await readFile(`src/static/${path}`)).byteLength).toBeGreaterThan(0);
    }
  });

  it("does not permit remotely hosted executable scripts", async () => {
    const source = await readFile("src/static/manifest.json", "utf8");
    const manifest = JSON.parse(source) as Manifest;
    const policy = manifest.content_security_policy?.extension_pages ?? "";
    expect(policy).toContain("script-src 'self'");
    expect(policy).not.toMatch(/https?:/);
    expect(policy).not.toMatch(/(?:^|\s)'unsafe-eval'/);
  });

  it("blocks extension-page network and image beacons outside the package", async () => {
    const source = await readFile("src/static/manifest.json", "utf8");
    const manifest = JSON.parse(source) as Manifest;
    const policy = manifest.content_security_policy?.extension_pages ?? "";
    for (const directive of [
      "default-src 'self'",
      "object-src 'none'",
      "connect-src 'self' data: blob:",
      "img-src 'self' data: blob:",
      "style-src 'self' 'unsafe-inline'",
      "worker-src 'self'",
    ]) {
      expect(policy).toContain(directive);
    }
    expect(policy).not.toMatch(/https?:/u);
  });
});
