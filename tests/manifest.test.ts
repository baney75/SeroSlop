import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

interface Manifest {
  manifest_version: number;
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
  });

  it("does not permit remotely hosted executable scripts", async () => {
    const source = await readFile("src/static/manifest.json", "utf8");
    const manifest = JSON.parse(source) as Manifest;
    const policy = manifest.content_security_policy?.extension_pages ?? "";
    expect(policy).toContain("script-src 'self'");
    expect(policy).not.toMatch(/https?:/);
    expect(policy).not.toMatch(/(?:^|\s)'unsafe-eval'/);
  });
});
