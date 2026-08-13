import { describe, expect, it } from "vitest";
import { assertLocalImageUrl } from "../src/shared/url-policy";

describe("image acquisition policy", () => {
  it("allows only locally rendered image pixels", () => {
    expect(() => assertLocalImageUrl("data:image/png;base64,AA==")).not.toThrow();
  });

  it("blocks every page-controlled network and privileged protocol", () => {
    for (const value of [
      "http://127.0.0.1/model",
      "http://10.1.2.3/image",
      "http://192.168.1.2/image",
      "http://[::ffff:127.0.0.1]/image",
      "http://[::1]/image",
      "https://images.example.com/photo.jpg",
      "file:///tmp/image.png",
      "blob:https://images.example.com/id",
      "data:text/plain;base64,AA==",
    ]) {
      expect(() => assertLocalImageUrl(value)).toThrow();
    }
  });
});
