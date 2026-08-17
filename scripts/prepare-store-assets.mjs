import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const root = process.cwd();
const output = path.join(root, "store", "assets");
await mkdir(output, { recursive: true });

const modeSheet = await sharp(path.join(root, "docs", "images", "seroslop-modes.png"))
  .extract({ left: 0, top: 0, width: 750, height: 900 })
  .resize(667, 800, { fit: "fill" })
  .png()
  .toBuffer();
await sharp({
  create: { width: 1280, height: 800, channels: 4, background: "#f4f4f1" },
})
  .composite([{ input: modeSheet, left: 306, top: 0 }])
  .png()
  .toFile(path.join(output, "screenshot-modes-1280x800.png"));

const logo = await readFile(path.join(root, "src", "static", "seroslop.svg"));
const promoSvg = (width, height, titleSize, subtitleSize, logoSize, logoX, textX) => Buffer.from(`
<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <rect width="${width}" height="${height}" rx="24" fill="#111311"/>
  <rect x="${logoX}" y="${Math.round((height - logoSize) / 2)}" width="${logoSize}" height="${logoSize}" rx="16" fill="#ff3338"/>
  <image href="data:image/svg+xml;base64,${logo.toString("base64")}" x="${logoX}" y="${Math.round((height - logoSize) / 2)}" width="${logoSize}" height="${logoSize}"/>
  <text x="${textX}" y="${Math.round(height * 0.46)}" fill="#ffffff" font-family="Arial, Helvetica, sans-serif" font-size="${titleSize}" font-weight="700">SeroSlop</text>
  <text x="${textX}" y="${Math.round(height * 0.65)}" fill="#c9cec9" font-family="Arial, Helvetica, sans-serif" font-size="${subtitleSize}">Local AI-image scores in Chrome</text>
</svg>`);

await sharp(promoSvg(440, 280, 39, 14, 138, 24, 184)).png().toFile(path.join(output, "small-promo-440x280.png"));
await sharp(promoSvg(1400, 560, 118, 42, 404, 70, 552)).png().toFile(path.join(output, "marquee-1400x560.png"));

const metadata = {
  files: [
    "screenshot-modes-1280x800.png",
    "small-promo-440x280.png",
    "marquee-1400x560.png",
  ],
  sourceScreenshot: "docs/images/seroslop-modes.png",
  sourceLogo: "src/static/seroslop.svg",
};
await writeFile(path.join(output, "README.json"), `${JSON.stringify(metadata, null, 2)}\n`);
