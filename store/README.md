# Chrome Web Store handoff

The stable GitHub ZIP and the Chrome Web Store package contain the same extension files. The Store listing is not published yet.

## Listing copy

**Name:** SeroSlop

**Single purpose:** Show a local AI-image model score beside images on webpages the user visits.

**Short description:** Local AI-image scores for webpage images. Runs in Chrome without a detector API or image upload.

**Category:** Productivity

The score is not proof of origin. Completed analyses show a numeric score; failures show unavailable.

## Permission explanations

- `contextMenus`: adds **Analyze this image with SeroSlop** to an image's right-click menu.
- `storage` and `unlimitedStorage`: store the verified packaged model, setup state, mode, and bounded local results.
- `offscreen`: run local ONNX inference outside the webpage.
- `<all_urls>`: discover and score supported images on ordinary HTTP and HTTPS pages after the user enables a scanning mode. Extension code does not fetch page-controlled image URLs.

Privacy policy: `https://baney75.github.io/SeroSlop/privacy.html` after that exact page is deployed. Until then, use the repository's `PRIVACY.md` and do not submit the listing.

## Prepared assets

Run `npm run prepare:store-assets`. It produces:

- `assets/screenshot-modes-1280x800.png`
- `assets/small-promo-440x280.png`
- `assets/marquee-1400x560.png`

The screenshot is derived from the project-owned mode-selection E2E capture. Store graphics use the repository interface and logo without third-party image fixtures.

## Human account gates

Publishing requires the owner's Google account, Chrome Web Store developer registration, acceptance of Google's Developer Agreement, any displayed registration fee, verified contact details, privacy disclosures, and a final listing review. Those account, identity, fee, and legal steps must be completed by Donovan. Do not claim Store installation or automatic Store updates before Google publishes the item.
