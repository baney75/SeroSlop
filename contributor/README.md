# SeroSlop Contributor Review (developer extension)

This is a separate, developer-only MV3 extension. It is not part of the public extension build or release archive.

The tool lets a contributor choose exactly one visible image, inspect a locally captured preview, record an AI false-negative or real false-positive, add bounded evidence, and prepare a JSON review file for a maintainer. It does not upload images. The prepared file is a local testing/export fallback.

The picker is injected only after the contributor presses **Choose an image**. It does not run on every page. Cross-origin images are captured from the visible tab; their remote URL is never loaded by the extension popup.

Raw intake is intentionally fail-closed. Enabling a first-party HTTPS quarantine service requires a reviewed endpoint, counsel-approved terms version, access controls, retention/deletion jobs, and a separate release decision. GitHub and Hugging Face are not raw-image intake destinations.

## Build

From the repository root:

```sh
npm run check:contributor
```

Load `contributor/dist` through `chrome://extensions` → Developer mode → Load unpacked.
