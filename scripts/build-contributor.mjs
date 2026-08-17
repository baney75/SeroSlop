import { cp, mkdir, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const source = resolve(root, 'contributor');
const out = resolve(source, 'dist');
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
for (const file of ['manifest.json', 'background.js', 'popup.html', 'popup.css', 'popup.js', 'content.js']) await cp(resolve(source, file), resolve(out, file));
await cp(resolve(root, 'src/static/seroslop.svg'), resolve(out, 'seroslop.svg'));
const endpoint = process.env.SEROSLOP_QUARANTINE_ENDPOINT || '';
const terms = process.env.SEROSLOP_COUNSEL_TERMS_VERSION || '';
const validEndpoint = /^https:\/\/([a-z0-9-]+\.)*seroslop\.[a-z]{2,}(\/|$)/i.test(endpoint);
await writeFile(resolve(out, 'intake-config.json'), JSON.stringify({ uploadEnabled: Boolean(validEndpoint && terms), quarantineEndpoint: validEndpoint && terms ? endpoint : null, counselTermsVersion: validEndpoint && terms ? terms : null, rawUploadDefault: false }, null, 2) + '\n');
console.log(`built contributor/dist (raw upload ${validEndpoint && terms ? 'eligible for separate review' : 'disabled'})`);
