import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const html = await readFile('site/index.html', 'utf8');
const privacy = await readFile('site/privacy.html', 'utf8');
const css = await readFile('site/styles.css', 'utf8');
const required = ['<title>', 'name="viewport"', '<main', 'aria-label="Primary"', 'A score is not proof of origin.', './styles.css', './assets/seroslop.svg'];
for (const marker of required) if (!html.includes(marker)) throw new Error(`site missing ${marker}`);
if (!html.includes('https://github.com/baney75/SeroSlop')) throw new Error('site missing canonical SeroSlop repository link');
if (!html.includes('https://github.com/baney75/SeroSlop/releases/latest') || !html.includes('Browse nightly builds')) throw new Error('stable/nightly install links missing');
if (!html.includes('Download the stable ZIP above and unzip it')) throw new Error('primary install path is not the release ZIP');
if (html.includes('https://github.com/baney75/prooflens')) throw new Error('site contains the retired repository link');
if (/<script\b/i.test(html) || /https?:\/\/(?!github\.com)/i.test(html + css)) throw new Error('site contains an external script or request');
if (!existsSync('site/assets/seroslop.svg')) throw new Error('site logo missing');
if (!html.includes('./privacy.html') || !privacy.includes('does not upload webpage images')) throw new Error('deployable privacy page missing');
if (!/prefers-color-scheme\s*:\s*dark/.test(css) || !/@media\s*\(max-width:375px\)/.test(css)) throw new Error('responsive theme gates missing');
if ((html.match(/class="step-content"/g) ?? []).length !== 4) throw new Error('install steps are not wrapped in one mobile-safe content column');
if (!/\.step-content\s*\{[^}]*min-width\s*:\s*0/isu.test(css)) throw new Error('install step content can overflow its mobile grid column');
console.log('site check: PASS (local assets, landmarks, limitation, responsive themes)');
