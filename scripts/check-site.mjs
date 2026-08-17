import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';

const html = await readFile('site/index.html', 'utf8');
const css = await readFile('site/styles.css', 'utf8');
const required = ['<title>', 'name="viewport"', '<main', 'aria-label="Primary"', 'A score is not proof of origin.', './styles.css', './assets/seroslop.svg'];
for (const marker of required) if (!html.includes(marker)) throw new Error(`site missing ${marker}`);
if (!html.includes('https://github.com/baney75/SeroSlop')) throw new Error('site missing canonical SeroSlop repository link');
if (html.includes('https://github.com/baney75/prooflens')) throw new Error('site contains the retired repository link');
if (/<script\b/i.test(html) || /https?:\/\/(?!github\.com)/i.test(html + css)) throw new Error('site contains an external script or request');
if (!existsSync('site/assets/seroslop.svg')) throw new Error('site logo missing');
if (!/prefers-color-scheme\s*:\s*dark/.test(css) || !/@media\s*\(max-width:375px\)/.test(css)) throw new Error('responsive theme gates missing');
console.log('site check: PASS (local assets, landmarks, limitation, responsive themes)');
