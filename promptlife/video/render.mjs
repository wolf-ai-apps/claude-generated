// Renders explainer.html to an MP4, frame by frame.
// Usage: node render.mjs [out.mp4] [fps] [--from s] [--to s] [--stills t1,t2,...]
// Needs Node 18+, Playwright (with Chromium) and ffmpeg with libx264.
import { createRequire } from 'node:module';
import { spawn, execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require('playwright')); }
catch { ({ chromium } = require(path.join(execSync('npm root -g').toString().trim(), 'playwright'))); }

const here = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (n, d) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : d; };
const out = args[0] && !args[0].startsWith('--') ? args[0] : path.join(here, 'life-of-a-prompt.mp4');
const fps = Number(args[1] && !args[1].startsWith('--') ? args[1] : 30);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
await page.addInitScript(() => { window.__RENDER__ = true; });
await page.goto('file://' + path.join(here, 'explainer.html'));
await page.evaluate(() => document.fonts.ready);
const total = await page.evaluate(() => window.TOTAL);

const grab = (T) => page.evaluate((T) => { window.renderFrame(T); return document.getElementById('c').toDataURL('image/png'); }, T);

const stills = flag('--stills');
if (stills) {
  for (const s of stills.split(',')) {
    const data = await grab(Number(s));
    fs.writeFileSync(path.join(here, `still-${s}.png`), Buffer.from(data.split(',')[1], 'base64'));
  }
  await browser.close();
  process.exit(0);
}

const from = Number(flag('--from', 0)), to = Number(flag('--to', total));
const frames = Math.round((to - from) * fps);
const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'png', '-i', '-',
  '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out], { stdio: ['pipe', 'inherit', 'inherit'] });

const started = Date.now();
for (let f = 0; f < frames; f++) {
  const data = await grab(from + f / fps);
  const buf = Buffer.from(data.split(',')[1], 'base64');
  if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
  if (f % (fps * 10) === 0) console.log(`frame ${f}/${frames} (${((Date.now() - started) / 1000).toFixed(0)}s)`);
}
ff.stdin.end();
await new Promise(r => ff.on('close', r));
await browser.close();
console.log(`wrote ${out} — ${(to - from).toFixed(1)}s at ${fps} fps`);
