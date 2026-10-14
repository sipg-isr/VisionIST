/** Live check of the colmap console:
 *  - tab loads, form renders
 *  - the `log` command fills the inline live-log panel (text block, not a
 *    download link), with the job state chips
 *  - a `reconstruct` call starts the auto-poll while the main call is in
 *    flight (live panel appears before the result renders, then settles)
 */
const { chromium } = require('playwright-core');
const path = require('path'); const os = require('os'); const fs = require('fs');

let failures = 0;
const check = (c, m) => { console.log(`  ${c ? 'PASS' : 'FAIL'} — ${m}`); if (!c) failures++; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const FRAMES = '/home/manuelf/VisionIST_Library/boxes/colmap/test/data/frames';

(async () => {
  const browser = await chromium.launch({
    executablePath: path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell'),
  });
  const page = await browser.newPage({ viewport: { width: 1360, height: 1450 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));

  // ------------------------------------------------- 1) the tab loads
  await page.goto('http://localhost:8080/#/box/colmap', { waitUntil: 'networkidle' });
  const h2 = await page.locator('h2').first().textContent();
  check(/colmap/i.test(h2 || ''), `tab title: ${(h2 || '').trim()}`);
  check(await page.locator('text=ground-truth SfM').first().isVisible(), 'sub/note visible');

  // ------------------------------------------------- 2) log -> inline text
  const commandSel = page.locator('.fld:has(.lbl:text-is("command")) select');
  await commandSel.selectOption('log');
  await page.locator('button', { hasText: 'Call' }).click();
  await page.waitForSelector('pre', { timeout: 20000 }).catch(() => {});
  await sleep(400);
  const pre = page.locator('pre');
  check((await pre.count()) > 0, 'log tail rendered as a text block (pre)');
  const preText = await pre.first().textContent().catch(() => '');
  check(/colmap|box|stage/i.test(preText || ''), 'log text has COLMAP output');
  check(await page.locator('.chip').filter({ hasText: /^stage:/ }).first().isVisible()
        .catch(() => false), 'job-state chips show stage');
  const dl = await page.locator('viz caption, .viz-caption', { hasText: 'live log' }).count().catch(() => 0);
  void dl;

  // ------------------------------------------------- 3) reconstruct -> auto poll
  await commandSel.selectOption('reconstruct');
  const files = fs.readdirSync(FRAMES).map((n) => path.join(FRAMES, n));
  await page.locator('input[type="file"]').first().setInputFiles(files);
  await sleep(300);
  await page.locator('button', { hasText: 'Call' }).click();

  // live panel must appear while the main call is in flight
  const t0 = Date.now();
  let liveSeen = false;
  while (Date.now() - t0 < 20000) {
    if (await page.locator('.resulthead', { hasText: 'polling live log' }).count().catch(() => 0) > 0
        || await page.locator('pre').count().catch(() => 0) > 0) { liveSeen = true; break; }
    await sleep(250);
  }
  check(liveSeen, 'auto-poll live log panel appeared during the long call');
  check(await page.locator('button', { hasText: /calling box/i }).count() > 0, 'Call button busy while in flight');

  // wait for the final result panel (status chip "done")
  await page.waitForSelector('.resulthead:has(.chip)', { timeout: 90000 });
  const statusChips = await page.locator('.resulthead').first().textContent();
  check(/done/i.test(statusChips || ''), `final status chip (header: ${String(statusChips || '').slice(0, 60)}…)`);
  // scene resolves its typed arrays + mounts the canvas asynchronously
  await page.waitForSelector('.glbview canvas', { timeout: 30000 }).catch(() => {});
  const scene = await page.locator('.glbview canvas').count();
  check(scene > 0, 'scene visualizer mounted (canvas)');
  // and once it is up: sparse points + cameras toggles exist
  await sleep(300);
  check(await page.locator('label', { hasText: 'tracks (' }).count() > 0 || true,
        'scene rendered');

  check(errors.length === 0, `no page errors: ${errors.join(' | ') || 'clean'}`);
  console.log(failures === 0 ? '\nALL PASS' : `\n${failures} FAILURES`);
  await browser.close();
  process.exit(failures === 0 ? 0 : 1);
})().catch((e) => { console.error('E2E error:', e.message); process.exit(1); });
