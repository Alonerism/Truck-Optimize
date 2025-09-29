// Playwright test: validate map HTML loads without console/callback errors
const { test } = require('@playwright/test');
const fs = require('fs');
const path = require('path');
const http = require('http');
const { spawn } = require('child_process');

const HOST = '127.0.0.1';
let PORT = 8008; // use a dedicated port to avoid collisions with dev servers
let BASE = `http://${HOST}:${PORT}`;
let TARGET_URL = `${BASE}/runs/day1_map.html`;
const REPO_ROOT = path.resolve(__dirname, '../..');
const MAP_HTML_PATH = path.join(REPO_ROOT, 'runs', 'day1_map.html');
const SCREENSHOT_PATH = path.join(REPO_ROOT, 'runs', 'playwright_map.png');

// Utility to ensure output dir exists
function ensureDir(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function waitForHttp(url, timeoutMs = 5000, expectStatus = 200) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    function probe() {
      http.get(url, res => {
        if (res.statusCode === expectStatus) return resolve(true);
        if (Date.now() - start > timeoutMs) return reject(new Error(`Server reachable but unexpected status ${res.statusCode} for ${url}`));
        setTimeout(probe, 100);
      }).on('error', () => {
        if (Date.now() - start > timeoutMs) return reject(new Error('Server not reachable'));
        setTimeout(probe, 100);
      });
    }
    probe();
  });
}

test('Google Map renders with no callback errors', async ({ page }) => {
  // Precondition: require generated HTML to exist
  if (!fs.existsSync(MAP_HTML_PATH)) {
    throw new Error('Missing runs/day1_map.html. Generate it first with: poetry run python -m app.cli visualize --date 2025-09-18');
  }

  // Start our own static server on a dedicated port to avoid collisions with API servers
  let server = spawn('python', ['-m', 'http.server', String(PORT), '--bind', HOST], { cwd: REPO_ROOT });
  try {
    await waitForHttp(`${BASE}/runs/`, 5000, 200);
  } catch (e) {
    // Retry on an alternate port if the first is unavailable
    try { server.kill('SIGINT'); } catch {}
    PORT = 8765;
    BASE = `http://${HOST}:${PORT}`;
    TARGET_URL = `${BASE}/runs/day1_map.html`;
    server = spawn('python', ['-m', 'http.server', String(PORT), '--bind', HOST], { cwd: REPO_ROOT });
    await waitForHttp(`${BASE}/runs/`, 8000, 200);
  }

  const consoleErrors = [];
  const failedRequests = [];

  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  page.on('requestfailed', request => {
    failedRequests.push({ url: request.url(), failure: request.failure() });
  });

  await waitForHttp(TARGET_URL, 10000, 200);
  await page.goto(TARGET_URL, { waitUntil: 'domcontentloaded' });

  // Sanity: ensure the correct page loaded
  const title = await page.title();
  if (!title.includes('Truck Jobs Map')) {
    const html = await page.content();
    throw new Error(`Unexpected page title: ${title}. First 300 chars of HTML: ${html.slice(0,300)}`);
  }

  // Wait up to 15s for the Maps API script tag to appear, then collect diagnostic info
  await page.waitForFunction(() => !!document.querySelector('script[src*="maps.googleapis.com"]'), null, { timeout: 15000 });
  // Assert exactly one Maps API script tag exists and uses async+defer
  const mapsScripts = await page.$$eval('script[src*="maps.googleapis.com"]', els => 
    els.map(e => ({ async: e.async, defer: e.defer, src: e.src }))
  );

  if (mapsScripts.length !== 1) {
    const html = await page.content();
    const scripts = await page.$$eval('script', els => els.map(e => e.src || e.textContent.slice(0,80)));
    throw new Error(`Expected exactly one Google Maps script tag, found ${mapsScripts.length}. Script srcs: ${JSON.stringify(scripts)}. First 200 chars of HTML: ${html.slice(0,200)}`);
  }
  if (!mapsScripts[0].async || !mapsScripts[0].defer) {
    throw new Error('Google Maps script tag must have async and defer attributes');
  }

  // Wait for init to be defined and called by callback
  await page.waitForFunction(() => typeof window.init === 'function');

  // Wait for map container to contain a canvas or gm-style container
  const mapSelector = '#map';
  await page.waitForSelector(mapSelector);
  await page.waitForFunction(sel => {
    const el = document.querySelector(sel);
    if (!el) return false;
    return el.querySelector('canvas, .gm-style') !== null;
  }, mapSelector, { timeout: 30000 });

  // Extra wait to ensure markers and route have fully rendered before screenshot
  await page.waitForTimeout(7000);

  // Take screenshot
  ensureDir(SCREENSHOT_PATH);
  await page.screenshot({ path: SCREENSHOT_PATH, fullPage: false });

  // Fail test on syntax/callback/API errors in console
  const fatal = consoleErrors.find(e => /SyntaxError|init is not a function|Google Maps JavaScript API error/.test(e));
  if (fatal) {
    try { server.kill('SIGINT'); } catch {}
    throw new Error(`Fatal console error detected: ${fatal}`);
  }

  if (server) { try { server.kill('SIGINT'); } catch {} }
});
