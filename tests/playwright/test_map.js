// Playwright test: validate map HTML loads without console/callback errors
const { test } = require('@playwright/test');
const fs = require('fs');
const path = require('path');

const TARGET_URL = 'http://localhost:8000/runs/day1_map.html';
const REPO_ROOT = path.resolve(__dirname, '../..');
const SCREENSHOT_PATH = path.join(REPO_ROOT, 'runs', 'playwright_map.png');

// Utility to ensure output dir exists
function ensureDir(filePath) {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

test('Google Map renders with no callback errors', async ({ page, browserName }) => {
  // Web server is managed by Playwright config

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

  await page.goto(TARGET_URL, { waitUntil: 'domcontentloaded' });

  // Assert exactly one Maps API script tag exists and uses async+defer
  const mapsScripts = await page.$$eval('script[src*="maps.googleapis.com/maps/api/js"]', els => 
    els.map(e => ({ async: e.async, defer: e.defer, src: e.src }))
  );

  if (mapsScripts.length !== 1) {
    throw new Error(`Expected exactly one Google Maps script tag, found ${mapsScripts.length}`);
  }
  if (!mapsScripts[0].async || !mapsScripts[0].defer) {
    throw new Error('Google Maps script tag must have async and defer attributes');
  }

  // Wait for init to be defined and called by callback
  await page.waitForFunction(() => typeof window.init === 'function');

  // Wait for map container to contain a canvas
  const mapSelector = '#map';
  await page.waitForSelector(mapSelector);
  await page.waitForFunction(sel => {
    const el = document.querySelector(sel);
    if (!el) return false;
    return el.querySelector('canvas, .gm-style') !== null;
  }, mapSelector, { timeout: 30000 });

  // Take screenshot
  ensureDir(SCREENSHOT_PATH);
  await page.screenshot({ path: SCREENSHOT_PATH, fullPage: false });

  // Log console errors and failed requests but don't fail test unless there are syntax/callback errors
  const fatal = consoleErrors.find(e => /SyntaxError|init is not a function|Failed to load resource|Google Maps JavaScript API error/.test(e));
  if (fatal) {
    throw new Error(`Fatal console error detected: ${fatal}`);
  }

  // Attach logs to stdout for debugging
  if (consoleErrors.length) console.log('Console errors:', consoleErrors);
  if (failedRequests.length) console.log('Failed requests:', failedRequests);
  // Server is managed by Playwright
});
