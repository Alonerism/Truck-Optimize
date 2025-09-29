// @ts-check
/** @type {import('@playwright/test').PlaywrightTestConfig} */
const config = {
  testDir: 'tests/playwright',
  timeout: 60_000,
  use: {
    headless: true,
    ignoreHTTPSErrors: true,
    viewport: { width: 1280, height: 800 },
    actionTimeout: 30_000,
    navigationTimeout: 30_000,
    video: 'off',
    screenshot: 'off'
  },
  reporter: [['list']],
  testMatch: ['map.spec.js']
};

module.exports = config;
