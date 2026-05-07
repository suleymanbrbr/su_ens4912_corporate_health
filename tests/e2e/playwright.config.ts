import { defineConfig, devices } from '@playwright/test'

/**
 * Smoke tests — optional. Requires frontend dev server or static preview.
 *
 *   cd tests/e2e && E2E_BASE_URL=http://localhost:5173 npx playwright test
 *
 * Or set E2E_SKIP=1 to skip (CI without UI).
 */
export default defineConfig({
  testDir: '.',
  timeout: 60_000,
  forbidOnly: !!process.env.CI,
  retries: 0,
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://localhost:5173',
    trace: 'on-first-retry',
    ...devices['Desktop Chrome'],
  },
})
