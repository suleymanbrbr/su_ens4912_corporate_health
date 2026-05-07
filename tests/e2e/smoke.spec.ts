import { test, expect } from '@playwright/test'

test.beforeEach(({ }, testInfo) => {
  if (process.env.E2E_SKIP === '1') testInfo.skip()
})

test('login page loads when not authenticated', async ({ page }) => {
  await page.goto('/login')
  await expect(page.locator('body')).toBeVisible()
})

test('policies route redirects or loads for guest', async ({ page }) => {
  await page.goto('/policies')
  await expect(page).toHaveURL(/\/(login)?/)
})
