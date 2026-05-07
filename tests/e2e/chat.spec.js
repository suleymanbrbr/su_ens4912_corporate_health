const { test, expect } = require('@playwright/test');

test.describe('Chat Interactions', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('sut_onboarding_done', '1');
    });

    await page.goto('/login');
    await page.locator('input[type="text"]').fill('admin');
    await page.locator('input[type="password"]').fill('Admin@1234!');
    await page.getByRole('button', { name: /Giriş Yap/i }).click();
    await expect(page).toHaveURL('/', { timeout: 25000 });
  });

  test('should send a message and receive a streamed response', async ({ page }) => {
    const input = page.getByPlaceholder(/SUT mevzuatı hakkında soru sorun/i);
    await input.fill('SUT nedir?');
    await page.locator('button[type="submit"]').click();

    await expect(page.locator('main').getByText('SUT nedir?')).toBeVisible({ timeout: 20000 });
    await expect(page.locator('.premium-card').first()).toBeVisible({ timeout: 45000 });
  });

  test('should switch to knowledge graph tab', async ({ page }) => {
    await page.getByRole('button', { name: /Bilgi Grafiği/i }).click();
    await expect(page.locator('canvas')).toBeVisible({ timeout: 25000 });
  });

  test('should open command palette via search button', async ({ page }) => {
    // Click header button
    await page.locator('button.no-print[aria-label="Komut paleti"]').click();
    
    // Explicitly target the input inside the dialog to avoid duplicates
    const dialog = page.locator('[role="dialog"]');
    await expect(dialog).toBeVisible({ timeout: 15000 });
    await expect(dialog.locator('input')).toBeVisible();
  });
});
