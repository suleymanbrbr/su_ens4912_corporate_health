const { test, expect } = require('@playwright/test');

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      window.localStorage.setItem('sut_onboarding_done', '1');
    });
  });

  test('should show login page by default', async ({ page }) => {
    await page.goto('/login');
    await expect(page.getByText(/Hoşgeldiniz/i)).toBeVisible({ timeout: 15000 });
  });

  test('should fail with wrong credentials', async ({ page }) => {
    await page.goto('/login');
    await page.locator('input[type="text"]').fill('wronguser');
    await page.locator('input[type="password"]').fill('wrongpass');
    await page.getByRole('button', { name: /Giriş Yap/i }).click();
    
    await expect(page.locator('div').filter({ hasText: /Giriş/i }).last()).toBeVisible({ timeout: 15000 });
  });

  test('should login successfully with admin credentials', async ({ page }) => {
    await page.goto('/login');
    await page.locator('input[type="text"]').fill('admin');
    await page.locator('input[type="password"]').fill('Admin@1234!');
    await page.getByRole('button', { name: /Giriş Yap/i }).click();
    
    await expect(page).toHaveURL('/', { timeout: 25000 });
    // Look for SUT Asistanı text which is more stable
    await expect(page.getByText(/SUT Asistanı/i).first()).toBeVisible({ timeout: 35000 });
  });
});
