# End-to-end smoke (Playwright)

```bash
cd tests/e2e
npm install
npx playwright install   # first time only
E2E_BASE_URL=http://localhost:5173 npm run test:e2e
```

Set `E2E_SKIP=1` to skip all tests in CI without a UI server.

Requires the Vite dev server (or any build served at `E2E_BASE_URL`).
