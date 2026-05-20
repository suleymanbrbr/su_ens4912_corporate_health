# MAHIKS-TR Frontend

Vite + React app for the multi-tenant MAHIKS-TR (SUT Asistanı) UI.

## Local development

```bash
npm install
cp .env.example .env   # adjust VITE_API_BASE_URL if your backend isn't on :8000
npm run dev
```

The Vite dev server proxies `/api/*` to `VITE_API_BASE_URL` (default
`http://localhost:8000`).

## Environment variables

| Variable             | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `VITE_API_BASE_URL`  | Backend base URL. Used by the dev proxy and by `src/services/api.js`. |

Files:
- `.env.example`       — checked in template
- `.env.production`    — checked in default for production builds (override on Vercel)

## Production deployment (Vercel)

`vercel.json` rewrites `/api/*` to the backend (intended to be a Hugging Face
Space). **Replace `YOUR_HF_SPACE` in `vercel.json` with your actual Space
subdomain** before deploying — the placeholder is intentional so the file can be
committed without leaking infra detail.

Example, after you create the Space `huseyin-mahiks-api`:

```json
{ "source": "/api/:path*", "destination": "https://huseyin-mahiks-api.hf.space/api/:path*" }
```

Alternatively, instead of using the rewrite, set `VITE_API_BASE_URL` to the full
backend URL in the Vercel project's Environment Variables — `src/services/api.js`
uses it directly, so the rewrite becomes optional.

## API key model

Each authenticated user stores their own LLM API key via the Settings page
(`/settings`, "LLM API Anahtarları" card). Backend endpoints (provided by the
backend service):

- `POST   /api/user/api-keys`            — save/update key for a provider
- `GET    /api/user/api-keys`            — list saved keys (hint only; key never returned)
- `DELETE /api/user/api-keys/{provider}` — remove key
- `POST   /api/user/api-keys/test`       — validate without persisting

If no key is configured, `POST /api/chat/query` responds 400 and the UI shows a
toast with a CTA to the Settings page.
