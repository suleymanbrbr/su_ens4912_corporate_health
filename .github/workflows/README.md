# GitHub Actions for MAHIKS-TR

This directory holds the CI/CD definitions for the MAHIKS-TR repository.

## Currently active workflows

| File | Trigger | What it does |
| ---- | ------- | ------------ |
| `ci.yml` | `push` to `main`, every `pull_request` to `main` | Installs backend + test deps, imports `api_server` as a smoke test, runs the database-free unit tests (`test_auth_utils.py`, `test_api_models.py`). |
| `frontend-build.yml` | `pull_request` touching `frontend/**` | `npm ci` + `npm run build` for the Vite app, uploads `frontend/dist` as an artifact (7-day retention). |

`../dependabot.yml` opens grouped weekly PRs for pip, npm, and `actions/*`
version updates. Major bumps for LangChain, FastAPI, React, Vite, and Torch are
intentionally ignored — they need a human in the loop.

## Disabled deploy workflow

`deploy.yml.disabled` is a template for automatic production deployment. It is
**not picked up by GitHub Actions** because of its file extension. To enable it:

1. Add these repository secrets (Settings → Secrets and variables → Actions):
   - `VERCEL_TOKEN` — from <https://vercel.com/account/tokens>
   - `VERCEL_ORG_ID` — from `frontend/.vercel/project.json` after running
     `vercel link` locally, or the Vercel project dashboard
   - `VERCEL_PROJECT_ID` — same source as above
   - `HF_TOKEN` — write-scope token from
     <https://huggingface.co/settings/tokens>
   - `HF_USERNAME` — e.g. `suleyman42`
   - `HF_SPACE_NAME` — e.g. `mahiks-tr`
2. Rename the file:
   ```bash
   git mv .github/workflows/deploy.yml.disabled .github/workflows/deploy.yml
   git commit -m "ci: enable deploy workflow"
   ```
3. Push and watch the next `main` build deploy both surfaces.

> **Why disabled by default?** Without the secrets, every push to `main` would
> fail. Disabling avoids noisy red checks on the jury's view of the repo until
> the maintainer is ready to flip the switch.

## Manual deploy (today)

Until the deploy workflow is enabled, the human workflow is:

```bash
# Frontend
cd frontend && vercel --prod

# Backend (HF Space)
git checkout hf-deploy
git push --force https://<user>:<HF_TOKEN>@huggingface.co/spaces/<user>/mahiks-tr hf-deploy:main
git checkout main
```
