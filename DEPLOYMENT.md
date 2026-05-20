# MAHIKS-TR — Production deployment guide

This is the **single source of truth** for getting MAHIKS-TR live on the
free-tier stack:

| Layer       | Host                  | Why                                            |
|-------------|-----------------------|------------------------------------------------|
| Frontend    | Vercel (Hobby)        | Free static hosting, instant Vite deploys      |
| Backend     | Hugging Face Spaces   | Free Docker container, 16 GB RAM, port 7860    |
| Database    | Neon Postgres 16      | Free 3 GB + pgvector extension, EU regions     |
| Keep-alive  | cron-job.org          | Pings `/health` so the Space doesn't sleep     |

End-to-end time on a fresh laptop: **≈ 45 minutes**, of which ≈ 15 min is
the first HF Spaces build (model pre-warm).

---

## Part 1 — Prerequisites

Create accounts (all free, no credit card):

- [ ] Hugging Face — <https://huggingface.co/join>
- [ ] Neon — <https://neon.tech>
- [ ] Vercel — <https://vercel.com/signup>
- [ ] GitHub — <https://github.com/signup>  (push this repo first)

Local tooling:

```bash
python3 --version    # 3.11+
git --version
docker --version     # only for local dev / sanity checking
```

---

## Part 2 — Provision the database (Neon)

1. Sign in at <https://console.neon.tech> → **Create project**.
2. Settings:
   - **Postgres version:** 16
   - **Region:** `Europe (Frankfurt) — aws-eu-central-1`
     *(stays inside the EU for KVKK / GDPR alignment)*
   - **Project name:** `mahiks-tr`
3. After creation, open the project → **Extensions** → toggle `vector` on.
4. Open **SQL Editor**, paste the contents of
   [`migrations/001_user_api_keys.sql`](./migrations/001_user_api_keys.sql),
   and click **Run**. Verify no errors.

   > Most tables are auto-created by `init_system_tables()` on the
   > backend's first boot — you only need this migration for the
   > `vector` extension and the new `user_api_keys` table.

5. Click **Dashboard → Connection Details**, choose:
   - **Pooled connection** (recommended for serverless apps)
   - **Connection string** dropdown → copy the value that ends in
     `?sslmode=require`.

   Save it as `DATABASE_URL`, e.g.:

   ```
   postgresql://mahiks_owner:XXXXXX@ep-cool-river-12345.eu-central-1.aws.neon.tech/mahiks?sslmode=require
   ```

---

## Part 3 — Generate the two production secrets

You need two strong random secrets that **never leave your secret manager**.
Run these on any machine with Python 3 + `cryptography` installed:

```bash
python -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))"
python -c "from cryptography.fernet import Fernet; print('API_KEY_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"
```

Save both lines in a password manager (1Password, Bitwarden, etc.).

> **⚠ Critical:** If you ever lose `API_KEY_ENCRYPTION_KEY`, every
> user-stored LLM key in the database becomes permanently unrecoverable
> (encrypted blobs with no key to decrypt them). Users will have to
> re-enter their API keys. Treat it like a vault root key.

---

## Part 4 — Deploy the backend (Hugging Face Spaces)

### 4.1 Create the Space

1. Go to <https://huggingface.co/new-space>.
2. Fill in:
   - **Owner:** your HF username (or an org)
   - **Space name:** `mahiks-tr`
   - **License:** MIT
   - **Select the Space SDK:** **Docker** → choose the "Blank" template
   - **Hardware:** CPU basic (free)
   - **Visibility:** Public *(or Private if you prefer)*
3. Click **Create Space**. HF prepares an empty git repo at
   `https://huggingface.co/spaces/<user>/mahiks-tr`.

### 4.2 Push the code to the Space

The cleanest path is to add the Space as a **second git remote** on the
existing GitHub repo so both stay in sync:

```bash
cd /path/to/su_ens4912_corporate_health

# One-time
git remote add hf https://huggingface.co/spaces/<user>/mahiks-tr

# When prompted, sign in with your HF username and a write-scoped
# access token from https://huggingface.co/settings/tokens
git push hf main
```

HF detects the root `Dockerfile` and starts building the Space.

> If `git push hf main` is rejected because the Space starts with its own
> initial commit, run it once with `--force` (only the very first push):
> `git push hf main --force`.

### 4.3 Add Space secrets

While the build is running, configure the runtime environment:

1. In the Space, open **Settings** → **Variables and secrets**.
2. Add the following as **secrets** (encrypted, hidden):

   | Key                    | Value                                                                 |
   |------------------------|-----------------------------------------------------------------------|
   | `JWT_SECRET_KEY`       | from Part 3                                                           |
   | `API_KEY_ENCRYPTION_KEY` | from Part 3                                                         |
   | `DATABASE_URL`         | the Neon pooled connection string from Part 2                         |
   | `CORS_ORIGINS`         | *(leave empty for now — you'll set this in Part 5 after Vercel)*      |

3. Add the following as a **public variable** (not secret, fine to expose):

   | Key                     | Value                                            |
   |-------------------------|--------------------------------------------------|
   | `SUT_EMBEDDING_MODEL`   | `paraphrase-multilingual-MiniLM-L12-v2`          |

   *(`HF_HOME=/data/hf_cache` is already hard-baked into the Dockerfile,
   so you do not need to set it as a Space variable.)*

4. After saving the secrets, click **Settings → Factory rebuild** so the
   running container picks them up. (For purely environment changes, a
   simple **Restart** is enough.)

### 4.4 Wait for the build

The first build is **slow (~10–15 min)** because it pre-downloads the
embedding model and the cross-encoder reranker into `/data/hf_cache`.

Open the Space's **App** tab and switch to **Logs**. You should see the
familiar uvicorn banner end-of-log:

```
INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

### 4.5 Smoke-test the backend

```bash
curl https://<user>-mahiks-tr.hf.space/health
# → {"status":"ok","db":"ok"}
```

If the response is `{"status":"ok","db":"error"}`, double-check the
`DATABASE_URL` secret and that you enabled the `vector` extension in Neon
(Part 2 step 3).

---

## Part 5 — Deploy the frontend (Vercel)

1. Go to <https://vercel.com/new>.
2. **Import Git Repository** → choose the GitHub repo containing this
   project.
3. **Configure Project**:
   - **Framework Preset:** *Vite*
   - **Root Directory:** `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Add **Environment Variable** (Production + Preview + Development):

   | Key                  | Value                                              |
   |----------------------|----------------------------------------------------|
   | `VITE_API_BASE_URL`  | `https://<user>-mahiks-tr.hf.space`                |

5. Click **Deploy**. After ~60 s you'll get a URL such as
   `https://mahiks-tr-<random>.vercel.app`. (You can claim a friendlier
   subdomain via **Settings → Domains**.)

6. **Back to the HF Space** → **Settings → Variables and secrets** →
   set `CORS_ORIGINS` to a comma-separated list:

   ```
   https://mahiks-tr.vercel.app,http://localhost:5173
   ```

   Then **Settings → Restart** the Space so the new CORS list is loaded.

7. Open the Vercel URL in a browser, register an account, then go to
   **Settings → API Keys**, paste a Google AI Studio Gemini key, save,
   and ask a Turkish SUT question (e.g. *"İmatinib hangi kanserlerde
   geri ödenir?"*). You should see a streamed answer with `[Madde X.X.X]`
   citations.

---

## Part 6 — Promote your first user to admin

Self-service signup creates regular `role='user'` accounts. To make
yourself the admin:

1. Sign up on the live site at least once.
2. In the Neon **SQL Editor**, run:

   ```sql
   UPDATE users
   SET role = 'admin', is_approved = TRUE
   WHERE username = 'your-username';
   ```

3. Log out and log back in to refresh the JWT.

The Admin panel is now visible in the sidebar — use it to ingest the
SUT corpus, approve other users, view audit logs, and so on.

---

## Part 7 — Keep the Space awake (optional)

HF Spaces on the free tier sleep after **48 h** with no traffic. During
the 1-month demo we just ping `/health` every 6 hours:

1. Sign up at <https://cron-job.org> (free, no card).
2. **Cronjobs → Create cronjob**:
   - **Title:** `mahiks-tr keep-alive`
   - **URL:** `https://<user>-mahiks-tr.hf.space/health`
   - **Schedule:** every 6 hours
   - **Notifications:** on failure → your email
3. **Save & enable**.

> A single GET every 6 hours uses essentially zero of Neon's compute-hours
> quota and costs nothing on cron-job.org's free plan.

---

## Part 8 — Updating production

| Change kind                | What to do                                                   |
|---------------------------|--------------------------------------------------------------|
| Backend code (Python)     | `git push hf main` — HF rebuilds (~3–10 min).                 |
| Frontend code (React)     | `git push origin main` — Vercel auto-redeploys (~60 s).       |
| DB schema change          | Add `migrations/NNN_*.sql`, run it in the Neon SQL Editor.    |
| New env var / secret      | HF Space **Settings → Variables and secrets**, then Restart.  |
| New Vercel env var        | Vercel **Settings → Environment Variables**, then redeploy.   |

Keep `git push hf main` and `git push origin main` in sync — both remotes
point at the same `Dockerfile`/`backend/`.

---

## Part 9 — Cost monitoring (everything stays free)

| Service        | Free quota                                       | Watch out for                              |
|----------------|--------------------------------------------------|--------------------------------------------|
| HF Space       | CPU basic, 16 GB RAM, unlimited time             | Auto-sleep after 48 h idle (Part 7 fixes)  |
| Neon Postgres  | 3 GB storage, ~190 compute-hours/mo on 1 endpoint| Heavy KG growth could approach 3 GB        |
| Vercel Hobby   | 100 GB bandwidth/mo, unlimited builds            | Don't enable team/commercial use           |
| cron-job.org   | 1 job free                                       | n/a — we only need one                     |

If you ever exceed the Neon storage limit, the console will warn before
read-only mode kicks in; you can purge old `query_history` or upgrade.

---

## Part 10 — Tear-down (after the demo)

When the project deliverable is signed off and you no longer need
production hosting:

1. **HF Space:** Settings → **Delete Space**.
2. **Neon:** Project Settings → **Delete Project**. (This also drops the
   pooled-connection string; any leaked copy becomes useless.)
3. **Vercel:** Project Settings → **Delete Project**.
4. **cron-job.org:** Cronjobs → delete the keep-alive job.
5. **Rotate the leaked secrets:** even though the services are gone,
   purge `JWT_SECRET_KEY` and `API_KEY_ENCRYPTION_KEY` from your password
   manager so they can't be reused if the DB backup ever leaks.

No credit card was ever attached → no chance of incurring charges.

---

## Appendix A — Environment variable reference

| Variable                  | Where set        | Required? | Example / default                                                        |
|--------------------------|------------------|-----------|--------------------------------------------------------------------------|
| `JWT_SECRET_KEY`         | HF Space secret  | yes       | `python -c "import secrets; print(secrets.token_urlsafe(64))"`           |
| `API_KEY_ENCRYPTION_KEY` | HF Space secret  | yes       | `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| `DATABASE_URL`           | HF Space secret  | yes       | `postgresql://user:pass@host/db?sslmode=require`                         |
| `CORS_ORIGINS`           | HF Space var     | yes       | `https://mahiks-tr.vercel.app,http://localhost:5173`                     |
| `SUT_EMBEDDING_MODEL`    | HF Space var     | no        | `paraphrase-multilingual-MiniLM-L12-v2` (default)                        |
| `HF_HOME`                | Dockerfile (ENV) | no        | `/data/hf_cache` (do not override on HF)                                 |
| `VITE_API_BASE_URL`      | Vercel env       | yes       | `https://<user>-mahiks-tr.hf.space`                                      |

See [`.env.example`](./.env.example) at the repo root for a copy-pasteable
template.

---

## Appendix B — Troubleshooting

| Symptom                                                | Likely cause / fix                                                     |
|-------------------------------------------------------|------------------------------------------------------------------------|
| HF Space stuck "Building…" >20 min on first deploy    | Model pre-warm is slow first time. Watch logs; only abort after 30 min. |
| `/health` returns `db:"error"`                        | `DATABASE_URL` wrong, or `vector` extension not enabled in Neon.        |
| Browser console: CORS error                           | `CORS_ORIGINS` missing the Vercel URL. Update and restart the Space.    |
| "401 Unauthorized" on every request after redeploy    | `JWT_SECRET_KEY` rotated — old tokens are invalid. Tell users to log in again. |
| Chat replies say "Lütfen API anahtarınızı ayarlayın"  | The user hasn't saved a personal Gemini/OpenRouter key yet (Settings). |
| Space woke from sleep, first request takes 30 s       | Cold start. Subsequent requests are warm. Enable Part 7 to avoid.       |
| Neon storage warning                                  | `DELETE FROM query_history WHERE created_at < NOW() - INTERVAL '90 days';` or upgrade. |

---

Done. You now have a live multi-tenant SaaS RAG running on €0/month.
