# Database migrations

This folder holds **one-off SQL files** that must be applied to the Neon
(or any) Postgres database that backs MAHIKS-TR.

The bulk of the schema (`users`, `chunks`, `query_history`, `conversations`,
`kg_nodes`, `kg_edges`, …) is created automatically by
`init_system_tables()` in `backend/api_server.py` on first boot. The files
in this folder cover the two cases that startup-time DDL cannot:

1. **Enabling Postgres extensions** (`vector`) — must be granted by an
   admin before any `CREATE TABLE` referencing `vector(N)` runs.
2. **Documenting and recreating schema for disaster recovery** — keep the
   user-data table definitions in version control even though they are
   normally idempotently created by the app.

## When to apply

| Situation                                       | Run what?                                        |
|------------------------------------------------|--------------------------------------------------|
| Brand-new Neon project, first deploy           | `001_user_api_keys.sql`                          |
| Restoring from a broken DB / new DB region     | every `NNN_*.sql` file, in numerical order       |
| Adding a new migration                         | create the next `NNN_<short_name>.sql` here       |
| Local docker-compose dev                       | nothing — `init_system_tables()` handles it      |

## How to apply (Neon)

1. Sign in at <https://console.neon.tech>.
2. Open your project → **SQL Editor**.
3. Paste the contents of the desired migration file.
4. Press **Run**. Confirm there are no errors.

> Tip: Neon also exposes a one-click toggle for the `vector` extension at
> **Project → Extensions → vector**. Either that toggle *or* the
> `CREATE EXTENSION` statement inside `001_*.sql` works — they're idempotent.

## How to apply (any psql client)

```bash
export DATABASE_URL='postgresql://USER:PASS@HOST/DB?sslmode=require'
psql "$DATABASE_URL" -f migrations/001_user_api_keys.sql
```

## Naming convention

`NNN_<snake_case_subject>.sql`, zero-padded, monotonically increasing.
Add a short header comment block describing **what** and **why**.

## Migration list

| #   | File                          | Purpose                                                    |
|-----|-------------------------------|------------------------------------------------------------|
| 001 | `001_user_api_keys.sql`       | Enable `vector` ext + create `user_api_keys` (multi-tenant) |
