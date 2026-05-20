-- ============================================================================
-- Migration 001 — pgvector extension + user_api_keys table
-- ----------------------------------------------------------------------------
-- Run this ONCE against your Neon database (or any fresh Postgres) before
-- the backend boots for the first time. Most other tables are created
-- automatically by init_system_tables() in backend/api_server.py on first
-- startup; this file exists so:
--   1. The pgvector extension is enabled (Neon also exposes a toggle in
--      Project → Extensions → vector — either path works).
--   2. The schema for the new multi-tenant `user_api_keys` table is
--      documented in version control for disaster-recovery purposes.
--
-- How to apply (Neon):
--   1. Open the Neon console (https://console.neon.tech).
--   2. Select your project → SQL Editor.
--   3. Paste the contents of this file and click "Run".
--
-- How to apply (any psql):
--   psql "$DATABASE_URL" -f migrations/001_user_api_keys.sql
-- ============================================================================

-- 1. Enable pgvector (required for chunks.embedding and kg_nodes.embedding).
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Per-user API keys (multi-tenant SaaS) -----------------------------------
--    NOTE: users.id in this project is TEXT (UUID), not INT. The original
--    cross-team contract used INT; we keep the realised schema here so this
--    file matches what init_system_tables() actually creates. Do not change
--    the type without coordinating a users.id migration.
CREATE TABLE IF NOT EXISTS user_api_keys (
    id              SERIAL PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider        VARCHAR(50)  NOT NULL CHECK (provider IN ('gemini','openrouter','local')),
    encrypted_key   BYTEA        NOT NULL,
    base_url        VARCHAR(500),
    key_hint        VARCHAR(8)   NOT NULL,
    is_active       BOOLEAN      DEFAULT TRUE,
    last_used_at    TIMESTAMP,
    created_at      TIMESTAMP    DEFAULT NOW(),
    updated_at      TIMESTAMP    DEFAULT NOW(),
    UNIQUE (user_id, provider)
);

CREATE INDEX IF NOT EXISTS idx_user_api_keys_user_id
    ON user_api_keys (user_id);

-- 3. (Optional) trigger to keep updated_at fresh on UPDATE -------------------
CREATE OR REPLACE FUNCTION set_user_api_keys_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_user_api_keys_updated_at ON user_api_keys;
CREATE TRIGGER trg_user_api_keys_updated_at
    BEFORE UPDATE ON user_api_keys
    FOR EACH ROW EXECUTE FUNCTION set_user_api_keys_updated_at();

-- ============================================================================
-- End of migration 001
-- ============================================================================
