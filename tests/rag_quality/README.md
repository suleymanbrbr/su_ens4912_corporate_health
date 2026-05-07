# RAG quality tests

These tests call a **live** API (default `http://localhost:8000`) and are marked `slow`.

## Environment

| Variable | Description |
|----------|-------------|
| `RAG_BASE_URL` | API base URL (default: `http://localhost:8000`) |
| `RAG_TEST_USER` | Login username (default: `admin`) |
| `RAG_TEST_PASSWORD` | Login password (default: `Admin@1234!`) |

Start the stack (e.g. Docker Compose) and ensure the user exists and is approved. If login returns 401, tests are **skipped** with a message to fix credentials or start the backend.

## Run

```bash
cd tests
pytest rag_quality/test_rag_quality.py -m slow -v
```

Or from repo root:

```bash
pytest tests/rag_quality/test_rag_quality.py -m slow -v
```
