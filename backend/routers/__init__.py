# routers package — modular FastAPI routers split out of api_server.py.
#
# Each submodule defines a single ``APIRouter`` named ``router`` that is
# included by ``api_server.py``. Endpoint paths are unchanged from the
# original monolithic file — the frontend depends on them as-is.
