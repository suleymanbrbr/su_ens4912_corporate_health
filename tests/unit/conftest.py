"""
Unit-level conftest.py
Overrides the session-scoped app/test_client fixtures with no-ops
so unit tests can run without FastAPI, database, or ML models installed.
"""
import pytest


# These overrides prevent the parent conftest from trying to
# import api_server / FastAPI during unit test collection.
@pytest.fixture(scope="session")
def app():
    pytest.skip("app fixture not available in unit tests — use integration suite")


@pytest.fixture(scope="session")
def test_client():
    pytest.skip("test_client not available in unit tests — use integration suite")
