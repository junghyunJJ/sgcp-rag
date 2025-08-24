import os
import sys

import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mcp.mcp_server as mcp_mod


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Clear environment keys and module-level constants
    monkeypatch.delenv("SUPABASE_JWT_SECRET", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(mcp_mod, "SUPABASE_JWT_SECRET", "")
    monkeypatch.setattr(mcp_mod, "OPENAI_API_KEY", "")


@pytest.fixture
def dummy_request(monkeypatch):
    """Monkeypatch mcp_mod.client.request to return given response data.
    Usage: dummy_request(data)
    """

    def _setter(response_data):
        async def _dummy_request(method, endpoint, **kwargs):
            return response_data

        monkeypatch.setattr(mcp_mod.client, "request", _dummy_request)

    return _setter


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class DummyAsyncClient:
    def __init__(self, response_data):
        self._response = DummyResponse(response_data)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return self._response


@pytest.fixture
def dummy_http_client(monkeypatch):
    """Monkeypatch httpx.AsyncClient for add_documents.
    Usage: dummy_http_client(response_data)
    """

    def _setter(response_data):
        # ensure token in headers
        monkeypatch.setenv("SUPABASE_JWT_SECRET", "token")
        monkeypatch.setattr(
            mcp_mod, "client", mcp_mod.LangConnectClient(mcp_mod.API_BASE_URL, "token")
        )
        dummy = DummyAsyncClient(response_data)
        monkeypatch.setattr(mcp_mod.httpx, "AsyncClient", lambda *args, **kwargs: dummy)

    return _setter
