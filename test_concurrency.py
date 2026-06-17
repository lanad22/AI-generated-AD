# test_concurrency.py
import os
import asyncio
import pytest
from fastapi.testclient import TestClient

# Force the cap BEFORE importing server so the module-level semaphore picks it up.
os.environ["MAX_CONCURRENT_PIPELINES"] = "1"

import server  # noqa: E402


@pytest.fixture
def client():
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def reset_state():
    # Re-create the semaphore for each test so prior state can't leak.
    server.pipeline_semaphore = asyncio.Semaphore(1)
    yield


def test_returns_503_when_semaphore_at_cap(client, monkeypatch):
    # Force the "no existing results" branches: skip past glob + S3 checks.
    monkeypatch.setattr(server.glob, "glob", lambda *_: [])
    monkeypatch.setattr(server, "check_and_download_final_data_from_s3", lambda *_: False)

    # Make the semaphore have zero slots — any acquire attempt fails immediately.
    server.pipeline_semaphore = asyncio.Semaphore(0)

    response = client.post(
        "/api/generate-ai-description",
        json={"youtube_id": "abc123", "data_type": "gpt"},
    )

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "busy"


def test_accepts_request_when_slot_free(client, monkeypatch):
    # Make the background pipeline a no-op so the test doesn't actually run anything.
    async def fake_run(*args, **kwargs):
        pass

    monkeypatch.setattr(server, "run_pipeline_and_forward", fake_run)

    # Force the "no existing results" branches.
    monkeypatch.setattr(server.glob, "glob", lambda *_: [])
    monkeypatch.setattr(server, "check_and_download_final_data_from_s3", lambda *_: False)

    response = client.post(
        "/api/generate-ai-description",
        json={"youtube_id": "xyz789", "data_type": "gpt"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
