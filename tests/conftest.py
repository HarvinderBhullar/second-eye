from __future__ import annotations

import io
import json

import httpx
import pytest
import pytest_asyncio
from PIL import Image

from second_eye.config import Settings, get_settings
from second_eye.main import create_api

VLLM = "http://vllm.test/v1"
CLM = "http://clm.test"

GOOD_READING = {
    "fracture_suspected": "yes",
    "confidence": "medium",
    "body_region": "hand_wrist",
    "hardware_visible": False,
    "image_quality": "good",
    "findings": "Lucent line through the distal radius.",
    "reasoning": "Cortical break is visible on this view.",
    "limitations": "Single view only.",
}


def chat_reply(content: str) -> dict:
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]}


def clm_reply(p: float, threshold: float | None = 0.5, choice: str | None = None) -> dict:
    fx: dict = {"probabilities": {"fracture": p, "no_fracture": 1 - p}}
    if threshold is not None:
        fx["threshold"] = threshold
    if choice is not None:
        fx["choice"] = choice
    return {"answers": {"fx": fx}}


def make_png(size: tuple[int, int] = (64, 48), mode: str = "L") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, 128).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def png() -> bytes:
    return make_png()


@pytest.fixture
def settings() -> Settings:
    return Settings(vllm_base_url=VLLM, vllm_model="nvidia/test-vl", vllm_timeout=5)


@pytest.fixture
def clm_settings(settings: Settings) -> Settings:
    return Settings(**{**settings.__dict__, "clm_url": CLM})


@pytest_asyncio.fixture
async def make_client():
    """Return an async HTTP client for the API wired to the given Settings."""
    clients: list[httpx.AsyncClient] = []

    async def _make(s: Settings) -> httpx.AsyncClient:
        api = create_api()
        api.dependency_overrides[get_settings] = lambda: s
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://app.test")
        clients.append(client)
        return client

    yield _make
    for c in clients:
        await c.aclose()


def sent_json(route) -> dict:
    return json.loads(route.calls.last.request.content)
