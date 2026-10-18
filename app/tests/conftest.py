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


YES, NO = "Yes, there is a fracture.", "No, there is no fracture."


def clm_reply(p: float, threshold: float | None = 0.5, decision: str | None = None) -> dict:
    """A reply shaped like `clm.vision.serve` POST /predict."""
    return {
        "question": "Is there a bone fracture in this X-ray?",
        "options": [YES, NO],
        "probabilities": {YES: p, NO: 1 - p},
        "calibrated": threshold is not None,
        "decision": decision or (YES if threshold is not None and p >= threshold else NO),
        "threshold": threshold,
        "operating_point": "val sensitivity >= 0.90" if threshold is not None else "argmax",
        "latency_ms": 37.5,
    }


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
    return Settings(**{**settings.__dict__, "clm_url": CLM, "clm_api_key": "clm-secret"})


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
