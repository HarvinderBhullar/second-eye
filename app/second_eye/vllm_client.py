"""Client for an OpenAI-compatible vLLM server (chat completions with an image)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any

import httpx
from pydantic import ValidationError

from .config import Settings
from .prompts import FALLBACK_SCHEMA_HINT, SYSTEM_PROMPT, USER_PROMPT
from .schemas import RESPONSE_FORMAT, Reading, ServiceHealth

log = logging.getLogger(__name__)

MOCK_REPLY = json.dumps(
    {
        "fracture_suspected": "yes",
        "confidence": "low",
        "body_region": "unclear",
        "hardware_visible": False,
        "image_quality": "limited",
        "findings": "MOCK RESPONSE, not a real reading: this text is canned and ignores the image.",
        "reasoning": "MOCK_VLLM=1 is set, so no model was called. Use this only to try the interface.",
        "limitations": "No model looked at this image. Every value here is fake.",
    }
)


class VLLMError(Exception):
    """The vLLM server could not be reached or returned an unusable reply (maps to HTTP 502)."""


def _headers(settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.vllm_api_key}"} if settings.vllm_api_key else {}


def build_payload(jpeg_bytes: bytes, model: str, structured: bool = True) -> dict[str, Any]:
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")
    text = USER_PROMPT if structured else f"{USER_PROMPT}\n{FALLBACK_SCHEMA_HINT}"
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
    }
    if structured:
        payload["response_format"] = RESPONSE_FORMAT
    return payload


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_text(text: str) -> str:
    """Strip <think> blocks and ``` fences, then return the outermost {...} span."""
    cleaned = _THINK_BLOCK.sub("", text or "")
    # Some reasoning models emit only the closing tag; drop everything before it.
    if "</think>" in cleaned.lower():
        cleaned = cleaned[cleaned.lower().rindex("</think>") + len("</think>") :]
    fence = _FENCE.search(cleaned)
    if fence:
        cleaned = fence.group(1)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def parse_reading(text: str) -> Reading | None:
    """Return a validated Reading, or None if the text is not JSON matching the schema."""
    try:
        return Reading.model_validate(json.loads(extract_json_text(text)))
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def _short(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


async def _post_chat(client: httpx.AsyncClient, settings: Settings, payload: dict[str, Any]) -> httpx.Response:
    url = f"{settings.vllm_base_url}/chat/completions"
    try:
        return await client.post(url, json=payload, headers=_headers(settings), timeout=settings.vllm_timeout)
    except httpx.TimeoutException as exc:
        raise VLLMError(f"vLLM did not answer within {settings.vllm_timeout:g} s.") from exc
    except httpx.HTTPError as exc:
        raise VLLMError(f"Cannot reach vLLM at {settings.vllm_base_url}: {type(exc).__name__}.") from exc


async def review_image(jpeg_bytes: bytes, settings: Settings, client: httpx.AsyncClient) -> tuple[str, float]:
    """Ask the VLM for a reading. Returns (raw model text, vLLM round-trip ms)."""
    started = time.perf_counter()
    if settings.mock_vllm:
        await asyncio.sleep(0.6)  # long enough to see the loading state
        return MOCK_REPLY, (time.perf_counter() - started) * 1000

    if not settings.vllm_base_url or not settings.vllm_model:
        raise VLLMError("VLLM_BASE_URL and VLLM_MODEL must be set (or set MOCK_VLLM=1).")

    response = await _post_chat(client, settings, build_payload(jpeg_bytes, settings.vllm_model))
    if response.status_code in (400, 422):
        # Most likely this server or model does not support response_format / guided decoding.
        log.info("vLLM rejected response_format (HTTP %s); retrying without it", response.status_code)
        response = await _post_chat(client, settings, build_payload(jpeg_bytes, settings.vllm_model, structured=False))
    latency_ms = (time.perf_counter() - started) * 1000

    if response.status_code >= 400:
        raise VLLMError(f"vLLM returned HTTP {response.status_code}: {_short(response.text)}")
    try:
        message = response.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise VLLMError(f"vLLM sent an unexpected response: {_short(response.text)}") from exc
    return message.get("content") or "", latency_ms


async def health(settings: Settings, client: httpx.AsyncClient) -> ServiceHealth:
    if settings.mock_vllm:
        return ServiceHealth(ok=True, model_listed=True, models=[settings.model_name])
    if not settings.vllm_base_url:
        return ServiceHealth(ok=False, error="VLLM_BASE_URL is not set.")
    try:
        response = await client.get(f"{settings.vllm_base_url}/models", headers=_headers(settings), timeout=10)
        response.raise_for_status()
        models = [m.get("id", "") for m in response.json().get("data", [])]
    except httpx.HTTPStatusError as exc:
        return ServiceHealth(ok=False, status_code=exc.response.status_code, error=f"HTTP {exc.response.status_code}")
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return ServiceHealth(ok=False, error=type(exc).__name__)
    return ServiceHealth(ok=True, model_listed=settings.vllm_model in models, models=models)
