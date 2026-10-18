"""Optional client for clm-serve (a calibrated classifier head). Never raises."""

from __future__ import annotations

import base64
from typing import Any

import httpx

from .config import Settings
from .schemas import ClmResult, ServiceHealth

# The head was trained and calibrated on exactly these option texts. Do not edit them.
FX_QUESTION: dict[str, Any] = {
    "type": "choice",
    "instructions": "Is there a bone fracture?",
    "criteria": {
        "fracture": "Radiograph showing an acute bone fracture.",
        "no_fracture": "Radiograph of intact bones with no fracture.",
    },
}

CLM_TIMEOUT_S = 30.0


def build_payload(original_bytes: bytes, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "state": {"type": "image", "image": base64.b64encode(original_bytes).decode("ascii")},
        "questions": {"fx": FX_QUESTION},
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_answer(body: dict[str, Any], latency_header: str | None) -> ClmResult:
    fx = body["answers"]["fx"]
    p = _float_or_none(fx["probabilities"]["fracture"])
    if p is None:
        raise ValueError("answers.fx.probabilities.fracture is not a number")
    threshold = _float_or_none(fx.get("threshold"))
    flagged = p >= threshold if threshold is not None else fx.get("choice") == "fracture"
    return ClmResult(p=p, threshold=threshold, flagged=flagged, latency_ms=_float_or_none(latency_header))


async def score(original_bytes: bytes, settings: Settings, client: httpx.AsyncClient) -> ClmResult:
    """Score the original upload. Any failure comes back as ClmResult(error=...)."""
    try:
        response = await client.post(
            f"{settings.clm_url}/v1/systemone",
            json=build_payload(original_bytes, settings.clm_model),
            timeout=CLM_TIMEOUT_S,
        )
        response.raise_for_status()
        return parse_answer(response.json(), response.headers.get("X-CLM-Latency-Ms"))
    except httpx.TimeoutException:
        return ClmResult(error=f"CLM did not answer within {CLM_TIMEOUT_S:g} s.")
    except httpx.HTTPStatusError as exc:
        return ClmResult(error=f"CLM returned HTTP {exc.response.status_code}.")
    except httpx.HTTPError as exc:
        return ClmResult(error=f"Cannot reach CLM: {type(exc).__name__}.")
    except (ValueError, KeyError, TypeError) as exc:
        return ClmResult(error=f"CLM sent an unexpected response ({type(exc).__name__}).")


async def health(settings: Settings, client: httpx.AsyncClient) -> ServiceHealth | None:
    if not settings.clm_enabled:
        return None
    try:
        response = await client.get(f"{settings.clm_url}/health", timeout=10)
    except httpx.HTTPError as exc:
        return ServiceHealth(ok=False, error=type(exc).__name__)
    ok = response.status_code < 400
    return ServiceHealth(
        ok=ok, status_code=response.status_code, models=[settings.clm_model], error=None if ok else f"HTTP {response.status_code}"
    )
