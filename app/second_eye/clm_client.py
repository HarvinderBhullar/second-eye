"""Optional client for the Second Eye image head (`python -m clm.vision.serve`). Never raises.

The server takes the image as a multipart upload on POST /predict and answers with a calibrated
distribution over its trained options. We send no `options`, so it uses the trained pair: Platt
calibration and the threshold only hold for those exact texts.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .schemas import ClmResult, ServiceHealth

# The head's trained options; index 0 is the positive class. The server echoes the options it used,
# and we prefer those, but fall back to these if a response omits them.
TRAINED_OPTIONS = ("Yes, there is a fracture.", "No, there is no fracture.")


def _headers(settings: Settings) -> dict[str, str]:
    # One key works for both the Nebius endpoint token and the server's own SECOND_EYE_API_KEY.
    return {"Authorization": f"Bearer {settings.clm_api_key}"} if settings.clm_api_key else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_answer(body: dict[str, Any]) -> ClmResult:
    options = body.get("options") or list(TRAINED_OPTIONS)
    positive = options[0]
    p = _float_or_none(body["probabilities"][positive])
    if p is None:
        raise ValueError("probability of the positive option is not a number")
    threshold = _float_or_none(body.get("threshold"))
    flagged = p >= threshold if threshold is not None else body.get("decision") == positive
    return ClmResult(
        p=p,
        threshold=threshold,
        flagged=flagged,
        latency_ms=_float_or_none(body.get("latency_ms")),
        calibrated=body.get("calibrated"),
        operating_point=body.get("operating_point"),
    )


async def score(original_bytes: bytes, settings: Settings, client: httpx.AsyncClient) -> ClmResult:
    """Score the original upload. Any failure comes back as ClmResult(error=...)."""
    try:
        response = await client.post(
            f"{settings.clm_url}/predict",
            files={"image": ("upload", original_bytes, "application/octet-stream")},
            headers=_headers(settings),
            timeout=settings.clm_timeout,
        )
        response.raise_for_status()
        return parse_answer(response.json())
    except httpx.TimeoutException:
        return ClmResult(error=f"Image model did not answer within {settings.clm_timeout:g} s.")
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        hint = " (check CLM_API_KEY)" if status in (401, 403) else ""
        return ClmResult(error=f"Image model returned HTTP {status}{hint}.")
    except httpx.HTTPError as exc:
        return ClmResult(error=f"Cannot reach the image model: {type(exc).__name__}.")
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return ClmResult(error=f"Image model sent an unexpected response ({type(exc).__name__}).")


async def health(settings: Settings, client: httpx.AsyncClient) -> ServiceHealth | None:
    if not settings.clm_enabled:
        return None
    try:
        response = await client.get(f"{settings.clm_url}/health", headers=_headers(settings), timeout=10)
    except httpx.HTTPError as exc:
        return ServiceHealth(ok=False, error=type(exc).__name__)
    if response.status_code >= 400:
        return ServiceHealth(ok=False, status_code=response.status_code, error=f"HTTP {response.status_code}")
    try:
        variant = response.json().get("variant")
    except (ValueError, AttributeError):
        variant = None
    return ServiceHealth(ok=True, status_code=response.status_code, models=[variant or settings.clm_model])
