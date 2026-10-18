"""The one analysis path shared by POST /api/analyze and the Gradio UI."""

from __future__ import annotations

import asyncio
import io
import time
import warnings
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from typing import Any

import httpx
from PIL import Image, ImageOps

from . import clm_client, vllm_client
from .config import JPEG_QUALITY, MAX_SIDE_PX, MAX_UPLOAD_BYTES, Settings, get_settings
from .schemas import Agreement, AnalyzeResponse, ClmResult, HealthResponse, Reading


class InvalidUpload(ValueError):
    """The upload is empty, too large or not a decodable image (maps to HTTP 422)."""


def _to_8bit(img: Image.Image) -> Image.Image:
    """Stretch 16-bit / 32-bit greyscale (common for radiographs) to 8-bit instead of clipping."""
    img = img.convert("I") if img.mode.startswith("I;16") else img
    lo, hi = img.getextrema()
    if hi <= lo:
        return img.convert("L")
    scale = 255.0 / (hi - lo)
    return img.point(lambda v: v * scale - lo * scale).convert("L")


def prepare_image(data: bytes) -> bytes:
    """Validate the upload and return an RGB JPEG whose longest side is at most MAX_SIDE_PX. In memory only."""
    if not data:
        raise InvalidUpload("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise InvalidUpload(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            img.load()
    except Exception as exc:  # Pillow raises many types for bad input
        raise InvalidUpload("The file is not an image that can be read (try PNG or JPEG).") from exc

    img = ImageOps.exif_transpose(img)
    if img.mode in ("I", "F") or img.mode.startswith("I;16"):
        img = _to_8bit(img)
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY)
    return out.getvalue()


def compute_agreement(clm: ClmResult | None, reading: Reading | None) -> Agreement | None:
    if clm is None or clm.flagged is None or reading is None:
        return None
    if reading.fracture_suspected == "uncertain":
        return "uncertain"
    return "agree" if clm.flagged == (reading.fracture_suspected == "yes") else "disagree"


async def analyze_stream(
    data: bytes,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Run both readers concurrently and report each one as it finishes.

    Yields, in completion order:
      ("clm", ClmResult)                                      only when CLM_URL is set
      ("vlm", {"raw": str, "latency_ms": float, "reading": Reading | None})  or  ("vlm_error", VLLMError)
    and finally ("done", AnalyzeResponse | None), where None means the VLM failed.
    Raises InvalidUpload before yielding anything.
    """
    settings = settings or get_settings()
    started = time.perf_counter()
    jpeg = prepare_image(data)

    async with AsyncExitStack() as stack:
        http = client or await stack.enter_async_context(httpx.AsyncClient())
        tasks = {asyncio.create_task(vllm_client.review_image(jpeg, settings, http)): "vlm"}
        if settings.clm_enabled:
            tasks[asyncio.create_task(clm_client.score(data, settings, http))] = "clm"

        clm: ClmResult | None = None
        vlm: dict[str, Any] | None = None
        pending = set(tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    if tasks[task] == "clm":
                        clm = task.result()
                        yield "clm", clm
                        continue
                    try:
                        raw, latency_ms = task.result()
                    except vllm_client.VLLMError as exc:
                        yield "vlm_error", exc
                    else:
                        vlm = {"raw": raw, "latency_ms": latency_ms, "reading": vllm_client.parse_reading(raw)}
                        yield "vlm", vlm
        finally:
            for task in pending:
                task.cancel()

    if vlm is None:
        yield "done", None
        return
    reading = vlm["reading"]
    yield "done", AnalyzeResponse(
        reading=reading,
        parse_error=reading is None,
        raw=vlm["raw"],
        model=settings.model_name,
        latency_ms=round(vlm["latency_ms"], 1),
        total_ms=round((time.perf_counter() - started) * 1000, 1),
        mock=settings.mock_vllm,
        clm=clm,
        agreement=compute_agreement(clm, reading),
    )


async def analyze(
    data: bytes,
    settings: Settings | None = None,
    client: httpx.AsyncClient | None = None,
) -> AnalyzeResponse:
    """Run the full review. Raises InvalidUpload or vllm_client.VLLMError; a CLM failure is reported, not raised."""
    result: AnalyzeResponse | None = None
    error: vllm_client.VLLMError | None = None
    async for kind, value in analyze_stream(data, settings, client):
        if kind == "vlm_error":
            error = value
        elif kind == "done":
            result = value
    if error is not None:
        raise error
    assert result is not None
    return result


async def check_health(settings: Settings | None = None, client: httpx.AsyncClient | None = None) -> HealthResponse:
    settings = settings or get_settings()

    async def run(http: httpx.AsyncClient) -> HealthResponse:
        vllm, clm = await asyncio.gather(vllm_client.health(settings, http), clm_client.health(settings, http))
        return HealthResponse(vllm=vllm, clm=clm, mock=settings.mock_vllm)

    if client is not None:
        return await run(client)
    async with httpx.AsyncClient() as http:
        return await run(http)
