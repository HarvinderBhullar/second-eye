"""FastAPI app: JSON API under /api, Gradio UI mounted at /."""

from __future__ import annotations

import logging
import os

from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from .analyze import InvalidUpload, analyze, check_health
from .config import MAX_UPLOAD_BYTES, Settings, get_settings
from .schemas import AnalyzeResponse, HealthResponse
from .vllm_client import VLLMError

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_api() -> FastAPI:
    api = FastAPI(title="Second Eye", description="Research prototype, not a medical device.", version="0.1.0")

    @api.exception_handler(InvalidUpload)
    async def _invalid_upload(_: Request, exc: InvalidUpload) -> JSONResponse:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    @api.exception_handler(VLLMError)
    async def _vllm_error(_: Request, exc: VLLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"error": str(exc)})

    @api.get("/api/health", response_model=HealthResponse)
    async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
        return await check_health(settings)

    @api.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_endpoint(file: UploadFile = File(...), settings: Settings = Depends(get_settings)) -> AnalyzeResponse:
        # Read one byte past the limit so oversized uploads are rejected without buffering all of them.
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        return await analyze(data, settings)

    return api


def create_app() -> FastAPI:
    import gradio as gr

    from .ui import CSS, JS_ON_LOAD, THEME, build_ui

    return gr.mount_gradio_app(
        create_api(),
        build_ui(get_settings()),
        path="/",
        theme=THEME,
        css=CSS,
        js=JS_ON_LOAD,
        max_file_size=MAX_UPLOAD_BYTES,
        footer_links=[],
    )


app = create_app()
