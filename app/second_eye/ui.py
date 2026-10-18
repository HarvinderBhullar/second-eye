"""Gradio Blocks UI. Layout and event wiring only: HTML comes from ui_render, styles from ui_style,
and all model logic from analyze.py."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import gradio as gr

from .analyze import InvalidUpload, analyze_stream, check_health
from .config import Settings
from .ui_render import (
    VIEWER_PLACEHOLDER,
    ViewContext,
    render_chips,
    render_evidence,
    render_header,
    render_mode_banner,
    render_paraphrase,
    render_pipeline,
    render_reading,
    render_two_readers,
    render_verdict,
    render_vlm_details,
)
from .ui_style import CSS, JS_COPY_NOTE, JS_ON_LOAD, THEME

__all__ = ["CSS", "JS_ON_LOAD", "THEME", "build_ui"]

log = logging.getLogger(__name__)

FOOTER = "<footer class='se-footer'>Second Eye supports a clinician's review and does not replace it.</footer>"
TICK_S = 0.5


def view_context(settings: Settings) -> ViewContext:
    return ViewContext(
        clm_enabled=settings.clm_enabled,
        clm_mock=settings.clm_enabled and settings.mock_clm,
        vlm_mock=settings.mock_vllm,
        vlm_label=settings.vlm_label,
    )


def chips_for(settings: Settings) -> list[str]:
    chips = []
    if settings.clm_enabled:
        if settings.encoder_label:
            chips.append(f"Encoder: {settings.encoder_label}")
        chips.append(f"Head: {settings.head_label or settings.clm_model}")
    if settings.served_on:
        chips.append(f"Served on {settings.served_on}")
    chips.append(f"VLM: {settings.model_name or 'not configured'}")
    return chips


def mock_services(settings: Settings) -> list[str]:
    services = []
    if settings.clm_enabled and settings.mock_clm:
        services.append("Image model (fake CLM)")
    if settings.mock_vllm:
        services.append("Vision reader (MOCK_VLLM=1)")
    return services


def build_ui(settings: Settings) -> gr.Blocks:
    ctx = view_context(settings)
    demos = {label: path for label, path in settings.demo_images if Path(path).is_file()}
    for label, path in settings.demo_images:
        if label not in demos:
            log.warning("DEMO_IMAGES entry %r skipped: %s not found", label, path)

    def render_all(result: dict[str, Any] | None, filename: str | None) -> tuple:
        evidence, paraphrase = render_evidence(result), render_paraphrase(result)
        final = result is not None and not result.get("running") and "raw" in result
        return (
            render_header(filename, result),
            render_pipeline(result, ctx),
            render_verdict(result, ctx),
            render_two_readers(result, ctx),
            render_vlm_details(result),
            render_reading(result),
            gr.HTML(evidence, visible=bool(evidence)),
            gr.HTML(paraphrase, visible=bool(paraphrase)),
            result.get("raw", "") if result else "",
            json.dumps({k: v for k, v in result.items() if k != "running"}, indent=2) if final else "",
        )

    async def run_review(path: str | None):
        if not path:
            yield render_all(None, None)
            return
        filename = Path(path).name
        started = time.perf_counter()
        state: dict[str, Any] = {"running": True, "elapsed_ms": 0.0}
        yield render_all(state, filename)

        try:
            stream = analyze_stream(Path(path).read_bytes(), settings)
        except OSError:
            gr.Warning("The uploaded file could not be read.")
            yield render_all({"vlm_error": "The uploaded file could not be read.", "total_ms": 0}, filename)
            return
        pending = asyncio.ensure_future(anext(stream))
        try:
            while True:
                done, _ = await asyncio.wait({pending}, timeout=TICK_S)
                state["elapsed_ms"] = (time.perf_counter() - started) * 1000
                if not done:
                    yield render_all(state, filename)
                    continue
                try:
                    kind, value = pending.result()
                except StopAsyncIteration:
                    break
                if kind == "clm":
                    state["clm"] = value.model_dump(mode="json")
                elif kind == "vlm":
                    reading = value["reading"]
                    state.update(
                        raw=value["raw"],
                        latency_ms=round(value["latency_ms"], 1),
                        reading=reading.model_dump(mode="json") if reading else None,
                        parse_error=reading is None,
                    )
                elif kind == "vlm_error":
                    state["vlm_error"] = str(value)
                    gr.Warning(str(value))
                elif kind == "done":
                    if value is not None:
                        state = value.model_dump(mode="json")
                    else:
                        state.pop("running", None)
                        state["total_ms"] = round(state["elapsed_ms"], 1)
                yield render_all(state, filename)
                pending = asyncio.ensure_future(anext(stream))
        except InvalidUpload as exc:
            gr.Warning(str(exc))
            yield render_all({"vlm_error": str(exc), "total_ms": state["elapsed_ms"]}, filename)
        except Exception:
            log.exception("Unexpected error during review")
            message = "Unexpected server error. Check the server logs."
            gr.Warning(message)
            yield render_all({**state, "running": False, "vlm_error": message, "total_ms": state["elapsed_ms"]}, filename)
        finally:
            pending.cancel()
            await stream.aclose()

    async def load_page() -> str:
        # The mode banner reflects configuration; health is checked so the log shows reachability early.
        await check_health(settings)
        return render_mode_banner(mock_services(settings))

    def pick_demo(label: str | None) -> str | None:
        return demos.get(label or "")

    def toggle_raw(is_open: bool) -> tuple:
        is_open = not is_open
        return (
            gr.Accordion(open=is_open, visible=is_open),
            gr.Button("Hide raw response" if is_open else "Raw response"),
            is_open,
        )

    with gr.Blocks(title="Second Eye", delete_cache=(600, 900)) as demo:
        with gr.Row(elem_id="se-header", equal_height=False):
            header = gr.HTML(render_header(None), elem_classes="se-plain", apply_default_css=False)
            with gr.Row(elem_id="se-head-right"):
                demo_pick = gr.Radio(
                    choices=list(demos),
                    value=None,
                    label="Sample X-ray",
                    show_label=False,
                    container=False,
                    elem_id="se-demo",
                    visible=bool(demos),
                )
                upload_btn = gr.UploadButton(
                    "Upload X-ray", file_types=["image"], variant="primary", size="lg", elem_id="se-upload"
                )

        banner = gr.HTML(render_mode_banner(mock_services(settings)), elem_classes="se-plain", apply_default_css=False)
        pipeline = gr.HTML(render_pipeline(None, ctx), elem_classes="se-plain", apply_default_css=False)

        with gr.Row(elem_id="se-main", equal_height=False):
            with gr.Column(elem_id="se-left", scale=3, min_width=0):
                image = gr.Image(
                    type="filepath",
                    image_mode=None,  # hand the backend the original file, untouched
                    label="X-ray image",
                    sources=["upload", "clipboard"],
                    placeholder=VIEWER_PLACEHOLDER,
                    elem_id="se-viewer",
                    buttons=["fullscreen"],
                    container=True,
                )
                gr.HTML(render_chips(chips_for(settings)), elem_classes="se-plain", apply_default_css=False)

            with gr.Column(elem_id="se-results", scale=2, min_width=0):
                plain = {"elem_classes": "se-plain", "apply_default_css": False}
                verdict = gr.HTML(render_verdict(None, ctx), **plain)
                readers = gr.HTML(render_two_readers(None, ctx), visible=ctx.clm_enabled, **plain)
                details = gr.HTML(render_vlm_details(None), **plain)
                with gr.Column(elem_id="se-reading-card"):
                    reading = gr.HTML(render_reading(None), **plain)
                    with gr.Row(elem_id="se-note-actions"):
                        copy_btn = gr.Button("Copy note", variant="secondary", elem_id="se-copy")
                        raw_btn = gr.Button("Raw response", variant="secondary", elem_id="se-raw-toggle")
                    with gr.Accordion("Raw response", open=False, visible=False, elem_id="se-raw") as raw_box:
                        raw = gr.Textbox(label="Raw model text", lines=6, max_lines=20, interactive=False, buttons=["copy"])
                        full = gr.Textbox(label="Full JSON response", lines=10, max_lines=30, interactive=False, buttons=["copy"])
                evidence = gr.HTML("", visible=False, **plain)
                paraphrase = gr.HTML("", visible=False, **plain)

        gr.HTML(FOOTER, elem_classes="se-plain", apply_default_css=False)
        raw_open = gr.State(False)

        outputs = [header, pipeline, verdict, readers, details, reading, evidence, paraphrase, raw, full]
        review = {"fn": run_review, "inputs": image, "outputs": outputs, "show_progress": "hidden", "concurrency_limit": 4}

        image.upload(**review)
        image.clear(lambda: render_all(None, None), outputs=outputs, show_progress="hidden")
        upload_btn.upload(lambda f: f, inputs=upload_btn, outputs=image, show_progress="hidden").then(**review)
        demo_pick.input(pick_demo, inputs=demo_pick, outputs=image, show_progress="hidden").then(**review)
        copy_btn.click(None, js=JS_COPY_NOTE)
        raw_btn.click(toggle_raw, inputs=raw_open, outputs=[raw_box, raw_btn, raw_open], show_progress="hidden")
        demo.load(load_page, outputs=banner, show_progress="hidden")
    return demo
