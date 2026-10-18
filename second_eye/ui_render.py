"""Pure functions that turn an analyze result dict into HTML for the Gradio UI.

`result` is the JSON shape of POST /api/analyze (AnalyzeResponse.model_dump(mode="json")), or a
partial version of it while a review is running:

  None                    nothing uploaded yet (empty state)
  {"running": True, ...}  review in progress; keys appear as each reader finishes:
                          "clm" (CLM done), "reading"/"raw"/"latency_ms"/"parse_error" (VLM done),
                          "vlm_error" (VLM failed), "elapsed_ms"
  final dict              no "running" key; "total_ms" is set. A failed VLM leaves "vlm_error".

Every piece of model or user text goes through html.escape before it reaches the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any

Result = dict[str, Any] | None

REGION_LABELS = {
    "hand_wrist": "Hand / wrist",
    "leg_knee_ankle_foot": "Leg / knee / ankle / foot",
    "hip_pelvis": "Hip / pelvis",
    "shoulder": "Shoulder",
    "other": "Other region",
    "unclear": "Region unclear",
}
VERDICT_COLOUR = {"yes": "red", "no": "teal", "uncertain": "amber"}
AGREEMENT_COLOUR = {"agree": "teal", "disagree": "red", "uncertain": "amber"}

LOGO_SVG = (
    '<svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true" focusable="false">'
    '<rect x="1" y="1" width="26" height="26" rx="7" fill="#0E5E57"/>'
    '<circle cx="12" cy="12" r="6" fill="none" stroke="#FFFFFF" stroke-width="2.5"/>'
    '<path d="M16.5 16.5L22 22" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round"/></svg>'
)
NO_LOCALISATION = "The models give one reading for the whole image. They do not show where a fracture is."


@dataclass(frozen=True)
class ViewContext:
    """Settings the renderers need, kept separate so the functions stay pure and testable."""

    clm_enabled: bool = False
    clm_mock: bool = False
    vlm_mock: bool = False
    vlm_label: str = "Nemotron"


# --- small helpers ---------------------------------------------------------------------------------


def _running(r: Result) -> bool:
    return bool(r and r.get("running"))


def _vlm_done(r: Result) -> bool:
    return bool(r) and "raw" in r


def _clm_state(r: Result, ctx: ViewContext) -> str:
    if not ctx.clm_enabled:
        return "off"
    clm = (r or {}).get("clm")
    if clm is None:
        return "running" if _running(r) else "waiting"
    return "failed" if clm.get("error") else "done"


def _vlm_state(r: Result) -> str:
    if r is None:
        return "waiting"
    if r.get("vlm_error"):
        return "failed"
    if _vlm_done(r):
        return "done"
    return "running" if _running(r) else "waiting"


def _ms(value: Any) -> str:
    return f"{float(value):.0f} ms" if isinstance(value, (int, float)) else ""


def _seconds(ms: Any) -> str:
    return f"{float(ms) / 1000:.1f} s" if isinstance(ms, (int, float)) else ""


def _cap(value: Any) -> str:
    return str(value).capitalize() if value else ""


def _card(body: str, cls: str = "", label: str = "") -> str:
    aria = f' aria-label="{escape(label)}"' if label else ""
    return f'<article class="se-card {cls}"{aria}>{body}</article>'


def _placeholder(text: str) -> str:
    return _card(f"<p>{escape(text)}</p>", "placeholder")


def _mock_pill(is_mock: bool) -> str:
    return '<span class="se-pill amber">Mock, not real</span>' if is_mock else ""


# --- header, banner, chips -------------------------------------------------------------------------


def render_header(filename: str | None, result: Result = None) -> str:
    study = escape(filename) if filename else "No image yet"
    region = ((result or {}).get("reading") or {}).get("body_region")
    if region:
        study += f" · {escape(REGION_LABELS.get(region, str(region)))}"
    return (
        '<div class="se-head-left">'
        f'<div class="se-brand">{LOGO_SVG}<span>Second Eye</span></div>'
        f'<div class="se-study"><span class="k">Study</span><span class="v">{study}</span></div>'
        '<span class="se-badge" role="note">Research prototype, not a medical device</span>'
        "</div>"
    )


def render_mode_banner(mock_services: list[str]) -> str:
    """Amber banner listing every mocked service, or "" when everything is real."""
    if not mock_services:
        return ""
    chips = "".join(f'<span class="chip">{escape(s)}</span>' for s in mock_services)
    return f'<div class="se-banner" role="status"><span>Mock services in use, numbers are not real:</span>{chips}</div>'


def render_chips(chips: list[str]) -> str:
    spans = "".join(f"<span>{escape(c)}</span>" for c in chips)
    return f'<div class="se-chips">{spans}</div><p class="se-note-line">{NO_LOCALISATION}</p>'


VIEWER_PLACEHOLDER = "Drop an X-ray here, upload one, or pick a sample"


# --- pipeline strip --------------------------------------------------------------------------------


def _tile(n: int, title: str, state: str, status: str) -> str:
    state_word = {"waiting": "Waiting", "running": "Running", "done": "Done", "failed": "Failed", "off": "Off"}[state]
    css = "waiting" if state == "off" else state
    return (
        f'<div class="se-tile {css}"><span class="dot" aria-hidden="true">{n}</span>'
        f'<div><span class="t">{escape(title)}</span>'
        f'<span class="s"><span class="se-sr-only">{state_word}: </span>{status}</span></div></div>'
    )


def render_pipeline(result: Result, ctx: ViewContext) -> str:
    r = result or {}
    elapsed = _seconds(r.get("elapsed_ms"))

    # 1 Image model
    clm_state = _clm_state(result, ctx)
    clm = r.get("clm") or {}
    clm_status = {
        "off": "Not configured",
        "waiting": "Waiting for an image",
        "running": "Scoring the image…",
        "failed": "Failed: " + escape(str(clm.get("error", ""))),
    }.get(clm_state, "")
    if clm_state == "done":
        clm_status = f'p <span class="se-mono">{clm.get("p", 0):.2f}</span>'
        if clm.get("latency_ms") is not None:
            clm_status += f' · <span class="se-mono">{_ms(clm["latency_ms"])}</span>'

    # 2 Vision reader
    vlm_state = _vlm_state(result)
    if vlm_state == "done":
        reading = r.get("reading")
        verdict = _cap(reading["fracture_suspected"]) if reading else "Reply not parsed"
        vlm_status = f'{escape(verdict)} · <span class="se-mono">{_ms(r.get("latency_ms"))}</span>'
    elif vlm_state == "failed":
        vlm_status = "Failed: " + escape(str(r.get("vlm_error")))
    else:
        vlm_status = "Reading the image…" if vlm_state == "running" else "Waiting for an image"

    # 3 Compare
    agreement = r.get("agreement")
    if not ctx.clm_enabled:
        cmp_state, cmp_status = "off", "Needs the image model"
    elif agreement:
        cmp_state, cmp_status = "done", escape(_cap(agreement))
    elif "failed" in (clm_state, vlm_state):
        cmp_state, cmp_status = "waiting", "Skipped: a reader failed"
    elif clm_state == "done" and vlm_state == "done" and not _running(result):
        cmp_state, cmp_status = "waiting", "Not available: reply not parsed"
    else:
        cmp_state, cmp_status = "waiting", "Waits for both readers"

    # 4 End to end
    if _running(result):
        total_state, total_status = "running", f"Running · <span class='se-mono'>{elapsed or '0.0 s'}</span>"
    elif result is None:
        total_state, total_status = "waiting", "Waiting for an image"
    else:
        total = _seconds(r.get("total_ms"))
        total_state = "failed" if vlm_state == "failed" else "done"
        total_status = f"{'Stopped after' if total_state == 'failed' else 'Total'} <span class='se-mono'>{total}</span>"

    tiles = [
        _tile(1, "Image model · CLM", clm_state, clm_status),
        _tile(2, f"Vision reader · {ctx.vlm_label}", vlm_state, vlm_status),
        _tile(3, "Compare", cmp_state, cmp_status),
        _tile(4, "End to end", total_state, total_status),
    ]
    return f'<section class="se-pipeline" aria-label="Pipeline status">{"".join(tiles)}</section>'


# --- verdict card --------------------------------------------------------------------------------------


def _threshold_text(threshold: float | None) -> str:
    if threshold is None:
        return "No threshold; using the model's choice"
    return f"Threshold {threshold:.3f} (calibrated, chosen on validation)"


def render_verdict(result: Result, ctx: ViewContext) -> str:
    if ctx.clm_enabled:
        return _clm_verdict(result, ctx)
    return _vlm_verdict(result, ctx)


def _clm_verdict(result: Result, ctx: ViewContext) -> str:
    state = _clm_state(result, ctx)
    if state == "waiting":
        return _placeholder("The image model's fracture probability appears here.")
    if state == "running":
        return _placeholder("Image model is scoring the image…")
    clm = result["clm"]
    if state == "failed":
        return _card(f"<h2>Image model failed</h2><p>{escape(str(clm['error']))}</p>", "error", "Verdict")

    p, threshold, flagged = float(clm["p"]), clm.get("threshold"), bool(clm.get("flagged"))
    colour = "red" if flagged else "teal"
    pill = "Flagged · second look advised" if flagged else "Not flagged"
    marker = (
        f'<div class="marker" style="left:calc({float(threshold) * 100:.1f}% - 1px)"></div>' if threshold is not None else ""
    )
    bar_label = f"Probability {p:.2f}" + (f", threshold {float(threshold):.3f}" if threshold is not None else "")
    return _card(
        '<div class="head">'
        f'<span class="se-pill {colour}">{pill}</span>'
        f'<span class="meta"><span class="se-latency">{_ms(clm.get("latency_ms"))}</span></span>'
        "</div>"
        f'<div class="se-big"><span class="n">{p:.2f}</span><span class="l">probability of fracture</span>'
        f"{_mock_pill(ctx.clm_mock)}</div>"
        "<div>"
        f'<div class="se-bar" role="img" aria-label="{escape(bar_label)}">'
        f'<div class="fill {colour}" style="width:{max(0.0, min(p, 1.0)) * 100:.1f}%"></div>{marker}</div>'
        f'<div class="se-scale"><span>0</span><span>{escape(_threshold_text(threshold))}</span><span>1</span></div>'
        "</div>",
        f"ring-{colour}",
        "Verdict",
    )


def _vlm_verdict(result: Result, ctx: ViewContext) -> str:
    state = _vlm_state(result)
    if state == "waiting":
        return _placeholder("The vision reader's verdict appears here.")
    if state == "running":
        return _placeholder("Vision reader is reading the image…")
    if state == "failed":
        return _card(f"<h2>Vision reader failed</h2><p>{escape(str(result['vlm_error']))}</p>", "error", "Verdict")
    reading = result.get("reading")
    latency = f'<span class="se-latency">{_ms(result.get("latency_ms"))}</span>'
    if not reading:
        return _card(
            f'<div class="head"><span class="se-pill amber">Reply not parsed</span><span class="meta">{latency}</span></div>'
            "<p>The model's reply was not valid JSON. See the Reading card for the raw text.</p>",
            "ring-amber",
            "Verdict",
        )
    verdict = reading["fracture_suspected"]
    colour = VERDICT_COLOUR.get(verdict, "amber")
    return _card(
        f'<div class="head"><span class="se-pill grey">Fracture suspected?</span>'
        f'<span class="meta">{latency}</span></div>'
        f'<div class="se-big"><span class="n {colour}">{escape(_cap(verdict))}</span>'
        f'<span class="l">{escape(_cap(reading["confidence"]))} confidence</span>{_mock_pill(ctx.vlm_mock)}</div>'
        '<p class="aside">The vision reader gives a category, not a probability.</p>',
        f"ring-{colour}",
        "Verdict",
    )


# --- two readers, details -----------------------------------------------------------------------------


def _row(label: str, value: str) -> str:
    return f'<div class="se-row"><span>{escape(label)}</span><span class="v">{value}</span></div>'


def render_two_readers(result: Result, ctx: ViewContext) -> str:
    if not ctx.clm_enabled:
        return ""
    if result is None:
        return _card("<h2>Two readers</h2><p class='aside'>Both readers' answers and whether they agree appear here.</p>")

    clm_state = _clm_state(result, ctx)
    clm = result.get("clm") or {}
    if clm_state == "done":
        clm_value = f"{'Flagged' if clm.get('flagged') else 'Not flagged'} · <span class='se-mono'>{float(clm['p']):.2f}</span>"
    else:
        clm_value = {"running": "Running…", "failed": "Failed"}.get(clm_state, "Waiting")

    vlm_state = _vlm_state(result)
    reading = result.get("reading")
    if vlm_state == "done" and reading:
        vlm_value = f"{escape(_cap(reading['fracture_suspected']))} · {escape(reading['confidence'])}"
    elif vlm_state == "done":
        vlm_value = "Reply not parsed"
    else:
        vlm_value = {"running": "Running…", "failed": "Failed"}.get(vlm_state, "Waiting")

    agreement = result.get("agreement")
    if agreement:
        badge = f'<div><span class="se-pill {AGREEMENT_COLOUR[agreement]}">{escape(_cap(agreement))}</span></div>'
    elif _running(result):
        badge = '<p class="aside">Agreement appears when both readers finish.</p>'
    else:
        badge = '<p class="aside">Agreement not available.</p>'
    warning = (
        '<div class="se-callout" role="alert">The readers disagree. Review this image carefully.</div>'
        if agreement == "disagree"
        else ""
    )
    return _card(
        "<h2>Two readers</h2>"
        + _row("Image model (CLM)", clm_value)
        + _row(f"Vision reader ({ctx.vlm_label})", vlm_value)
        + badge
        + warning,
        label="Two readers",
    )


def render_vlm_details(result: Result) -> str:
    title = "<h2>Vision reader details</h2>"
    state = _vlm_state(result)
    reading = (result or {}).get("reading")
    if state == "done" and reading:
        return _card(
            title
            + _row("Body region", escape(REGION_LABELS.get(reading["body_region"], str(reading["body_region"]))))
            + _row("Hardware visible", "Yes" if reading["hardware_visible"] else "No")
            + _row("Image quality", escape(_cap(reading["image_quality"]))),
            label="Vision reader details",
        )
    text = {
        "waiting": "Body region, hardware and image quality appear here.",
        "running": "Waiting for the vision reader…",
        "failed": "No details: the vision reader failed.",
    }.get(state, "No details: the reply could not be parsed.")
    return _card(f"{title}<p class='aside'>{text}</p>", "placeholder")


# --- reading card ----------------------------------------------------------------------------------------


def note_text(result: Result) -> str:
    """Plain text that "Copy note" puts on the clipboard."""
    reading = (result or {}).get("reading")
    if not reading:
        return ""
    return (
        f"Findings: {reading['findings']}\n"
        f"Reasoning: {reading['reasoning']}\n"
        f"Uncertainty: {reading['limitations']}\n"
        "Draft from Second Eye, a research prototype. Not a medical device; for clinician review only."
    )


def render_reading(result: Result) -> str:
    head = (
        '<div class="head"><h2>Reading</h2><span class="aside">Draft for clinician review</span></div>'
    )
    state = _vlm_state(result)
    if state == "waiting":
        return _card(head + "<p class='aside'>Findings, reasoning and uncertainty appear here.</p>", "se-reading")
    if state == "running":
        return _card(head + "<p class='aside'>Waiting for the vision reader…</p>", "se-reading")
    if state == "failed":
        return _card(
            head + f'<div class="se-error-box" role="alert">{escape(str(result["vlm_error"]))}</div>', "se-reading"
        )
    reading = result.get("reading")
    if not reading:
        return _card(
            head
            + '<div class="se-callout" role="alert">The model\'s reply could not be read as the expected JSON. '
            "Its raw text is shown below.</div>"
            f'<pre>{escape(str(result.get("raw", "")))}</pre>',
            "se-reading",
        )
    return _card(
        head
        + f"<p>{escape(reading['findings'])}</p>"
        + f"<p>{escape(reading['reasoning'])}</p>"
        + f'<div class="se-callout"><strong>Uncertainty:</strong> {escape(reading["limitations"])}</div>'
        + f'<div id="se-note-text" hidden>{escape(note_text(result))}</div>',
        "se-reading",
    )


# --- future slots: only shown when the analyze result carries these fields ---------------------------------------


def _safe_url(url: Any) -> str:
    url = str(url or "")
    return escape(url, quote=True) if url.startswith(("https://", "http://")) else ""


def render_evidence(result: Result) -> str:
    """Expects result["evidence"] as {"quote", "title", "url", "via"} or a list of those."""
    evidence = (result or {}).get("evidence")
    if not evidence:
        return ""
    items = evidence if isinstance(evidence, list) else [evidence]
    body = "<h2>Guideline evidence</h2>"
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("quote"):
            body += f'<p class="se-quote">“{escape(str(item["quote"]))}”</p>'
        title = escape(str(item.get("title") or "Source"))
        via = f" (via {escape(str(item['via']))})" if item.get("via") else ""
        url = _safe_url(item.get("url"))
        body += f'<a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a>{via}' if url else f"<span>{title}{via}</span>"
    return _card(body, label="Guideline evidence")


def render_paraphrase(result: Result) -> str:
    """Expects result["paraphrase"] as a list of {"question", "p"} (or {"items": [...], "spread": float})."""
    para = (result or {}).get("paraphrase")
    if not para:
        return ""
    items = para.get("items", []) if isinstance(para, dict) else para
    rows = [i for i in items if isinstance(i, dict) and isinstance(i.get("p"), (int, float))]
    if not rows:
        return ""
    ps = [float(i["p"]) for i in rows]
    spread = para.get("spread") if isinstance(para, dict) and para.get("spread") is not None else (max(ps) - min(ps)) / 2
    body = (
        '<div class="head"><h2>Paraphrase check</h2>'
        f'<span class="aside">Spread <b class="se-mono">± {float(spread):.2f}</b></span></div>'
    )
    body += "".join(
        f'<div class="se-para"><span>{escape(str(i.get("question", "")))}</span><span class="p">{float(i["p"]):.2f}</span></div>'
        for i in rows
    )
    return _card(body, label="Paraphrase check")
