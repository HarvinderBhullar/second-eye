"""Theme and the one CSS string for the Gradio UI. Design tokens are defined once, as CSS variables."""

from __future__ import annotations

import gradio as gr

# Design tokens (from design/second-eye-reference.html). The theme below repeats a few of them
# because Gradio's theme API takes literal values, not CSS variables.
BG = "#F4F5F2"
PANEL = "#FFFFFF"
BORDER = "#D9DCD6"
TEXT = "#16202A"
TEXT_2 = "#2E3A45"
MUTED = "#45505B"
TEAL = "#0E5E57"
TEAL_BG = "#EAF3F1"
BTN_BORDER = "#C4C9C2"
FOCUS = "#7FB3D5"

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "-apple-system", "Segoe UI", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
    text_size=gr.themes.sizes.text_md,
).set(
    body_background_fill=BG,
    body_text_color=TEXT,
    body_text_color_subdued=MUTED,
    background_fill_primary=PANEL,
    background_fill_secondary=BG,
    block_background_fill=PANEL,
    block_border_color=BORDER,
    block_border_width="1px",
    block_label_text_color=MUTED,
    block_title_text_color=TEXT,
    border_color_primary=BORDER,
    border_color_accent=TEAL,
    color_accent=TEAL,
    color_accent_soft=TEAL_BG,
    link_text_color="#1F5F8B",
    link_text_color_hover="#123E5C",
    button_primary_background_fill=TEAL,
    button_primary_background_fill_hover="#0A4A44",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color=TEAL,
    button_secondary_background_fill=PANEL,
    button_secondary_background_fill_hover=BG,
    button_secondary_text_color=TEXT,
    button_secondary_border_color=BTN_BORDER,
    button_border_width="1px",
    button_large_radius="8px",
    button_medium_radius="8px",
    button_small_radius="8px",
    checkbox_background_color_selected=TEAL,
    checkbox_border_color_focus=FOCUS,
    input_border_color_focus=FOCUS,
    loader_color=TEAL,
    slider_color=TEAL,
    shadow_drop="none",
    shadow_drop_lg="none",
)

CSS = """
:root {
  --se-bg: #F4F5F2; --se-panel: #FFFFFF; --se-border: #D9DCD6; --se-divider: #E4E6E1;
  --se-text: #16202A; --se-text-2: #2E3A45; --se-muted: #45505B;
  --se-teal: #0E5E57; --se-teal-bg: #EAF3F1;
  --se-blue: #2F5D8A; --se-blue-bg: #E6EEF8;
  --se-red: #B42318; --se-red-pill: #F9D6D2; --se-red-text: #8A1A12; --se-red-bg: #FDF0EE;
  --se-amber-bg: #FBEEDD; --se-amber-text: #6E3F00;
  --se-grey-bg: #F0F1EE; --se-grey-dot: #6B7580; --se-seg-bg: #ECEEEA;
  --se-link: #1F5F8B; --se-link-hover: #123E5C; --se-viewer: #0F1419;
  --se-btn-border: #C4C9C2; --se-focus: #7FB3D5;
  --se-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --se-r-card: 12px; --se-r-tile: 10px; --se-r-btn: 8px; --se-r-pill: 999px;
}

/* ---- page frame: undo Gradio's container padding and footer ---- */
body, gradio-app, .gradio-container { background: var(--se-bg) !important; }
.gradio-container { max-width: none !important; padding: 0 !important; margin: 0 !important;
  color: var(--se-text); font-size: 15px; line-height: 1.5; }
.gradio-container .main { max-width: none !important; }
.gradio-container > .main, .gradio-container .main > .wrap, .gradio-container .contain { padding: 0 !important; gap: 0 !important; }
.gradio-container .contain > .column { gap: 0 !important; }
footer:not(.se-footer) { display: none !important; }  /* Gradio's own footer */
.se-plain, .se-plain > .html-container { padding: 0 !important; border: 0 !important; background: none !important; box-shadow: none !important; }
.gradio-container a { color: var(--se-link); } .gradio-container a:hover { color: var(--se-link-hover); }
.se-mono { font-family: var(--se-mono); }

/* ---- focus: 3px ring everywhere ---- */
.gradio-container :focus-visible, .gradio-container button:focus-visible {
  outline: 3px solid var(--se-focus) !important; outline-offset: 2px !important; box-shadow: none !important; }
.se-sr-only { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0; }

/* ---- header ---- */
#se-header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 16px;
  padding: 16px 32px; background: var(--se-panel); border-bottom: 1px solid var(--se-border); }
#se-header > div { flex: 0 1 auto !important; min-width: 0 !important; width: auto !important; }
.se-head-left { display: flex; flex-wrap: wrap; align-items: center; gap: 24px; }
.se-brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 20px; letter-spacing: -0.01em; }
.se-study { display: flex; flex-direction: column; min-width: 0; }
.se-study .k { font-size: 12px; color: var(--se-muted); text-transform: uppercase; letter-spacing: .06em; }
.se-study .v { font-weight: 600; overflow-wrap: anywhere; }
.se-badge { padding: 4px 10px; border-radius: var(--se-r-pill); background: var(--se-amber-bg); color: var(--se-amber-text);
  font-size: 13px; font-weight: 500; }
#se-head-right { display: flex !important; flex-direction: row !important; flex-wrap: wrap; align-items: center; gap: 12px !important; }
#se-head-right > * { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }

/* segmented control (a gr.Radio) */
#se-demo { padding: 4px !important; border-radius: var(--se-r-tile) !important; background: var(--se-seg-bg) !important;
  border: 0 !important; }
#se-demo .wrap { display: flex; flex-wrap: nowrap; gap: 0 !important; }
#se-demo label { min-height: 40px; padding: 0 14px !important; margin: 0 !important; border-radius: var(--se-r-btn) !important;
  border: 0 !important; background: transparent !important; box-shadow: none !important; color: var(--se-muted) !important;
  font-size: 15px !important; display: flex; align-items: center; cursor: pointer; }
#se-demo label.selected { background: var(--se-panel) !important; color: var(--se-text) !important; font-weight: 600;
  box-shadow: 0 1px 2px rgba(0,0,0,.12) !important; }
#se-demo input[type=radio] { position: absolute; opacity: 0; width: 1px; height: 1px; outline: none !important; }
#se-demo label:has(input:focus-visible) { outline: 3px solid var(--se-focus); outline-offset: 1px; }

/* buttons */
.gradio-container button.primary, .gradio-container button.secondary { min-height: 44px; border-radius: var(--se-r-btn) !important;
  font-weight: 600; font-size: 15px; box-shadow: none !important; }
.gradio-container button.primary { background: var(--se-teal) !important; color: #fff !important; border: 0 !important; padding: 0 20px !important; }
.gradio-container button.primary:hover { background: #0A4A44 !important; }
.gradio-container button.secondary { background: var(--se-panel) !important; color: var(--se-text) !important;
  border: 1px solid var(--se-btn-border) !important; font-weight: 400; font-size: 14px; padding: 0 14px !important; }
.gradio-container button.secondary:hover { background: var(--se-bg) !important; }

/* ---- mode banner ---- */
.se-banner { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 10px 32px; background: var(--se-amber-bg);
  color: var(--se-amber-text); font-weight: 600; font-size: 14px; border-bottom: 1px solid var(--se-border); }
.se-banner .chip { padding: 2px 10px; border-radius: var(--se-r-pill); background: #fff; border: 1px solid #E8C99A; font-weight: 500; font-size: 13px; }

/* ---- pipeline strip ---- */
.se-pipeline { display: flex; flex-wrap: wrap; gap: 12px; padding: 16px 32px; background: var(--se-panel); border-bottom: 1px solid var(--se-border); }
.se-tile { flex: 1 1 220px; display: flex; align-items: center; gap: 12px; padding: 12px 16px; border-radius: var(--se-r-tile);
  background: var(--se-grey-bg); min-width: 0; }
.se-tile .dot { flex: none; width: 28px; height: 28px; border-radius: 999px; background: var(--se-grey-dot); color: #fff;
  display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 14px; }
.se-tile .t { font-weight: 600; } .se-tile .s { font-size: 13px; color: var(--se-muted); overflow-wrap: anywhere; }
.se-tile > div { display: flex; flex-direction: column; min-width: 0; }
.se-tile.running { background: var(--se-blue-bg); } .se-tile.running .dot { background: var(--se-blue); animation: se-pulse 1.2s ease-in-out infinite; }
.se-tile.done { background: var(--se-teal-bg); } .se-tile.done .dot { background: var(--se-teal); }
.se-tile.failed { background: var(--se-red-bg); } .se-tile.failed .dot { background: var(--se-red); }
@keyframes se-pulse { 0%, 100% { box-shadow: 0 0 0 0 rgba(47,93,138,.45); } 50% { box-shadow: 0 0 0 6px rgba(47,93,138,0); } }
@media (prefers-reduced-motion: reduce) { .se-tile.running .dot { animation: none; } }

/* ---- main area ---- */
#se-main { display: flex; flex-wrap: wrap; gap: 24px; padding: 24px 32px; align-items: flex-start; }
#se-left { flex: 999 1 560px !important; min-width: 0 !important; gap: 16px !important; }
#se-results { flex: 1 1 440px !important; min-width: 0 !important; gap: 16px !important; }

/* X-ray viewer (a gr.Image) */
#se-viewer { background: var(--se-viewer) !important; border: 0 !important; border-radius: var(--se-r-card) !important;
  overflow: hidden; height: 620px !important; padding: 0 !important; }
#se-viewer > label, #se-viewer [data-testid="block-label"] { position: absolute !important; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; }
#se-viewer .image-container { background: var(--se-viewer) !important; height: 100% !important; display: flex; flex-direction: column; }
#se-viewer .image-container > :not(.source-selection):not(.icon-button-wrapper) { flex: 1 1 auto; min-height: 0; }
#se-viewer .upload-container, #se-viewer .wrap, #se-viewer button.center { background: var(--se-viewer) !important; }
#se-viewer button.center, #se-viewer .image-frame { height: 100% !important; }
#se-viewer .source-selection { flex: none; }
#se-viewer img { object-fit: contain !important; width: 100%; height: 100%; }
#se-viewer .icon-buttons button, #se-viewer .icon-button { background: rgba(255,255,255,.92) !important; color: var(--se-text) !important; }
#se-viewer .source-selection { background: var(--se-viewer) !important; border-top: 1px solid #2A323C !important; }
#se-viewer .source-selection button { color: #C9D1D9 !important; }
#se-viewer .source-selection button.selected { color: #FFFFFF !important; }
/* empty state: Gradio's placeholder is plain text, so the icon and second line come from CSS */
#se-viewer [data-testid="upload-text"] { display: flex; flex-direction: column; align-items: center; gap: 12px; color: #C9D1D9;
  font-size: 16px; font-weight: 500; text-align: center; padding: 24px; }
#se-viewer [data-testid="upload-icon"] { width: 56px; height: 56px; border: 0; background: no-repeat center / 56px 56px
  url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2'/%3E%3Ccircle cx='8.5' cy='8.5' r='1.5'/%3E%3Cpath d='M21 15l-5-5L5 21'/%3E%3C/svg%3E"); }
#se-viewer [data-testid="upload-icon"] svg { display: none; }
#se-viewer [data-testid="upload-text"] p { margin: 0; color: #C9D1D9; }
#se-viewer [data-testid="upload-text"]::after { content: "PNG or JPEG, up to 20 MB"; font-size: 13px; font-weight: 400; color: #8B949E; }
.se-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; color: #C9D1D9;
  text-align: center; padding: 24px; font-size: 16px; font-weight: 500; }
.se-empty small { font-size: 13px; color: #8B949E; font-weight: 400; }
.se-chips { display: flex; flex-wrap: wrap; gap: 8px; font-size: 13px; color: var(--se-muted); }
.se-chips span { padding: 4px 10px; border-radius: 6px; background: var(--se-panel); border: 1px solid var(--se-border); overflow-wrap: anywhere; }
.se-note-line { margin: 8px 0 0; font-size: 13px; color: var(--se-muted); }

/* ---- cards ---- */
.se-card { background: var(--se-panel); border: 1px solid var(--se-border); border-radius: var(--se-r-card); padding: 20px;
  display: flex; flex-direction: column; gap: 12px; color: var(--se-text); }
.se-card h2 { margin: 0; font-size: 16px; font-weight: 600; line-height: 1.4; color: var(--se-text); }
.se-card p { margin: 0; }
.se-card .head { display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap; }
.se-card .aside { font-size: 12px; color: var(--se-muted); }
.se-card .head .meta { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.se-card.ring-red { box-shadow: 0 0 0 2px var(--se-red); } .se-card.ring-teal { box-shadow: 0 0 0 2px var(--se-teal); }
.se-card.ring-amber { box-shadow: 0 0 0 2px #B26B00; }
.se-card.placeholder { color: var(--se-muted); font-size: 14px; }
.se-card.error { border-color: var(--se-red); background: var(--se-red-bg); color: var(--se-red-text); font-size: 14px; }
.se-pill { padding: 3px 12px; border-radius: var(--se-r-pill); font-size: 14px; font-weight: 600; }
.se-pill.red { background: var(--se-red-pill); color: var(--se-red-text); }
.se-pill.teal { background: var(--se-teal-bg); color: var(--se-teal); }
.se-pill.amber { background: var(--se-amber-bg); color: var(--se-amber-text); }
.se-pill.grey { background: var(--se-grey-bg); color: var(--se-text-2); }
.se-latency { font-family: var(--se-mono); font-size: 12px; color: var(--se-muted); }
.se-big { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.se-big .n { font-size: 48px; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
.se-big .l { font-size: 15px; color: var(--se-muted); }
.se-big .n.red { color: var(--se-red); } .se-big .n.teal { color: var(--se-teal); } .se-big .n.amber { color: var(--se-amber-text); }
.se-bar { position: relative; height: 10px; border-radius: 999px; background: var(--se-divider); }
.se-bar .fill { height: 10px; border-radius: 999px; background: var(--se-teal); }
.se-bar .fill.red { background: var(--se-red); }
.se-bar .marker { position: absolute; top: -5px; width: 2px; height: 20px; background: var(--se-text); }
.se-scale { display: flex; justify-content: space-between; gap: 8px; font-size: 12px; color: var(--se-muted); margin-top: 6px; }
.se-scale span:nth-child(2) { text-align: center; }
.se-row { display: flex; justify-content: space-between; gap: 12px; padding: 10px 12px; border-radius: var(--se-r-btn); background: var(--se-bg); }
.se-row .v { font-weight: 600; text-align: right; }
.se-error-box { padding: 10px 12px; border-radius: var(--se-r-btn); background: var(--se-red-bg); color: var(--se-red-text);
  border: 1px solid var(--se-red-pill); font-size: 14px; overflow-wrap: anywhere; }
.se-big .se-pill { align-self: center; font-size: 13px; }
.se-callout { padding: 10px 12px; border-radius: var(--se-r-btn); background: var(--se-amber-bg); color: var(--se-amber-text); font-size: 14px; }
.se-quote { font-size: 14px; color: var(--se-text-2); background: var(--se-bg); padding: 10px 12px; border-radius: var(--se-r-btn); }
.se-reading p { font-size: 14px; line-height: 1.6; }
.se-reading pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font-family: var(--se-mono); font-size: 13px;
  background: var(--se-bg); padding: 10px 12px; border-radius: var(--se-r-btn); max-height: 280px; overflow: auto; }
.se-para { display: flex; justify-content: space-between; gap: 12px; font-size: 14px; padding: 6px 0; border-bottom: 1px solid var(--se-divider); }
.se-para:last-child { border-bottom: 0; } .se-para .p { font-family: var(--se-mono); }

/* Reading card = a gr.Column holding HTML, buttons and the raw-response accordion */
#se-reading-card { background: var(--se-panel) !important; border: 1px solid var(--se-border) !important; border-radius: var(--se-r-card) !important;
  padding: 20px !important; gap: 12px !important; }
#se-reading-card .se-card { border: 0; padding: 0; border-radius: 0; }
#se-note-actions { display: flex; flex-wrap: wrap; gap: 8px !important; }
#se-note-actions > button { flex: 0 0 auto !important; min-width: 0 !important; width: auto !important; }
#se-raw { border: 1px solid var(--se-divider) !important; border-radius: var(--se-r-btn) !important; background: var(--se-panel) !important; }
#se-raw textarea { font-family: var(--se-mono) !important; font-size: 13px !important; }

.se-footer { padding: 8px 32px 32px; font-size: 13px; color: var(--se-muted); }

/* ---- narrow screens ---- */
@media (max-width: 1000px) {
  #se-left, #se-results { flex-basis: 100% !important; }
  #se-viewer { height: 460px !important; }
}
@media (max-width: 640px) {
  #se-header, .se-pipeline, #se-main, .se-banner { padding-left: 16px; padding-right: 16px; }
  .se-footer { padding: 8px 16px 24px; }
  .se-head-left { gap: 12px 24px; }
  .se-tile { flex-basis: 100%; }
  .se-card { padding: 18px; }
  #se-reading-card { padding: 18px !important; }
  #se-viewer { height: 420px !important; }
}
"""

# Runs once on page load: keep the clinical light look even when the OS is in dark mode,
# and make the results column a polite live region (gr.Column cannot set ARIA attributes).
JS_ON_LOAD = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.toString());
    return;
  }
  const results = document.getElementById('se-results');
  if (results) { results.setAttribute('aria-live', 'polite'); results.setAttribute('aria-label', 'Results'); }
  const demo = document.getElementById('se-demo');
  if (demo) { demo.setAttribute('role', 'radiogroup'); demo.setAttribute('aria-label', 'Sample X-ray'); }
}
"""

# "Copy note" button: copies the plain-text note that render_reading() embeds in the page.
JS_COPY_NOTE = """
async () => {
  const el = document.getElementById('se-note-text');
  const btn = document.getElementById('se-copy');
  if (!el || !el.textContent.trim()) return;
  try { await navigator.clipboard.writeText(el.textContent.trim()); } catch (e) { return; }
  if (btn) { const old = btn.textContent; btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = old; }, 1500); }
}
"""
