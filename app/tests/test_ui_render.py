from __future__ import annotations

import pytest

from second_eye.ui_render import (
    ViewContext,
    note_text,
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
from tests.conftest import GOOD_READING

WITH_CLM = ViewContext(clm_enabled=True, vlm_label="Nemotron")
NO_CLM = ViewContext(clm_enabled=False, vlm_label="Nemotron")


def result(fracture="yes", clm=None, agreement=None, **overrides) -> dict:
    base = {
        "reading": {**GOOD_READING, "fracture_suspected": fracture},
        "parse_error": False,
        "raw": "{}",
        "model": "nvidia/test-vl",
        "latency_ms": 812.0,
        "total_ms": 900.0,
        "mock": False,
        "clm": clm,
        "agreement": agreement,
    }
    return {**base, **overrides}


FLAGGED = {"p": 0.87, "threshold": 0.112, "flagged": True, "latency_ms": 41.0, "error": None,
           "calibrated": True, "operating_point": "val sensitivity >= 0.90"}
NOT_FLAGGED = {"p": 0.04, "threshold": 0.112, "flagged": False, "latency_ms": 39.0, "error": None}


# --- verdict ----------------------------------------------------------------------------------------


def test_flagged_clm_verdict():
    html = render_verdict(result(clm=FLAGGED, agreement="agree"), WITH_CLM)
    assert "Flagged · second look advised" in html
    assert "ring-red" in html and "fill red" in html
    assert "0.87" in html and "probability of fracture" in html
    assert "left:calc(11.2% - 1px)" in html
    assert "Threshold 0.112 (calibrated, val sensitivity &gt;= 0.90)" in html
    assert "41 ms" in html


def test_not_flagged_clm_verdict():
    html = render_verdict(result("no", clm=NOT_FLAGGED, agreement="agree"), WITH_CLM)
    assert "Not flagged" in html and "Flagged · second look" not in html
    assert "ring-teal" in html and "fill teal" in html
    assert "0.04" in html


def test_clm_without_threshold_says_so():
    clm = {**FLAGGED, "threshold": None}
    html = render_verdict(result(clm=clm), WITH_CLM)
    assert "marker" not in html and "No threshold" in html


def test_no_clm_shows_vlm_verdict_without_probability():
    html = render_verdict(result("no"), NO_CLM)
    assert ">No<" in html and "Medium confidence" in html
    assert "se-bar" not in html and "probability of fracture" not in html
    assert render_two_readers(result("no"), NO_CLM) == ""


@pytest.mark.parametrize("verdict, colour", [("yes", "red"), ("no", "teal"), ("uncertain", "amber")])
def test_vlm_verdict_colours(verdict, colour):
    html = render_verdict(result(verdict), NO_CLM)
    assert f"n {colour}" in html and f"ring-{colour}" in html


def test_clm_failure_shows_error_and_vlm_still_renders():
    clm = {"p": None, "threshold": None, "flagged": None, "latency_ms": None, "error": "Cannot reach CLM: ConnectError."}
    r = result(clm=clm)
    assert "Cannot reach CLM" in render_verdict(r, WITH_CLM)
    readers = render_two_readers(r, WITH_CLM)
    assert "Failed" in readers and "Yes · medium" in readers
    pipeline = render_pipeline(r, WITH_CLM)
    assert 'se-tile failed' in pipeline and pipeline.count("se-tile done") >= 2


# --- two readers ---------------------------------------------------------------------------------------


def test_vlm_uncertain():
    r = result("uncertain", clm=FLAGGED, agreement="uncertain")
    html = render_two_readers(r, WITH_CLM)
    assert "Uncertain · medium" in html
    assert 'se-pill amber">Uncertain' in html
    assert "readers disagree" not in html
    assert ">No<" not in render_verdict(r, NO_CLM)
    assert "n amber" in render_verdict(r, NO_CLM)


def test_readers_disagree():
    html = render_two_readers(result("no", clm=FLAGGED, agreement="disagree"), WITH_CLM)
    assert "Flagged · <span class='se-mono'>0.87</span>" in html
    assert "No · medium" in html
    assert 'se-pill red">Disagree' in html
    assert "The readers disagree. Review this image carefully." in html


def test_readers_agree_has_no_warning():
    html = render_two_readers(result("no", clm=NOT_FLAGGED, agreement="agree"), WITH_CLM)
    assert "Not flagged" in html and 'se-pill teal">Agree' in html
    assert "readers disagree" not in html


# --- unparsed reply ------------------------------------------------------------------------------------


def test_unparsed_vlm_reply():
    r = result(reading=None, parse_error=True, raw="I think it is broken <maybe>", clm=FLAGGED, agreement=None)
    reading = render_reading(r)
    assert "could not be read" in reading
    assert "I think it is broken &lt;maybe&gt;" in reading
    assert "Reply not parsed" in render_two_readers(r, WITH_CLM)
    assert "Agreement not available" in render_two_readers(r, WITH_CLM)
    assert "Reply not parsed" in render_verdict(r, NO_CLM)
    assert "could not be parsed" in render_vlm_details(r)
    assert note_text(r) == ""


# --- escaping ------------------------------------------------------------------------------------------


def test_model_and_user_text_is_escaped():
    evil = "<img src=x onerror=alert(1)>"
    reading = {**GOOD_READING, "findings": evil, "reasoning": evil, "limitations": evil}
    r = result(reading=reading, clm={**FLAGGED, "error": None})
    html = "".join(
        [
            render_reading(r),
            render_header(f"{evil}.png", r),
            render_mode_banner([evil]),
            render_evidence({"evidence": {"quote": evil, "title": evil, "url": "javascript:alert(1)", "via": evil}}),
            render_paraphrase({"paraphrase": [{"question": evil, "p": 0.5}]}),
        ]
    )
    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html
    assert "javascript:" not in html


def test_errors_are_escaped():
    html = render_pipeline({"vlm_error": "<b>boom</b>", "total_ms": 5}, WITH_CLM)
    assert "<b>boom</b>" not in html and "&lt;b&gt;boom&lt;/b&gt;" in html


# --- reading, details, header, pipeline states --------------------------------------------------------


def test_reading_card_content_and_note():
    r = result()
    html = render_reading(r)
    assert "Draft for clinician review" in html
    assert GOOD_READING["findings"] in html and GOOD_READING["reasoning"] in html
    assert "Uncertainty:</strong> Single view only." in html
    assert note_text(r).startswith("Findings: ")
    assert "Uncertainty: Single view only." in note_text(r)


def test_details_rows():
    html = render_vlm_details(result())
    assert "Hand / wrist" in html and "Hardware visible" in html and "Good" in html


def test_header_shows_file_and_region_after_analysis():
    assert "No image yet" in render_header(None)
    html = render_header("wrist.png", result())
    assert "wrist.png · Hand / wrist" in html
    assert "Research prototype, not a medical device" in html


def test_mode_banner():
    assert render_mode_banner([]) == ""
    html = render_mode_banner(["Vision reader (MOCK_VLLM=1)"])
    assert "Mock services in use, numbers are not real:" in html and "MOCK_VLLM=1" in html


def test_pipeline_states_progress():
    empty = render_pipeline(None, WITH_CLM)
    assert empty.count("se-tile waiting") == 4
    running = render_pipeline({"running": True, "elapsed_ms": 1200}, WITH_CLM)
    assert running.count("se-tile running") == 3 and "1.2 s" in running
    half = render_pipeline({"running": True, "clm": FLAGGED, "elapsed_ms": 300}, WITH_CLM)
    assert half.count("se-tile done") == 1 and half.count("se-tile running") == 2
    final = render_pipeline(result(clm=FLAGGED, agreement="agree"), WITH_CLM)
    assert final.count("se-tile done") == 4 and "Agree" in final and "0.9 s" in final


def test_pipeline_without_clm():
    html = render_pipeline(result(), NO_CLM)
    assert "Not configured" in html and "Needs the image model" in html


def test_vlm_failure_marks_tile_failed():
    r = {"clm": FLAGGED, "vlm_error": "vLLM did not answer within 90 s.", "total_ms": 90000}
    pipeline = render_pipeline(r, WITH_CLM)
    assert pipeline.count("se-tile failed") == 2  # vision reader and end-to-end
    assert "did not answer" in render_reading(r)
    assert "Flagged" in render_verdict(r, WITH_CLM)  # the other reader still renders


# --- future slots --------------------------------------------------------------------------------------


def test_evidence_and_paraphrase_hidden_unless_present():
    assert render_evidence(result()) == "" and render_paraphrase(result()) == ""
    ev = render_evidence({"evidence": [{"quote": "Q", "title": "T", "url": "https://x.org", "via": "Tavily"}]})
    assert 'href="https://x.org"' in ev and "(via Tavily)" in ev
    pa = render_paraphrase({"paraphrase": [{"question": "a", "p": 0.87}, {"question": "b", "p": 0.83}]})
    assert "Paraphrase check" in pa and "± 0.02" in pa
