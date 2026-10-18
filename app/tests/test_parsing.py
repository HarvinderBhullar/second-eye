import json

from second_eye.config import _parse_demo_images
from second_eye.vllm_client import extract_json_text, parse_reading
from tests.conftest import GOOD_READING


def test_strips_think_and_fences():
    text = "<think>\nhmm {not this}\n</think>\nHere you go:\n```json\n" + json.dumps(GOOD_READING) + "\n```"
    assert json.loads(extract_json_text(text)) == GOOD_READING


def test_handles_dangling_close_think_tag():
    text = "reasoning without an opening tag</think>" + json.dumps(GOOD_READING)
    assert parse_reading(text).fracture_suspected == "yes"


def test_missing_field_is_not_a_reading():
    partial = {k: v for k, v in GOOD_READING.items() if k != "limitations"}
    assert parse_reading(json.dumps(partial)) is None


def test_demo_images_parsing():
    assert _parse_demo_images("Wrist=a.png; Hip = b/c.jpg;bad;=x") == [("Wrist", "a.png"), ("Hip", "b/c.jpg")]
    assert _parse_demo_images(None) == []


def test_parses_a_real_image_head_reply():
    # Captured from `python -m clm.vision.serve --model harvinderbhullar/second-eye --no-custom-options`.
    from second_eye.clm_client import parse_answer

    body = {
        "question": "Is there a bone fracture in this X-ray?",
        "options": ["Yes, there is a fracture.", "No, there is no fracture."],
        "probabilities": {"Yes, there is a fracture.": 0.5471389293670654, "No, there is no fracture.": 0.45286107063293457},
        "calibrated": True,
        "decision": "Yes, there is a fracture.",
        "threshold": 0.0674779936671257,
        "operating_point": "val sensitivity >= 0.90",
        "latency_ms": 7669.9,
        "disclaimer": "Research prototype, not a medical device. Not for clinical use.",
    }
    result = parse_answer(body)
    assert result.p == body["probabilities"]["Yes, there is a fracture."]
    assert result.flagged is True and result.calibrated is True
    assert result.threshold == body["threshold"] and result.latency_ms == 7669.9
