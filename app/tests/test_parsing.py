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
