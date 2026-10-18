from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
import respx
from PIL import Image

from second_eye.config import MAX_UPLOAD_BYTES, Settings
from tests.conftest import CLM, GOOD_READING, VLLM, chat_reply, clm_reply, make_png, sent_json

pytestmark = pytest.mark.asyncio


def upload(data: bytes, name: str = "xray.png", ctype: str = "image/png") -> dict:
    return {"file": (name, data, ctype)}


# --- vLLM replies -----------------------------------------------------------------------


@respx.mock
async def test_valid_json_reply(make_client, settings, png):
    route = respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply(json.dumps(GOOD_READING)))
    client = await make_client(settings)

    r = await client.post("/api/analyze", files=upload(png))

    assert r.status_code == 200
    body = r.json()
    assert body["reading"] == GOOD_READING
    assert body["parse_error"] is False
    assert body["model"] == "nvidia/test-vl"
    assert body["mock"] is False
    assert body["clm"] is None and body["agreement"] is None
    assert body["latency_ms"] >= 0 and body["total_ms"] >= body["latency_ms"]

    req = sent_json(route)
    assert req["model"] == "nvidia/test-vl"
    assert req["temperature"] == 0.1 and req["max_tokens"] == 800
    assert req["response_format"]["type"] == "json_schema"
    assert req["messages"][0]["role"] == "system"
    image_url = req["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")


@respx.mock
async def test_response_format_rejected_then_retried(make_client, settings, png):
    fenced = "<think>let me look</think>\n```json\n" + json.dumps(GOOD_READING) + "\n```"
    route = respx.post(f"{VLLM}/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}}),
            httpx.Response(200, json=chat_reply(fenced)),
        ]
    )
    client = await make_client(settings)

    r = await client.post("/api/analyze", files=upload(png))

    assert r.status_code == 200
    assert r.json()["reading"] == GOOD_READING
    assert route.call_count == 2
    first, second = (json.loads(c.request.content) for c in route.calls)
    assert "response_format" in first
    assert "response_format" not in second


@respx.mock
async def test_invalid_json_reply_is_returned_raw(make_client, settings, png):
    respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply("I think there might be a fracture."))
    client = await make_client(settings)

    r = await client.post("/api/analyze", files=upload(png))

    assert r.status_code == 200
    body = r.json()
    assert body["reading"] is None
    assert body["parse_error"] is True
    assert body["raw"] == "I think there might be a fracture."


@respx.mock
async def test_json_with_wrong_enum_counts_as_parse_error(make_client, settings, png):
    respx.post(f"{VLLM}/chat/completions").respond(
        json=chat_reply(json.dumps({**GOOD_READING, "fracture_suspected": "probably"}))
    )
    client = await make_client(settings)

    body = (await client.post("/api/analyze", files=upload(png))).json()

    assert body["parse_error"] is True and body["reading"] is None


@respx.mock
async def test_vllm_timeout_returns_502(make_client, settings, png):
    respx.post(f"{VLLM}/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    client = await make_client(settings)

    r = await client.post("/api/analyze", files=upload(png))

    assert r.status_code == 502
    assert "did not answer" in r.json()["error"]


@respx.mock
async def test_vllm_down_returns_502(make_client, settings, png):
    respx.post(f"{VLLM}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    client = await make_client(settings)

    r = await client.post("/api/analyze", files=upload(png))

    assert r.status_code == 502
    assert "Cannot reach vLLM" in r.json()["error"]


@respx.mock
async def test_vllm_server_error_returns_502(make_client, settings, png):
    respx.post(f"{VLLM}/chat/completions").respond(500, text="CUDA out of memory")
    client = await make_client(settings)

    r = await client.post("/api/analyze", files=upload(png))

    assert r.status_code == 502
    assert "HTTP 500" in r.json()["error"]


# --- upload validation -------------------------------------------------------------------


async def test_oversized_upload_is_422(make_client, settings):
    client = await make_client(settings)
    r = await client.post("/api/analyze", files=upload(b"\0" * (MAX_UPLOAD_BYTES + 1)))
    assert r.status_code == 422
    assert "larger than 20 MB" in r.json()["error"]


async def test_non_image_upload_is_422(make_client, settings):
    client = await make_client(settings)
    r = await client.post("/api/analyze", files=upload(b"%PDF-1.4 not an image", "a.pdf", "application/pdf"))
    assert r.status_code == 422
    assert "not an image" in r.json()["error"]


async def test_empty_upload_is_422(make_client, settings):
    client = await make_client(settings)
    r = await client.post("/api/analyze", files=upload(b""))
    assert r.status_code == 422
    assert "empty" in r.json()["error"]


@respx.mock
async def test_large_image_is_downscaled_to_jpeg(make_client, settings):
    route = respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply(json.dumps(GOOD_READING)))
    client = await make_client(settings)

    await client.post("/api/analyze", files=upload(make_png((3000, 1000), mode="I;16")))

    url = sent_json(route)["messages"][1]["content"][1]["image_url"]["url"]
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert img.format == "JPEG" and img.mode == "RGB"
    assert max(img.size) == 1536 and img.size == (1536, 512)


# --- CLM -----------------------------------------------------------------------------------


@respx.mock
async def test_clm_down_does_not_fail_request(make_client, clm_settings, png):
    respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply(json.dumps(GOOD_READING)))
    respx.post(f"{CLM}/v1/systemone").mock(side_effect=httpx.ConnectError("refused"))
    client = await make_client(clm_settings)

    r = await client.post("/api/analyze", files=upload(png))

    assert r.status_code == 200
    body = r.json()
    assert body["reading"] == GOOD_READING
    assert "Cannot reach CLM" in body["clm"]["error"]
    assert body["agreement"] is None


@respx.mock
async def test_clm_request_uses_original_upload_and_exact_criteria(make_client, clm_settings, png):
    respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply(json.dumps(GOOD_READING)))
    route = respx.post(f"{CLM}/v1/systemone").respond(
        json=clm_reply(0.81, 0.42), headers={"X-CLM-Latency-Ms": "37.5"}
    )
    client = await make_client(clm_settings)

    body = (await client.post("/api/analyze", files=upload(png))).json()

    req = sent_json(route)
    assert req["model"] == "fracatlas-medsiglip"
    assert base64.b64decode(req["state"]["image"]) == png
    assert req["questions"]["fx"]["criteria"] == {
        "fracture": "Radiograph showing an acute bone fracture.",
        "no_fracture": "Radiograph of intact bones with no fracture.",
    }
    assert body["clm"] == {"p": 0.81, "threshold": 0.42, "flagged": True, "latency_ms": 37.5, "error": None}
    assert body["agreement"] == "agree"


@pytest.mark.parametrize(
    "vlm_says, clm_body, expected",
    [
        ("yes", clm_reply(0.9, 0.5), "agree"),
        ("no", clm_reply(0.1, 0.5), "agree"),
        ("no", clm_reply(0.9, 0.5), "disagree"),
        ("yes", clm_reply(0.2, 0.5), "disagree"),
        ("uncertain", clm_reply(0.9, 0.5), "uncertain"),
        ("no", clm_reply(0.3, None, choice="fracture"), "disagree"),  # no threshold: use choice
        ("no", clm_reply(0.3, None, choice="no_fracture"), "agree"),
    ],
)
@respx.mock
async def test_agreement(make_client, clm_settings, png, vlm_says, clm_body, expected):
    reading = {**GOOD_READING, "fracture_suspected": vlm_says}
    respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply(json.dumps(reading)))
    respx.post(f"{CLM}/v1/systemone").respond(json=clm_body)
    client = await make_client(clm_settings)

    body = (await client.post("/api/analyze", files=upload(png))).json()

    assert body["agreement"] == expected


@respx.mock
async def test_agreement_null_when_vlm_unparseable(make_client, clm_settings, png):
    respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply("not json"))
    respx.post(f"{CLM}/v1/systemone").respond(json=clm_reply(0.9))
    client = await make_client(clm_settings)

    body = (await client.post("/api/analyze", files=upload(png))).json()

    assert body["clm"]["flagged"] is True
    assert body["agreement"] is None


# --- health and mock --------------------------------------------------------------------------


@respx.mock
async def test_health_lists_models(make_client, clm_settings):
    respx.get(f"{VLLM}/models").respond(json={"data": [{"id": "nvidia/test-vl"}, {"id": "other"}]})
    respx.get(f"{CLM}/health").respond(json={"ok": True})
    client = await make_client(clm_settings)

    body = (await client.get("/api/health")).json()

    assert body["vllm"]["ok"] is True and body["vllm"]["model_listed"] is True
    assert body["vllm"]["models"] == ["nvidia/test-vl", "other"]
    assert body["clm"]["ok"] is True
    assert body["mock"] is False


@respx.mock
async def test_health_reports_vllm_down(make_client, settings):
    respx.get(f"{VLLM}/models").mock(side_effect=httpx.ConnectError("refused"))
    client = await make_client(settings)

    body = (await client.get("/api/health")).json()

    assert body["vllm"]["ok"] is False and body["clm"] is None


async def test_mock_mode_needs_no_server(make_client, png):
    client = await make_client(Settings(mock_vllm=True))

    body = (await client.post("/api/analyze", files=upload(png))).json()

    assert body["mock"] is True
    assert body["reading"]["findings"].startswith("MOCK RESPONSE")
    health = (await client.get("/api/health")).json()
    assert health["mock"] is True and health["vllm"]["ok"] is True


# --- streaming helper used by the UI -----------------------------------------------------------------


@respx.mock
async def test_analyze_stream_reports_each_reader(clm_settings, png):
    from second_eye.analyze import analyze_stream

    respx.post(f"{VLLM}/chat/completions").respond(json=chat_reply(json.dumps(GOOD_READING)))
    respx.post(f"{CLM}/v1/systemone").respond(json=clm_reply(0.9))

    events = [kind async for kind, _ in analyze_stream(png, clm_settings)]

    assert sorted(events[:2]) == ["clm", "vlm"] and events[-1] == "done"


@respx.mock
async def test_analyze_stream_vlm_failure_still_reports_clm(clm_settings, png):
    from second_eye.analyze import analyze_stream

    respx.post(f"{VLLM}/chat/completions").mock(side_effect=httpx.ConnectError("refused"))
    respx.post(f"{CLM}/v1/systemone").respond(json=clm_reply(0.9))

    events = {kind: value async for kind, value in analyze_stream(png, clm_settings)}

    assert events["clm"].flagged is True
    assert "Cannot reach vLLM" in str(events["vlm_error"])
    assert events["done"] is None
