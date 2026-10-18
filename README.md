---
title: Second Eye
emoji: 🦴
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
license: apache-2.0
short_description: Research prototype second reading of X-rays for fractures
---

# Second Eye

**Research prototype, not a medical device.** You upload an X-ray and a vision-language model (VLM)
served by vLLM gives a second reading for possible fractures. You can also add a calibrated
classifier (clm-serve) as a second reader. Second Eye supports a clinician's review; it does not replace it.

```
browser ──▶ Second Eye (FastAPI + Gradio, CPU only) ──HTTP──▶ vLLM  /v1/chat/completions
                                                     └─HTTP──▶ clm-serve /v1/systemone (optional)
```

The app holds no model weights and needs no GPU. It only talks to the model endpoints over HTTP.

## Layout

| Path | Purpose |
|---|---|
| `second_eye/config.py` | Env vars / `.env` |
| `second_eye/prompts.py` | System and user prompts (tune here) |
| `second_eye/schemas.py` | `Reading` model, JSON schema sent to vLLM, API response models |
| `second_eye/vllm_client.py` | Chat-completions call, `response_format` fallback, JSON extraction, mock |
| `second_eye/clm_client.py` | Optional clm-serve call (never raises) |
| `second_eye/analyze.py` | The one analysis path: validate → downscale → vLLM ∥ CLM → agreement (`analyze_stream` reports each reader as it finishes) |
| `second_eye/ui.py` | Gradio Blocks layout and event wiring |
| `second_eye/ui_render.py` | Pure functions that turn an analyze result dict into card HTML |
| `second_eye/ui_style.py` | Gradio theme and the one CSS string (design tokens as CSS variables) |
| `design/second-eye-reference.html` | Static reference mockup the UI follows |
| `second_eye/main.py` | FastAPI app, `/api/*` routes, Gradio mounted at `/` |
| `app.py` | Hugging Face Space entry point (same app) |
| `tests/` | pytest suite (respx-mocked vLLM/CLM) and `fake_clm.py` demo server |

## Run locally

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then edit

./run.sh --mock                 # UI only, canned VLM reply, http://localhost:8080
./run.sh --mock --fake-clm      # also starts tests/fake_clm.py so you can see the Two readers panel
./run.sh                        # real vLLM, using .env
pytest                          # test suite
```

In mock mode, every reading is labelled **MOCK, not real**, and the status line says the values are fake.
The fake CLM's "probability" is the image's mean brightness (threshold 0.112, with a 0.25 s delay set by
`FAKE_CLM_DELAY_S`, so you can watch the readers finish one by one). It exists only to drive the UI.
`--fake-clm` also sets `MOCK_CLM=1`, so the mode banner says those numbers are not real.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_BASE_URL` | none | OpenAI-compatible base URL including `/v1`, e.g. `http://10.0.0.5:8000/v1` |
| `VLLM_API_KEY` | empty | Sent as `Authorization: Bearer …` if set |
| `VLLM_MODEL` | none | Exact served model id, as listed by `GET /v1/models` |
| `VLLM_TIMEOUT` | `90` | Seconds per vLLM request |
| `CLM_URL` | empty | clm-serve base URL. Empty hides the Two readers panel |
| `CLM_MODEL` | `fracatlas-medsiglip` | Model name sent to clm-serve |
| `DEMO_IMAGES` | empty | `Label=path;Label=path` one-click examples (missing files are skipped with a log warning) |
| `MOCK_VLLM` | `0` | `1` returns a canned, clearly labelled fake VLM reply |
| `MOCK_CLM` | `0` | `1` when `CLM_URL` points at a stand-in such as `tests/fake_clm.py`. It only labels the CLM numbers as not real in the UI |
| `ENCODER_LABEL`, `HEAD_LABEL`, `SERVED_ON` | empty | Display-only chips under the X-ray viewer, e.g. `MedSigLIP-448, frozen` |

Real environment variables take precedence over `.env`.

## Point it at vLLM

Start vLLM on the GPU machine with the **multimodal settings from the model's card**. Vision-language
models usually need some of `--trust-remote-code`, `--limit-mm-per-prompt`, a suitable
`--max-model-len`, and sometimes a specific chat template or `--mm-processor-kwargs`, and the right
values differ from model to model. Copy them from the card for your exact checkpoint. The general
shape is:

```bash
vllm serve <hf-repo-id-of-the-VL-model> \
  --served-model-name <id-you-will-put-in-VLLM_MODEL> \
  --api-key "$VLLM_API_KEY" \
  --host 0.0.0.0 --port 8000 \
  <multimodal flags from the model card>
```

Then check it from the app machine:

```bash
curl -H "Authorization: Bearer $VLLM_API_KEY" http://<gpu-host>:8000/v1/models
```

Set `VLLM_MODEL` to an `id` from that list. The status line in the UI turns red if the model is not listed.

The app asks for structured output with `response_format: {type: "json_schema"}` (vLLM guided
decoding). If the server answers 400/422, the app retries once without it, adds a short schema hint
to the prompt, and extracts JSON from the text, stripping ```` ``` ```` fences and `<think>…</think>` blocks.

## Deploy on a Nebius VM (separate from the model endpoint)

Run the app on a small CPU VM. vLLM runs on its own GPU VM in the same VPC network.

```bash
# on the CPU VM
git clone <repo> second-eye && cd second-eye
docker build -t second-eye .
cp .env.example .env && vi .env     # VLLM_BASE_URL=http://<gpu-vm-private-ip>:8000/v1, etc.
docker run -d --name second-eye --restart unless-stopped --env-file .env -p 8080:8080 second-eye
curl localhost:8080/api/health
```

Recommendations:

- Use the GPU VM's **private IP** for `VLLM_BASE_URL`, and restrict port 8000 on the GPU VM
  (security group / firewall) so only the app VM can reach it. Also set `--api-key` on vLLM.
- Expose only the app VM publicly. Put TLS in front of it, for example Caddy
  (`caddy reverse-proxy --from your.domain --to localhost:8080`). The container already trusts
  forwarded headers.
- The Docker `HEALTHCHECK` calls `/api/health`, which returns 200 even when vLLM is down. Read the
  JSON to see whether vLLM is up.

## Publish as a Hugging Face Space

This repo is a **Gradio SDK** Space (see the front matter above). I chose Gradio SDK over Docker
because it runs on the free CPU hardware with no extra configuration and rebuilds quickly. `app.py`
imports the same FastAPI app (`second_eye.main:app`, with Gradio mounted at `/` and `/api/*` still
available) and serves it with uvicorn on port 7860, which is what the Space expects.

1. Create a Space with the Gradio SDK, then push this repo to it:
   `git remote add space https://huggingface.co/spaces/<user>/second-eye && git push space main`.
2. Under *Settings → Variables and secrets*, add `VLLM_BASE_URL`, `VLLM_MODEL`, and `VLLM_API_KEY`
   (as a secret). For a demo without a model, set `MOCK_VLLM=1`. Leave `CLM_URL` unset to hide the
   Two readers panel.
3. The vLLM endpoint must be reachable from the public internet for a Space, so protect it with an
   API key and TLS.

To use the Docker SDK instead, set `sdk: docker` and `app_port: 8080` and remove `sdk_version` and
`app_file`. The existing Dockerfile works as is.

## API contract

### `GET /api/health`

```json
{
  "vllm": {"ok": true, "model_listed": true, "models": ["<id>"], "error": null, "status_code": null},
  "clm":  {"ok": true, "model_listed": null, "models": ["fracatlas-medsiglip"], "error": null, "status_code": 200},
  "mock": false
}
```

`vllm` comes from `GET {VLLM_BASE_URL}/models`. `clm` is `null` when `CLM_URL` is unset; otherwise
it comes from `GET {CLM_URL}/health`.

### `POST /api/analyze`

`multipart/form-data` with field `file`.

```bash
curl -F file=@wrist.png http://localhost:8080/api/analyze
```

200 response:

```json
{
  "reading": {
    "fracture_suspected": "yes | no | uncertain",
    "confidence": "low | medium | high",
    "body_region": "hand_wrist | leg_knee_ankle_foot | hip_pelvis | shoulder | other | unclear",
    "hardware_visible": false,
    "image_quality": "good | limited | poor",
    "findings": "…", "reasoning": "…", "limitations": "…"
  },
  "parse_error": false,
  "raw": "<model text>",
  "model": "<VLLM_MODEL>",
  "latency_ms": 2140.3,
  "total_ms": 2188.9,
  "mock": false,
  "clm": {"p": 0.81, "threshold": 0.42, "flagged": true, "latency_ms": 37.5, "error": null},
  "agreement": "agree | disagree | uncertain | null"
}
```

- If the reply is not valid JSON matching the schema: `reading: null`, `parse_error: true`, and `raw`
  holds the text. The status is still 200.
- `clm` is `null` without `CLM_URL`. If CLM fails, it is `{"error": "…"}` with the other fields null,
  and the request still succeeds.
- `agreement` is `uncertain` if the VLM said uncertain, `null` if there is no usable CLM result or
  the VLM reply could not be parsed, and otherwise `agree`/`disagree` (CLM `flagged` vs VLM `yes`).
- `latency_ms` is the vLLM round trip, including the one `response_format` retry if it happened.

Errors, always as JSON `{"error": "…"}`:

| Status | When |
|---|---|
| 422 | Empty file, larger than 20 MB, or not decodable by Pillow (missing `file` field gives FastAPI's standard 422) |
| 502 | vLLM timed out, unreachable, returned an HTTP error, or sent a malformed completion |

Uploads are converted to RGB (16-bit greyscale is contrast-stretched first) and downscaled so the
longest side is at most 1536 px. They are re-encoded as JPEG quality 90 in memory for the VLM. CLM
receives the original upload bytes. The API never writes uploads to disk or logs image bytes.
