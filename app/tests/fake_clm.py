"""A stand-in for the Second Eye image head (`clm.vision.serve`) for local demos:
`uvicorn tests.fake_clm:app --port 9100`.

It copies the real server's /health and /predict shapes, but its "probability" is just the image's
mean brightness, so it is NOT a real score. It exists only to exercise the Two readers panel.
"""

from __future__ import annotations

import asyncio
import io
import os
import time

from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image, ImageStat

app = FastAPI(title="fake second-eye image head")
OPTIONS = ["Yes, there is a fracture.", "No, there is no fracture."]
THRESHOLD = 0.112
DELAY_S = float(os.getenv("FAKE_CLM_DELAY_S", "0.25"))  # so the UI shows the readers finishing one by one


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "variant": "fake", "device": "cpu", "custom_options": False, "auth": False}


@app.post("/predict")
async def predict(image: UploadFile = File(...), question: str | None = Form(None)) -> dict:
    started = time.perf_counter()
    await asyncio.sleep(DELAY_S)
    img = Image.open(io.BytesIO(await image.read())).convert("L")
    p = round(ImageStat.Stat(img).mean[0] / 255, 3)
    return {
        "question": question or "Is there a bone fracture in this X-ray?",
        "options": OPTIONS,
        "probabilities": {OPTIONS[0]: p, OPTIONS[1]: round(1 - p, 3)},
        "calibrated": True,
        "decision": OPTIONS[0] if p >= THRESHOLD else OPTIONS[1],
        "threshold": THRESHOLD,
        "operating_point": "val sensitivity >= 0.90",
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "disclaimer": "Fake stand-in. Not a real model.",
    }
