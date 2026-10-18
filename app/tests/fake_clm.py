"""A stand-in clm-serve for local demos: `uvicorn tests.fake_clm:app --port 9100`.

The "probability" is just the image's mean brightness, so it is NOT a real score. It
exists only to exercise the Two readers panel.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import time

from fastapi import FastAPI, Response
from PIL import Image, ImageStat

app = FastAPI(title="fake clm-serve")
THRESHOLD = 0.112
DELAY_S = float(os.getenv("FAKE_CLM_DELAY_S", "0.25"))  # so the UI shows the readers finishing one by one


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "fake": True}


@app.post("/v1/systemone")
async def systemone(body: dict, response: Response) -> dict:
    started = time.perf_counter()
    await asyncio.sleep(DELAY_S)
    img = Image.open(io.BytesIO(base64.b64decode(body["state"]["image"]))).convert("L")
    p = round(ImageStat.Stat(img).mean[0] / 255, 3)
    response.headers["X-CLM-Latency-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return {
        "model": body.get("model"),
        "answers": {
            "fx": {
                "choice": "fracture" if p >= THRESHOLD else "no_fracture",
                "probabilities": {"fracture": p, "no_fracture": round(1 - p, 3)},
                "threshold": THRESHOLD,
            }
        },
    }
