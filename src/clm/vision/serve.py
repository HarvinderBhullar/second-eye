"""Second Eye HTTP API: image -> calibrated answer distribution.  Separate from ``clm-serve``.

    python -m clm.vision.serve --port 8701                                         # in-process Qwen3-8B
    python -m clm.vision.serve --model harvinderbhullar/second-eye                 # head from the HF Hub
    python -m clm.vision.serve --port 8701 --emb-url http://127.0.0.1:8090/v1/embeddings   # vLLM encoder
    python -m clm.vision.serve --model harvinderbhullar/second-eye --no-custom-options     # lean: no Qwen3-8B

    export SECOND_EYE_API_KEY=...    # optional; then /predict needs "Authorization: Bearer <key>"
    export CLM_EMB_API_KEY=...       # token for a remote (e.g. Nebius) vLLM embeddings endpoint

    curl -s -X POST localhost:8701/predict -F image=@xray.jpg
    curl -s -X POST localhost:8701/predict -F image=@xray.jpg \\
         -F 'options=["A fracture is present.", "No fracture is present."]'

The state is the image alone: the head was trained for one question (``QUESTION``), so a
``question`` field is echoed back but does not change the state embedding.  Custom ``options``
are embedded on the fly by the action tower (LRU-cached).  Platt calibration and the
>=0.90-sensitivity decision were fit on val for the trained options only, and the paraphrase
ablation shows they do not transfer to rewordings (ranking holds, the threshold does not), so for
any other option set the raw softmax and the argmax are returned with ``calibrated: false``.
Limits: uploads up to MAX_UPLOAD_MB, images up to Image.MAX_IMAGE_PIXELS, at most MAX_OPTIONS
options of at most MAX_OPTION_CHARS characters.  Research prototype, not a medical device.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from PIL import Image

from . import OPTIONS, QUESTION
from .calibrate import Platt
from .embed_image import ENCODERS, TILE, device
from .embed_options import LocalQwenEmbedder, action_head, embed_texts
from .head import StateHead
from .train import CKPT_DIR

DISCLAIMER = "Research prototype, not a medical device. Not for clinical use."
MAX_UPLOAD_MB = 20
MAX_OPTIONS = 8
MAX_OPTION_CHARS = 300
Image.MAX_IMAGE_PIXELS = 50_000_000      # decompression-bomb guard (FracAtlas max is 2880x2304)


class Predictor:
    def __init__(self, emb_url: str | None = None, ckpt_dir: str = CKPT_DIR, cache_size: int = 256,
                 model: str | None = None, custom_options: bool = True, emb_api_key: str | None = None):
        self.dev = device()
        if model:                       # a Hub repo id or a `clm.vision.hub export` dir
            from .hub import load
            self.head, trained_options, self.cal, cfg = load(model, self.dev)
            self.variant = cfg["variant"]
        else:
            with open(os.path.join(ckpt_dir, "calibration.json")) as f:
                self.cal = json.load(f)
            self.variant = self.cal["variant"]
            self.head, ck = StateHead.load(os.path.join(ckpt_dir, f"{self.variant}.pt"), self.dev)
            trained_options = ck["options"]
        self.platt = Platt(**self.cal["platt"])
        self.threshold = self.cal["threshold_sens90"]
        self.encoder = ENCODERS[self.variant.split("_")[0]](self.dev)
        self.custom_options = custom_options
        if custom_options:              # the action tower is only needed for options other than the trained ones
            self.action = action_head(dev=self.dev)
            if emb_url:
                from ..embedder import Embedder
                self.embedder = Embedder(url=emb_url, api_key=emb_api_key)
            else:
                self.embedder = LocalQwenEmbedder(dev=self.dev)
        self.cache: OrderedDict[tuple[str, ...], torch.Tensor] = OrderedDict()
        self.cache[tuple(OPTIONS)] = trained_options.to(self.dev)        # the trained options, as trained
        self.cache_size = cache_size

    def options(self, texts: list[str]) -> torch.Tensor:
        key = tuple(texts)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if not self.custom_options:
            raise HTTPException(400, "this server only answers the trained options (started with --no-custom-options)")
        _, proj = embed_texts(texts, self.embedder, self.action)
        v = torch.from_numpy(proj).float().to(self.dev)
        self.cache[key] = v
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return v

    @torch.no_grad()
    def __call__(self, image: Image.Image, options: list[str]) -> dict:
        tile = np.array(image.convert("RGB").resize((TILE, TILE), Image.BICUBIC))
        x = torch.from_numpy(self.encoder([tile])[self.variant]).to(self.dev)
        probs = self.head(x, self.options(options)).softmax(-1)[0].cpu().numpy()
        if list(options) == list(OPTIONS):
            p_pos = float(self.platt(probs[:1])[0])
            return {"probabilities": {options[0]: p_pos, options[1]: 1 - p_pos}, "calibrated": True,
                    "decision": options[0] if p_pos >= self.threshold else options[1],
                    "threshold": self.threshold, "operating_point": "val sensitivity >= 0.90"}
        return {"probabilities": dict(zip(options, map(float, probs))), "calibrated": False,
                "decision": options[int(probs.argmax())], "threshold": None, "operating_point": "argmax"}


def create_app(emb_url: str | None = None, model: str | None = None, custom_options: bool = True,
               api_key: str | None = None, emb_api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="Second Eye", description=DISCLAIMER)
    pred = Predictor(emb_url, model=model, custom_options=custom_options, emb_api_key=emb_api_key)   # loads once

    def auth(authorization: str | None = Header(None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "missing or wrong API key (Authorization: Bearer <key>)")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "variant": pred.variant, "device": pred.dev, "custom_options": pred.custom_options,
                "auth": bool(api_key), "disclaimer": DISCLAIMER}

    @app.post("/predict", dependencies=[Depends(auth)])
    def predict(image: UploadFile = File(...), question: str | None = Form(None),
                options: str | None = Form(None)) -> dict:
        # a plain `def`: FastAPI runs it in a worker thread, so model compute does not block the event loop
        t0 = time.perf_counter()
        data = image.file.read(MAX_UPLOAD_MB * 2**20 + 1)
        if len(data) > MAX_UPLOAD_MB * 2**20:
            raise HTTPException(413, f"image larger than {MAX_UPLOAD_MB} MB")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"cannot read image: {e}") from e
        opts = OPTIONS
        if options:
            try:
                opts = json.loads(options)
            except json.JSONDecodeError as e:
                raise HTTPException(400, "options must be a JSON list of strings") from e
            if not (isinstance(opts, list) and 2 <= len(opts) <= MAX_OPTIONS
                    and all(isinstance(o, str) and 0 < len(o) <= MAX_OPTION_CHARS for o in opts)):
                raise HTTPException(400, f"options must be a JSON list of 2..{MAX_OPTIONS} non-empty strings "
                                         f"of at most {MAX_OPTION_CHARS} characters")
        out = pred(img, opts)
        if pred.dev == "cuda":
            torch.cuda.synchronize()
        return {"question": question or QUESTION, "options": opts, **out,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "disclaimer": DISCLAIMER}

    return app


def main() -> None:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--emb-url", default=os.environ.get("CLM_EMB_URL"))
    ap.add_argument("--model", default=None, help="Hub repo id or export dir (default: local checkpoints/vision)")
    ap.add_argument("--no-custom-options", action="store_true",
                    help="serve only the trained options; skips loading Qwen3-8B and the action head")
    a = ap.parse_args()
    api_key = os.environ.get("SECOND_EYE_API_KEY")
    print(f"[second-eye] auth {'on' if api_key else 'OFF'}, custom options {'off' if a.no_custom_options else 'on'}",
          flush=True)
    uvicorn.run(create_app(a.emb_url, a.model, not a.no_custom_options, api_key, os.environ.get("CLM_EMB_API_KEY")),
                host=a.host, port=a.port)


if __name__ == "__main__":
    main()
