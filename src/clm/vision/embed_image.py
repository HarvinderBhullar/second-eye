"""Image embeddings for the vision state head.

    python -m clm.vision.embed_image --probe                 # one image, print shapes (checkpoint 1)
    python -m clm.vision.embed_image --encoder radio         # embed every image in data/vision/manifest.csv

Encoders:
  radio     C-RADIOv2-H, the vision tower Nemotron Nano 2 VL uses (vision_config.auto_map
            -> nvidia/C-RADIOv2-H).  One fixed 512x512 tile per image, the tile size Nemotron
            feeds it (force_image_size=512), so every image gives 32x32 = 1024 patch tokens.
            Stores two variants: ``radio_mean`` (mean over patch tokens) and ``radio_summary``
            (the summary/CLS vector).
  nemotron  the full Nemotron-Nano-12B-v2-VL (needs a CUDA GPU; run on Nebius).

Each variant is saved as data/vision/emb/<variant>.npy ([n, d] float32, L2-normalised,
rows in manifest order) with an index file <variant>.index.csv (row -> path).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

RADIO_REPO = "nvidia/C-RADIOv2-H"
TILE = 512
EMB_DIR = os.path.join("data", "vision", "emb")


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def load_tile(path: str) -> np.ndarray:
    """Greyscale X-ray -> RGB, resized to one fixed TILE x TILE tile, uint8 [H, W, 3]."""
    return np.array(Image.open(path).convert("RGB").resize((TILE, TILE), Image.BICUBIC))


class RadioEncoder:
    variants = ("radio_mean", "radio_summary")

    def __init__(self, dev: str | None = None):
        from transformers import AutoModel
        self.dev = dev or device()
        self.model = AutoModel.from_pretrained(RADIO_REPO, trust_remote_code=True).eval().to(self.dev)

    @torch.no_grad()
    def __call__(self, tiles: list[np.ndarray]) -> dict[str, np.ndarray]:
        # RADIO takes [0, 1] RGB and normalises internally (its input conditioner).
        x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float().div(255).to(self.dev)
        summary, spatial = self.model(x)                       # [b, 3840], [b, 1024, 1280]
        return {"radio_mean": l2(spatial.mean(1).float().cpu().numpy()),
                "radio_summary": l2(summary.float().cpu().numpy())}


ENCODERS = {"radio": RadioEncoder}


def probe(encoder: str) -> None:
    import glob
    path = sorted(glob.glob("data/FracAtlas/images/Fractured/*.jpg"))[0]
    enc = ENCODERS[encoder]()
    if encoder == "radio":
        x = torch.from_numpy(load_tile(path)[None]).permute(0, 3, 1, 2).float().div(255).to(enc.dev)
        with torch.no_grad():
            summary, spatial = enc.model(x)
        print(f"{path}\n  summary {tuple(summary.shape)}  spatial (patch tokens) {tuple(spatial.shape)}  "
              f"nan={bool(torch.isnan(spatial).any())}")
    for k, v in enc([load_tile(path)]).items():
        print(f"  {k}: {v.shape}")


def embed_all(encoder: str, manifest: str, batch: int) -> None:
    import pandas as pd
    from tqdm import tqdm
    df = pd.read_csv(manifest)
    enc = ENCODERS[encoder]()
    out: dict[str, list[np.ndarray]] = {v: [] for v in enc.variants}
    for i in tqdm(range(0, len(df), batch), desc=f"embed {encoder}"):
        for k, v in enc([load_tile(p) for p in df.path[i:i + batch]]).items():
            out[k].append(v)
    os.makedirs(EMB_DIR, exist_ok=True)
    for k, chunks in out.items():
        X = np.concatenate(chunks).astype(np.float32)
        np.save(os.path.join(EMB_DIR, f"{k}.npy"), X)
        df[["path"]].to_csv(os.path.join(EMB_DIR, f"{k}.index.csv"), index_label="row")
        print(f"{k}: {X.shape} -> {EMB_DIR}/{k}.npy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="radio", choices=sorted(ENCODERS))
    ap.add_argument("--manifest", default="data/vision/manifest.csv")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--probe", action="store_true", help="run one image and print shapes")
    a = ap.parse_args()
    torch.manual_seed(0)
    probe(a.encoder) if a.probe else embed_all(a.encoder, a.manifest, a.batch)


if __name__ == "__main__":
    main()
