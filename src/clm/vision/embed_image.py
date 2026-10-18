"""Image embeddings for the vision state head.

    python -m clm.vision.embed_image --probe                 # one image, print shapes (checkpoint 1)
    python -m clm.vision.embed_image --encoder radio         # embed every image in data/vision/manifest.csv
    python -m clm.vision.embed_image --check                 # sanity-check every cached variant

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
    """``CLM_DEVICE`` if set (as in ``clm.heads``), else cuda, mps or cpu."""
    if os.environ.get("CLM_DEVICE"):
        return os.environ["CLM_DEVICE"]
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
    def raw(self, tiles: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        """-> (summary [b, 3840], spatial patch features [b, 1024, 1280]) on the device."""
        # RADIO takes [0, 1] RGB and normalises internally (its input conditioner).
        x = torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float().div(255).to(self.dev)
        return self.model(x)

    def pool(self, summary: torch.Tensor, spatial: torch.Tensor) -> dict[str, np.ndarray]:
        return {"radio_mean": l2(spatial.mean(1).float().cpu().numpy()),
                "radio_summary": l2(summary.float().cpu().numpy())}

    def __call__(self, tiles: list[np.ndarray]) -> dict[str, np.ndarray]:
        return self.pool(*self.raw(tiles))


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


def load_variant(variant: str) -> tuple[np.ndarray, list[str]]:
    """-> (embeddings [n, d], paths in row order)."""
    import pandas as pd
    X = np.load(os.path.join(EMB_DIR, f"{variant}.npy"))
    return X, pd.read_csv(os.path.join(EMB_DIR, f"{variant}.index.csv")).path.tolist()


def check(manifest: str) -> None:
    """Dims, NaNs, and mean cosine for same-class vs different-class pairs (2000 random pairs each)."""
    import glob
    import pandas as pd
    lab = pd.read_csv(manifest).set_index("path").label
    rng = np.random.default_rng(0)
    for f in sorted(glob.glob(os.path.join(EMB_DIR, "*.index.csv"))):
        v = os.path.basename(f)[:-len(".index.csv")]
        X, paths = load_variant(v)
        y = lab.loc[paths].to_numpy()
        i, j = rng.integers(0, len(X), (2, 20000))
        cos = (X[i] * X[j]).sum(1)
        ff, nn, fn = cos[(y[i] == 1) & (y[j] == 1)], cos[(y[i] == 0) & (y[j] == 0)], cos[y[i] != y[j]]
        print(f"{v}: {X.shape}  nan={bool(np.isnan(X).any())}  norm={np.linalg.norm(X, axis=1).mean():.3f}  "
              f"cos frac-frac {ff[:2000].mean():.4f}  non-non {nn[:2000].mean():.4f}  frac-non {fn[:2000].mean():.4f}")
    opt = os.path.join(EMB_DIR, "options.npz")
    if os.path.exists(opt):
        o = np.load(opt)
        print(f"options: raw {o['raw'].shape}  proj {o['proj'].shape}  nan={bool(np.isnan(o['proj']).any())}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="radio", choices=sorted(ENCODERS))
    ap.add_argument("--manifest", default="data/vision/manifest.csv")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--probe", action="store_true", help="run one image and print shapes")
    ap.add_argument("--check", action="store_true", help="sanity-check the cached embeddings")
    a = ap.parse_args()
    torch.manual_seed(0)
    if a.check:
        check(a.manifest)
    elif a.probe:
        probe(a.encoder)
    else:
        embed_all(a.encoder, a.manifest, a.batch)


if __name__ == "__main__":
    main()
