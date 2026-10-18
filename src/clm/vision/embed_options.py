"""Option (action-side) embeddings: Qwen3-8B last-token pooling + the released CLM action head.

    python -m clm.vision.embed_options                        # in-process Qwen3-8B (no vLLM, e.g. macOS)
    python -m clm.vision.embed_options --emb-url http://127.0.0.1:8090/v1/embeddings   # vLLM, as clm-serve
    python -m clm.vision.embed_options --emb-url ... --compare  # check both paths give the same vectors

The encoder call goes through the package's ``Embedder`` when a vLLM pooling server is up.
Without one, ``LocalQwenEmbedder`` computes the same thing in-process: the last-layer hidden
state (after the final norm) of the last token, L2-normalised, which is what
``vllm serve Qwen/Qwen3-8B --runner pooling`` returns.  Either way the 4096-d vectors go
through the released action head via ``clm.heads.HeadPair`` -- the action tower is reused,
not reimplemented.

Writes data/vision/emb/options.npz: ``texts``, ``raw`` [n, 4096] and ``proj`` [n, 512]
(L2-normalised action-head outputs), for the task options followed by the paraphrases.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from ..embedder import Embedder, l2
from ..heads import HeadPair, default_checkpoint, download
from . import OPTIONS

QWEN = "Qwen/Qwen3-8B"
OUT = os.path.join("data", "vision", "emb", "options.npz")

# Step 5 ablation: reworded option pairs (positive first), never used for training.
PARAPHRASES = [
    ["A fracture is present.", "No fracture is present."],
    ["Yes, the bone is broken.", "No, the bone is intact."],
    ["This radiograph shows a fractured bone.", "This radiograph shows no fracture."],
    ["Fracture detected.", "No fracture detected."],
    ["Yes, I can see a break in the bone.", "No, I cannot see any break in the bone."],
]


class LocalQwenEmbedder:
    """Duck-types ``clm.embedder.Embedder.embed`` with an in-process Qwen3-8B."""

    def __init__(self, model: str = QWEN, dev: str | None = None, max_tokens: int = 2048):
        from transformers import AutoModel, AutoTokenizer
        from .embed_image import device
        self.dev, self.max_tokens = dev or device(), max_tokens
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, dtype=torch.bfloat16).eval().to(self.dev)
        self.cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def embed(self, texts: list[str]) -> tuple[np.ndarray, int]:
        todo = [t for t in dict.fromkeys(texts) if t not in self.cache]
        tokens = 0
        for t in todo:   # one at a time: no padding, so the last position is the last real token
            ids = self.tok(t, return_tensors="pt", truncation=True, max_length=self.max_tokens).input_ids.to(self.dev)
            h = self.model(input_ids=ids).last_hidden_state[0, -1].float().cpu().numpy()
            self.cache[t] = l2(h)
            tokens += ids.shape[1]
        return np.stack([self.cache[t] for t in texts]), tokens


def action_head(ckpt: str | None = None, dev: str | None = None) -> HeadPair:
    from .embed_image import device
    path = ckpt or default_checkpoint() or download()
    return HeadPair("clm-latest", path, dev or device()).ensure()


def embed_texts(texts: list[str], embedder, head: HeadPair) -> tuple[np.ndarray, np.ndarray]:
    """-> (raw [n, 4096], proj [n, proj_dim]) both L2-normalised."""
    raw, _ = embedder.embed(texts)
    return raw.astype(np.float32), head.project_actions(raw.astype(np.float32)).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emb-url", default=None, help="vLLM /v1/embeddings URL; default: in-process Qwen3-8B")
    ap.add_argument("--emb-model", default="qwen3-8b")
    ap.add_argument("--ckpt", default=None, help="CLM checkpoint (default: the released head, downloaded if missing)")
    ap.add_argument("--compare", action="store_true", help="also run the other path and print the cosine")
    a = ap.parse_args()
    torch.manual_seed(0)
    texts = list(OPTIONS) + [t for pair in PARAPHRASES for t in pair]
    head = action_head(a.ckpt)
    print(f"action head: {head.path}  proj_dim={head.proj_dim}  scale={head.scale:.2f}")
    emb = Embedder(url=a.emb_url, model=a.emb_model) if a.emb_url else LocalQwenEmbedder()
    raw, proj = embed_texts(texts, emb, head)
    if a.compare and a.emb_url:
        raw2, _ = embed_texts(texts, LocalQwenEmbedder(), head)
        print("cosine(vLLM, in-process) per text: min %.4f  mean %.4f" % (
            (raw * raw2).sum(1).min(), (raw * raw2).sum(1).mean()))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, texts=np.array(texts), raw=raw, proj=proj)
    print(f"raw {raw.shape}  proj {proj.shape}  nan={bool(np.isnan(proj).any())}  -> {OUT}")
    print(f"cos(option yes, option no): raw {raw[0] @ raw[1]:.4f}  proj {proj[0] @ proj[1]:.4f}")
    P = proj[2:].reshape(len(PARAPHRASES), 2, -1)
    print("paraphrase vs task option (proj cosine): yes-side %s  no-side %s" % (
        np.round(P[:, 0] @ proj[0], 3).tolist(), np.round(P[:, 1] @ proj[1], 3).tolist()))


if __name__ == "__main__":
    main()
