"""Hugging Face Hub export / load of the selected image state head (no pickles).

    python -m clm.vision.hub export                                  # -> hf_export/ (safetensors, config, card)
    python -m clm.vision.hub push --repo harvinderbhullar/second-eye --private
    python -m clm.vision.serve --model harvinderbhullar/second-eye   # serve straight from the Hub

Files: ``model.safetensors`` (head weights incl. input standardisation, and the two frozen option
vectors it was trained against), ``config.json``, ``calibration.json`` and ``README.md`` (model
card generated from results/vision/metrics.json).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

from . import OPTIONS, QUESTION
from .embed_image import RADIO_REPO, TILE
from .head import StateHead
from .train import CKPT_DIR, RESULTS

EXPORT_DIR = "hf_export"
UPSTREAM = "https://github.com/Contrastive-LM/CLM"
CODE = "https://github.com/HarvinderBhullar/second-eye/tree/hackathon"


def export(out: str = EXPORT_DIR, ckpt_dir: str = CKPT_DIR) -> str:
    from safetensors.torch import save_file
    with open(os.path.join(ckpt_dir, "calibration.json")) as f:
        cal = json.load(f)
    variant = cal["variant"]
    head, ck = StateHead.load(os.path.join(ckpt_dir, f"{variant}.pt"))
    os.makedirs(out, exist_ok=True)
    tensors = {k: v.contiguous() for k, v in head.state_dict().items()}
    tensors["options"] = ck["options"].contiguous()
    save_file(tensors, os.path.join(out, "model.safetensors"))
    config = {**head.cfg, "variant": variant, "encoder": RADIO_REPO, "tile": TILE,
              "pooling": "RADIO summary vector" if variant.endswith("summary") else "mean over patch tokens",
              "question": QUESTION, "options": OPTIONS, "positive_option": 0,
              "action_tower": {"encoder": "Qwen/Qwen3-8B (last-token pooling)",
                               "head": "Contrastive-LM/CLM-v0.1-8B/CLM_v0.1-8B.pt"}}
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    shutil.copy(os.path.join(ckpt_dir, "calibration.json"), os.path.join(out, "calibration.json"))
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(model_card(config, cal))
    return out


def load(model: str, device: str = "cpu") -> tuple[StateHead, torch.Tensor, dict, dict]:
    """``model``: a local export dir or a Hub repo id -> (head, options [2, 512], calibration, config)."""
    from safetensors.torch import load_file
    if not os.path.isdir(model):
        from huggingface_hub import snapshot_download
        model = snapshot_download(model)
    with open(os.path.join(model, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(model, "calibration.json")) as f:
        cal = json.load(f)
    t = load_file(os.path.join(model, "model.safetensors"))
    options = t.pop("options")
    head = StateHead(config["in_dim"], config["out_dim"], config["hidden"])
    head.load_state_dict(t)
    return head.eval().to(device), options, cal, config


def model_card(config: dict, cal: dict) -> str:
    with open(os.path.join(RESULTS, "metrics.json")) as f:
        m = json.load(f)
    rows = []
    for r in m["results"] + m["ablations"]:
        lo, hi = r["test_auroc_ci95"]
        rows.append(f"| {r['method']} | {r['test_auroc']:.3f} [{lo:.3f}, {hi:.3f}] | "
                    f"{r['sens90']['sensitivity']:.3f} / {r['sens90']['specificity']:.3f} | "
                    f"{r['ece_raw']:.3f} → {r['ece_platt']:.3f} |")
    table = "\n".join(rows)
    return f"""---
license: apache-2.0
library_name: pytorch
pipeline_tag: image-classification
tags:
- medical-imaging
- x-ray
- fracture-detection
- contrastive
- clm
- research-prototype
datasets:
- FracAtlas
---

# Second Eye: CLM image state head for fracture detection (FracAtlas, C-RADIOv2-H)

> **Research prototype, not a medical device.** Not validated for clinical use. Do not use it to
> diagnose or treat anyone.

A small trainable image **state head** for [CLM (Contrastive Language Models)]({UPSTREAM}).
Frozen `{config['encoder']}` image features (the vision tower of NVIDIA Nemotron Nano 2 VL) go
through this head into CLM's 512-d action space. There they are scored by `scale · cos` against
CLM's frozen **action tower** (Qwen3-8B last-token pooling + the released
[CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) action head) embedding the
answer options:

- Question: *"{config['question']}"*
- Options: *"{config['options'][0]}"* / *"{config['options'][1]}"*

Code, training and evaluation: [{CODE}]({CODE}) (see `docs/VISION.md`).

## Files

| file | |
|---|---|
| `model.safetensors` | head weights (incl. train-set input standardisation) and `options`: the 2 frozen option vectors it was trained against |
| `config.json` | architecture ({config['in_dim']} → {config['hidden']} → {config['out_dim']}, GELU), encoder, tile size, pooling, question and options |
| `calibration.json` | Platt scaling (`a`, `b`) and two thresholds on calibrated p(fracture), all fit on the validation split |

## Usage

```bash
git clone -b hackathon https://github.com/HarvinderBhullar/second-eye && cd second-eye
scripts/vision/step1_env.sh
python -m clm.vision.serve --model harvinderbhullar/second-eye --port 8701
curl -s -X POST localhost:8701/predict -F image=@xray.jpg
```

Input: one `{config['tile']}×{config['tile']}` RGB tile per image (greyscale X-rays converted to
RGB, bicubic resize), {config['pooling']}, L2-normalised. Calibrated probabilities and the
decision (`p ≥ {cal['threshold_sens90']:.4f}`, the val operating point with sensitivity ≥ 0.90) are
valid only for the two trained options. Reworded options keep the ranking but not the calibration.

## Evaluation

FracAtlas after integrity cleaning (truncated JPEGs, uniform bottom fills and byte duplicates
removed): 3,938 images, 70/15/15 split stratified on fracture, seed 42. Every choice was made on
validation. The test split ({m['n_test']} images, {m['n_test_fractured']} fractured) was evaluated
once. AUROC CI: 1,000-sample stratified bootstrap.

| method | test AUROC [95% CI] | sens / spec @ val ≥0.90-sens threshold | ECE raw → Platt |
|---|---:|---:|---:|
{table}

**Limitations.**
- With two fixed options, random option vectors do about as well as the CLM option embeddings
  (see the last row). The head is effectively a classifier, and its value is the CLM interface,
  not accuracy.
- The head is ahead of a linear probe in point estimate, but the CIs overlap.
- It is trained on one public dataset from a limited set of sources, with no patient IDs and no
  external validation.
- Image size alone reaches about 0.62 AUROC on this data (a body-region confounder).

## Requirements and licences

- These head weights: Apache-2.0.
- Using them requires [`{config['encoder']}`](https://huggingface.co/{config['encoder']}) (NVIDIA,
  under its own licence) for the features, and Qwen3-8B plus the CLM action head only for custom
  options.
- Training data: **FracAtlas**, Abedeen et al., *Scientific Data* (2023),
  [doi:10.6084/m9.figshare.22363012](https://doi.org/10.6084/m9.figshare.22363012), CC BY 4.0.
- Built on [Contrastive-LM/CLM]({UPSTREAM}) (Apache-2.0) at the Nebius × NVIDIA hackathon,
  Berlin, October 2026.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "push"])
    ap.add_argument("--repo", default="harvinderbhullar/second-eye")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--out", default=EXPORT_DIR)
    a = ap.parse_args()
    out = export(a.out)
    print(f"exported -> {out}: {sorted(os.listdir(out))}")
    if a.cmd == "push":
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.repo, private=a.private, exist_ok=True)
        api.upload_folder(folder_path=out, repo_id=a.repo, commit_message="Second Eye: FracAtlas state head (C-RADIOv2-H)")
        print(f"pushed -> https://huggingface.co/{a.repo}" + ("  (private)" if a.private else ""))


if __name__ == "__main__":
    main()
