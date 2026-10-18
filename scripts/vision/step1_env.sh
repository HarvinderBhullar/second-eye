#!/bin/bash
# Step 1: environment + one image through the image encoder.
# macOS/CPU: vLLM does not install, so the package goes in with --no-deps and its
# other dependencies are listed explicitly.  On a CUDA box `pip install -e .` also works.
set -eu
cd "$(dirname "$0")/../.."
python3 -m venv .venv-vision
.venv-vision/bin/pip install -q --upgrade pip
.venv-vision/bin/pip install -q torch torchvision transformers accelerate scikit-learn pandas pillow fastapi \
    uvicorn gradio numpy requests pyarrow huggingface_hub tqdm einops timm open_clip_torch python-multipart matplotlib
.venv-vision/bin/pip install -q --no-deps -e .
.venv-vision/bin/python -m clm.vision.embed_image --encoder "${ENCODER:-radio}" --probe
