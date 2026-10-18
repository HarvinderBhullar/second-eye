#!/bin/bash
# Start script for the Second Eye image head on a Nebius AI endpoint (pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime).
# Code comes from GitHub, pinned to a commit; weights download from Hugging Face on first start.
# Injected with: --inject-file "./start.sh:/opt/start.sh" --container-command bash --args /opt/start.sh
set -euo pipefail

CODE_REF="${CODE_REF:-f575b34a408eeac9a4d001d6a6b4b47c9eb10828}"   # HarvinderBhullar/second-eye main, Oct 2026

echo "[start] installing dependencies (torch comes from the image and must stay 2.5.1)"
# transformers 5.x needs torch >= 2.6 and silently disables torch on this image, so stay on 4.x.
# These pins were rehearsed end to end (health + calibrated /predict) on torch 2.5.1 / Python 3.11.
pip install -q --no-cache-dir "torch==$(python -c 'import torch; print(torch.__version__.split("+")[0])')" \
    "transformers==4.57.6" "timm==1.0.30" "open_clip_torch==3.3.0" \
    accelerate safetensors huggingface_hub numpy requests pillow \
    fastapi uvicorn python-multipart einops scikit-learn pandas
echo "[start] installing second-eye code at ${CODE_REF}"
pip install -q --no-cache-dir --no-deps "https://github.com/HarvinderBhullar/second-eye/archive/${CODE_REF}.zip"
python -c "import torch, transformers; print('[start] torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"

ARGS="--host 0.0.0.0 --port 8701 --model ${MODEL_REPO:-harvinderbhullar/second-eye}"
[ "${MODE:-lean}" = lean ] && ARGS="$ARGS --no-custom-options"
echo "[start] python -m clm.vision.serve $ARGS  (weights download from Hugging Face on first start)"
exec python -m clm.vision.serve $ARGS
