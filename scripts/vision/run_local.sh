#!/bin/bash
# Run Second Eye locally on a Mac (Apple MPS) or any Linux box, for manual testing.
#
#   scripts/vision/run_local.sh            # start the API on http://127.0.0.1:8701 (Ctrl+C stops it)
#   scripts/vision/run_local.sh notebook   # start the API in the background + open the demo notebook
#
# Env knobs:
#   MODE=full|lean   full (default): custom answer options work (loads Qwen3-8B, ~16 GB; slow first call on a 24 GB Mac)
#                    lean: only the built-in fracture question, starts in ~1 min, ~3 GB
#   MODEL=...        Hugging Face repo (default harvinderbhullar/second-eye) or "local" for checkpoints/vision
#   PORT=8701
#   SECOND_EYE_API_KEY=...   optional; if set, requests need "Authorization: Bearer <key>"
#   CLM_DEVICE=cpu           force CPU instead of MPS/CUDA
set -euo pipefail
cd "$(dirname "$0")/../.."
CMD=${1:-server}
MODE=${MODE:-full}
MODEL=${MODEL:-harvinderbhullar/second-eye}
PORT=${PORT:-8701}
PY=.venv-vision/bin/python
URL="http://127.0.0.1:$PORT"

if [ ! -x "$PY" ]; then
    echo "creating .venv-vision (first run only)..."
    python3 -m venv .venv-vision
    .venv-vision/bin/pip install -q --upgrade pip
    .venv-vision/bin/pip install -q torch torchvision transformers accelerate safetensors huggingface_hub numpy \
        requests pillow fastapi uvicorn python-multipart einops timm open_clip_torch scikit-learn pandas matplotlib tqdm
    .venv-vision/bin/pip install -q --no-deps -e .
fi

ARGS=(--host 127.0.0.1 --port "$PORT")
[ "$MODEL" != local ] && ARGS+=(--model "$MODEL")
[ "$MODE" = lean ] && ARGS+=(--no-custom-options)

if [ "$CMD" = server ]; then
    echo "Second Eye on $URL  (mode $MODE, model $MODEL)  -- API docs: $URL/docs"
    exec "$PY" -m clm.vision.serve "${ARGS[@]}"
fi

if [ "$CMD" = notebook ]; then
    "$PY" -c "import jupyterlab" 2>/dev/null || .venv-vision/bin/pip install -q jupyterlab ipykernel
    mkdir -p logs
    if curl -sf "$URL/health" >/dev/null; then
        echo "API already running on $URL"
    else
        echo "starting the API in the background (log: logs/run_local.log)..."
        "$PY" -m clm.vision.serve "${ARGS[@]}" > logs/run_local.log 2>&1 &
        SERVER=$!
        trap 'echo "stopping the API"; kill $SERVER 2>/dev/null' EXIT
        printf "waiting for the models to load"
        until curl -sf "$URL/health" >/dev/null; do
            kill -0 $SERVER 2>/dev/null || { echo; echo "the API exited; see logs/run_local.log"; tail -20 logs/run_local.log; exit 1; }
            printf "."; sleep 3
        done
        echo " ready"
    fi
    SECOND_EYE_URL="$URL" .venv-vision/bin/jupyter lab examples/second_eye_demo.ipynb
    exit 0
fi

echo "usage: $0 [server|notebook]"; exit 2
