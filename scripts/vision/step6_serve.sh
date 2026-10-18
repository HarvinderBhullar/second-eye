#!/bin/bash
# Step 6: the Second Eye API on :8701 (all models load once at startup).
#   EMB_URL=http://127.0.0.1:8090/v1/embeddings   use a vLLM Qwen3-8B pooling server for custom options
#                                                 (recommended on a GPU box; see serve_qwen3_8b.sh)
# Latency:  python scripts/vision/bench_latency.py --n 30
set -eu
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-vision/bin/python}
exec $PY -m clm.vision.serve --host "${HOST:-127.0.0.1}" --port "${PORT:-8701}" ${EMB_URL:+--emb-url "$EMB_URL"}
