#!/bin/bash
# Step 3: image embeddings (one encoder at a time), option embeddings, sanity check.
#   ENCODERS="radio"            local / Mac (C-RADIOv2-H = Nemotron VL's vision tower)
#   ENCODERS="radio nemotron"   CUDA box: also the full Nemotron-Nano-12B-v2-VL
#   EMB_URL=http://127.0.0.1:8090/v1/embeddings   use a vLLM pooling server for Qwen3-8B
set -eu
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-vision/bin/python}
for e in ${ENCODERS:-radio}; do
    $PY -m clm.vision.embed_image --encoder "$e" --batch "${BATCH:-16}"   # each run frees its GPU memory on exit
done
$PY -m clm.vision.embed_options ${EMB_URL:+--emb-url "$EMB_URL"}
$PY -m clm.vision.embed_image --check
