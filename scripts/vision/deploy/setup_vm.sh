#!/bin/bash
# Set up the Second Eye endpoint on a fresh Ubuntu VM (Nebius or any cloud), as a systemd service.
# Serves the public head harvinderbhullar/second-eye; no training, no dataset needed.
#
#   curl -fsSL https://raw.githubusercontent.com/HarvinderBhullar/second-eye/hackathon/scripts/vision/deploy/setup_vm.sh | bash
#   # or, from a clone:  scripts/vision/deploy/setup_vm.sh
#
# Env knobs:
#   MODE=full|lean     full (default): custom options via in-process Qwen3-8B (~16 GB; GPU recommended)
#                      lean: trained question only, no Qwen3-8B (fast start, ~3 GB)
#   EMB_URL=...        use a remote vLLM embeddings endpoint (e.g. Nebius) instead of in-process Qwen3-8B
#   CLM_EMB_API_KEY=   its token
#   BIND=127.0.0.1     use 0.0.0.0 to expose :8701 directly (no TLS; prefer Caddy, see Caddyfile)
set -euo pipefail
REPO=https://github.com/HarvinderBhullar/second-eye.git
DIR=${DIR:-$HOME/second-eye}
MODE=${MODE:-full}
BIND=${BIND:-127.0.0.1}

sudo apt-get update -qq && sudo apt-get install -y -qq git python3-venv python3-dev curl >/dev/null
[ -d "$DIR/.git" ] || git clone -q -b hackathon "$REPO" "$DIR"
git -C "$DIR" pull -q --ff-only
cd "$DIR"

python3 -m venv .venv-vision
PIP=".venv-vision/bin/pip"
$PIP install -q --upgrade pip
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    echo "GPU found:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    $PIP install -q torch torchvision                     # CUDA wheels from PyPI
else
    echo "no GPU: installing CPU-only torch"
    $PIP install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
$PIP install -q transformers accelerate safetensors huggingface_hub numpy requests pillow fastapi uvicorn \
    python-multipart einops timm open_clip_torch scikit-learn pandas
$PIP install -q --no-deps -e .

# secrets / settings, kept out of the unit file
ENV=/etc/second-eye.env
if [ ! -f "$ENV" ]; then
    KEY=$(openssl rand -hex 24)
    printf 'SECOND_EYE_API_KEY=%s\nCLM_EMB_API_KEY=%s\nHF_HOME=%s/.cache/huggingface\n' \
        "$KEY" "${CLM_EMB_API_KEY:-}" "$HOME" | sudo tee "$ENV" >/dev/null
    sudo chmod 600 "$ENV"
    echo "generated API key (also in $ENV): $KEY"
fi

ARGS="--host $BIND --port 8701 --model harvinderbhullar/second-eye"
[ "$MODE" = lean ] && ARGS="$ARGS --no-custom-options"
[ -n "${EMB_URL:-}" ] && ARGS="$ARGS --emb-url $EMB_URL"
sed -e "s#@DIR@#$DIR#g" -e "s#@USER@#$USER#g" -e "s#@ARGS@#$ARGS#g" \
    scripts/vision/deploy/second-eye.service | sudo tee /etc/systemd/system/second-eye.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now second-eye
sudo systemctl restart second-eye

echo "waiting for the models to load (the first start downloads the weights; full mode also Qwen3-8B, ~16 GB)..."
curl -s --retry 120 --retry-delay 5 --retry-connrefused --retry-all-errors "http://127.0.0.1:8701/health"; echo
echo "logs: journalctl -u second-eye -f"
