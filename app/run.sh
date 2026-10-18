#!/usr/bin/env bash
# Run Second Eye locally. Usage: ./run.sh [--mock] [--fake-clm]
set -euo pipefail
cd "$(dirname "$0")"

PORT="${PORT:-8080}"
for arg in "$@"; do
  case "$arg" in
    --mock) export MOCK_VLLM=1 ;;
    --fake-clm)
      export CLM_URL="http://127.0.0.1:9100" MOCK_CLM=1
      python -m uvicorn tests.fake_clm:app --host 127.0.0.1 --port 9100 --log-level warning &
      trap 'kill $! 2>/dev/null || true' EXIT
      ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

exec python -m uvicorn second_eye.main:app --host "${HOST:-0.0.0.0}" --port "$PORT"
