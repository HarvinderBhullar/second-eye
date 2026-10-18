#!/bin/bash
# Step 5: Platt scaling + thresholds on val, then ONE evaluation on test (refuses to run twice).
# Run with DRY_RUN=1 first: the same pipeline with val standing in for test, nothing written.
set -eu
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-vision/bin/python}
if [ -n "${DRY_RUN:-}" ]; then
    $PY -m clm.vision.evaluate --dry-run
else
    $PY -m clm.vision.evaluate | tee results/vision/metrics_summary.md
fi
