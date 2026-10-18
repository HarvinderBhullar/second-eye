#!/bin/bash
# Step 4: train the image state head + linear probe for every cached variant, pick the best on val;
# then the random-option-vector head for ablation (a).  Test is not read here.
set -eu
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-vision/bin/python}
$PY -m clm.vision.train
$PY -m clm.vision.train --random-options
