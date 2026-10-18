#!/bin/bash
# Step 2: download FracAtlas (CC BY 4.0), build the manifest, clean it, split 70/15/15.
# The figshare link in the brief (files/43283628) now returns an empty bot-check page;
# the current file id comes from the figshare API (article 22363012).
set -eu
cd "$(dirname "$0")/../.."
PY=${PY:-.venv-vision/bin/python}
if [ ! -f data/FracAtlas/dataset.csv ]; then
    mkdir -p data/fracatlas_raw
    curl -fsSL -A "Mozilla/5.0" -o data/fracatlas_raw/fracatlas.zip https://ndownloader.figshare.com/files/65518038
    unzip -q -o data/fracatlas_raw/fracatlas.zip -d data
fi
$PY -m clm.vision.data
