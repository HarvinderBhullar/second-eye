# Second Eye: a vision state tower for CLM

> **Research prototype, not a medical device.** Nothing here is validated for clinical use;
> do not use it to diagnose or treat anyone.

Second Eye is a vision extension of **CLM (Contrastive Language Models)**. This repository is a
mirror of the original CLM repository, [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM),
including its full history. GitHub does not show a "forked from" link for it, so to be explicit:
everything outside the files listed below is the original CLM work, and the original
[LICENSE](../LICENSE) (Apache-2.0) is unchanged.

## What it does

CLM scores candidate actions by the cosine between a projected *state* and each projected
*action* (answer option). Second Eye swaps the text state for an image:

```
X-ray ──► frozen image encoder ──► trainable image STATE head ──► 512-d ─┐
                                                                         ├─ scale · cos ─► softmax ─► Platt ─► p(option)
option text ──► Qwen3-8B (frozen, last-token) ──► released CLM ACTION head (frozen) ──► 512-d ─┘
```

Task: binary fracture detection on FracAtlas, posed as a CLM typed question:

- Question: *"Is there a bone fracture in this X-ray?"*
- Options: *"Yes, there is a fracture."* / *"No, there is no fracture."*

The action side is CLM's own tower, reused as-is: `clm.embedder.Embedder` (vLLM) or an
in-process equivalent with the same last-token pooling, then `clm.heads.HeadPair.project_actions`
with the released `CLM_v0.1-8B.pt` head. Only the image state head is trained.

### Image encoder

The target encoder is **NVIDIA Nemotron Nano 2 VL** (`nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16`).
Its vision tower is **C-RADIOv2-H** (its `vision_config.auto_map` points at `nvidia/C-RADIOv2-H`;
ViT-H/16, 1280-d, 512×512 tiles).

The results below were produced on a 24 GB Apple-silicon laptop. The full 12B VLM does not fit
there: its BF16 weights alone are about 24 GB, and its Mamba kernels (`mamba-ssm`, `causal-conv1d`)
need CUDA. So this run uses the **vision-tower-only** path, C-RADIOv2-H on one fixed 512×512 tile
per image (the tile size Nemotron feeds it). Two variants come from the same forward pass:

| variant | what | dim |
|---|---|---:|
| `radio_mean` | mean over the 32×32 = 1024 patch tokens | 1280 |
| `radio_summary` | RADIO summary (CLS) vector | 3840 |

**Status of the full-VLM variant:** `--encoder nemotron` (last-layer hidden states of the full VLM,
pooled over image tokens) is planned for the Nebius H100 run and is **not implemented in this
commit**. It will get its own single test pass, reported as a separate row, so the result below
is never re-selected after looking at test.

## What was added

Everything is new code; no upstream file was modified apart from `.gitignore` (embedding / weight
patterns) and a short note at the top of `README.md`.

| path | |
|---|---|
| `src/clm/vision/data.py` | manifest, integrity cleaning, stratified splits |
| `src/clm/vision/embed_image.py` | C-RADIOv2-H embeddings (`--probe`, `--check`) |
| `src/clm/vision/embed_options.py` | option embeddings via CLM's action tower, plus the paraphrase set |
| `src/clm/vision/head.py` | the image state head |
| `src/clm/vision/train.py` | head + linear-probe training, variant selection on val |
| `src/clm/vision/calibrate.py` | Platt scaling, thresholds, ECE |
| `src/clm/vision/evaluate.py` | the single test evaluation and ablations |
| `src/clm/vision/serve.py` | FastAPI app (separate from `clm-serve`) |
| `scripts/vision/step{1..6}_*.sh`, `bench_latency.py` | one script per step |
| `splits/{train,val,test}.csv` | the committed splits |
| `results/vision/` | `dropped.csv`, `train_summary*.json`, `metrics.json`, `metrics_summary.md` |

## How to run

```bash
scripts/vision/step1_env.sh        # venv + deps, one image through C-RADIOv2-H
scripts/vision/step2_data.sh       # download FracAtlas, clean, split  -> splits/*.csv
scripts/vision/step3_embed.sh      # image + option embeddings, sanity check  (EMB_URL=... to use vLLM)
scripts/vision/step4_train.sh      # state heads + linear probes, pick best on val; random-option ablation
DRY_RUN=1 scripts/vision/step5_evaluate.sh   # whole evaluation with val standing in for test
scripts/vision/step5_evaluate.sh   # the one look at test (refuses to run twice)
scripts/vision/step6_serve.sh      # API on :8701
```

On macOS, vLLM does not install, so step 1 installs the package with `--no-deps` and lists the
other dependencies explicitly. On a CUDA machine, `pip install -e .` plus the same extras works,
and `EMB_URL=http://127.0.0.1:8090/v1/embeddings` makes the action tower use a vLLM Qwen3-8B
pooling server (`serve_qwen3_8b.sh`). The in-process path reproduces the README's `engine.rank`
example (0.993 vs 0.997 with vLLM). `embed_options --emb-url … --compare` checks the two paths
against each other.

## Data

**FracAtlas** (Abedeen et al., 2023), CC BY 4.0,
[doi:10.6084/m9.figshare.22363012](https://doi.org/10.6084/m9.figshare.22363012):
4,083 musculoskeletal radiographs, 719 with fractures.
The original download link (`ndownloader/files/43283628`) now returns an empty bot-check page; the
current file (`files/65518038`) comes from the figshare API.

Cleaning (every drop with its reason in `results/vision/dropped.csv`):

| reason | n | fractured | |
|---|---:|---:|---|
| truncated | 59 | 0 | does not decode with `LOAD_TRUNCATED_IMAGES = False`. All non-fractured; decoded leniently they show a mid-grey bottom fill, which would leak the label. |
| grey_band | 73 | 20 | decodes, but the bottom rows are a uniform non-black fill (≥ 2% of height, row std < 1). Mostly the small 373×454 scans; 27% fractured against 20% for that size. Black background and textured plate edges are kept. |
| duplicate | 13 | 4 | byte-identical (SHA-256) to another file; no label conflicts. FracAtlas has no patient IDs, so this is the leakage check available. |

That leaves 3,938 images (695 fractured, 17.6%), split 70/15/15 with seed 42. The split is
stratified on fracture only: the smallest fracture × region group (10 images) was too small for
all three splits. No image ID or file hash appears in two splits.

| split | n | fracture rate | hardware | hand | leg | hip | shoulder | mixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2756 | 0.176 | 0.024 | 0.304 | 0.527 | 0.046 | 0.025 | 0.097 |
| val | 591 | 0.178 | 0.025 | 0.311 | 0.511 | 0.058 | 0.029 | 0.091 |
| test | 591 | 0.176 | 0.030 | 0.293 | 0.540 | 0.029 | 0.022 | 0.117 |

## Training

- **State head:** standardise (train mean/std, stored in the checkpoint) → Linear(d, 1024) → GELU →
  Linear(1024, 512) → L2.
- **Logits:** `scale · cos(state, option_k)`, with a learnable `scale = exp(logit_scale)`
  initialised to 1/0.07 and clamped at 100.
- **Loss:** class-weighted (balanced) cross-entropy over the two options; the action side is frozen.
- **Optimiser:** AdamW, lr 3e-5, weight decay 0.1, batch 64, early stopping on val AUROC
  (patience 25). No hyperparameter sweep. All seeds are fixed.
- **Baseline:** logistic regression (balanced, C = 1) on the same standardised embeddings.

| variant | CLM head val AUROC | linear probe val AUROC |
|---|---:|---:|
| `radio_mean` | 0.923 | 0.900 |
| **`radio_summary`** (selected on val) | **0.935** | 0.927 |

The learnable scale barely moves at this learning rate (14.3 → 14.35); calibration is handled by
Platt scaling instead.

## Results (test, evaluated once)

591 images, 104 fractured. Platt scaling and both thresholds were fit on val:

- **Youden:** the val threshold that maximises J.
- **≥0.90-sens:** the *highest* val threshold whose sensitivity is still ≥ 0.90. Any lower
  threshold also reaches 0.90; this one keeps the most specificity.

The AUROC CI is a 1,000-sample stratified bootstrap. ECE uses 15 equal-width bins on p(fracture).

| method | val AUROC | test AUROC [95% CI] | sens / spec @ Youden | sens / spec @ ≥0.90-sens | ECE raw | ECE Platt |
|---|---:|---:|---:|---:|---:|---:|
| **CLM head · C-RADIOv2-H summary** | 0.935 | **0.925 [0.891, 0.953]** | 0.760 / 0.938 | **0.913 / 0.723** | 0.062 | **0.020** |
| Linear probe · C-RADIOv2-H summary | 0.926 | 0.893 [0.850, 0.931] | 0.750 / 0.906 | 0.885 / 0.612 | 0.085 | 0.041 |
| Resolution only (leakage check) | 0.595 | 0.616 [0.577, 0.656] | 0.933 / 0.269 | 0.933 / 0.269 | 0.323 | 0.025 |
| *Full Nemotron VL* | | *pending (Nebius run)* | | | | |

### Ablations

| ablation | test AUROC [95% CI] | sens / spec @ ≥0.90-sens |
|---|---:|---:|
| (a) random option vectors instead of the CLM option embeddings (retrained) | 0.913 [0.878, 0.945] | 0.875 / 0.733 |

| (b) reworded options, trained head, no retraining | test AUROC | sens / spec @ original ≥0.90-sens threshold |
|---|---:|---:|
| A fracture is present. / No fracture is present. | 0.918 | 0.981 / 0.115 |
| Yes, the bone is broken. / No, the bone is intact. | 0.918 | 0.981 / 0.232 |
| This radiograph shows a fractured bone. / This radiograph shows no fracture. | 0.922 | 1.000 / 0.000 |
| Fracture detected. / No fracture detected. | 0.913 | 0.971 / 0.290 |
| Yes, I can see a break in the bone. / No, I cannot see any break in the bone. | 0.926 | 0.981 / 0.444 |

### Reading the results

- The calibrated head meets the ≥0.90-sensitivity target on test (0.913, specificity 0.723), and
  Platt scaling brings ECE from 0.062 down to 0.020.
- The head is ahead of the linear probe in point estimate (0.925 vs 0.893), but the CIs overlap,
  so this is not evidence that it is better.
- **The text tower adds little measurable accuracy here.** With only two fixed options, random
  option vectors reach 0.913, with an overlapping CI. The option embeddings are effectively a
  fixed output direction, and the head acts as a classifier. This matches the earlier MedSigLIP /
  Qwen3-VL work (see *Prior work*). What the CLM head buys is the interface: the image becomes a
  typed question answered in CLM's action space.
- **Rewordings keep the ranking, not the calibration.** Paraphrased options give AUROC
  0.913–0.926 with no retraining, so the learned direction lines up with fracture / no-fracture
  wording generally. But the probabilities shift with the wording, so the val calibration and
  threshold do not transfer. The server therefore calibrates only the trained options.
- Image size alone reaches 0.616 AUROC. That is mostly body region (the small scans are largely
  hands, which have a higher fracture rate), a confounder rather than a pixel shortcut.

## Released model

The selected head (`radio_summary`) is on the Hugging Face Hub:
[harvinderbhullar/second-eye](https://huggingface.co/harvinderbhullar/second-eye). It contains
`model.safetensors` (head and trained option vectors), `config.json`, `calibration.json` and a
model card. `python -m clm.vision.hub export` rebuilds it from the local checkpoint.

```bash
python -m clm.vision.serve --model harvinderbhullar/second-eye --port 8701
```

## Serving

```bash
scripts/vision/step6_serve.sh                    # python -m clm.vision.serve --port 8701
```

`POST /predict` takes an image upload and optional `question` and `options` (a JSON list) form
fields. It returns per-option probabilities, the decision, the threshold and `latency_ms`. All
models load once at startup.

```bash
curl -s -X POST localhost:8701/predict -F image=@data/FracAtlas/images/Fractured/IMG0000019.jpg
```

```json
{
  "question": "Is there a bone fracture in this X-ray?",
  "options": ["Yes, there is a fracture.", "No, there is no fracture."],
  "probabilities": {"Yes, there is a fracture.": 0.589, "No, there is no fracture.": 0.411},
  "calibrated": true,
  "decision": "Yes, there is a fracture.",
  "threshold": 0.0675,
  "operating_point": "val sensitivity >= 0.90",
  "latency_ms": 307.5,
  "disclaimer": "Research prototype, not a medical device. Not for clinical use."
}
```

Custom options are embedded on the fly by the action tower and LRU-cached:

```bash
curl -s -X POST localhost:8701/predict -F image=@xray.jpg \
     -F 'options=["A fracture is present.", "No fracture is present."]'
```

For any option set other than the trained one, the response has `"calibrated": false`, raw softmax
probabilities and an argmax decision (see ablation (b)). The `question` field is echoed back. The
state is the image alone, and the head was trained for the one question above.

### Latency

Measured with `python scripts/vision/bench_latency.py --n 30` on val images:

| device | request | server p50 | p90 |
|---|---|---:|---:|
| Apple M5 Pro (MPS, 24 GB) | trained options | 307 ms | 309 ms |
| Apple M5 Pro (MPS, 24 GB) | custom options, cached | 308 ms | 311 ms |
| Apple M5 Pro (MPS, 24 GB) | custom options, first time | 80–90 s | |
| Apple M5 Pro, CPU only (`CLM_DEVICE=cpu`, lean mode) | trained options | 1,175 ms | 1,256 ms |
| Nebius GPU VM | | *pending* | |

On the laptop, the first request with new options is dominated by Qwen3-8B (16 GB in BF16) paging
back in from swap next to C-RADIO. On a GPU box, point `EMB_URL` at a vLLM pooling server instead.

## Deploying as an endpoint

The endpoint only needs the public head on the Hub, not the dataset or any training. On a fresh
Ubuntu VM (GPU or CPU), one script installs it and runs it as a systemd service on `:8701`:

```bash
git clone -b hackathon https://github.com/HarvinderBhullar/second-eye && cd second-eye
scripts/vision/deploy/setup_vm.sh               # MODE=lean to skip Qwen3-8B; EMB_URL=... for a remote vLLM encoder
```

- **API key.** The script generates `SECOND_EYE_API_KEY` and stores it in `/etc/second-eye.env`.
  `/predict` then requires `Authorization: Bearer <key>`; `/health` stays open.
- **Modes.** *full* answers custom options with an in-process Qwen3-8B (about 16 GB; a GPU is
  recommended). *lean* answers only the trained question and never loads Qwen3-8B.
- **Remote encoder.** With `EMB_URL` and `CLM_EMB_API_KEY`, custom options use a vLLM pooling
  endpoint instead, e.g. a Nebius Serverless vLLM endpoint running
  `--model Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048`.
- **Network.** The service binds to `127.0.0.1`. For HTTPS, put Caddy in front
  (`scripts/vision/deploy/Caddyfile`, which needs a DNS name). For a quick demo, use
  `ssh -L 8701:localhost:8701 <vm>`.
- **Limits.** Uploads up to 20 MB, images up to 50 MP, and 2–8 options of at most 300 characters.

```bash
curl -s -X POST https://<host>/predict -H "Authorization: Bearer $SECOND_EYE_API_KEY" -F image=@xray.jpg
```

## Prior work

A previous version of this idea, using **MedSigLIP-448** and **Qwen3-VL-4B** as image encoders,
existed before the event: [HarvinderBhullar/CLM](https://github.com/HarvinderBhullar/CLM). It was
used here as a design reference only (data cleaning, training objective, calibration and the
evaluation protocol); no code was copied from it. Its splits differ, so its numbers are not
directly comparable to the ones above.

The Nemotron VL / C-RADIO integration, all the code in this document and the Nebius deployment were
built at the Nebius × NVIDIA hackathon in Berlin (October 2026), on top of the original
[Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM).

## Credits and licences

- CLM: [Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM), Apache-2.0 (see [LICENSE](../LICENSE)).
- FracAtlas: Abedeen, I. et al., *FracAtlas: A Dataset for Fracture Classification, Localization and
  Segmentation of Musculoskeletal Radiographs*, Scientific Data (2023). CC BY 4.0.
- C-RADIOv2-H and Nemotron Nano 2 VL: NVIDIA, under their model licences on Hugging Face.
- Qwen3-8B: Qwen team, Apache-2.0.
