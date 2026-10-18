"""FracAtlas: manifest, integrity cleaning and stratified splits.

    python -m clm.vision.data            # -> data/vision/manifest.csv, results/vision/dropped.csv, splits/*.csv

Cleaning, every drop logged with its reason:
  truncated    the JPEG does not fully decode with ImageFile.LOAD_TRUNCATED_IMAGES = False.
               In FracAtlas all of these are non-fractured and decode leniently with a mid-grey
               (128) fill at the bottom, a visual shortcut that leaks the label.
  grey_band    decodes, but the bottom rows are a uniform non-black fill (same artefact baked in).
               Black background and the textured bright plate edge of the hand scans are kept.
  duplicate    byte-identical to an earlier file (FracAtlas has no patient IDs; this is the
               leakage check we can do).  If duplicates disagree on the label, all copies go.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ROOT = os.path.join("data", "FracAtlas")
OUT_DIR = os.path.join("data", "vision")
RESULTS = os.path.join("results", "vision")
SPLITS = "splits"
SEED = 42
REGIONS = ["hand", "leg", "hip", "shoulder"]
VIEWS = ["frontal", "lateral", "oblique"]

# grey-band rule (see module docstring); thresholds picked by inspecting the bottom-row statistics
ROW_STD = 1.0          # a row is "uniform" below this pixel std
BAND_MIN_FRAC = 0.02   # uniform run must cover >= 2% of the image height
BAND_MIN_MEAN = 60.0   # ... and not be black background


def region_of(r) -> str:
    set_ = [k for k in REGIONS if r[k]]
    return "mixed" if r["mixed"] or len(set_) > 1 else (set_[0] if set_ else "unknown")


def view_of(r) -> str:
    return "+".join(k for k in VIEWS if r[k]) or "unknown"


def build_manifest() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(ROOT, "dataset.csv"))
    print(f"dataset.csv: {df.shape[0]} rows, columns: {list(df.columns)}")
    print(df.head().to_string(index=False))
    m = pd.DataFrame({
        "image_id": df.image_id,
        "path": [os.path.join(ROOT, "images", "Fractured" if f else "Non_fractured", i)
                 for i, f in zip(df.image_id, df.fractured)],
        "label": df.fractured.astype(int),
        "region": df.apply(region_of, axis=1),
        "hardware": df.hardware.astype(int),
        "view": df.apply(view_of, axis=1),
    })
    missing = [p for p in m.path if not os.path.exists(p)]
    assert not missing, f"{len(missing)} manifest paths do not exist, e.g. {missing[:3]}"
    print(f"\ntotal {len(m)} images, {m.label.sum()} fractured ({m.label.mean():.1%}); all paths exist")
    return m


def check_image(path: str) -> tuple[str | None, dict]:
    """-> (drop reason or None, info).  Fully decodes the file."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        im = Image.open(path)
        im.load()
    except Exception as e:  # noqa: BLE001 - any decode failure is a drop
        return "truncated", {"detail": str(e)[:100]}
    a = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = a.shape
    run = 0
    for s in a.std(axis=1)[::-1]:
        if s >= ROW_STD:
            break
        run += 1
    info = {"width": w, "height": h}
    if run >= BAND_MIN_FRAC * h:
        band = a[h - run:]
        if band.std() < ROW_STD and band.mean() >= BAND_MIN_MEAN:
            return "grey_band", {**info, "detail": f"{run} uniform rows ({run / h:.1%}), mean {band.mean():.0f}"}
    return None, info


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def clean(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """-> (manifest with size/hash columns, kept rows, dropped log)."""
    from tqdm import tqdm
    dropped, info = [], []
    for r in tqdm(m.itertuples(), total=len(m), desc="integrity"):
        reason, d = check_image(r.path)
        info.append({"width": d.get("width"), "height": d.get("height"), "sha256": sha256(r.path)})
        if reason:
            dropped.append({"image_id": r.image_id, "path": r.path, "label": r.label, "reason": reason,
                            "detail": d.get("detail", "")})
    m = pd.concat([m, pd.DataFrame(info, index=m.index)], axis=1)
    bad = {d["image_id"] for d in dropped}
    keep = m[~m.image_id.isin(bad)]
    for h, g in keep.groupby("sha256"):
        if len(g) < 2:
            continue
        drop = g if g.label.nunique() > 1 else g.iloc[1:]
        for r in drop.itertuples():
            dropped.append({"image_id": r.image_id, "path": r.path, "label": r.label, "reason": "duplicate",
                            "detail": f"same bytes as {g.image_id.iloc[0]}" + (" (label conflict)" if len(drop) == len(g) else "")})
    dropped = pd.DataFrame(dropped, columns=["image_id", "path", "label", "reason", "detail"])
    return m, m[~m.image_id.isin(set(dropped.image_id))].reset_index(drop=True), dropped


def split(m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    strata = m.label.astype(str) + "_" + m.region
    # each stratum needs enough members to land in all three splits (15% of it >= 2)
    if strata.value_counts().min() < 14:
        print(f"smallest fracture x region stratum has {strata.value_counts().min()} images; stratifying on fracture only")
        strata = m.label.astype(str)
    else:
        print("stratifying on fracture x body region")
    train, rest = train_test_split(m, test_size=0.30, stratify=strata, random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata[rest.index], random_state=SEED)
    return {"train": train, "val": val, "test": test}


def summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, d in splits.items():
        reg = d.region.value_counts(normalize=True)
        rows.append({"split": name, "n": len(d), "fractured": int(d.label.sum()), "fracture_rate": d.label.mean(),
                     "hardware_rate": d.hardware.mean(),
                     **{f"region_{k}": reg.get(k, 0.0) for k in REGIONS + ["mixed"]}})
    return pd.DataFrame(rows)


def main() -> None:
    m = build_manifest()
    print("\nregion counts:", m.region.value_counts().to_dict())
    print("view counts:", m.view.value_counts().to_dict())
    m, kept, dropped = clean(m)
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True); os.makedirs(SPLITS, exist_ok=True)
    m.to_csv(os.path.join(OUT_DIR, "manifest_all.csv"), index=False)
    kept.to_csv(os.path.join(OUT_DIR, "manifest.csv"), index=False)
    dropped.to_csv(os.path.join(RESULTS, "dropped.csv"), index=False)
    print(f"\ndropped {len(dropped)} images:")
    print(dropped.groupby("reason").label.agg(n="size", fractured="sum").to_string())
    print(f"kept {len(kept)} images, {kept.label.sum()} fractured ({kept.label.mean():.1%})\n")

    splits = split(kept)
    ids = {k: set(v.image_id) for k, v in splits.items()}
    hashes = {k: set(v.sha256) for k, v in splits.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not ids[a] & ids[b], f"image overlap between {a} and {b}"
        assert not hashes[a] & hashes[b], f"duplicate bytes across {a} and {b}"
    assert sum(len(v) for v in splits.values()) == len(kept)
    cols = ["image_id", "path", "label", "region", "hardware", "view", "width", "height", "sha256"]
    for k, v in splits.items():
        v.sort_values("image_id")[cols].to_csv(os.path.join(SPLITS, f"{k}.csv"), index=False)
    print(summary(splits).to_markdown(index=False, floatfmt=".3f"))
    print("\nno overlap between splits (image ids and file hashes); saved splits/{train,val,test}.csv")


if __name__ == "__main__":
    main()
