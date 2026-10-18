"""Step 5: calibrate on val, then evaluate on test exactly once.

    python -m clm.vision.evaluate              # refuses to run twice (results/vision/.test_evaluated_<variant>)
    python -m clm.vision.evaluate --dry-run    # whole pipeline with val standing in for test (no test read)

Uses the variant picked on val by ``clm.vision.train``.  Reports, on test:
AUROC with a 1000-sample stratified bootstrap 95% CI, sensitivity / specificity at the val
Youden and val >=0.90-sensitivity thresholds, ECE before and after val Platt scaling; the same
for the linear probe; plus a resolution-only baseline (leakage check) and two ablations:
(a) the head retrained with random option vectors, (b) the trained head scored against five
reworded option pairs without retraining.  Writes results/vision/metrics.json and
checkpoints/vision/calibration.json (used by the server).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import calibrate as cal
from .embed_image import EMB_DIR
from .embed_options import PARAPHRASES
from .head import StateHead
from .train import CKPT_DIR, RESULTS, p_fracture, split_xy

N_BOOT = 1000


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, n: int = N_BOOT, seed: int = 0) -> list[float]:
    """Stratified bootstrap: resample positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    aucs = []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        aucs.append(roc_auc_score(y[i], p[i]))
    return [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))]


def report(name: str, p_val, y_val, p_test, y_test) -> tuple[dict, dict]:
    """Fit calibration + thresholds on val, apply to test.  -> (metrics, calibration)."""
    c = cal.fit(p_val, y_val)
    platt = cal.Platt(**c["platt"])
    pc = platt(p_test)
    m = {"method": name,
         "val_auroc": float(roc_auc_score(y_val, p_val)),
         "test_auroc": float(roc_auc_score(y_test, p_test)),
         "test_auroc_ci95": bootstrap_auroc(y_test, p_test),
         "ece_raw": cal.ece(p_test, y_test), "ece_platt": cal.ece(pc, y_test)}
    for k in ("youden", "sens90"):
        t = c[f"threshold_{k}"]
        se, sp = cal.sens_spec(pc, y_test, t)
        m[k] = {"threshold": t, "sensitivity": se, "specificity": sp}
    return m, c


def resolution_features(split: str) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(os.path.join("splits", f"{split}.csv"))
    X = np.stack([d.width, d.height, d.width / d.height, np.log(d.width * d.height)], 1).astype(float)
    return X, d.label.to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="evaluate on test again (breaks the one-look rule)")
    ap.add_argument("--dry-run", action="store_true", help="use val as the test set; writes nothing")
    ap.add_argument("--variant", default=None, help="default: the best variant on val (train_summary.json)")
    a = ap.parse_args()
    if a.variant:
        variant = a.variant
    else:
        with open(os.path.join(RESULTS, "train_summary.json")) as f:
            variant = json.load(f)["best_variant"]
    marker = os.path.join(RESULTS, f".test_evaluated_{variant}")
    if os.path.exists(marker) and not (a.force or a.dry_run):
        raise SystemExit(f"test was already evaluated for {variant} ({marker}); refusing to look again")
    print(f"variant: {variant}" + ("   [DRY RUN: val stands in for test]" if a.dry_run else ""))
    test_split = "val" if a.dry_run else "test"
    Xva, yva = split_xy(variant, "val")
    Xte, yte = split_xy(variant, test_split)
    rows = []

    head, ck = StateHead.load(os.path.join(CKPT_DIR, f"{variant}.pt"))
    opts = ck["options"]
    p_va, p_te = p_fracture(head, Xva, opts), p_fracture(head, Xte, opts)
    m, calib = report(f"CLM head · {variant}", p_va, yva, p_te, yte)
    rows.append(m)

    with open(os.path.join(CKPT_DIR, f"{variant}_linprobe.pkl"), "rb") as f:
        lp = pickle.load(f)
    pr = lambda X: lp["lr"].predict_proba(lp["scaler"].transform(X))[:, 1]
    rows.append(report(f"Linear probe · {variant}", pr(Xva), yva, pr(Xte), yte)[0])

    Rtr, rtr = resolution_features("train"); Rva, _ = resolution_features("val"); Rte, _ = resolution_features(test_split)
    res = LogisticRegression(class_weight="balanced", max_iter=5000).fit((Rtr - Rtr.mean(0)) / Rtr.std(0), rtr)
    rp = lambda R: res.predict_proba((R - Rtr.mean(0)) / Rtr.std(0))[:, 1]
    rows.append(report("Resolution only (leakage check)", rp(Rva), yva, rp(Rte), yte)[0])

    ablations = []
    rnd = os.path.join(CKPT_DIR, f"{variant}_randopt.pt")
    if os.path.exists(rnd):
        h2, ck2 = StateHead.load(rnd)
        ablations.append(report(f"(a) random option vectors · {variant}", p_fracture(h2, Xva, ck2["options"]), yva,
                                p_fracture(h2, Xte, ck2["options"]), yte)[0])
    proj = np.load(os.path.join(EMB_DIR, "options.npz"))["proj"]
    platt = cal.Platt(**calib["platt"])
    para = []
    for i, pair in enumerate(PARAPHRASES):
        o = torch.from_numpy(proj[2 + 2 * i: 4 + 2 * i]).float()
        p = p_fracture(head, Xte, o)
        se, sp = cal.sens_spec(platt(p), yte, calib["threshold_sens90"])
        para.append({"options": pair, "test_auroc": float(roc_auc_score(yte, p)),
                     "sens_at_sens90_threshold": se, "spec_at_sens90_threshold": sp})

    out = {"variant": variant, "n_test": int(len(yte)), "n_test_fractured": int(yte.sum()),
           "n_bootstrap": N_BOOT, "results": rows, "ablations": ablations, "paraphrases": para,
           "calibration": calib}
    metrics_path = os.path.join(RESULTS, "metrics.json" if variant.startswith("radio") else f"metrics_{variant}.json")
    if not a.dry_run:
        with open(metrics_path, "w") as f:
            json.dump(out, f, indent=2)
        with open(os.path.join(CKPT_DIR, "calibration.json"), "w") as f:
            json.dump({"variant": variant, **calib}, f, indent=2)
        open(marker, "w").write("evaluated\n")

    print(f"\n{test_split}: {len(yte)} images, {yte.sum()} fractured\n")
    print("| method | val AUROC | test AUROC [95% CI] | sens / spec @ Youden | sens / spec @ ≥0.90-sens | ECE raw | ECE Platt |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for m in rows + ablations:
        lo, hi = m["test_auroc_ci95"]
        print(f"| {m['method']} | {m['val_auroc']:.3f} | {m['test_auroc']:.3f} [{lo:.3f}, {hi:.3f}] | "
              f"{m['youden']['sensitivity']:.3f} / {m['youden']['specificity']:.3f} | "
              f"{m['sens90']['sensitivity']:.3f} / {m['sens90']['specificity']:.3f} | {m['ece_raw']:.3f} | {m['ece_platt']:.3f} |")
    print("\n| paraphrased options (no retraining) | test AUROC | sens / spec @ ≥0.90-sens threshold |")
    print("|---|---:|---:|")
    for r in para:
        print(f"| {r['options'][0]} / {r['options'][1]} | {r['test_auroc']:.3f} | "
              f"{r['sens_at_sens90_threshold']:.3f} / {r['spec_at_sens90_threshold']:.3f} |")
    if not a.dry_run:
        print(f"\n-> {metrics_path}, {CKPT_DIR}/calibration.json")


if __name__ == "__main__":
    main()
