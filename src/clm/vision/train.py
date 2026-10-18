"""Train the image state head (and the linear-probe baseline) for every cached image variant.

    python -m clm.vision.train                          # all variants in data/vision/emb
    python -m clm.vision.train --variants radio_mean    # just one
    python -m clm.vision.train --random-options         # ablation (a): random option vectors

Objective: class-weighted cross-entropy over the two options, logits = scale * cos(state, option);
the action side is frozen.  AdamW lr 3e-5, wd 0.1, early stopping on val AUROC.  Baseline: a
class-balanced logistic regression on the same standardised embeddings (C=1, no sweep).
The best variant is picked on val AUROC only; test is not read here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .embed_image import EMB_DIR, load_variant
from .head import StateHead

SEED = 0
CKPT_DIR = os.path.join("checkpoints", "vision")
RESULTS = os.path.join("results", "vision")


def seed_all(seed: int = SEED) -> None:
    np.random.seed(seed); torch.manual_seed(seed)


def split_xy(variant: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (embeddings, label) for a split, rows in the split CSV's order."""
    X, paths = load_variant(variant)
    row = {p: i for i, p in enumerate(paths)}
    d = pd.read_csv(os.path.join("splits", f"{split}.csv"))
    return X[[row[p] for p in d.path]], d.label.to_numpy()


def option_vectors(random: bool = False) -> torch.Tensor:
    """[2, 512] L2-normalised: index 0 = "Yes, there is a fracture.", 1 = "No, ..."."""
    o = torch.from_numpy(np.load(os.path.join(EMB_DIR, "options.npz"))["proj"][:2]).float()
    if random:
        g = torch.Generator().manual_seed(SEED)
        o = F.normalize(torch.randn(o.shape, generator=g), dim=-1)
    return o


def p_fracture(head: StateHead, X: np.ndarray, options: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return head(torch.from_numpy(X).float(), options).softmax(-1)[:, 0].numpy()


def train_head(Xtr, ytr, Xva, yva, options, epochs=300, patience=25, batch=64, lr=3e-5) -> tuple[StateHead, dict]:
    seed_all()
    head = StateHead(Xtr.shape[1], options.shape[1])
    head.set_norm(torch.from_numpy(Xtr).float())
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.1)
    target = torch.from_numpy(1 - ytr).long()                 # fracture -> option 0 ("Yes")
    counts = np.bincount(1 - ytr, minlength=2)
    weight = torch.tensor(len(ytr) / (2 * counts), dtype=torch.float32)   # balanced class weights
    X = torch.from_numpy(Xtr).float()
    best, best_state, best_ep, hist = -1.0, None, 0, []
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = F.cross_entropy(head(X[idx], options), target[idx], weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        auc = roc_auc_score(yva, p_fracture(head, Xva, options))
        hist.append({"epoch": ep, "loss": loss.item(), "val_auroc": auc, "scale": head.scale.item()})
        if auc > best:
            best, best_ep, best_state = auc, ep, {k: v.clone() for k, v in head.state_dict().items()}
        elif ep - best_ep >= patience:
            break
    head.load_state_dict(best_state)
    return head.eval(), {"val_auroc": best, "best_epoch": best_ep, "epochs_run": len(hist),
                         "scale": head.scale.item(), "history": hist}


def linear_probe(Xtr, ytr) -> tuple[StandardScaler, LogisticRegression]:
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=SEED).fit(sc.transform(Xtr), ytr)
    return sc, lr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--random-options", action="store_true")
    a = ap.parse_args()
    variants = a.variants or sorted(os.path.basename(f)[:-len(".index.csv")]
                                    for f in glob.glob(os.path.join(EMB_DIR, "*.index.csv")))
    options = option_vectors(a.random_options)
    tag = "_randopt" if a.random_options else ""
    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    summary = {}
    for v in variants:
        Xtr, ytr = split_xy(v, "train")
        Xva, yva = split_xy(v, "val")
        head, info = train_head(Xtr, ytr, Xva, yva, options)
        head.save(os.path.join(CKPT_DIR, f"{v}{tag}.pt"), variant=v, random_options=a.random_options,
                  options=options, val_auroc=info["val_auroc"])
        row = {k: info[k] for k in ("val_auroc", "best_epoch", "epochs_run", "scale")}
        if not a.random_options:
            sc, lr = linear_probe(Xtr, ytr)
            with open(os.path.join(CKPT_DIR, f"{v}_linprobe.pkl"), "wb") as f:
                pickle.dump({"scaler": sc, "lr": lr, "variant": v}, f)
            row["linprobe_val_auroc"] = roc_auc_score(yva, lr.predict_proba(sc.transform(Xva))[:, 1])
        summary[v] = row
        print(f"{v}{tag}: dim {Xtr.shape[1]}  head val AUROC {row['val_auroc']:.4f} (epoch {row['best_epoch']}, "
              f"scale {row['scale']:.1f})" + (f"  linear probe val AUROC {row['linprobe_val_auroc']:.4f}"
                                              if "linprobe_val_auroc" in row else ""))
    out = os.path.join(RESULTS, f"train_summary{tag}.json")
    if not a.random_options:
        best = max(summary, key=lambda k: summary[k]["val_auroc"])
        summary = {"best_variant": best, "selected_on": "val AUROC of the CLM head", "variants": summary}
        print(f"\nbest variant on val: {best}")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
