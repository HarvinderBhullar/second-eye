"""Where is the fracture?  A patch heatmap by feature-space occlusion, checked against FracAtlas boxes.

    python -m clm.vision.localize            # pointing game on the val fractured images -> results/vision/localization.json

C-RADIOv2-H sees one 512x512 tile as a 32x32 grid of patch features.  The ``radio_mean`` state head
was trained on the L2-normalised *mean* of those 1024 patches.  For every patch we remove the 3x3
window around it from that mean, re-pool, re-score, and record how much the fracture logit
(scale * (cos(state, "Yes") - cos(state, "No"))) drops.  Removing a few patches keeps the input
close to what the head was trained on, unlike scoring single patches on their own.  Positive
values mean "this region pushed the answer towards fracture".

This is an explanation of the classifier, not a trained localiser.  The check is the pointing game
on val (the test split is not used): does the heatmap peak fall inside an annotated fracture box
(within one patch, 16 px at 512, fixed in advance)?  Compared with chance (the fraction of patch
positions that would hit) and with always pointing at the image centre.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from .head import StateHead
from .train import CKPT_DIR, RESULTS

GRID = 32          # 512 / 16 patches per side
WINDOW = 3         # occlusion window, in patches (fixed in advance)
TOL = 1            # pointing-game tolerance, in patches (16 px at 512)
COCO = os.path.join("data", "FracAtlas", "Annotations", "COCO JSON", "COCO_fracture_masks.json")


class Localizer:
    def __init__(self, path: str = os.path.join(CKPT_DIR, "radio_mean.pt"), device: str = "cpu"):
        self.head, ck = StateHead.load(path, device)
        assert ck.get("variant") in (None, "radio_mean"), "the localiser needs the radio_mean head"
        self.options = ck["options"].to(device)
        self.dev = device

    def _logit(self, pooled: torch.Tensor) -> torch.Tensor:
        """pooled [n, 1280] (un-normalised means) -> fracture logit [n]."""
        z = self.head(F.normalize(pooled, dim=-1), self.options)
        return z[:, 0] - z[:, 1]

    @torch.no_grad()
    def heatmap(self, spatial: torch.Tensor) -> np.ndarray:
        """spatial [1024, 1280] patch features of one tile -> [32, 32] drop in fracture logit."""
        f = spatial.float().to(self.dev)
        n = f.shape[0]
        total = f.sum(0)
        grid = f.T.reshape(1, -1, GRID, GRID)                                    # [1, 1280, 32, 32]
        ones = torch.ones(1, 1, GRID, GRID, device=self.dev)
        k = torch.ones(1, 1, WINDOW, WINDOW, device=self.dev)
        win_sum = F.conv2d(grid.transpose(0, 1), k, padding=WINDOW // 2).transpose(0, 1)   # window sums
        win_cnt = F.conv2d(ones, k, padding=WINDOW // 2)                         # patches per window (edges)
        occluded = (total[:, None, None] - win_sum[0]) / (n - win_cnt[0])         # [1280, 32, 32]
        drop = self._logit(total[None] / n) - self._logit(occluded.reshape(f.shape[1], -1).T)
        return drop.reshape(GRID, GRID).cpu().numpy()


def peak(hm: np.ndarray) -> tuple[float, float]:
    """-> (x, y) of the heatmap maximum, normalised to [0, 1] (patch centre)."""
    r, c = np.unravel_index(int(np.argmax(hm)), hm.shape)
    return (c + 0.5) / GRID, (r + 0.5) / GRID


def load_boxes(path: str = COCO) -> dict[str, list[list[float]]]:
    """{file name: [[x0, y0, x1, y1], ...]} normalised to [0, 1] (the tile is a plain resize, so this maps 1:1)."""
    with open(path) as f:
        d = json.load(f)
    imgs = {i["id"]: i for i in d["images"]}
    out: dict[str, list[list[float]]] = {}
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        out.setdefault(im["file_name"], []).append(
            [x / im["width"], y / im["height"], (x + w) / im["width"], (y + h) / im["height"]])
    return out


def hits(x: float, y: float, boxes: list[list[float]], tol: float = TOL / GRID) -> bool:
    return any(b[0] - tol <= x <= b[2] + tol and b[1] - tol <= y <= b[3] + tol for b in boxes)


def chance(boxes: list[list[float]]) -> float:
    """Fraction of the 32x32 patch centres that would count as a hit."""
    c = (np.arange(GRID) + 0.5) / GRID
    return float(np.mean([[hits(x, y, boxes) for x in c] for y in c]))


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p, d = k / n, 1 + z * z / n
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(p + z * z / (2 * n) - m) / d, (p + z * z / (2 * n) + m) / d]


def main() -> None:
    import pandas as pd
    from tqdm import tqdm
    from .embed_image import RadioEncoder, device, load_tile
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["train", "val"], help="test is never used here")
    ap.add_argument("--examples", type=int, default=8, help="images in the example figure")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = device()
    loc, enc, boxes = Localizer(device=dev), RadioEncoder(dev), load_boxes()
    d = pd.read_csv(os.path.join("splits", f"{a.split}.csv"))
    d = d[(d.label == 1) & d.image_id.isin(boxes)].reset_index(drop=True)
    rows, maps = [], {}
    for r in tqdm(d.itertuples(), total=len(d), desc="heatmaps"):
        _, spatial = enc.raw([load_tile(r.path)])
        hm = loc.heatmap(spatial[0])
        x, y = peak(hm)
        b = boxes[r.image_id]
        rows.append({"image_id": r.image_id, "region": r.region, "hit": hits(x, y, b), "chance": chance(b),
                     "centre_hit": hits(0.5, 0.5, b), "peak_x": x, "peak_y": y, "n_boxes": len(b)})
        maps[r.image_id] = hm
    res = pd.DataFrame(rows)
    k, n = int(res.hit.sum()), len(res)
    out = {"split": a.split, "n_images": n, "window_patches": WINDOW, "tolerance_px_at_512": TOL * 16,
           "pointing_game_hit_rate": k / n, "hit_rate_ci95_wilson": wilson(k, n),
           "chance_hit_rate": float(res.chance.mean()), "centre_baseline_hit_rate": float(res.centre_hit.mean()),
           "by_region": res.groupby("region").agg(n=("hit", "size"), hit_rate=("hit", "mean"),
                                                  chance=("chance", "mean")).round(3).to_dict(orient="index")}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "localization.json"), "w") as f:
        json.dump(out, f, indent=2)
    res.to_csv(os.path.join(RESULTS, f"localization_{a.split}.csv"), index=False)
    lo, hi = out["hit_rate_ci95_wilson"]
    print(f"\npointing game on {a.split} ({n} fractured images with boxes, window {WINDOW}x{WINDOW}, tol {TOL * 16}px):")
    print(f"  heatmap peak in a fracture box: {k}/{n} = {k / n:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
    print(f"  chance (random patch):          {out['chance_hit_rate']:.3f}")
    print(f"  always the image centre:        {out['centre_baseline_hit_rate']:.3f}")
    print(pd.DataFrame(out["by_region"]).T.to_string())
    figure(d.head(a.examples), maps, boxes, os.path.join(RESULTS, "localization_examples.png"))


def figure(d, maps, boxes, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    from .embed_image import TILE
    cols = 4
    rws = math.ceil(len(d) / cols)
    fig, axes = plt.subplots(rws, cols, figsize=(3.2 * cols, 3.4 * rws))
    for ax, r in zip(np.ravel(axes), d.itertuples()):
        ax.imshow(Image.open(r.path).convert("L").resize((TILE, TILE)), cmap="gray")
        hm = np.clip(maps[r.image_id], 0, None)
        ax.imshow(np.kron(hm, np.ones((16, 16))), cmap="inferno", alpha=0.45, vmin=0, vmax=max(hm.max(), 1e-6))
        for b in boxes[r.image_id]:
            ax.add_patch(Rectangle((b[0] * TILE, b[1] * TILE), (b[2] - b[0]) * TILE, (b[3] - b[1]) * TILE,
                                   fill=False, ec="#4cc3d4", lw=1.5))
        x, y = peak(maps[r.image_id])
        ok = hits(x, y, boxes[r.image_id])
        ax.plot(x * TILE, y * TILE, marker="x", ms=10, mew=2.5, color="#7CFC00" if ok else "#ff4d4d")
        ax.set_title(f"{r.image_id} · {'hit' if ok else 'miss'}", fontsize=9)
        ax.set_axis_off()
    for ax in np.ravel(axes)[len(d):]:
        ax.set_axis_off()
    fig.suptitle("Occlusion heatmap (orange), annotated fracture boxes (cyan), peak (x)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
