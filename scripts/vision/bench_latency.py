"""p50 / p90 latency of POST /predict over val images (server-side latency_ms and client wall time).

    python scripts/vision/bench_latency.py --url http://127.0.0.1:8701 --n 30 [--options '["A", "B"]']
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8701")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--options", default=None)
ap.add_argument("--api-key", default=None, help="default: $SECOND_EYE_API_KEY")
a = ap.parse_args()
key = a.api_key or os.environ.get("SECOND_EYE_API_KEY")
headers = {"Authorization": f"Bearer {key}"} if key else {}
paths = pd.read_csv("splits/val.csv").path.sample(a.n + a.warmup, random_state=0).tolist()
server, wall = [], []
for i, p in enumerate(paths):
    t = time.perf_counter()
    with open(p, "rb") as f:
        r = requests.post(f"{a.url}/predict", files={"image": f}, headers=headers, data={"options": a.options} if a.options else None)
    r.raise_for_status()
    if i >= a.warmup:
        wall.append((time.perf_counter() - t) * 1000); server.append(r.json()["latency_ms"])
dev = requests.get(f"{a.url}/health").json()["device"]
out = {"device": dev, "n": a.n, "options": "custom" if a.options else "default",
       "server_p50_ms": float(np.median(server)), "server_p90_ms": float(np.percentile(server, 90)),
       "wall_p50_ms": float(np.median(wall)), "wall_p90_ms": float(np.percentile(wall, 90))}
print(json.dumps(out))
