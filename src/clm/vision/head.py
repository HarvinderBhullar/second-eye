"""The image state head: frozen image embedding -> CLM action space.

    logits_k = scale * cos(state_head(image), option_k)

``option_k`` are the frozen action-tower vectors (Qwen3-8B + released action head, 512-d,
L2-normalised).  The head standardises its input with train-set statistics (stored as buffers,
so a checkpoint is self-contained), then Linear -> GELU -> Linear, L2-normalised.
``scale = exp(logit_scale)``, learnable, initialised to 1/0.07 and clamped at 100 as in CLIP / CLM.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SCALE = 100.0


class StateHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.cfg = {"in_dim": in_dim, "out_dim": out_dim, "hidden": hidden}
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def set_norm(self, x: torch.Tensor) -> None:
        self.mean.copy_(x.mean(0)); self.std.copy_(x.std(0).clamp_min(1e-6))

    @property
    def scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=MAX_SCALE)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net((x - self.mean) / self.std), dim=-1)

    def forward(self, x: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """x [b, in_dim], options [k, out_dim] (L2-normalised) -> logits [b, k]."""
        return self.scale * self.embed(x) @ options.T

    def save(self, path: str, **extra) -> None:
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> tuple["StateHead", dict]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        h = cls(**ck["cfg"])
        h.load_state_dict(ck["state_dict"])
        return h.eval().to(device), ck
