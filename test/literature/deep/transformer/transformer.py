"""Transformer on raw EEG (Siddhad et al., arXiv 2202.05170 style): non-overlapping raw patches (0.2 s = 50 samples) ->
linear patch embedding (d=64) + learnable [CLS] + learnable positional encoding -> 4-layer TransformerEncoder (4 heads,
FFN 128, dropout .1, pre-LN) -> LayerNorm(CLS) -> linear. Input: 4 s crop = 20 patches.
Run: python test/literature/deep/transformer/transformer.py [--smoke] [--seeds 0,1,2]
"""
# --- test layout bootstrap (generated 2026-11-02): locate test/ root, shared modules and cache ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TEST = _HERE
while _TEST.name != "test" and _TEST.parent != _TEST:
    _TEST = _TEST.parent
CACHE_DIR = _TEST / "cache"
if str(_TEST) not in _sys.path:
    _sys.path.append(str(_TEST))
for _d in sorted(_TEST.iterdir()):
    if _d.is_dir() and _d.name not in {"cache", "results", "pretrained_cache", "reports", "experiments", "_orphans", "__pycache__"}:
        if str(_d) not in _sys.path:
            _sys.path.append(str(_d))
# --- end bootstrap ---

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import run, CROP  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

PATCH, D, HEADS, LAYERS, FFN = 50, 64, 4, 4, 128


class EEGTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        n = CROP // PATCH
        self.embed = nn.Linear(PATCH, D)
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.randn(1, n + 1, D) * .02)
        layer = nn.TransformerEncoderLayer(D, HEADS, FFN, .1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D); self.fc = nn.Linear(D, 1)

    def forward(self, x):
        z = self.embed(x[:, 0].unfold(-1, PATCH, PATCH))
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos
        return self.fc(self.norm(self.enc(z)[:, 0]))


if __name__ == '__main__':
    run('transformer', EEGTransformer)
