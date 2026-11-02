"""E16：单被试加权组合搜索。取该人单候选 bACC 前 M 名，在 3 候选单纯形（步长 0.1）× EMA h 上搜索。
运行：.venv\\Scripts\\python.exe test/16_subject_push/wsearch.py 040 [M]
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.argv.append("--sf2")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "15_nested_select"))
import nested_select as ns  # noqa: E402
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps2")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps3")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps4")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps5")
ns.EXTRAS.append(Path(__file__).resolve().parent / "scores_ps6")
if "--longh" in sys.argv:
    ns.HS = ns.HS + (60, 90)

BIASES = tuple(np.round(np.arange(-0.4, 0.41, 0.1), 2)) if "--bias" in sys.argv else (0.0,)
q = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 8
y, blk, S = ns.load(q)
dev = "cuda"
yt = torch.tensor(y, device=dev)


def bacc_t(P):  # P: (n, T) bool
    return 0.5 * ((P[:, yt == 1]).float().mean(1) + (~P[:, yt == 0]).float().mean(1))


cands = sorted(S)
E = {c: {h: torch.tensor(ns.ema(S[c] / (S[c].std() + 1e-9), h), device=dev, dtype=torch.float32)
         for h in ns.HS} for c in cands}
single = {c: max(float(bacc_t((E[c][h] >= 0)[None])[0]) for h in ns.HS) for c in cands}
top = sorted(cands, key=lambda c: -single[c])[:M]
print("top", [(c, round(single[c], 3)) for c in top])
W = np.array([w for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=3) if abs(sum(w) - 1) < 1e-6])
Wt = torch.tensor(W, device=dev, dtype=torch.float32)
best = (0, None)
for a, b, c in itertools.combinations(top, 3):
    for h in ns.HS:
        X = torch.stack([E[a][h], E[b][h], E[c][h]])  # (3,T)
        V = Wt @ X
        for bias in BIASES:
            r = bacc_t(V >= bias)
            i = int(r.argmax())
            if float(r[i]) > best[0]:
                best = (float(r[i]), (a, b, c, h, W[i].round(1).tolist(), bias))
print(f"[{q}] best {best[0]:.3f} {best[1]}")

# 记录（若优于已有结果）
import json  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
a, b, c, h, w, bias = best[1]
v = (w[0] * E[a][h] + w[1] * E[b][h] + w[2] * E[c][h]).cpu().numpy() - bias
rp = Path(__file__).resolve().parent / "results.json"
rec = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
if best[0] > rec.get(q, {}).get("bacc", 0):
    rec[q] = dict(bacc=best[0], auc=float(roc_auc_score(y, v)), f1=float(f1_score(y, v >= 0)),
                  cfg=[a, b, c, int(h), w, float(bias)])
    rp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", q, round(best[0], 3), "auc", round(rec[q]["auc"], 3), "f1", round(rec[q]["f1"], 3))
