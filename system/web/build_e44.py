"""把 E42/E44 的 LOSO 逐帧分数转换成查看器 npz（system/web/cache/e44/pred_<sid>.npz）。

帧时间 t / 标签 y 取自 test/11_individual_20261102/sliding1/sub-<sid>.npz（与 E42/E44 的 L.load 同序）。
字段：raw（E42 LOSO 融合分数）、hmm（E42 基线 hmm_d300）、dt1200_hmm（E44 最佳）。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

import model_eval as ME

ROOT = Path(__file__).resolve().parents[2]
SLID = ROOT / "test" / "11_individual_20261102" / "sliding1"
SRC = ROOT / "test" / "42_e17_loso_hmm" / "e42_scores.npz"
OUT = ME.CACHE / "e44"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ema_mean(S, hl):
    a = 1 - 0.5 ** (1 / hl)
    m = S[:, 0].clone()
    out = torch.empty_like(S)
    out[:, 0] = m
    for t in range(1, S.shape[1]):
        m = m + a * (S[:, t] - m)
        out[:, t] = m
    return out


def hmm(S, dwell=300):
    lw = -math.log(dwell)
    ls = math.log1p(-1 / dwell)
    lp = torch.full((S.shape[0], 2), -math.log(2), device=S.device)
    out = torch.empty_like(S)
    sp = torch.nn.functional.softplus
    for t in range(S.shape[1]):
        z = S[:, t]
        em = torch.stack((-sp(z), -sp(-z)), 1)
        if t:
            lp = torch.stack((torch.logaddexp(lp[:, 0] + ls, lp[:, 1] + lw),
                              torch.logaddexp(lp[:, 0] + lw, lp[:, 1] + ls)), 1)
        u = lp + em
        lp = u - torch.logsumexp(u, 1, keepdim=True)
        out[:, t] = lp[:, 1] - lp[:, 0]
    return out


def main():
    z = np.load(SRC)
    subs = [str(s) for s in z["subjects"]]
    T = max(len(z[q]) for q in subs)
    S = torch.zeros(len(subs), T, device=DEV)
    for i, q in enumerate(subs):
        S[i, :len(z[q])] = torch.tensor(z[q], device=DEV)
    H = hmm(S).cpu().numpy()
    DT = hmm(S - ema_mean(S, 1200)).cpu().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    for i, q in enumerate(subs):
        d = np.load(SLID / f"sub-{q}.npz")
        o = np.argsort(d["t"])
        n = len(z[q])
        assert len(o) == n, (q, len(o), n)
        np.savez_compressed(OUT / f"pred_{q}.npz", t=np.round(d["t"][o]).astype(np.float32),
                            y=d["y"][o].astype(np.int8), raw=z[q].astype(np.float32),
                            hmm=H[i, :n].astype(np.float32), dt1200_hmm=DT[i, :n].astype(np.float32))
    for f in ("raw", "hmm", "dt1200_hmm"):
        rows = [ME.metrics(np.load(OUT / f"pred_{q}.npz")["y"], np.load(OUT / f"pred_{q}.npz")[f]) for q in subs]
        print(f, {k: round(float(np.mean([r[k] for r in rows])), 4) for k in ("bacc", "auc", "f1")},
              "n>=.9", sum(r["bacc"] >= 0.9 for r in rows))


if __name__ == "__main__":
    main()
