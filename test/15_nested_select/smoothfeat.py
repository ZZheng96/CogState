"""E15-2：在因果 EMA 平滑后的特征上训练本人 LR（留一块），缓存为新候选分数。

假设：先对分数做 EMA 等价于固定权重的线性滤波；而在平滑特征上重训，权重会针对"平滑后"的
信噪比优化——慢噪声强的特征会被降权。若成立，sf_h* 候选的逐人 bACC 应高于 own_k + EMA。

输出 scores_sf/sub-XXX.npz：score__sf_h{h}_k{k}
运行：py -3.11 test/15_nested_select/smoothfeat.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "11_individual_20261102"
sys.path.insert(0, str(SRC))
from feat_cstd import running_stats  # noqa: E402

warnings.filterwarnings("ignore")
SLID = SRC / "sliding1"
if "--v2" in sys.argv:
    # 第二轮：围绕 h10_k50 细扫 + 多尺度堆叠（让 LR 自己学因果带通）
    OUT = HERE / "scores_sf2"
    VARIANTS = [(f"sf_h{h}_k{k}_C{C}", (h,), k, C)
                for h in (8, 12, 15) for k in (20, 50, 100) for C in (0.01,)] + \
               [(f"sf_h10_k{k}_C{C}", (10,), k, C) for k in (20, 50, 100) for C in (0.001, 0.1)] + \
               [(f"ms_{'-'.join(map(str, hs))}_k{k}", hs, k, 0.01)
                for hs in ((3, 10, 30), (5, 15), (2, 10, 60), (10, 30, 90)) for k in (50, 150)]
else:
    OUT = HERE / "scores_sf"
    VARIANTS = [(f"sf_h{h}_k{k}", (h,), k, 0.01) for h in (5, 10, 20, 30) for k in (50, 200)]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def ema_mat(X, h):
    a = 0.5 ** (1.0 / h)
    return lfilter([1 - a], [1, -a], X, axis=0, zi=(a * X[:1]))[0].astype(np.float32)


def lr_fit(A, y, B, C=0.01):
    """GPU L2 逻辑回归，目标函数与 sklearn LogisticRegression(C) 一致（按样本均值缩放）。"""
    A = torch.tensor(A, device=DEV, dtype=torch.float32)
    B = torch.tensor(B, device=DEV, dtype=torch.float32)
    t = torch.tensor(y, device=DEV, dtype=torch.float32)
    w = torch.zeros(A.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, tolerance_grad=1e-6,
                            line_search_fn="strong_wolfe")
    reg = 0.5 / (C * len(y))

    def clo():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(A @ w + b, t) + reg * (w @ w)
        loss.backward()
        return loss
    opt.step(clo)
    with torch.no_grad():
        return (A @ w + b).cpu().numpy(), (B @ w + b).cpu().numpy()


def fscore(A, y):
    m1, m0 = A[y == 1].mean(0), A[y == 0].mean(0)
    v = A[y == 1].var(0) + A[y == 0].var(0) + 1e-9
    return (m1 - m0) ** 2 / v


def main():
    OUT.mkdir(exist_ok=True)
    subs = sorted(p.stem.replace("sub-", "") for p in SLID.glob("sub-*.npz"))
    for q in subs:
        if (OUT / f"sub-{q}.npz").exists():
            continue
        S = np.load(SLID / f"sub-{q}.npz")
        X = np.nan_to_num(S["F"].astype(np.float32))
        y, blk, t = S["y"].astype(int), S["blk"].astype(int), S["t"]
        o = np.argsort(t)
        X, y, blk, t = X[o], y[o], blk[o], t[o]
        if len(np.unique(y)) < 2:
            continue
        m, sd = running_stats(X, w=300)
        Xc = np.nan_to_num((X - m) / sd).astype(np.float32)
        out = {}
        cache = {}
        for name, hs, k, C in VARIANTS:
            for h in hs:
                if h not in cache:
                    cache[h] = ema_mat(Xc, h)
            Xs = np.hstack([cache[h] for h in hs])
            s = np.zeros(len(y))
            for b in np.unique(blk):
                te, tr = blk == b, blk != b
                if len(np.unique(y[tr])) < 2:
                    continue
                mu, sg = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
                A, B = (Xs[tr] - mu) / sg, (Xs[te] - mu) / sg
                sel = np.argsort(-fscore(A, y[tr]))[:k]
                dtr, dte = lr_fit(A[:, sel], y[tr], B[:, sel], C=C)
                s[te] = (dte - dtr.mean()) / (dtr.std() + 1e-12)
            out[f"score__{name}"] = s.astype(np.float32)
        np.savez_compressed(OUT / f"sub-{q}.npz", y=y.astype(np.int8),
                            blk=blk.astype(np.int16), t=t, **out)
        print(f"[{q}] saved", flush=True)
    print("[done]")


if __name__ == "__main__":
    main()
