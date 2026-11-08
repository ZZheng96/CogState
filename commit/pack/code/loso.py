"""E17：合规基线。跨被试训练（只用训练被试标签），测试被试只做无标签因果自适应，单窗 6 s 决策。

档位：
  w=0      纯 6 s：特征只来自当前窗，训练集统计量做 z-score
  w>0      6 s + 因果标准化：用测试被试过去 w 帧（含当前）无标签特征的均值/标准差（细则（2）在线自适应）
  thr=tw   决策阈值 = 过去 tw 帧分数的中位数（无标签，因果）；tw=0 为固定阈值 0
超参 (w, C, tw) 在外层训练被试上做内层 group-CV 选择，被评被试不参与任何选择。

运行：.venv\\Scripts\\python.exe test/17_compliant/loso.py [--outer 5] [--inner 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLID = ROOT / "test" / "11_individual_20261102" / "sliding1"
dev = "cuda"


def run_stats_t(X, w):
    """GPU 因果滑动均值/标准差（含当前帧）。X (n,d) tensor。"""
    n = X.shape[0]
    z = torch.zeros(1, X.shape[1], device=X.device, dtype=torch.float64)
    c1 = torch.cat([z, X.double().cumsum(0)])
    c2 = torch.cat([z, (X.double() ** 2).cumsum(0)])
    idx = torch.arange(1, n + 1, device=X.device)
    lo = torch.clamp(idx - w, min=0)
    cnt = (idx - lo)[:, None].double()
    m = (c1[idx] - c1[lo]) / cnt
    v = (c2[idx] - c2[lo]) / cnt - m ** 2
    return m.float(), v.clamp_min(1e-12).sqrt().float()


def causal_median(s, w):
    """过去 w 帧（含当前）分数的中位数；w=0 返回 0。"""
    if w == 0:
        return torch.zeros_like(s)
    pad = torch.cat([torch.full((w - 1,), float("nan"), device=s.device), s])
    return pad.unfold(0, w, 1).nanmedian(1).values


def load():
    subs = sorted(p.stem[4:] for p in SLID.glob("sub-*.npz"))
    D = {}
    for q in subs:
        S = np.load(SLID / f"sub-{q}.npz")
        o = np.argsort(S["t"])
        D[q] = (torch.tensor(np.nan_to_num(S["F"][o].astype(np.float32)), device=dev),
                S["y"][o].astype(int), S["blk"][o].astype(int))
    return subs, D


def feats(D, subs, w):
    out = {}
    for q in subs:
        X = D[q][0]
        if w:
            m, sd = run_stats_t(X, w)
            X = (X - m) / sd
        out[q] = X.clamp(-10, 10)
    return out


def fit_lr(X, y, C, iters=200):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    wv = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    yt = torch.tensor(y, device=dev, dtype=torch.float32)
    pw = (yt == 0).sum() / (yt == 1).sum()
    opt = torch.optim.LBFGS([wv, b], max_iter=iters, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(y))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ wv + b, yt, pos_weight=pw) \
            + 0.5 * lam * (wv ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    wv, b = wv.detach(), b.detach()
    return lambda X: ((X - mu) / sd) @ wv + b


def subj_metrics(y, s, thr):
    p = (s > thr).cpu().numpy().astype(int)
    sc = (s - thr).cpu().numpy()
    tp = p[y == 1].mean()
    tn = 1 - p[y == 0].mean()
    return dict(bacc=float(0.5 * (tp + tn)), auc=float(roc_auc_score(y, sc)), f1=float(f1_score(y, p)))


def evaluate(F, D, train, test, C, tws):
    X = torch.cat([F[q] for q in train])
    y = np.concatenate([D[q][1] for q in train])
    f = fit_lr(X, y, C)
    res = {tw: {} for tw in tws}
    for q in test:
        s = f(F[q])
        for tw in tws:
            res[tw][q] = subj_metrics(D[q][1], s, causal_median(s, tw))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--ws", default="0,60,120,300,600")
    ap.add_argument("--Cs", default="0.001,0.01,0.1")
    ap.add_argument("--tws", default="0,120,300,600")
    a = ap.parse_args()
    WS = [int(x) for x in a.ws.split(",")]
    CS = [float(x) for x in a.Cs.split(",")]
    TWS = [int(x) for x in a.tws.split(",")]
    subs, D = load()
    rng = np.random.RandomState(0)
    perm = list(rng.permutation(subs))
    outer = [perm[k::a.outer] for k in range(a.outer)]
    FW = {w: feats(D, subs, w) for w in WS}
    final, chosen, grid_all = {}, [], {}
    for k, te in enumerate(outer):
        tr = [q for q in subs if q not in te]
        inner = [tr[j::a.inner] for j in range(a.inner)]
        score = {}
        for w, C in itertools.product(WS, CS):
            acc = {tw: [] for tw in TWS}
            for iv in inner:
                r = evaluate(FW[w], D, [q for q in tr if q not in iv], iv, C, TWS)
                for tw in TWS:
                    acc[tw] += [m["bacc"] for m in r[tw].values()]
            for tw in TWS:
                score[(w, C, tw)] = float(np.mean(acc[tw]))
        best = max(score, key=score.get)
        chosen.append(dict(fold=k, w=best[0], C=best[1], tw=best[2], inner_bacc=score[best]))
        print(f"[fold {k}] best {best} inner {score[best]:.3f}", flush=True)
        # 外层：报告所有档（供分析），主结果用内层选出的配置
        for w, C in itertools.product(WS, CS):
            r = evaluate(FW[w], D, tr, te, C, TWS)
            for tw in TWS:
                grid_all.setdefault(f"w{w}_C{C}_t{tw}", {}).update(r[tw])
        key = f"w{best[0]}_C{best[1]}_t{best[2]}"
        final.update({q: grid_all[key][q] for q in te})
    macro = {m: float(np.mean([v[m] for v in final.values()])) for m in ("bacc", "auc", "f1")}
    n90 = sum(v["bacc"] >= 0.9 for v in final.values())
    print(f"[nested] bACC {macro['bacc']:.3f} AUC {macro['auc']:.3f} F1 {macro['f1']:.3f} n>=.9 {n90}")
    table = {g: {m: float(np.mean([v[m] for v in r.values()])) for m in ("bacc", "auc", "f1")}
             for g, r in grid_all.items()}
    for g, v in sorted(table.items(), key=lambda kv: -kv[1]["bacc"])[:10]:
        print(f"  {g:22s} bACC {v['bacc']:.3f} AUC {v['auc']:.3f} F1 {v['f1']:.3f}")
    (HERE / "loso.json").write_text(json.dumps(dict(chosen=chosen, macro=macro, n90=n90, per_subject=final,
                                                    grid_macro=table, grid_subject=grid_all),
                                               ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
