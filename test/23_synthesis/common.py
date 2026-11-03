"""E23 公共模块：数据/特征加载、E22 第 1–3(+6) 轮候选的统一重建、全配置评估缓存。

候选（全部只用本人块 0–3 + 其他被试数据学习，z 参考 = 校准段）：
  joint   E19 联合 LR 8 个（xblock2_scores.npz）+ ens8
  own     本人校准 LR，一致性前 k 维，k∈{20,50,200,800,4390}，C∈{1e-3,1e-2,1e-1}
  idx     一致性前 m 维按校准极性平均，m∈{1,3,10,30,100}
  f       单特征 f0–f4
  g       特征组 LR（stft/dwt/hilbert/oth/emg，k=50/200）
  idxg    特征组指数（组内前 10）
  sim     校准效应量方向最近 10 名其他被试 + 本人联合 LR（k=800/4390，C=1e-3/1e-2）
  b23     本人块 2–3 LR（r=块0–3一致性排序 / n=块2–3效应量排序，k=100/400，C=.1）
  以及全部两两 z 平均。
阈值（因果、无标签）：t0；过去 w 帧中位数 m120/m300/m600；过去 w 帧 q 分位数 q300/q600 × {.4,.6}。
034（块 0–2 缺失）：joint→跨被试+块3残段联合 LR；g/idx/f→跨被试版本；own/sim/b23 不可用（NaN）。
运行：.venv\\Scripts\\python.exe test/23_synthesis/common.py   → cands_scores.npz / cands_eval.npz / cands_meta.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19_xblock"))
sys.path.insert(0, str(HERE.parent / "22_subject"))
from xblock import L, WS, dev, dval, z, fit_wlr  # noqa: E402
from xblock2 import KS, WOS  # noqa: E402
from round6 import groups  # noqa: E402

NF = 878
THR = [("t0", 0, None), ("m120", 120, .5), ("m300", 300, .5), ("m600", 600, .5),
       ("q300_.4", 300, .4), ("q300_.6", 300, .6), ("q600_.4", 600, .4), ("q600_.6", 600, .6)]
JK = [f"k{k}_w{wo}" for wo in WOS for k in KS]
S034 = "034"


def family(n):
    for p, f in (("ens8", "ens8"), ("k", "joint"), ("own", "own"), ("idxg", "idxg"), ("idx", "idx"),
                 ("f", "f"), ("g", "g"), ("sim", "sim"), ("b23", "b23")):
        if n.startswith(p):
            return f
    return "?"


def base_names():
    n = list(JK) + ["ens8"]
    n += [f"own{k}_C{C}" for k in (20, 50, 200, 800, 4390) for C in (1e-3, 1e-2, 1e-1)]
    n += [f"idx{m}" for m in (1, 3, 10, 30, 100)] + [f"f{m}" for m in range(5)]
    n += [f"g{g}_k{k}" for g in ("dwt", "emg", "hilbert", "oth", "stft") for k in (50, 200)]
    n += [f"idxg{g}10" for g in ("dwt", "emg", "hilbert", "oth", "stft")]
    n += [f"sim{k}_C{C}" for k in (800, 4390) for C in (1e-3, 1e-2)]
    n += [f"b23{t}{k}_C0.1" for t in ("r", "n") for k in (100, 400)]
    return n


def load_all():
    subs, D = L.load()
    F = None
    for w in WS:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
    subs6 = [q for q in subs if len(np.unique(D[q][2])) == 6]
    return subs, subs6, D, F


def cons_order(X, y, blk):
    p1, p2 = blk <= 1, (blk >= 2) & (blk <= 3)
    d1, d2 = dval(X[p1], y[p1]), dval(X[p2], y[p2])
    cons = torch.where(torch.sign(d1) == torch.sign(d2), torch.minimum(d1.abs(), d2.abs()), torch.zeros_like(d1))
    return d1, d2, cons, torch.argsort(cons, descending=True)


def cands_regular(q, subs6, D, F, dall, SC, G):
    i = subs6.index(q)
    y, blk = D[q][1], D[q][2]
    cal = blk <= 3
    cal_t = torch.tensor(cal, device=dev)
    X = F[q]
    d1, d2, cons, order = cons_order(X, y, blk)
    S = {k: torch.tensor(SC[f"{q}|{k}"], device=dev) for k in JK}
    S["ens8"] = z(torch.stack([S[k] for k in JK]).mean(0), cal_t)
    for k in (20, 50, 200, 800, 4390):
        c = order[:k]
        for C in (1e-3, 1e-2, 1e-1):
            S[f"own{k}_C{C}"] = z(L.fit_lr(X[cal][:, c], y[cal], C)(X[:, c]), cal_t)
    sg = torch.sign(d1 + d2)
    Zc = (X - X[cal].mean(0)) / (X[cal].std(0) + 1e-6) * sg
    for m in (1, 3, 10, 30, 100):
        S[f"idx{m}"] = z(Zc[:, order[:m]].mean(1), cal_t)
    for m in range(5):
        S[f"f{m}"] = z(Zc[:, order[m]], cal_t)
    for g, cols in G.items():
        og = cols[torch.argsort(cons[cols], descending=True)]
        for k in (50, 200):
            S[f"g{g}_k{k}"] = z(L.fit_lr(X[cal][:, og[:k]], y[cal], 1e-2)(X[:, og[:k]]), cal_t)
        S[f"idxg{g}10"] = z(Zc[:, og[:10]].mean(1), cal_t)
    del Zc
    dq = dval(X[cal], y[cal])
    cos = torch.nn.functional.cosine_similarity(dall, dq[None], 1)
    cos[i] = -9
    sim = [subs6[j] for j in torch.argsort(cos, descending=True)[:10].tolist()]
    Xo = torch.cat([F[o] for o in sim])
    yo = np.concatenate([D[o][1] for o in sim])
    yj = np.concatenate([yo, y[cal]])
    sw = np.concatenate([np.full(len(yo), .5 / len(yo)), np.full(cal.sum(), .5 / cal.sum())])
    for k in (800, 4390):
        c = order[:k]
        Xj = torch.cat([Xo[:, c], X[cal][:, c]])
        for C in (1e-3, 1e-2):
            S[f"sim{k}_C{C}"] = z(fit_wlr(Xj, yj, sw, C)(X[:, c]), cal_t)
        del Xj
    del Xo
    p2 = (blk >= 2) & (blk <= 3)
    o2 = torch.argsort(d2.abs(), descending=True)
    for t, oo in (("r", order), ("n", o2)):
        for k in (100, 400):
            S[f"b23{t}{k}_C0.1"] = z(L.fit_lr(X[p2][:, oo[:k]], y[p2], .1)(X[:, oo[:k]]), cal_t)
    return S


def cands_034(subs6, D, F, G):
    X, y, blk = F[S034], D[S034][1], D[S034][2]
    cal_t = torch.tensor(blk == 3, device=dev)
    dall = torch.stack([dval(F[o], D[o][1]) for o in subs6])
    cons = dall.mean(0) / (dall.std(0) + 1e-6)
    order = torch.argsort(cons.abs(), descending=True)
    Xo = torch.cat([F[o] for o in subs6])
    yo = np.concatenate([D[o][1] for o in subs6])
    n3 = int(cal_t.sum())
    S = {}
    for k in KS:
        c = order[:k]
        Xj = torch.cat([Xo[:, c], X[cal_t][:, c]])
        yj = np.concatenate([yo, y[cal_t.cpu().numpy()]])
        for wo in WOS:
            sw = np.concatenate([np.full(len(yo), (1 - wo) / len(yo)), np.full(n3, wo / n3)])
            S[f"k{k}_w{wo}"] = z(fit_wlr(Xj, yj, sw, 1e-2)(X[:, c]), cal_t)
        del Xj
    S["ens8"] = z(torch.stack([S[k] for k in JK]).mean(0), cal_t)
    Zc = X * torch.sign(cons)
    for m in (1, 3, 10, 30, 100):
        S[f"idx{m}"] = z(Zc[:, order[:m]].mean(1), cal_t)
    for m in range(5):
        S[f"f{m}"] = z(Zc[:, order[m]], cal_t)
    for g, cols in G.items():
        og = cols[torch.argsort(cons[cols].abs(), descending=True)]
        for k in (50, 200):
            S[f"g{g}_k{k}"] = z(L.fit_lr(Xo[:, og[:k]], yo, 1e-2)(X[:, og[:k]]), cal_t)
        S[f"idxg{g}10"] = z(Zc[:, og[:10]].mean(1), cal_t)
    return S


def thresholds(M, name_w_q):
    """M (nc,T)。返回 (nc,T) 因果阈值。"""
    _, w, qq = name_w_q
    if w == 0:
        return torch.zeros_like(M)
    out = torch.empty_like(M)
    pad = torch.cat([torch.full((M.shape[0], w - 1), float("nan"), device=dev), M], 1)
    for a in range(0, M.shape[0], 48):
        U = pad[a:a + 48].unfold(1, w, 1)  # (b,T,w)
        out[a:a + 48] = U.nanquantile(qq, 2)
    return out


def metrics_batch(M, Th, y, mask):
    """逐行 bACC/AUC/F1（在 mask 帧上）。M、Th (nc,T)。"""
    s = (M - Th)[:, mask]
    yy = torch.tensor(y[mask.cpu().numpy()], device=dev).bool()
    P = s > 0
    tp = (P & yy).sum(1).float()
    fp = (P & ~yy).sum(1).float()
    npos, nneg = yy.sum().float(), (~yy).sum().float()
    bacc = 0.5 * (tp / npos + 1 - fp / nneg)
    f1 = 2 * tp / (2 * tp + fp + (npos - tp))
    if npos == 0 or nneg == 0:
        return bacc, torch.full_like(bacc, float("nan")), f1
    r = s.argsort(1).argsort(1).float() + 1
    auc = (r[:, yy].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)
    return bacc, auc, f1


def pair_names(base):
    return [f"{a}+{b}" for a, b in itertools.combinations(base, 2)]


def expand(B, cal_t):
    """B (nb,T) → (nb+npair, T)：base + 两两 z 平均。"""
    nb = B.shape[0]
    ia, ib = zip(*itertools.combinations(range(nb), 2))
    P = B[list(ia)] + B[list(ib)]
    mu = P[:, cal_t].mean(1, keepdim=True)
    sd = P[:, cal_t].std(1, keepdim=True) + 1e-6
    return torch.cat([B, (P - mu) / sd])


def calref(q, D):
    blk = D[q][2]
    return torch.tensor(blk == 3 if q == S034 else blk <= 3, device=dev)


def evaluate_subject(B, q, D):
    """返回 test bacc/auc/f1 与 cal bacc，形状 (nc, nthr)。"""
    y, blk = D[q][1], D[q][2]
    cal_t = calref(q, D)
    tst = torch.tensor(blk >= 4, device=dev)
    calm = torch.tensor(blk <= 3, device=dev)
    M = expand(B, cal_t)
    out = np.full((4, M.shape[0], len(THR)), np.nan, np.float32)
    for j, t in enumerate(THR):
        Th = thresholds(M, t)
        b, a, f = metrics_batch(M, Th, y, tst)
        out[0, :, j], out[1, :, j], out[2, :, j] = b.cpu().numpy(), a.cpu().numpy(), f.cpu().numpy()
        if q != S034:
            out[3, :, j] = metrics_batch(M, Th, y, calm)[0].cpu().numpy()
    return out


def main():
    subs, subs6, D, F = load_all()
    G = {k: torch.tensor(v, device=dev) for k, v in groups().items()}
    SC = np.load(HERE.parent / "19_xblock" / "xblock2_scores.npz")
    dall = torch.stack([dval(F[q], D[q][1]) for q in subs6])
    names = base_names()
    allq = subs6 + [S034]
    scores, ev = {}, []
    for q in allq:
        S = cands_034(subs6, D, F, G) if q == S034 else cands_regular(q, subs6, D, F, dall, SC, G)
        T = len(D[q][1])
        B = torch.stack([S[n] if n in S else torch.full((T,), float("nan"), device=dev) for n in names])
        scores[q] = B.cpu().numpy()
        ev.append(evaluate_subject(B, q, D))
        e = ev[-1][0]
        print(q, f"oracle-in-set {np.nanmax(e):.3f} ens8|t0 {e[names.index('ens8'), 0]:.3f}", flush=True)
        del S, B
        torch.cuda.empty_cache()
    np.savez_compressed(HERE / "cands_scores.npz", **scores)
    np.savez_compressed(HERE / "cands_eval.npz", ev=np.stack(ev), subs=np.array(allq))
    json.dump(dict(base=names, cands=names + pair_names(names), thr=[t[0] for t in THR], subs=allq),
              open(HERE / "cands_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
