"""E28：块偏移处理。cat 4390 维因果多窗 z 特征（同 E27），本人块 0–3 校准，块 4–5 测试。

数据事实：每块单类（0/2=relax，1/3=task），所以"块内配对差"不存在，
块偏移只能由同类跨块差（m2−m0、m3−m1）及块内前后半差估计。

E28a 块偏移子空间投影 → own LR（λ 网格），投影到 B 的正交补：
  own1  span{(m2−m0)+(m3−m1)}            own2  span{m2−m0, m3−m1}
  own6  own2 + 4 个块内前/后半差           shK   其余 56 人同类跨块差/半块差的 PCA 前 K 维（K=3,10,30）
  own2+sh10
E28b 因果重定心（减过去 w 帧特征均值，w∈{0,120,300,600}）+ 校准 PCA 白化空间原型距离差
  （d∈{10,30,100}，每类 K∈{1,3} 个 k-means 原型）；
  pair：相邻块 task−relax 差均值为方向，块内中心化协方差（收缩 α）做 Mahalanobis 线性分。
E28c 块不变训练：LDA Σ = S_wblk + (1+γ)·S_blk + 收缩；LR 加 γ·Σ(w·u_k)² 惩罚（u_k = own6 方向）；
  rc{w}：在因果重定心特征上训练 own LR。
决策：分数用本人校准统计 z 化 → e26lib THR2 85 种因果阈值。配置×阈值由外层 LOSO（其余 56 人平均 bACC）选。
034 只有块 3（全 task）：所有配置退化为通用 w0（其余人逐人 LR 均值）分数。
运行：.venv\\Scripts\\python.exe test/28_block_shift/e28.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E27 = _load("e27", T / "27_common_individual" / "e27.py")
E = E27.E
L = E27.L
dev = "cuda"
S034 = "034"
LAM = E27.LAM_D  # .03 .1 .3 1 3


def cmean(X, w):
    if w == 0:
        return torch.zeros_like(X)
    return L.run_stats_t(X, w)[0]


def drift_vecs(X, y, blk):
    """同类跨块差 + 块内前/后半差（未单位化）。返回 (dlist2, dlist6)。"""
    m = {b: X[blk == b].mean(0) for b in range(4)}
    d2 = [m[2] - m[0], m[3] - m[1]]
    h = []
    for b in range(4):
        idx = np.where(blk == b)[0]
        n = len(idx) // 2
        h.append(X[idx[n:]].mean(0) - X[idx[:n]].mean(0))
    return d2, d2 + h


def orth(V):
    V = torch.stack(V, 1) if isinstance(V, list) else V
    return torch.linalg.qr(V).Q


def proj(X, Q):
    return X if Q is None else X - (X @ Q) @ Q.T


def lr_pen(X, y, sw, lams, U, gam, iters=120):
    """ridge LR + gam·Σ_k (w·u_k)²（U 列单位化），批量 λ。返回 (L,d) 权重与偏置。"""
    Lm, d = len(lams), X.shape[1]
    W = torch.zeros(Lm, d, device=dev, requires_grad=True)
    b = torch.zeros(Lm, device=dev, requires_grad=True)
    yt = torch.as_tensor(y, device=dev).float()[:, None]
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(X @ W.T + b, yt.expand(-1, Lm), reduction="none")
             * sw[:, None]).sum() + 0.5 * (lams[:, None] * W ** 2).sum() + 0.5 * gam * ((W @ U) ** 2).sum()
        l.backward()
        return l
    opt.step(closure)
    return W.detach(), b.detach()


def kmeans(X, K, it=30):
    if K == 1:
        return X.mean(0, keepdim=True)
    g = torch.Generator(device="cpu").manual_seed(0)
    C = X[torch.randperm(len(X), generator=g)[:K].to(dev)].clone()
    for _ in range(it):
        a = torch.cdist(X, C).argmin(1)
        for k in range(K):
            if (a == k).any():
                C[k] = X[a == k].mean(0)
    return C


def shrink_solve(S, v, a):
    d = S.shape[0]
    M = (1 - a) * S + a * (torch.trace(S) / d) * torch.eye(d, device=dev)
    return torch.linalg.solve(M, v)


def subject_scores(Z, y, blk, Qsh):
    """返回 {cfg: 全帧分数}（非 034）。"""
    cal = blk <= 3
    calt = torch.tensor(cal, device=dev)
    yc = y[cal]
    sw = E27.bal_w(yc)
    Xc = Z[calt]
    bc = blk[cal]
    d2, d6 = drift_vecs(Xc, yc, bc)
    S = {}
    # ---- E28a ----
    Qs = {"none": None, "own1": orth([d2[0] + d2[1]]), "own2": orth(d2), "own6": orth(d6)}
    for K, Q in Qsh.items():
        Qs[f"sh{K}"] = Q
    Qs["own2+sh10"] = orth(torch.cat([Qsh[10], orth(d2)], 1))
    for nm, Q in Qs.items():
        Xp = proj(Z, Q)
        f = E27.lbfgs_lr(Xp[calt], yc, sw, LAM)
        sc = f(Xp)
        for li, lam in enumerate(LAM.tolist()):
            S[f"a|{nm}|l{lam:g}"] = sc[:, li]
        del Xp
    # ---- E28c LR 惩罚 ----
    U = torch.stack([v / v.norm() for v in d6], 1)
    for g in (1.0, 10.0, 100.0, 1000.0):
        W, b = lr_pen(Xc, yc, sw, LAM, U, g)
        sc = Z @ W.T + b
        for li, lam in enumerate(LAM.tolist()):
            S[f"c|lrpen{g:g}|l{lam:g}"] = sc[:, li]
    # ---- E28c LDA：Σ = S_wblk + (1+γ) S_blk ----
    yct = torch.tensor(yc, device=dev).bool()
    Xw = Xc.clone()
    Sblk = torch.zeros(Z.shape[1], Z.shape[1], device=dev)
    for c in (0, 1):
        mc = Xc[yct == c].mean(0)
        for bb in np.unique(bc[yc == c]):
            mk = torch.tensor(bc == bb, device=dev)
            mb = Xc[mk].mean(0)
            Xw[mk] = Xc[mk] - mb
            Sblk += mk.sum() * torch.outer(mb - mc, mb - mc)
    Swb = Xw.T @ Xw / len(Xc)
    Sblk /= len(Xc)
    dm = Xc[yct].mean(0) - Xc[~yct].mean(0)
    for g in (0.0, 1.0, 10.0, 100.0, 1000.0):
        Sg = Swb + (1 + g) * Sblk
        for a in (0.1, 0.5):
            S[f"c|lda_g{g:g}_a{a}"] = Z @ shrink_solve(Sg, dm, a)
    # ---- E28b pair：相邻块 task−relax 差，块内中心化协方差 ----
    mb = {k: Xc[torch.tensor(bc == k, device=dev)].mean(0) for k in range(4)}
    dp = ((mb[1] - mb[0]) + (mb[1] - mb[2]) + (mb[3] - mb[2])) / 3
    for a in (0.1, 0.3, 0.9):
        S[f"b|pair_a{a}"] = Z @ shrink_solve(Swb, dp, a)
    del Sblk, Sg
    # ---- E28b 原型 & E28c rc own LR ----
    for w in (0, 120, 300, 600):
        Zr = Z - cmean(Z, w)
        Xr = Zr[calt]
        mu = Xr.mean(0)
        _, s, Vh = torch.linalg.svd(Xr - mu, full_matrices=False)
        for dd in (10, 30, 100):
            P = (Zr - mu) @ Vh[:dd].T / (s[:dd] / np.sqrt(len(Xr)))
            Pc = P[calt]
            for K in (1, 3):
                C0, C1 = kmeans(Pc[~yct], K), kmeans(Pc[yct], K)
                S[f"b|proto_w{w}_d{dd}_K{K}"] = torch.cdist(P, C0).min(1).values - torch.cdist(P, C1).min(1).values
        if w:
            S[f"b|pair_rc{w}_a0.3"] = Zr @ shrink_solve(Swb, dp, 0.3)
            f = E27.lbfgs_lr(Xr, yc, sw, LAM)
            sc = f(Zr)
            for li, lam in enumerate(LAM.tolist()):
                S[f"c|rc{w}|l{lam:g}"] = sc[:, li]
        del Zr, Xr
    return S


def main():
    subs, D = L.load()
    F = E27.build_set("cat", D, subs)
    reg = [q for q in subs if q != S034]
    Wj, DV = {}, {}
    for q in reg:
        y, blk = D[q][1], D[q][2]
        f = E27.lbfgs_lr(F[q], y, E27.bal_w(y), torch.tensor([E27.LAM_WJ], device=dev))
        Wj[q] = f.W[0] / f.W[0].norm()
        cal = blk <= 3
        _, d6 = drift_vecs(F[q][torch.tensor(cal, device=dev)], y[cal], blk[cal])
        DV[q] = torch.stack([v / v.norm() for v in d6])
    cfgs, EV = None, []
    for q in subs:
        y, blk = D[q][1], D[q][2]
        Z = F[q]
        cal = E.calref(q, D)
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        w0 = torch.stack([Wj[o] for o in reg if o != q]).mean(0)
        if q == S034:
            S = {c: Z @ w0 for c in cfgs}
        else:
            Vh = torch.linalg.svd(torch.cat([DV[o] for o in reg if o != q]), full_matrices=False).Vh
            Qsh = {K: Vh[:K].T.contiguous() for K in (3, 10, 30)}
            S = subject_scores(Z, y, blk, Qsh)
            S["k0"] = Z @ w0
            if cfgs is None:
                cfgs = list(S)
        M = torch.stack([E27.zc(S[c], cal) for c in cfgs])
        M = torch.nan_to_num(M, 0.0)
        EV.append(E.metrics(M, tst, yy).cpu().numpy())
        best = np.nanmax(EV[-1][:, :, 0], 1)
        print(q, f"oracle {best.max():.2f}", " ".join(f"{p}={best[[i for i, c in enumerate(cfgs) if c.startswith(p)]].max():.2f}"
                                                   for p in ("a|none", "a|own", "a|sh", "b|proto", "b|pair", "c|lda", "c|lrpen", "c|rc")),
              flush=True)
        torch.cuda.empty_cache()
    A = np.stack(EV, 1)  # (nc,nq,85,3)
    np.savez_compressed(HERE / "ev_e28.npz", A=A.astype(np.float32), cfgs=np.array(cfgs), subs=np.array(subs))
    report(A, cfgs, subs)


def nested(A, idx, subs):
    """在 idx 配置 × 85 阈值上嵌套 LOSO 选（其余人平均 bACC 最大）。"""
    B = A[idx]
    nq = B.shape[1]
    out, pick = np.zeros((nq, 3)), []
    for i in range(nq):
        m = np.nanmean(np.delete(B[:, :, :, 0], i, 1), 1)
        c, t = np.unravel_index(np.nanargmax(m), m.shape)
        out[i] = B[c, i, t]
        pick.append(f"{idx[c]}@{E.THR2[t][0]}")
    return out, pick


def report(A, cfgs, subs):
    res = dict(table={}, families={}, per_subject={})
    for ci, c in enumerate(cfgs):
        a, sel = E27.loso_thr(A[ci])
        res["table"][c] = dict(loso=E27.summ(a), thr_mode=E.THR2[int(np.bincount(sel).argmax())][0])
    fam = {"E28a_none(own基线)": "a|none", "E28a_own投影": "a|own", "E28a_共享投影": "a|sh",
           "E28b_原型": "b|proto", "E28b_pair": "b|pair", "E28c_LDA块惩罚": "c|lda", "E28c_LR块惩罚": "c|lrpen",
           "E28c_rc_ownLR": "c|rc", "E28a_all": "a|", "E28b_all": "b|", "E28c_all": "c|", "ALL": ""}
    for fn, p in fam.items():
        idx = [i for i, c in enumerate(cfgs) if c.startswith(p) and c != "k0"]
        out, pick = nested(A, idx, subs)
        pk = [cfgs[int(s.split("@")[0])] + "@" + s.split("@")[1] for s in pick]
        res["families"][fn] = dict(summ=E27.summ(out), picks_top=dict(zip(*np.unique(pk, return_counts=True))))
        res["families"][fn]["picks_top"] = {k: int(v) for k, v in res["families"][fn]["picks_top"].items()}
        res["per_subject"][fn] = {q: [float(x) for x in out[i]] for i, q in enumerate(subs)}
        s = res["families"][fn]["summ"]
        print(f"{fn:22s} bACC {s['bacc']:.3f} AUC {s['auc']:.3f} F1 {s['f1']:.3f} n90 {s['n90']:2d} min {s['min']:.3f}",
              max(res["families"][fn]["picks_top"], key=res["families"][fn]["picks_top"].get), flush=True)
    top = sorted(res["table"].items(), key=lambda kv: -kv[1]["loso"]["bacc"])[:15]
    for c, v in top:
        s = v["loso"]
        print(f"  {c:28s} {s['bacc']:.3f} {s['auc']:.3f} {s['f1']:.3f} n90 {s['n90']} {v['thr_mode']}")
    json.dump(res, open(HERE / "e28.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        Z = np.load(HERE / "ev_e28.npz")
        report(Z["A"], list(Z["cfgs"]), list(Z["subs"]))
    else:
        main()
