"""E28：块不变个体表征。三类方法 + E27 复现行，全部用 e26lib 85 种因果阈值，阈值/超参按 LOSO（其余 56 人）选。

输入：cat 特征（E27 build_set("cat")，4390 维，本人校准段 z）。
a) 块偏移投影 proj：
   B_own = 本人校准块的「类内块均值差」（R0−R1、T0−T1）+ 各块前后半均值差（块内漂移），共 ≤6 维；
   B0    = 其余人（全部 6 块）同样构造的偏移向量，单位化后 PCA 前 k 维（k=5/10）。
   变体 none / own2(仅类内块差) / own6 / g5 / g10 / own6+g5 / own6+g10：X ← X(I − BBᵀ) 后训练本人 LR；
   打分 own（仅本人）与 own+w0（再加 E27 的通用方向 w0 投影后打分，z 和）。λ∈LAMS 作为配置。
b) 块内中心化 ctr：每类块都是单类，「块内减均值」等价于删掉类别信息，因此训练时以相邻 (R,T) 块对为单元减类平衡均值；
   测试（以及另一变体的训练）用过去 w 帧因果均值中心化（w=120/300/600）。
   pair：训练 X−pair 均值，测试 X−m_w；caus：训练、测试都 X−m_w；
   pool：其余 56 人全部帧（X−m_w）+ 本人校准（X−m_w）联合 LR，本人权重占比 α∈{0,.25,.5,.75}（α=0 为纯跨人）。
c) 原型 proto：在 X−m_w 空间，本人 d_own = 各 (R,T) 块对的 μT−μR 平均；块内残差协方差 Σ（低秩+收缩 γ），
   方向 v = Σ_γ⁻¹·[(1−β)·d̂_own + β·d̂_grp]，d_grp 为其余人 d 的单位化均值（原型收缩到他人）；
   分数 = vᵀx（等价于两类共享协方差马氏距离差）。γ∈{0.1,1,10,∞(欧氏)}，β∈{0,.25,.5,.75,1}，可选先投影掉 B_own。
034 无放松标签：本人相关模型回退到对应的纯他人版本（a→g10 投影+w0，b→pool α=0，c→β=1）。
运行：.venv\\Scripts\\python.exe test/28_block_invariant/e28.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
sys.path.insert(0, str(T / "17_compliant"))
sys.path.insert(0, str(T / "18_calib"))
import loso as L  # noqa: E402


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _imp("e26lib", T / "26_merge2" / "e26lib.py")
E27 = _imp("e27", T / "27_common_individual" / "e27.py")
dev, S034 = "cuda", "034"
bal_w, lbfgs_lr, newton_lr, auc_cols, zc = E27.bal_w, E27.lbfgs_lr, E27.newton_lr, E27.auc_cols, E27.zc
LAMS = torch.tensor([0.03, 0.3, 3.0], device=dev)
CWS = (120, 300, 600)
ALPHAS = (0.0, 0.25, 0.5, 0.75)
GAMS = (0.1, 1.0, 10.0, float("inf"))
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
LAM_POOL = 0.01


def orth(V, tol=1e-3):
    """V (m,d) 行向量 → 正交基 (d,r)。"""
    if V is None or len(V) == 0:
        return None
    U, s, Vh = torch.linalg.svd(V, full_matrices=False)
    return Vh[s > tol * s.max()].T


def proj(X, B):
    return X if B is None else X - (X @ B) @ B.T


def offs_vecs(X, y, blk, blocks):
    """类内块均值差 + 块内前后半差。返回 (2 类内差向量列表, 半差列表)。"""
    mu = {b: X[torch.as_tensor(blk == b, device=dev)].mean(0) for b in blocks}
    cls, half = [], []
    for c in (0, 1):
        bs = [b for b in blocks if (y[blk == b] == c).all()]
        cls += [mu[b] - mu[bs[0]] for b in bs[1:]] if len(bs) > 1 else []
    for b in blocks:
        ix = np.where(blk == b)[0]
        h = len(ix) // 2
        half.append(X[ix[h:]].mean(0) - X[ix[:h]].mean(0))
    return cls, half


def pairs_of(y, blk, blocks):
    """相邻 (R,T) 块对（按块号顺序两两配对，要求两类各一）。"""
    out = []
    for a, b in zip(blocks[0::2], blocks[1::2]):
        ya, yb = int(y[blk == a][0]), int(y[blk == b][0])
        if ya != yb:
            out.append((a, b) if ya == 0 else (b, a))  # (R,T)
    return out


def lbfgs_multi(X, y, SW, lam, iters=150):
    """X (n,d)，SW (n,m) 每列一组样本权重 → W (m,d), b (m,)。"""
    m, d = SW.shape[1], X.shape[1]
    W = torch.zeros(m, d, device=dev, requires_grad=True)
    b = torch.zeros(m, device=dev, requires_grad=True)
    yt = torch.as_tensor(y, device=dev).float()[:, None].expand(-1, m)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        l = (torch.nn.functional.binary_cross_entropy_with_logits(X @ W.T + b, yt, reduction="none") * SW).sum() \
            + 0.5 * lam * (W ** 2).sum()
        l.backward()
        return l
    opt.step(closure)
    return W.detach(), b.detach()


def e27_kcv_own(Z, y, blk, cal, w0, U):
    """E27 cat|kcv+own 的分数（同 e27.run_set 中 kcv 与 own 分支）。"""
    yb, bt = torch.tensor(y, device=dev).bool(), torch.tensor(blk, device=dev)
    calnp = cal.cpu().numpy()
    s0n = zc(Z @ w0, cal)
    P = Z @ U
    P = (P - P[cal].mean(0)) / (P[cal].std(0) + 1e-6)
    Xd = torch.cat([torch.ones(len(y), 1, device=dev), s0n[:, None], P], 1)
    best = (-1, None)
    for k in E27.KS:
        pen = torch.tensor([0, 0] + [1] * k, device=dev).float()
        Xk = Xd[:, :2 + k]
        A = []
        for held, trn in E27.FOLDS:
            mt, mh = (bt == trn[0]) | (bt == trn[1]), (bt == held[0]) | (bt == held[1])
            mtn = mt.cpu().numpy()
            B = newton_lr(Xk[mt], y[mtn], bal_w(y[mtn]), E27.LAM_K, pen)
            A.append(auc_cols(Xk[mh] @ B.T, yb[mh]))
        A = torch.stack(A).mean(0)
        li = int(A.argmax())
        if float(A[li]) > best[0]:
            B = newton_lr(Xk[cal], y[calnp], bal_w(y[calnp]), E27.LAM_K[li:li + 1], pen)
            best = (float(A[li]), Xk @ B[0])
    A = []
    for held, trn in E27.FOLDS:
        mt, mh = (bt == trn[0]) | (bt == trn[1]), (bt == held[0]) | (bt == held[1])
        yt_ = y[mt.cpu().numpy()]
        A.append(auc_cols(lbfgs_lr(Z[mt], yt_, bal_w(yt_), E27.LAM_D)(Z[mh]), yb[mh]))
    li = int(torch.stack(A).mean(0).argmax())
    own = lbfgs_lr(Z[cal], y[calnp], bal_w(y[calnp]), E27.LAM_D[li:li + 1])(Z)[:, 0]
    return zc(best[1], cal) + zc(own, cal)


def main():
    t0 = time.time()
    subs, D = L.load()
    F = E27.build_set("cat", D, subs)
    reg = [q for q in subs if q != S034]
    Y = {q: D[q][1] for q in subs}
    BK = {q: D[q][2] for q in subs}
    CAL = {q: torch.tensor(BK[q] == 3 if q == S034 else BK[q] <= 3, device=dev) for q in subs}
    CB = {q: ([3] if q == S034 else [0, 1, 2, 3]) for q in subs}
    AB = {q: [int(b) for b in np.unique(BK[q]) if b >= 0] for q in subs}
    ROWS = {q: {} for q in subs}

    def put(q, name, s):
        ROWS[q][name] = s.float()

    # ---------- 通用 w_j（E27 同款，全 6 块 λ=0.3）与他人偏移向量 ----------
    Wj, OFF = {}, []
    for q in reg:
        w = lbfgs_lr(F[q], Y[q], bal_w(Y[q]), torch.tensor([E27.LAM_WJ], device=dev)).W[0]
        Wj[q] = w / w.norm()
        cls, half = offs_vecs(F[q], Y[q], BK[q], AB[q])
        for v in cls + half:
            OFF.append((q, v / (v.norm() + 1e-9)))
    Wall = torch.stack([Wj[q] for q in reg])
    OFFM = torch.stack([v for _, v in OFF])
    OFFQ = np.array([q for q, _ in OFF])
    print(f"w_j/offsets done {time.time() - t0:.0f}s", flush=True)

    # ---------- a) 投影 + E27 复现 ----------
    for q in subs:
        Z, y, blk, cal = F[q], Y[q], BK[q], CAL[q]
        Wo = Wall[[k for k, o in enumerate(reg) if o != q]]
        w0 = Wo.mean(0)
        U = torch.linalg.svd(Wo - w0, full_matrices=False).Vh[:max(E27.KS)].T
        G = torch.linalg.svd(OFFM[OFFQ != q], full_matrices=False).Vh  # 已按奇异值排序
        Bs = {"none": None, "g5": G[:5].T, "g10": G[:10].T}
        if q != S034:
            cls, half = offs_vecs(Z, y, blk, CB[q])
            Bs["own2"] = orth(torch.stack(cls))
            Bs["own6"] = orth(torch.stack(cls + half))
            Bs["own6+g5"] = orth(torch.cat([torch.stack(cls + half), G[:5]]))
            Bs["own6+g10"] = orth(torch.cat([torch.stack(cls + half), G[:10]]))
            put(q, "e27|kcv+own", e27_kcv_own(Z, y, blk, cal, w0, U))
        else:
            put(q, "e27|kcv+own", Z @ w0)
        calnp = cal.cpu().numpy()
        for bn in ("none", "own2", "own6", "g5", "g10", "own6+g5", "own6+g10"):
            if q == S034:
                s = zc(proj(Z, Bs["g10"]) @ w0, cal)
                for li in range(len(LAMS)):
                    put(q, f"a|{bn}|l{float(LAMS[li]):g}|own", s)
                    put(q, f"a|{bn}|l{float(LAMS[li]):g}|own+w0", s)
                continue
            B = Bs[bn]
            Zp = proj(Z, B)
            f = lbfgs_lr(Zp[cal], y[calnp], bal_w(y[calnp]), LAMS)
            So = f(Zp)  # (n,L)
            sw0 = zc(Zp @ w0, cal)
            for li in range(len(LAMS)):
                put(q, f"a|{bn}|l{float(LAMS[li]):g}|own", So[:, li])
                put(q, f"a|{bn}|l{float(LAMS[li]):g}|own+w0", zc(So[:, li], cal) + sw0)
        print(f"a {q} {time.time() - t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    # ---------- b) 中心化 / c) 原型：逐 w ----------
    lens = [len(Y[q]) for q in subs]
    off = np.concatenate([[0], np.cumsum(lens)])
    yall = np.concatenate([Y[q] for q in subs])
    sid = torch.tensor(np.concatenate([[i] * n for i, n in enumerate(lens)]), device=dev)
    BW = torch.cat([bal_w(Y[q]) for q in subs])  # 每人类平衡、和为 1
    CW = torch.cat([bal_w(Y[q], CAL[q]) if q != S034 else torch.zeros(len(Y[q]), device=dev) for q in subs])
    for w in CWS:
        Xall = torch.cat([F[q] for q in subs])
        for i, q in enumerate(subs):
            m, _ = L.run_stats_t(F[q], w)
            Xall[off[i]:off[i + 1]] -= m
        Fc = {q: Xall[off[i]:off[i + 1]] for i, q in enumerate(subs)}
        # 他人原型方向 d_j（全 6 块的 (R,T) 对）
        dj = {}
        for q in reg:
            P = pairs_of(Y[q], BK[q], AB[q])
            if not P:
                continue
            d = torch.stack([Fc[q][torch.as_tensor(BK[q] == t, device=dev)].mean(0)
                             - Fc[q][torch.as_tensor(BK[q] == r, device=dev)].mean(0) for r, t in P]).mean(0)
            dj[q] = d / d.norm()
        for i, q in enumerate(subs):
            Z, Zc, y, blk, cal = F[q], Fc[q], Y[q], BK[q], CAL[q]
            calnp = cal.cpu().numpy()
            # ---- b) pool：其余人 + 本人校准
            SW = []
            oth = (sid != i).float() * BW / (len(subs) - 1)
            for a in ALPHAS:
                if q == S034 and a > 0:
                    continue
                SW.append((1 - a) * oth + a * (sid == i).float() * CW)
            Wp, bp = lbfgs_multi(Xall, yall, torch.stack(SW, 1), LAM_POOL)
            Sp = Zc @ Wp.T + bp
            for k, a in enumerate(ALPHAS):
                put(q, f"b|pool|w{w}|a{a:g}", Sp[:, 0 if q == S034 else k])
            if q == S034:
                for li in range(len(LAMS)):
                    for tr in ("pair", "caus"):
                        put(q, f"b|{tr}|w{w}|l{float(LAMS[li]):g}", Sp[:, 0])
                        put(q, f"b|{tr}+pool|w{w}|l{float(LAMS[li]):g}", Sp[:, 0])
            else:
                # ---- b) pair / caus 本人训练
                Xpr = Z.clone()
                for r, t in pairs_of(y, blk, CB[q]):
                    mk = torch.as_tensor((blk == r) | (blk == t), device=dev)
                    mu = .5 * (Z[torch.as_tensor(blk == r, device=dev)].mean(0)
                               + Z[torch.as_tensor(blk == t, device=dev)].mean(0))
                    Xpr[mk] -= mu
                for tr, Xt in (("pair", Xpr), ("caus", Zc)):
                    f = lbfgs_lr(Xt[cal], y[calnp], bal_w(y[calnp]), LAMS)
                    So = f(Zc)
                    for li in range(len(LAMS)):
                        put(q, f"b|{tr}|w{w}|l{float(LAMS[li]):g}", So[:, li])
                        put(q, f"b|{tr}+pool|w{w}|l{float(LAMS[li]):g}",
                            zc(So[:, li], cal) + zc(Sp[:, 0], cal))
                del Xpr
            # ---- c) 原型
            dg = torch.stack([dj[o] for o in dj if o != q]).mean(0)
            dg = dg / dg.norm()
            if q == S034:
                R = Zc[cal] - Zc[cal].mean(0)
                do = dg
                Bown = None
            else:
                P = pairs_of(y, blk, CB[q])
                mus = {b: Zc[torch.as_tensor(blk == b, device=dev)].mean(0) for b in CB[q]}
                do = torch.stack([mus[t] - mus[r] for r, t in P]).mean(0)
                do = do / do.norm()
                R = torch.cat([Zc[torch.as_tensor(blk == b, device=dev)] - mus[b] for b in CB[q]])
                cls, half = offs_vecs(Z, y, blk, CB[q])
                Bown = orth(torch.stack(cls + half))
            ev, V = torch.linalg.eigh(R @ R.T / len(R))  # 低秩：Σ = Uₛ diag(s) Uₛᵀ
            keep = ev > 1e-6 * ev.max()
            s = ev[keep]
            Us = (R.T @ V[:, keep]) / (s * len(R)).sqrt()  # (d,r) 单位正交
            tr_ = float(s.sum()) / Z.shape[1]
            for beta in BETAS:
                if q == S034 and beta < 1:
                    continue
                dv = (1 - beta) * do + beta * dg
                for g in GAMS:
                    if g == float("inf"):
                        v = dv
                    else:
                        lam = g * tr_
                        c = Us.T @ dv
                        v = (dv - Us @ c) / lam + Us @ (c / (s + lam))
                    for pj in ("raw", "pB"):
                        vv = v if (pj == "raw" or Bown is None) else proj(v[None], Bown)[0]
                        sc = Zc @ vv
                        bl = [beta] if q != S034 else BETAS
                        for bb in bl:
                            put(q, f"c|w{w}|g{g:g}|b{bb:g}|{pj}", sc)
            del R, V, Us
            torch.cuda.empty_cache()
        print(f"w{w} done {time.time() - t0:.0f}s", flush=True)
        del Xall, Fc
        torch.cuda.empty_cache()

    # ---------- 评估 ----------
    names = list(ROWS[subs[0]])
    EV = []
    for q in subs:
        blk, cal = BK[q], CAL[q]
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(Y[q][blk >= 4], device=dev).bool()
        M = torch.stack([zc(ROWS[q][n], cal) for n in names])
        EV.append(E.metrics(M, tst, yy).cpu().numpy())
        ROWS[q] = M.half().cpu().numpy()
    EV = np.stack(EV, 1)  # (nr,nq,85,3)
    np.savez_compressed(HERE / "e28_ev.npz", EV=EV, names=np.array(names), subs=np.array(subs))
    np.savez_compressed(HERE / "e28_scores.npz", names=np.array(names), **{f"s{q}": ROWS[q] for q in subs})
    res = dict(table={}, per_subject={}, nested={})
    for r, n in enumerate(names):
        a, sel = E27.loso_thr(EV[r])
        res["table"][n] = dict(loso=E27.summ(a), thr_mode=E.THR2[int(np.bincount(sel).argmax())][0])
        res["per_subject"][n] = {q: [round(float(x), 4) for x in a[i]] for i, q in enumerate(subs)}
    fam = {"a": [n for n in names if n.startswith("a|")], "b": [n for n in names if n.startswith("b|")],
           "c": [n for n in names if n.startswith("c|")], "abc": [n for n in names if not n.startswith("e27")],
           "all": names}
    for fn, rows in fam.items():
        idx = [names.index(n) for n in rows]
        A = EV[idx][..., 0]
        out, pick = np.zeros((len(subs), 3)), {}
        for i, q in enumerate(subs):
            mm = np.nanmean(np.delete(A, i, 1), 1)
            c, t = np.unravel_index(np.nanargmax(mm), mm.shape)
            out[i] = EV[idx[c], i, t]
            pick[q] = f"{rows[c]}@{E.THR2[t][0]}"
        res["nested"][fn] = dict(summ=E27.summ(out), picks=pick,
                                 per_subject={q: [round(float(x), 4) for x in out[i]] for i, q in enumerate(subs)})
        print("nested", fn, E27.summ(out), sorted(set(pick.values()))[:5], flush=True)
    json.dump(res, open(HERE / "e28.json", "w"), indent=1)
    top = sorted(res["table"].items(), key=lambda kv: -kv[1]["loso"]["bacc"])
    for n, v in top[:25]:
        t = v["loso"]
        print(f"{n:32s} bACC {t['bacc']:.3f} AUC {t['auc']:.3f} F1 {t['f1']:.3f} n90 {t['n90']:2d} "
              f"min {t['min']:.3f} {v['thr_mode']}", flush=True)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
