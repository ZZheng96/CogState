"""E28 主实验：块偏移投影（E28a）+ 原型距离（E28b）+ 校准内 CV 诊断（E28c）+ 与 E27 组合。

E28a 块偏移子空间 B（本人块 0–3 估计），X ← X(I − BBᵀ) 后跑 E27 的 kcv / own / kcv+own：
  cls_{n|o}{r}：各块均值 − 该类总均值（4 个向量，秩 2），SVD 前 r 维；o = 先对类均值差 d̂ 正交化。
  clh_{n|o}{r}：cls + 各块后半−前半均值差（块内漂移，6 个向量）。
  blk_o{r}  ：块均值 − 校准总均值（不分类），必须对 d̂ 正交化（否则直接吃掉类别方向）。
  他人：w_j 在「各自块 0–3 估计的同配置子空间」投影后的全 6 块特征上训练（λ=.3，单位化），
        w0 / U 为其余 56 人均值 / PCA（同 E27），本人分数在本人投影空间计算。
  own λ ∈ LAM_D 作为独立行（λ 由外层 LOSO 选），另有 own_cvp（E27 留一对块 4 折选 λ）与 own_cvh（块内前/后半 2 折）。
E28b 原型：空间 = X − 过去 w 帧因果均值（w=0/300），可选先投影 cls_o1；d_own = μT − μR（本人校准），
  d_g = 其余人（全 6 块）单位化 d_j 的均值，d = (1−β)d̂_own + β d̂_g；Σ = 校准段类内协方差，
  v = ((1−a)Σ + a·tr(Σ)/D·I)⁻¹ d（a=1 即欧氏原型距离差），分数 = vᵀx（= 共享协方差马氏距离差）。
E28c 诊断：own_cvp / own_cvh / kcv 的校准 CV AUC 与测试 AUC 的跨人 Spearman ρ。
决策：分数按本人校准统计 z 化 → e26lib THR2 85 种因果阈值，阈值 / 行由 LOSO（其余 56 人平均 bACC）选。
034 只有块 3（全 task）：E28a 全部退化为 k0（w0 分数），E28b 用 β=1 且 Σ 取块 3 中心化协方差。
运行：.venv\\Scripts\\python.exe test/28_block_invariant/e28main.py raw|cat   （各存 ev/分数 npz）
      .venv\\Scripts\\python.exe test/28_block_invariant/e28main.py comb      （组合 + 嵌套选择 + 汇总 json）
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


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


E27 = _imp("e27", T / "27_common_individual" / "e27.py")
E, L = E27.E, E27.L
bal_w, lbfgs_lr, newton_lr, auc_cols, zc = E27.bal_w, E27.lbfgs_lr, E27.newton_lr, E27.auc_cols, E27.zc
dev, S034 = "cuda", "034"
LAM = E27.LAM_D
CFGS = ["none", "cls_n1", "cls_n2", "cls_o1", "cls_o2", "clh_n1", "clh_n2", "clh_n3",
        "clh_o1", "clh_o2", "clh_o3", "blk_o1", "blk_o2", "blk_o3"]
DIAG = {"none", "cls_o1", "clh_o2", "blk_o2"}
PW, PPJ, PA, PB = (0, 300), ("none", "cls_o1"), (0.1, 0.5, 1.0), (0.0, 0.25, 0.5, 1.0)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    with open(HERE / "e28main.log", "a", encoding="utf-8") as f:
        f.write(s + "\n")


def proj(X, B):
    return X if B is None else X - (X @ B) @ B.T


def basis(X, y, blk, blocks, cfg):
    if cfg == "none":
        return None
    kind, rest = cfg.split("_")
    o, r = rest[0] == "o", int(rest[1:])
    bt, yt = torch.as_tensor(blk, device=dev), torch.as_tensor(y, device=dev)
    inb = torch.zeros(len(y), dtype=torch.bool, device=dev)
    for b in blocks:
        inb |= bt == b
    mR, mT = X[inb & (yt == 0)].mean(0), X[inb & (yt == 1)].mean(0)
    dh = (mT - mR) / (mT - mR).norm()
    mu = {b: X[bt == b].mean(0) for b in blocks}
    V = []
    if kind in ("cls", "clh"):
        V += [mu[b] - (mT if int(y[blk == b][0]) == 1 else mR) for b in blocks]
    if kind == "clh":
        for b in blocks:
            ix = np.where(blk == b)[0]
            h = len(ix) // 2
            V.append(X[ix[h:]].mean(0) - X[ix[:h]].mean(0))
    if kind == "blk":
        mall = X[inb].mean(0)
        V, o = [mu[b] - mall for b in blocks], True
    V = torch.stack(V)
    if o:
        V = V - (V @ dh)[:, None] * dh[None]
    return torch.linalg.svd(V, full_matrices=False).Vh[:r].T.contiguous()


def folds_of(blk):
    bt = torch.as_tensor(blk, device=dev)
    pair = [(((bt == t[0]) | (bt == t[1])), ((bt == h[0]) | (bt == h[1]))) for h, t in E27.FOLDS]
    first = torch.zeros(len(blk), dtype=torch.bool, device=dev)
    for b in range(4):
        ix = np.where(blk == b)[0]
        first[torch.as_tensor(ix[:len(ix) // 2], device=dev)] = True
    cal = bt <= 3
    half = [(first, cal & ~first), (cal & ~first, first)]
    return pair, half


def kcv(Zp, y, blk, cal, w0, U, pair):
    yb = torch.tensor(y, device=dev).bool()
    calnp = cal.cpu().numpy()
    s0n = zc(Zp @ w0, cal)
    P = Zp @ U
    P = (P - P[cal].mean(0)) / (P[cal].std(0) + 1e-6)
    Xd = torch.cat([torch.ones(len(y), 1, device=dev), s0n[:, None], P], 1)
    best = (-1.0, None)
    for k in E27.KS:
        pen = torch.tensor([0, 0] + [1] * k, device=dev).float()
        Xk = Xd[:, :2 + k]
        A = []
        for mt, mh in pair:
            mtn = mt.cpu().numpy()
            B = newton_lr(Xk[mt], y[mtn], bal_w(y[mtn]), E27.LAM_K, pen)
            A.append(auc_cols(Xk[mh] @ B.T, yb[mh]))
        A = torch.stack(A).mean(0)
        li = int(A.argmax())
        if float(A[li]) > best[0]:
            B = newton_lr(Xk[cal], y[calnp], bal_w(y[calnp]), E27.LAM_K[li:li + 1], pen)
            best = (float(A[li]), Xk @ B[0])
    return best[1], best[0]


def own_all(Zp, y, cal, folds, diag):
    yb = torch.tensor(y, device=dev).bool()
    yc = y[cal.cpu().numpy()]
    S = lbfgs_lr(Zp[cal], yc, bal_w(yc), LAM)(Zp)
    cv = {}
    if diag:
        for nm, fl in zip(("p", "h"), folds):
            A = []
            for mt, mh in fl:
                yt_ = y[mt.cpu().numpy()]
                A.append(auc_cols(lbfgs_lr(Zp[mt], yt_, bal_w(yt_), LAM)(Zp[mh]), yb[mh]))
            cv[nm] = torch.stack(A).mean(0)
    return S, cv


def run_set(name):
    t0 = time.time()
    subs, D = L.load()
    F = E27.build_set(name, D, subs)
    reg = [q for q in subs if q != S034]
    Y = {q: D[q][1] for q in subs}
    BK = {q: D[q][2] for q in subs}
    CAL = {q: torch.tensor(BK[q] == 3 if q == S034 else BK[q] <= 3, device=dev) for q in subs}
    # ---- 他人 w_j（每配置）----
    Wall = {}
    for c in CFGS:
        W = []
        for q in reg:
            Xp = proj(F[q], basis(F[q], Y[q], BK[q], [0, 1, 2, 3], c))
            w = lbfgs_lr(Xp, Y[q], bal_w(Y[q]), torch.tensor([E27.LAM_WJ], device=dev)).W[0]
            W.append(w / w.norm())
        Wall[c] = torch.stack(W)
    log(f"[{name}] w_j × {len(CFGS)} cfg done {time.time() - t0:.0f}s")
    # ---- 他人原型方向 d_j ----
    DJ = {}
    for w in PW:
        for pj in PPJ:
            for q in reg:
                Zr = F[q] - (L.run_stats_t(F[q], w)[0] if w else 0)
                Zr = proj(Zr, basis(Zr, Y[q], BK[q], [0, 1, 2, 3], pj) if pj != "none" else None)
                yt = torch.as_tensor(Y[q], device=dev)
                d = Zr[yt == 1].mean(0) - Zr[yt == 0].mean(0)
                DJ[(w, pj, q)] = d / d.norm()
    ROWS, CVLOG = {}, {}
    for i, q in enumerate(subs):
        Z, y, blk, cal = F[q], Y[q], BK[q], CAL[q]
        oth = [k for k, o in enumerate(reg) if o != q]
        R, cvq = {}, {}
        folds = folds_of(blk) if q != S034 else None
        for c in CFGS:
            Wo = Wall[c][oth]
            w0 = Wo.mean(0)
            if q == S034:
                s = Z @ w0
                for nm in [f"own_l{float(l):g}" for l in LAM] + ["kcv"] + [f"kcv+own_l{float(l):g}" for l in LAM] + \
                        (["own_cvp", "own_cvh", "kcv+own_cvp", "kcv+own_cvh"] if c in DIAG else []):
                    R[f"a|{c}|{nm}"] = s
                continue
            U = torch.linalg.svd(Wo - w0, full_matrices=False).Vh[:max(E27.KS)].T
            Zp = proj(Z, basis(Z, y, blk, [0, 1, 2, 3], c))
            sk, cvk = kcv(Zp, y, blk, cal, w0, U, folds[0])
            So, cv = own_all(Zp, y, cal, folds, c in DIAG)
            R[f"a|{c}|kcv"] = sk
            zk = zc(sk, cal)
            for li, l in enumerate(LAM.tolist()):
                R[f"a|{c}|own_l{l:g}"] = So[:, li]
                R[f"a|{c}|kcv+own_l{l:g}"] = zk + zc(So[:, li], cal)
            cvq[c] = dict(kcv=cvk)
            for nm, A in cv.items():
                li = int(A.argmax())
                R[f"a|{c}|own_cv{nm}"] = So[:, li]
                R[f"a|{c}|kcv+own_cv{nm}"] = zk + zc(So[:, li], cal)
                cvq[c][f"own_cv{nm}"] = (float(A[li]), float(LAM[li]))
            del Zp
        # ---- E28b 原型 ----
        calb = [3] if q == S034 else [0, 1, 2, 3]
        for w in PW:
            Zr0 = Z - (L.run_stats_t(Z, w)[0] if w else 0)
            for pj in PPJ:
                B = basis(Zr0, y, blk, calb, pj) if (pj != "none" and q != S034) else None
                Zr = proj(Zr0, B)
                yt = torch.as_tensor(y, device=dev)
                Xc = Zr[cal]
                yc = yt[cal]
                dg = torch.stack([DJ[(w, pj, o)] for o in reg if o != q]).mean(0)
                dg = dg / dg.norm()
                if q == S034:
                    Rc, do = Xc - Xc.mean(0), dg
                else:
                    mR, mT = Xc[yc == 0].mean(0), Xc[yc == 1].mean(0)
                    do = (mT - mR) / (mT - mR).norm()
                    Rc = torch.where((yc == 1)[:, None], Xc - mT, Xc - mR)
                Sg = Rc.T @ Rc / len(Rc)
                mu_ = torch.trace(Sg) / Sg.shape[0]
                Id = torch.eye(Sg.shape[0], device=dev)
                for a in PA:
                    Mi = None if a == 1.0 else (1 - a) * Sg + a * mu_ * Id
                    for b in PB:
                        bb = 1.0 if q == S034 else b
                        dv = (1 - bb) * do + bb * dg
                        v = dv if Mi is None else torch.linalg.solve(Mi, dv)
                        R[f"p|w{w}|{pj}|a{a:g}|b{b:g}"] = Zr @ v
                del Sg, Id, Rc, Zr
        names = list(R)
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        M = torch.stack([zc(R[n].float(), cal) for n in names])
        ROWS.setdefault("names", names)
        assert ROWS["names"] == names
        ROWS.setdefault("EV", []).append(E.metrics(M, tst, yy).cpu().numpy())
        ROWS[q] = M.half().cpu().numpy()
        CVLOG[q] = cvq
        bst = np.nanmax(ROWS["EV"][-1][:, :, 0], 1)
        fam = lambda p: bst[[k for k, n in enumerate(names) if n.startswith(p)]].max()  # noqa: E731
        log(f"[{name}] {q} a|none={fam('a|none|'):.2f} a|cls={fam('a|cls'):.2f} a|clh={fam('a|clh'):.2f} "
            f"a|blk={fam('a|blk'):.2f} p={fam('p|'):.2f} {time.time() - t0:.0f}s")
        del M, R
        torch.cuda.empty_cache()
    names = ROWS["names"]
    EV = np.stack(ROWS["EV"], 1)  # (nr,nq,85,3)
    np.savez_compressed(HERE / f"e28m_ev_{name}.npz", EV=EV, names=np.array(names), subs=np.array(subs))
    np.savez_compressed(HERE / f"e28m_sc_{name}.npz", names=np.array(names), **{f"s{q}": ROWS[q] for q in subs})
    json.dump(CVLOG, open(HERE / f"e28m_cv_{name}.json", "w"), indent=1)
    log(f"[{name}] total {time.time() - t0:.0f}s")


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ok = ~(np.isnan(a) | np.isnan(b))
    ra, rb = a[ok].argsort().argsort(), b[ok].argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


def nested(A, subs):
    """A (nr,nq,85,3) → 嵌套 LOSO 选 (行, 阈值)。"""
    out, pick = np.zeros((A.shape[1], 3)), []
    for i in range(A.shape[1]):
        m = np.nanmean(np.delete(A[..., 0], i, 1), 1)
        c, t = np.unravel_index(np.nanargmax(m), m.shape)
        out[i] = A[c, i, t]
        pick.append((int(c), E.THR2[t][0]))
    return out, pick


def comb():
    subs, D = L.load()
    EV, NM, SC = [], [], {}
    for s in ("cat", "raw"):
        z = np.load(HERE / f"e28m_ev_{s}.npz")
        assert list(z["subs"]) == subs
        EV.append(z["EV"])
        NM += [f"{s}|{n}" for n in z["names"]]
        SC[s] = np.load(HERE / f"e28m_sc_{s}.npz")
    EV = np.concatenate(EV)
    res = dict(table={}, per_subject={}, nested={}, diag={})
    for r, n in enumerate(NM):
        a, sel = E27.loso_thr(EV[r])
        res["table"][n] = dict(loso=E27.summ(a), thr_mode=E.THR2[int(np.bincount(sel).argmax())][0])
        res["per_subject"][n] = {q: [round(float(x), 4) for x in a[i]] for i, q in enumerate(subs)}
    # ---- 组合：E27 复现（cat|a|none|kcv+own_cvp）+ 各族 top 行 z 和 ----
    base = "cat|a|none|kcv+own_cvp"
    tb = res["table"]
    cands = []
    for p in ("cat|a|", "raw|a|", "cat|p|", "raw|p|"):
        rows = sorted([n for n in NM if n.startswith(p) and n != base], key=lambda n: -tb[n]["loso"]["bacc"])
        cands += rows[:4]
    ib = NM.index(base)
    CEV, CNM = [], []
    for i, q in enumerate(subs):
        blk, y = D[q][2], D[q][1]
        cal = torch.tensor(blk == 3 if q == S034 else blk <= 3, device=dev)
        tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
        yy = torch.tensor(y[blk >= 4], device=dev).bool()
        get = lambda n: torch.tensor(SC[n.split("|")[0]][f"s{q}"][list(SC[n.split("|")[0]]["names"]).index(  # noqa: E731
            n.split("|", 1)[1])].astype(np.float32), device=dev)
        sb = get(base)
        rows = [zc(sb + get(c), cal) for c in cands]
        CEV.append(E.metrics(torch.stack(rows), tst, yy).cpu().numpy())
    CEV = np.stack(CEV, 1)
    CNM = [f"comb|{base}+{c}" for c in cands]
    for r, n in enumerate(CNM):
        a, sel = E27.loso_thr(CEV[r])
        res["table"][n] = dict(loso=E27.summ(a), thr_mode=E.THR2[int(np.bincount(sel).argmax())][0])
        res["per_subject"][n] = {q: [round(float(x), 4) for x in a[i]] for i, q in enumerate(subs)}
    ALL = np.concatenate([EV, CEV])
    ANM = NM + CNM
    fams = {
        "E27复现(cat none kcv+own_cvp)": [base],
        "a_cat": [n for n in ANM if n.startswith("cat|a|") and "|none|" not in n],
        "a_raw": [n for n in ANM if n.startswith("raw|a|") and "|none|" not in n],
        "a_cat_orth": [n for n in ANM if n.startswith("cat|a|") and ("_o" in n)],
        "a_cat_noorth": [n for n in ANM if n.startswith("cat|a|") and ("_n" in n)],
        "p_cat": [n for n in ANM if n.startswith("cat|p|")],
        "p_raw": [n for n in ANM if n.startswith("raw|p|")],
        "comb": CNM,
        "all": ANM,
    }
    for fn, rows in fams.items():
        idx = [ANM.index(n) for n in rows]
        out, pick = nested(ALL[idx], subs)
        from collections import Counter
        cnt = Counter(f"{rows[c]}@{t}" for c, t in pick)
        res["nested"][fn] = dict(summ=E27.summ(out), picks_top=dict(cnt.most_common(5)),
                                 per_subject={q: [round(float(x), 4) for x in out[i]] for i, q in enumerate(subs)})
        s = res["nested"][fn]["summ"]
        log(f"nested {fn:30s} bACC {s['bacc']:.3f} AUC {s['auc']:.3f} F1 {s['f1']:.3f} n90 {s['n90']:2d} "
            f"min {s['min']:.3f} {cnt.most_common(1)[0][0]}")
    # ---- E28c 诊断：CV AUC vs 测试 AUC（t0，阈值无关）----
    for s in ("cat", "raw"):
        CV = json.load(open(HERE / f"e28m_cv_{s}.json"))
        for c in sorted(DIAG):
            for k in ("kcv", "own_cvp", "own_cvh"):
                r = NM.index(f"{s}|a|{c}|{k}")
                qs = [q for q in subs if q != S034]
                cva = [CV[q][c][k] if k == "kcv" else CV[q][c][k][0] for q in qs]
                tsa = [EV[r, subs.index(q), 0, 1] for q in qs]
                res["diag"][f"{s}|{c}|{k}"] = dict(rho=spearman(cva, tsa), cv_mean=float(np.mean(cva)),
                                                   test_auc_mean=float(np.nanmean(tsa)))
                log(f"diag {s}|{c}|{k:8s} rho {res['diag'][f'{s}|{c}|{k}']['rho']:+.3f} "
                    f"cvAUC {np.mean(cva):.3f} testAUC {np.nanmean(tsa):.3f}")
    # ---- 042 / 034 逐人 ----
    top = sorted(res["table"].items(), key=lambda kv: -kv[1]["loso"]["bacc"])
    for n, v in top[:30]:
        t = v["loso"]
        log(f"{n:58s} bACC {t['bacc']:.3f} AUC {t['auc']:.3f} F1 {t['f1']:.3f} n90 {t['n90']:2d} "
            f"min {t['min']:.3f} {v['thr_mode']}")
    for q in ("042", "029", "053"):
        log(q, {n: res["per_subject"][n][q][0] for n in [base, top[0][0], "cat|p|w300|none|a0.1|b0",
                                                          "cat|p|w300|cls_o1|a0.1|b0", "cat|a|cls_o1|kcv+own_cvp"]
                if n in res["per_subject"]})
    json.dump(res, open(HERE / "e28main.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "comb":
        comb()
    else:
        run_set(sys.argv[1])
