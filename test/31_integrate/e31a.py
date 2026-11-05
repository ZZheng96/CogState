"""E31a：并集统一配方搜索（oracle 口径同 E26；配方本身因果可计算）。

U1 = ['025','028','010','053','043','054']（E26 G6+G11），U2 = ['011','004','055','052']（G7+G13）。
候选分数 = 72 个 b|pool 配置（重定心窗 w∈{300,600,1200} × 本人权重 α∈{.35,...,.7} ×
  他人余弦加权 τ∈{inf,.2,.5}，wo_j ∝ exp(cos(d_j,d_本人)/τ)，d 为校准块 (R,T) 对效应方向、
  在 w 重定心空间计算；034 无校准效应对照取 cos=0）
  + 全部 pool×15 组件两两 z 平均（z 平均 = 相加后按本人校准段重标准化，同 E26 zpair）。
  可选扩展：并集 min<.895 时，top20 pool × top6 组件取 C(6,2) 做三配置 z 平均。
组件 = cands_scores.npz 的 13 个 + 按 E25/excl.py 定义重建的 sc99999_k200、jw0.95_k1600_C0.1
  （重建正确性用 054 专属配方 sc+jw|t0 复现 excl.json bACC=0.91974 校验）。
pool 训练用 e28 同款加权 LBFGS LR（LAM_POOL=0.01，多配置列批量），以保证锚点复现：
  自检要求 b|pool|w600|a0.5(τ=inf) 在 U1 逐人 LOSO bACC 与 e28.json 偏差 ≤0.01（否则中止），
  另报 b|pool|w300|a0.5 在 U1+U2 的偏差。
评估 = e26lib 85 种因果阈值；oracle 口径：同一阈值下并集成员 min 测试 bACC 最大（阈值共享）。
运行：.venv\\Scripts\\python.exe test/31_integrate/e31a.py
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
T = HERE.parent
for p in ("17_compliant", "18_calib", "19_xblock", "23_synthesis"):
    sys.path.insert(0, str(T / p))
import loso as L  # noqa: E402
from xblock import fit_wlr  # noqa: E402


def _imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E28 = _imp("e28", T / "28_block_invariant" / "e28.py")
E = _imp("e26lib", T / "26_merge2" / "e26lib.py")
E27 = _imp("e27", T / "27_common_individual" / "e27.py")
from common import cons_order  # noqa: E402  (23_synthesis，与 excl.py 同一定义)

dev, S034 = "cuda", "034"
U1 = ["025", "028", "010", "053", "043", "054"]
U2 = ["011", "004", "055", "052"]
WGRID = (300, 600, 1200)
ALPHAS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
TAUS = (float("inf"), 0.2, 0.5)
TNAME = {float("inf"): "inf", 0.2: "0.2", 0.5: "0.5"}
LAM_POOL = 0.01
COMP13 = ["goth_k200", "gemg_k200", "ghilbert_k200", "gstft_k200", "idx10", "idxgdwt10",
          "own200_C0.1", "b23n100_C0.1", "b23n400_C0.1", "k4390_w0.6", "k1600_w0.85",
          "sim800_C0.01", "sim4390_C0.01"]
COMPS = COMP13 + ["sc99999_k200", "jw0.95_k1600_C0.1"]
WSP = [60, 120, 300, 600, 99999]  # calib2.WS：5 个因果尺度 × 878 = 4390
NF = 878
ANCHOR = {600: ("b|pool|w600|a0.5", U1), 300: ("b|pool|w300|a0.5", U1 + U2)}
TOL = 0.01

LOG = open(HERE / "e31a.log", "w", encoding="utf-8")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def zc(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-6)


class PoolCtx:
    """固定 w：全体 57 人重定心特征 Xall、类平衡权重 BW/CW、校准效应方向（重定心空间）。"""

    def __init__(self, w, F, subs, Y, BK):
        self.w, self.subs = w, subs
        Xall = torch.cat([F[q] for q in subs])
        off = np.concatenate([[0], np.cumsum([len(Y[q]) for q in subs])])
        for i, q in enumerate(subs):
            m, _ = L.run_stats_t(F[q], w)
            Xall[off[i]:off[i + 1]] -= m
        self.Xall = Xall
        self.Fc = {q: Xall[off[i]:off[i + 1]] for i, q in enumerate(subs)}
        self.yall = np.concatenate([Y[q] for q in subs])
        self.sid = torch.tensor(np.concatenate([[i] * len(Y[q]) for i, q in enumerate(subs)]), device=dev)
        self.BW = torch.cat([E27.bal_w(Y[q]) for q in subs])
        self.CW = torch.cat([E27.bal_w(Y[q], torch.tensor(BK[q] <= 3, device=dev)) if q != S034
                             else torch.zeros(len(Y[q]), device=dev) for q in subs])
        reg = [q for q in subs if q != S034]
        dj = {}
        for q in reg:
            P = E28.pairs_of(Y[q], BK[q], [0, 1, 2, 3])
            dv = torch.stack([self.Fc[q][torch.as_tensor(BK[q] == t, device=dev)].mean(0)
                              - self.Fc[q][torch.as_tensor(BK[q] == r, device=dev)].mean(0)
                              for r, t in P]).mean(0)
            dj[q] = dv / dv.norm()
        self.reg, self.dj = reg, dj
        self.DR = torch.stack([dj[q] for q in reg])

    def sw_columns(self, q):
        """目标 q 的 24 列样本权重（τ 主序 × α）。"""
        i = self.subs.index(q)
        own = (self.sid == i).float() * self.CW
        oths = []
        for tau in TAUS:
            if tau == float("inf"):
                oths.append((self.sid != i).float() * self.BW / (len(self.subs) - 1))
            else:
                cos = torch.zeros(len(self.subs), device=dev)  # 034 保持 cos=0 → wo=1
                c = self.DR @ self.dj[q]
                for k, o in enumerate(self.reg):
                    if o != q:
                        cos[self.subs.index(o)] = c[k]
                wo = torch.exp(cos / tau)
                wo[i] = 0.0
                oths.append(self.BW * (wo / wo.sum())[self.sid])
        return torch.stack([(1 - a) * o + a * own for o in oths for a in ALPHAS], 1)

    def free(self):
        del self.Xall, self.Fc
        torch.cuda.empty_cache()


def anchor_check(ctx, w, cfg, members, e28ref, Y, BK):
    """全体 57 人各一列（034 用 α=0，同 e28），LOSO 选阈值后与 e28.json 逐人对比。"""
    t0 = time.time()
    cols = []
    for i, q in enumerate(ctx.subs):
        oth = (ctx.sid != i).float() * ctx.BW / (len(ctx.subs) - 1)
        if q == S034:
            cols.append(oth)
        else:
            cols.append(0.5 * oth + 0.5 * (ctx.sid == i).float() * ctx.CW)
    Wp, bp = E28.lbfgs_multi(ctx.Xall, ctx.yall, torch.stack(cols, 1), LAM_POOL)
    EV = []
    for i, q in enumerate(ctx.subs):
        s = zc(ctx.Fc[q] @ Wp[i] + bp[i], torch.tensor(BK[q] == 3 if q == S034 else BK[q] <= 3, device=dev))
        tst = torch.tensor(np.where(BK[q] >= 4)[0], device=dev)
        yy = torch.tensor(Y[q][BK[q] >= 4], device=dev).bool()
        EV.append(E.metrics(s[None], tst, yy)[0].cpu().numpy())
    a, sel = E27.loso_thr(np.stack(EV))
    ref = e28ref["per_subject"][cfg]
    rows, mx = [], 0.0
    for q in members:
        i = ctx.subs.index(q)
        dev_i = float(a[i, 0]) - ref[q][0]
        mx = max(mx, abs(dev_i))
        rows.append(dict(sub=q, bacc=round(float(a[i, 0]), 4), auc=round(float(a[i, 1]), 4),
                         f1=round(float(a[i, 2]), 4), ref_bacc=ref[q][0], dev=round(dev_i, 4),
                         thr=E.THR2[int(sel[i])][0]))
    pr(f"[anchor {cfg}] max|dev|={mx:.4f} [{time.time() - t0:.0f}s]")
    for r in rows:
        pr(f"  {r['sub']} bACC {r['bacc']:.4f} (ref {r['ref_bacc']:.4f}, dev {r['dev']:+.4f}) "
           f"AUC {r['auc']:.4f} F1 {r['f1']:.4f} thr {r['thr']}")
    return dict(cfg=cfg, max_dev=mx, rows=rows)


def build_components(q, subs6, Fr, X6, y6, sid6, SCNPZ, base, D):
    """15 个组件分数（全部已按本人校准段 z）。"""
    y, blk = D[q][1], D[q][2]
    cal = blk <= 3
    cal_t = torch.tensor(cal, device=dev)
    X = Fr[q]
    S = {n: torch.tensor(SCNPZ[q][base.index(n)], device=dev, dtype=torch.float32) for n in COMP13}
    _, _, cons, order = cons_order(X, y, blk)
    j = WSP.index(99999)
    c200 = torch.argsort(cons[j * NF:(j + 1) * NF], descending=True) + j * NF
    c200 = c200[:200]
    S["sc99999_k200"] = zc(L.fit_lr(X[cal][:, c200], y[cal], 1e-2)(X[:, c200]), cal_t)
    i = subs6.index(q)
    oth = torch.tensor(sid6 != i, device=dev)
    c = order[:1600]
    n_o, n_c = int(oth.sum()), int(cal.sum())
    sw = np.concatenate([np.full(n_o, 0.05 / n_o), np.full(n_c, 0.95 / n_c)])
    Xj = torch.cat([X6[oth][:, c], X[cal][:, c]])
    yj = np.concatenate([y6[sid6 != i], y[cal]])
    S["jw0.95_k1600_C0.1"] = zc(fit_wlr(Xj, yj, sw, 0.1)(X[:, c]), cal_t)
    return torch.stack([S[n] for n in COMPS])


def rank_top(EVs, names, members, topn=50):
    """oracle 口径：min over 成员 → max over 阈值；同分时宏 bACC 高者优先。"""
    B = np.stack([e[..., 0] for e in EVs])          # (m, nc, 85)
    mn, mac = B.min(0), B.mean(0)
    A = np.stack([e[..., 1] for e in EVs]).mean(0)
    Fm = np.stack([e[..., 2] for e in EVs]).mean(0)
    nc = mn.shape[0]
    bt = mn.argmax(1)
    minb = mn[np.arange(nc), bt]
    order = sorted(range(nc), key=lambda c: (-minb[c], -mac[c, bt[c]]))
    top = []
    for c in order[:topn]:
        t = int(bt[c])
        per = {q: [round(float(x), 4) for x in EVs[i][c, t]] for i, q in enumerate(members)}
        top.append(dict(rank=len(top) + 1, cfg=names[c], thr=E.THR2[t][0],
                        min_bacc=round(float(minb[c]), 4),
                        macro=dict(bacc=round(float(mac[c, t]), 4), auc=round(float(A[c, t]), 4),
                                   f1=round(float(Fm[c, t]), 4)),
                        per_subject=per))
    return top, mn, order


def eval_union(uname, members, POOL, COMP, Y, BK, res):
    t0 = time.time()
    pool_names = [f"b|pool|w{w}|a{a:g}|t{TNAME[t]}" for w in WGRID for t in TAUS for a in ALPHAS]
    names = pool_names + [f"z[{p};{c}]" for p in pool_names for c in COMPS]
    EV = []
    for q in members:
        P = torch.cat([POOL[q][wi] for wi in range(len(WGRID))])     # (72,T)，行序 w×τ×α
        C = COMP[q]                                                  # (15, T)
        cal = torch.tensor(BK[q] <= 3, device=dev)
        S = (P[:, None] + C[None]).reshape(-1, P.shape[1])
        mu, sd = S[:, cal].mean(1, keepdim=True), S[:, cal].std(1, keepdim=True) + 1e-6
        M = torch.cat([P, (S - mu) / sd])
        tst = torch.tensor(np.where(BK[q] >= 4)[0], device=dev)
        yy = torch.tensor(Y[q][BK[q] >= 4], device=dev).bool()
        EV.append(E.metrics(M, tst, yy).cpu().numpy())
        pr(f"[{uname}] metrics {q} done [{time.time() - t0:.0f}s]")
    top, mn, order = rank_top(EV, names, members)
    best = top[0]
    pr(f"[{uname}] singles+pairs best min {best['min_bacc']:.4f} | {best['cfg']} | {best['thr']} "
       f"[{time.time() - t0:.0f}s]")
    ext = None
    if best["min_bacc"] < 0.895:
        # 可选扩展：top20 pool × top6 组件的 C(6,2) 三配置 z 平均
        pmn = mn[:72].max(1)
        top20 = np.argsort(-pmn)[:20]
        cval = np.zeros(len(COMPS))
        for ci in range(len(COMPS)):
            rows = [72 + p * len(COMPS) + ci for p in range(72)]  # pair 行偏移 72
            cval[ci] = mn[rows].max()
        top6 = np.argsort(-cval)[:6]
        tri = [(int(p), int(a), int(b)) for p in top20 for a, b in itertools.combinations(top6, 2)]
        pr(f"[{uname}] 扩展：top20 pool × top6 组件 {[COMPS[i] for i in top6]} → {len(tri)} 个三配置")
        EV3 = []
        for q in members:
            P = torch.cat([POOL[q][wi] for wi in range(len(WGRID))])
            C = COMP[q]
            cal = torch.tensor(BK[q] <= 3, device=dev)
            S3 = P[[t[0] for t in tri]] + C[[t[1] for t in tri]] + C[[t[2] for t in tri]]
            mu, sd = S3[:, cal].mean(1, keepdim=True), S3[:, cal].std(1, keepdim=True) + 1e-6
            tst = torch.tensor(np.where(BK[q] >= 4)[0], device=dev)
            yy = torch.tensor(Y[q][BK[q] >= 4], device=dev).bool()
            EV3.append(E.metrics((S3 - mu) / sd, tst, yy).cpu().numpy())
        names3 = [f"z3[{pool_names[p]};{COMPS[a]};{COMPS[b]}]" for p, a, b in tri]
        top3, mn3, _ = rank_top(EV3, names3, members)
        ext = dict(top20=[pool_names[i] for i in top20], top6=[COMPS[i] for i in top6], top=top3)
        EV = [np.concatenate([EV[i], EV3[i]]) for i in range(len(members))]
        top, _, _ = rank_top(EV, names + names3, members)
        best = top[0]
        pr(f"[{uname}] +三配置后 best min {best['min_bacc']:.4f} | {best['cfg']} | {best['thr']}")
    # 成员 standalone oracle（瓶颈分析）
    solo = {}
    for i, q in enumerate(members):
        e = EV[i][..., 0]
        c, t = np.unravel_index(np.nanargmax(e), e.shape)
        solo[q] = dict(bacc=round(float(e[c, t]), 4), thr=E.THR2[t][0])
    return dict(members=members, best=best, top50=top, triples=ext, solo_oracle=solo)


def main():
    t00 = time.time()
    pr(f"E31a start | gpu {torch.cuda.get_device_name(0)} "
       f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.0f}GB")
    subs, D = L.load()
    Y = {q: D[q][1] for q in subs}
    BK = {q: D[q][2] for q in subs}
    CAL = {q: torch.tensor(BK[q] == 3 if q == S034 else BK[q] <= 3, device=dev) for q in subs}
    res = dict(params=dict(U1=U1, U2=U2, WGRID=WGRID, ALPHAS=ALPHAS, TAUS=[str(t) for t in TAUS],
                           lam=LAM_POOL, comps=COMPS, tol=TOL),
               selfcheck={})

    # ---------- 特征：Fr = 4390 因果多窗 z（组件用）；F = 本人校准段再 z + clamp（pool 用，同 e28 cat） ----------
    Fr, F = {}, {}
    for k, q in enumerate(subs):
        X = torch.cat([L.feats(D, [q], w)[q] for w in WSP], 1)
        Fr[q] = X
        F[q] = ((X - X[CAL[q]].mean(0)) / (X[CAL[q]].std(0) + 1e-6)).clamp(-10, 10)
        if k % 20 == 19:
            pr(f"feats {k + 1}/57 [{time.time() - t00:.0f}s]")
    pr(f"feats done, gpu mem {torch.cuda.memory_allocated() / 2 ** 30:.2f}GB [{time.time() - t00:.0f}s]")

    # ---------- 组件（13 个 npz + 2 个重建；用 054 专属配方校验重建） ----------
    meta = json.load(open(T / "23_synthesis" / "cands_meta.json"))
    SCNPZ = np.load(T / "23_synthesis" / "cands_scores.npz")
    subs6 = [q for q in subs if len(np.unique(BK[q])) == 6]
    X6 = torch.cat([Fr[q] for q in subs6])
    y6 = np.concatenate([Y[q] for q in subs6])
    sid6 = np.concatenate([[i] * len(Y[q]) for i, q in enumerate(subs6)])
    COMP = {}
    for q in dict.fromkeys(U1 + U2):
        COMP[q] = build_components(q, subs6, Fr, X6, y6, sid6, SCNPZ, meta["base"], D)
        pr(f"components {q} done [{time.time() - t00:.0f}s]")
    q = "054"
    s = COMP[q][COMPS.index("sc99999_k200")] + COMP[q][COMPS.index("jw0.95_k1600_C0.1")]
    s = zc(s, CAL[q])
    m54 = L.subj_metrics(Y[q][BK[q] >= 4], s[BK[q] >= 4], torch.tensor(0.0, device=dev))
    ok54 = abs(m54["bacc"] - 0.9197361871577899) < 1e-3
    pr(f"组件重建校验 054 sc+jw|t0: bACC {m54['bacc']:.6f} (excl.json 0.919736) AUC {m54['auc']:.4f} "
       f"F1 {m54['f1']:.4f} → {'OK' if ok54 else 'FAIL'}")
    res["selfcheck"]["comp_054"] = dict(bacc=m54["bacc"], auc=m54["auc"], f1=m54["f1"], ok=bool(ok54))
    del X6
    torch.cuda.empty_cache()
    if not ok54:
        pr("ABORT: 组件重建与 excl.json 不一致")
        json.dump(res, open(HERE / "e31a.json", "w"), indent=1)
        sys.exit(1)

    # ---------- 逐 w：锚点自检 + 两个并集的 pool 模型（共享 Xall，结果先出 U1 后出 U2） ----------
    e28ref = json.load(open(T / "28_block_invariant" / "e28.json"))
    POOL = {q: [None] * len(WGRID) for q in dict.fromkeys(U1 + U2)}
    for wi, w in enumerate(WGRID):
        ctx = PoolCtx(w, F, subs, Y, BK)
        if w in ANCHOR:
            cfg, mem = ANCHOR[w]
            r = anchor_check(ctx, w, cfg, mem, e28ref, Y, BK)
            res["selfcheck"][cfg] = r
            json.dump(res, open(HERE / "e31a.json", "w"), indent=1)
            if r["max_dev"] > TOL and w == 600:
                pr(f"ABORT: 锚点 {cfg} 偏差 {r['max_dev']:.4f} > {TOL}，检查 m_w / z 参考段 / λ / 权重归一化")
                json.dump(res, open(HERE / "e31a.json", "w"), indent=1)
                sys.exit(1)
        for uname, members in (("U1", U1), ("U2", U2)):
            t0 = time.time()
            SW = torch.cat([ctx.sw_columns(q) for q in members], 1)
            Wp, bp = E28.lbfgs_multi(ctx.Xall, ctx.yall, SW, LAM_POOL)
            for mi, q in enumerate(members):
                S = ctx.Fc[q] @ Wp[mi * 24:(mi + 1) * 24].T + bp[mi * 24:(mi + 1) * 24]
                POOL[q][wi] = torch.stack([zc(S[:, k], CAL[q]) for k in range(24)])
            pr(f"w{w} {uname} pool {len(members)}人×24配置 done [{time.time() - t0:.0f}s]")
        ctx.free()
        del ctx
        torch.cuda.empty_cache()
        json.dump(res, open(HERE / "e31a.json", "w"), indent=1)

    # ---------- 评估：先 U1 后 U2 ----------
    for uname, members in (("U1", U1), ("U2", U2)):
        res[uname] = eval_union(uname, members, POOL, COMP, Y, BK, res)
        json.dump(res, open(HERE / "e31a.json", "w"), indent=1)
        pr(f"[{uname}] top10:")
        for e in res[uname]["top50"][:10]:
            per = " ".join(f"{q}:{v[0]:.3f}" for q, v in e["per_subject"].items())
            pr(f"  #{e['rank']} min {e['min_bacc']:.4f} macro {e['macro']['bacc']:.4f}/"
               f"{e['macro']['auc']:.4f}/{e['macro']['f1']:.4f} thr {e['thr']} | {e['cfg']} | {per}")
    pr(f"total {time.time() - t00:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
