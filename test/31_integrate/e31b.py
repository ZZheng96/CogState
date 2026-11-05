"""E31b：全组对合并搜索（oracle）+ 通用可部署评估（LOSO）。

候选（每人一次算好并缓存）= 72 个 b|pool 配置 + 72×15 个 pool⊕组件 z 平均 = 1152 配置 × 85 阈值，
  定义与 E31a 完全相同（PoolCtx / build_components / z 平均 / e26lib.metrics 代码拷贝自 e31a.py）；
  pool 训练集 = 其余 56 人全部帧 + 本人校准块（blk<=3），不使用目标被试测试块标签；
  034 无校准对照，本人权重 0、他人等权（同 e28 α=0）。
  另加 18 列 = round2 原组配方（固定阈值，逐人 bACC/AUC/F1，复现 round2 rec_metrics）。
第 1 部分：起点 17 组（round2 18 组，G7+G13 合并为 U2，配方 = E31a U2 最优）。
  每轮在 (57, 1152*85+18) bACC 张量上对所有组对取并集 min → max over 列，取可行（≥.9）且 min 最大者合并，
  直到无可行组对。
第 2 部分：每个配置 LOSO 选阈值（其余 56 人平均测试 bACC 最大）→ top10；
  外层嵌套 LOSO：(a) 联合选 (配置×阈值)；(b) 配置内先 LOSO 选阈值、再按其余 56 人的 LOSO bACC 选配置。
运行：.venv\\Scripts\\python.exe test/31_integrate/e31b.py  （--recompute 强制重算缓存）
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
from common import cons_order  # noqa: E402
sys.path.insert(0, str(T / "25_merge"))
from clib import load as load_r  # noqa: E402
from merge import build_pool  # noqa: E402

dev, S034 = "cuda", "034"
U1 = ["025", "028", "010", "053", "043", "054"]
U2 = ["011", "004", "055", "052"]
U2_CFG = "z[b|pool|w300|a0.7|tinf;b23n400_C0.1]"
WGRID = (300, 600, 1200)
ALPHAS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
TAUS = (float("inf"), 0.2, 0.5)
TNAME = {float("inf"): "inf", 0.2: "0.2", 0.5: "0.5"}
LAM_POOL = 0.01
COMP13 = ["goth_k200", "gemg_k200", "ghilbert_k200", "gstft_k200", "idx10", "idxgdwt10",
          "own200_C0.1", "b23n100_C0.1", "b23n400_C0.1", "k4390_w0.6", "k1600_w0.85",
          "sim800_C0.01", "sim4390_C0.01"]
COMPS = COMP13 + ["sc99999_k200", "jw0.95_k1600_C0.1"]
WSP = [60, 120, 300, 600, 99999]
NF = 878
TH = 0.9
NT = len(E.THR2)
CHUNK = 8
CACHE = HERE / "e31b_ev.npz"
POOL_NAMES = [f"b|pool|w{w}|a{a:g}|t{TNAME[t]}" for w in WGRID for t in TAUS for a in ALPHAS]
NAMES = POOL_NAMES + [f"z[{p};{c}]" for p in POOL_NAMES for c in COMPS]
NC = len(NAMES)

LOG = open(HERE / "e31b.log", "w", encoding="utf-8")


def pr(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def zc(s, ref):
    return (s - s[ref].mean()) / (s[ref].std() + 1e-6)


# ---------------- 拷贝自 e31a.py（034 分支为新增：本人无校准对照 → α 列全部取他人等权） ----------------
class PoolCtx:
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
        i = self.subs.index(q)
        if q == S034:
            oth = (self.sid != i).float() * self.BW / (len(self.subs) - 1)
            return oth[:, None].expand(-1, len(TAUS) * len(ALPHAS)).contiguous()
        own = (self.sid == i).float() * self.CW
        oths = []
        for tau in TAUS:
            if tau == float("inf"):
                oths.append((self.sid != i).float() * self.BW / (len(self.subs) - 1))
            else:
                cos = torch.zeros(len(self.subs), device=dev)
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


def build_components(q, subs6, Fr, X6, y6, sid6, SCNPZ, base, D):
    y, blk = D[q][1], D[q][2]
    X = Fr[q]
    S = {n: torch.tensor(SCNPZ[q][base.index(n)], device=dev, dtype=torch.float32) for n in COMP13}
    if q == S034:  # 无块 0–2：两个重建组件不可算 → NaN（下游 metrics 输出 NaN）
        nan = torch.full((len(y),), float("nan"), device=dev)
        S["sc99999_k200"] = S["jw0.95_k1600_C0.1"] = nan
        return torch.stack([S[n] for n in COMPS])
    cal = blk <= 3
    cal_t = torch.tensor(cal, device=dev)
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


# ---------------- 原组配方（复现 round2 rec_metrics） ----------------
def orig_recipes(groups):
    R = load_r()
    subs = R["subs"]
    M0, _, _, _ = build_pool(R)
    P0 = M0.shape[2]
    ev2 = np.load(T / "26_merge2" / "ev2.npz")["ev2"]          # (57,1653,85,3)
    out = np.full((len(subs), len(groups), 3), np.nan, np.float32)
    for gi, g in enumerate(groups):
        rec = g["rec"]
        for j, q in enumerate(subs):
            if rec[0] == "s":
                k = rec[1]
                if k < P0:
                    out[j, gi] = M0[j, :, k]
                else:
                    c, t = divmod(k - P0, NT)
                    out[j, gi] = ev2[j, c, t]
            else:
                _, a, b, t = rec
                C, cal_t, tst, yy = E.subj_data(R, q)
                M = E.zpair(C, cal_t, [a], [b])
                out[j, gi] = E.metrics(M, tst, yy)[0, t].cpu().numpy()
                del C
        torch.cuda.empty_cache()
    del M0, ev2, R
    return subs, out


# ---------------- 缓存张量：(57, 1152, 85, 3) ----------------
def compute_ev(subs, D, Y, BK, CAL):
    t00 = time.time()
    Fr, F = {}, {}
    for q in subs:
        X = torch.cat([L.feats(D, [q], w)[q] for w in WSP], 1)
        Fr[q] = X
        F[q] = ((X - X[CAL[q]].mean(0)) / (X[CAL[q]].std(0) + 1e-6)).clamp(-10, 10)
    pr(f"feats done, gpu {torch.cuda.memory_allocated() / 2 ** 30:.2f}GB [{time.time() - t00:.0f}s]")
    meta = json.load(open(T / "23_synthesis" / "cands_meta.json"))
    SCNPZ = np.load(T / "23_synthesis" / "cands_scores.npz")
    subs6 = [q for q in subs if len(np.unique(BK[q])) == 6]
    X6 = torch.cat([Fr[q] for q in subs6])
    y6 = np.concatenate([Y[q] for q in subs6])
    sid6 = np.concatenate([[i] * len(Y[q]) for i, q in enumerate(subs6)])
    COMP = {}
    for k, q in enumerate(subs):
        COMP[q] = build_components(q, subs6, Fr, X6, y6, sid6, SCNPZ, meta["base"], D)
        if k % 10 == 9:
            pr(f"components {k + 1}/57 [{time.time() - t00:.0f}s]")
    q = "054"
    s = zc(COMP[q][COMPS.index("sc99999_k200")] + COMP[q][COMPS.index("jw0.95_k1600_C0.1")], CAL[q])
    m54 = L.subj_metrics(Y[q][BK[q] >= 4], s[BK[q] >= 4], torch.tensor(0.0, device=dev))
    pr(f"组件重建校验 054 sc+jw|t0 bACC {m54['bacc']:.6f} (excl.json 0.919736)")
    assert abs(m54["bacc"] - 0.9197361871577899) < 1e-3
    del X6, Fr
    torch.cuda.empty_cache()

    # chunk 顺序：U1、U2（与 E31a 相同 → 结果可复现），其余每 CHUNK 人一批
    rest = [q for q in subs if q not in U1 + U2]
    chunks = [U1, U2] + [rest[i:i + CHUNK] for i in range(0, len(rest), CHUNK)]
    POOL = {q: [None] * len(WGRID) for q in subs}
    for wi, w in enumerate(WGRID):
        ctx = PoolCtx(w, F, subs, Y, BK)
        for ch in chunks:
            t0 = time.time()
            SW = torch.cat([ctx.sw_columns(q) for q in ch], 1)
            Wp, bp = E28.lbfgs_multi(ctx.Xall, ctx.yall, SW, LAM_POOL)
            for mi, q in enumerate(ch):
                S = ctx.Fc[q] @ Wp[mi * 24:(mi + 1) * 24].T + bp[mi * 24:(mi + 1) * 24]
                POOL[q][wi] = torch.stack([zc(S[:, k], CAL[q]) for k in range(24)])
            del SW, Wp, bp
            pr(f"w{w} pool {len(ch)}人 done [{time.time() - t0:.0f}s] tot [{time.time() - t00:.0f}s]")
        ctx.free()
        del ctx
        torch.cuda.empty_cache()
    del F
    torch.cuda.empty_cache()

    EV = np.zeros((len(subs), NC, NT, 3), np.float32)
    for j, q in enumerate(subs):
        P = torch.cat([POOL[q][wi] for wi in range(len(WGRID))])
        C = COMP[q]
        cal = CAL[q]
        Sm = (P[:, None] + C[None]).reshape(-1, P.shape[1])
        mu, sd = Sm[:, cal].mean(1, keepdim=True), Sm[:, cal].std(1, keepdim=True) + 1e-6
        M = torch.cat([P, (Sm - mu) / sd])
        tst = torch.tensor(np.where(BK[q] >= 4)[0], device=dev)
        yy = torch.tensor(Y[q][BK[q] >= 4], device=dev).bool()
        EV[j] = E.metrics(M, tst, yy).cpu().numpy()
        if j % 10 == 9:
            pr(f"metrics {j + 1}/57 [{time.time() - t00:.0f}s]")
    return EV


def colname(k, gnames):
    if k >= NC * NT:
        return f"原组配方[{gnames[k - NC * NT]}]", "fixed"
    c, t = divmod(k, NT)
    return NAMES[c], E.THR2[t][0]


def main():
    t00 = time.time()
    pr(f"E31b start | gpu {torch.cuda.get_device_name(0)}")
    r2 = json.load(open(T / "26_merge2" / "round2.json", encoding="utf-8"))["groups"]
    subs, D = L.load()
    Y = {q: D[q][1] for q in subs}
    BK = {q: D[q][2] for q in subs}
    CAL = {q: torch.tensor(BK[q] == 3 if q == S034 else BK[q] <= 3, device=dev) for q in subs}

    if CACHE.exists() and "--recompute" not in sys.argv:
        z = np.load(CACHE)
        EV, ORIG = z["EV"], z["ORIG"]
        pr(f"载入缓存 {CACHE.name} [{time.time() - t00:.0f}s]")
    else:
        s2, ORIG = orig_recipes(r2)
        s2 = [str(q) for q in s2]
        ORIG = ORIG[[s2.index(q) for q in subs]]          # clib 的被试顺序（034 在末尾）→ loso 顺序
        pr(f"原组配方 18 列 done [{time.time() - t00:.0f}s]")
        EV = compute_ev(subs, D, Y, BK, CAL)
        np.savez(CACHE, EV=EV, ORIG=ORIG)
        pr(f"缓存已写 {CACHE.name} [{time.time() - t00:.0f}s]")

    # 原组配方自检：每组成员 min 应 = round2 min_bacc
    for gi, g in enumerate(r2):
        idx = [subs.index(q) for q in g["members"]]
        mn = float(np.nanmin(ORIG[idx, gi, 0]))
        assert abs(mn - g["min_bacc"]) < 1e-4, (gi, mn, g["min_bacc"])
    pr("原组配方复现 round2 min_bacc：OK")

    # ---------- 全列张量 (57, K, 3) ----------
    Ball = np.concatenate([EV.reshape(len(subs), NC * NT, 3), ORIG], 1)
    Bt = torch.tensor(np.nan_to_num(Ball[..., 0], nan=-1.0), device=dev)       # 可行性用：NaN → -1
    gnames = [f"G{i + 1}:{g['recipe']}" for i, g in enumerate(r2)]

    # U2 自检
    iU2 = [subs.index(q) for q in U2]
    cU2 = NAMES.index(U2_CFG) * NT + [t[0] for t in E.THR2].index("m300")
    pr(f"U2 自检 {U2_CFG}@m300 min {Bt[iU2, cU2].min():.4f} (E31a 0.9268)")
    iU1 = [subs.index(q) for q in U1]
    pr(f"U1 自检 best min {Bt[iU1, :NC * NT].min(0).values.max():.4f} (E31a 0.8909)")

    # ---------- 第 1 部分：起点 17 组 ----------
    groups, recs = [], []
    for gi, g in enumerate(r2):
        if gi == 13:      # G13（055,052）并入 G7（011,004）；0 基索引
            continue
        if gi == 7:
            groups.append([subs.index(q) for q in r2[7]["members"] + r2[13]["members"]])
            recs.append(cU2)
        else:
            groups.append([subs.index(q) for q in g["members"]])
            recs.append(NC * NT + gi)
    assert sorted(sum(groups, [])) == list(range(len(subs)))

    def gstat(g, k):
        v = Ball[g, k]
        return dict(min=float(np.nanmin(v[:, 0])), bacc=float(np.nanmean(v[:, 0])),
                    auc=float(np.nanmean(v[:, 1])), f1=float(np.nanmean(v[:, 2])))

    def best_col(g):
        b = Bt[g]
        mn = b.min(0).values
        mx = mn.max()
        tie = mn >= mx - 1e-7
        k = int(torch.where(tie, b.mean(0), torch.tensor(-9.0, device=dev)).argmax())
        return float(mx), k

    res = dict(params=dict(WGRID=WGRID, ALPHAS=ALPHAS, TAUS=[str(t) for t in TAUS], comps=COMPS,
                           n_cfg=NC, n_thr=NT, n_orig=len(r2), TH=TH), rounds=[])
    pr(f"\n=== 第 1 部分：起点 {len(groups)} 组 ===")
    for g, k in zip(groups, recs):
        st = gstat(g, k)
        assert st["min"] >= TH - 1e-6, ([subs[m] for m in g], st)
    rnd = 0
    pair_log = None
    while True:
        ng = len(groups)
        Gm = torch.stack([Bt[g].min(0).values for g in groups])      # (ng, K)
        Gs = torch.stack([Bt[g].sum(0) for g in groups])
        Gn = torch.tensor([len(g) for g in groups], device=dev).float()
        S = torch.full((ng, ng), -1.0, device=dev)
        for a in range(ng - 1):
            mn = torch.minimum(Gm[a][None], Gm[a + 1:])
            S[a, a + 1:] = mn.max(1).values
        Sn = S.cpu().numpy()
        pairs = sorted([(Sn[a, b], a, b) for a in range(ng) for b in range(a + 1, ng)], reverse=True)
        if rnd == 0:
            pair_log = [dict(a=[subs[m] for m in groups[a]], b=[subs[m] for m in groups[b]], min=round(float(v), 4))
                        for v, a, b in pairs[:30]]
        feas = [(v, a, b) for v, a, b in pairs if v >= TH]
        pr(f"round {rnd}: {ng} 组，{len(pairs)} 组对，可行 {len(feas)}；top5 "
           + "; ".join(f"{'+'.join(subs[m] for m in groups[a][:2])}..|{'+'.join(subs[m] for m in groups[b][:2])}.. "
                       f"{v:.4f}" for v, a, b in pairs[:5]))
        if not feas:
            break
        # 同 min 时取并集宏 bACC 高者
        top_v = feas[0][0]
        tied = [(a, b) for v, a, b in feas if v >= top_v - 1e-7]
        cands = []
        for a, b in tied:
            u = groups[a] + groups[b]
            v, k = best_col(u)
            cands.append((v, gstat(u, k)["bacc"], a, b, k))
        v, _, a, b, k = max(cands, key=lambda x: (x[0], x[1]))
        u = groups[a] + groups[b]
        st = gstat(u, k)
        cfg, thr = colname(k, gnames)
        rec = dict(round=rnd + 1, a=[subs[m] for m in groups[a]], b=[subs[m] for m in groups[b]],
                   cfg=cfg, thr=thr, min=round(st["min"], 4), bacc=round(st["bacc"], 4),
                   auc=round(st["auc"], 4), f1=round(st["f1"], 4),
                   per_subject={subs[m]: round(float(Ball[m, k, 0]), 4) for m in u},
                   n_feasible=len(feas))
        res["rounds"].append(rec)
        pr(f"  合并 {rec['a']} + {rec['b']} → min {st['min']:.4f} macro {st['bacc']:.4f}/{st['auc']:.4f}/"
           f"{st['f1']:.4f} | {cfg} | {thr}")
        groups = [g for i, g in enumerate(groups) if i not in (a, b)] + [u]
        recs = [r for i, r in enumerate(recs) if i not in (a, b)] + [k]
        rnd += 1
        json.dump(res, open(HERE / "e31b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    res["round0_top_pairs"] = pair_log

    # 不可行剩余组对的瓶颈
    ng = len(groups)
    Gm = torch.stack([Bt[g].min(0).values for g in groups])
    bott = []
    for a in range(ng - 1):
        for b in range(a + 1, ng):
            u = groups[a] + groups[b]
            v, k = best_col(u)
            worst = min(u, key=lambda m: Ball[m, k, 0] if not np.isnan(Ball[m, k, 0]) else -1)
            bott.append((v, a, b, k, worst))
    bott.sort(key=lambda x: -x[0])
    res["final_top_infeasible"] = []
    from collections import Counter
    cnt = Counter()
    for v, a, b, k, worst in bott[:20]:
        cfg, thr = colname(k, gnames)
        res["final_top_infeasible"].append(dict(a=[subs[m] for m in groups[a]], b=[subs[m] for m in groups[b]],
                                                min=round(v, 4), cfg=cfg, thr=thr, worst=subs[worst]))
        cnt[subs[worst]] += 1
    res["bottleneck_count_top20"] = dict(cnt.most_common())

    # 最终分组
    per = np.zeros((len(subs), 3))
    fg = []
    order = sorted(range(ng), key=lambda i: -len(groups[i]))
    pr(f"\n=== 最终 {ng} 组 ===")
    for i in order:
        g, k = groups[i], recs[i]
        st = gstat(g, k)
        cfg, thr = colname(k, gnames)
        fg.append(dict(members=[subs[m] for m in g], cfg=cfg, thr=thr, min=round(st["min"], 4),
                       bacc=round(st["bacc"], 4), auc=round(st["auc"], 4), f1=round(st["f1"], 4)))
        for m in g:
            per[m] = np.nan_to_num(Ball[m, k], nan=.5)
        pr(f"[{len(g)}] min {st['min']:.4f} mean {st['bacc']:.4f} | {cfg} | {thr} | {' '.join(subs[m] for m in g)}")
    fm = dict(bacc=float(per[:, 0].mean()), auc=float(per[:, 1].mean()), f1=float(per[:, 2].mean()),
              n90=int((per[:, 0] >= TH).sum()), min=float(per[:, 0].min()))
    res["final_groups"], res["final_metrics"] = fg, fm
    res["final_per_subject"] = {q: [round(float(x), 4) for x in per[j]] for j, q in enumerate(subs)}
    pr(f"最终 {ng} 组：宏 bACC {fm['bacc']:.4f} AUC {fm['auc']:.4f} F1 {fm['f1']:.4f} n90 {fm['n90']}/57 "
       f"min {fm['min']:.4f}")
    pr("剩余最佳不可行组对 top5：" + "; ".join(f"{d['min']:.4f}(瓶颈 {d['worst']})" for d in res["final_top_infeasible"][:5]))
    pr(f"瓶颈人计数（top20 不可行组对）：{res['bottleneck_count_top20']}")
    json.dump(res, open(HERE / "e31b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---------- 第 2 部分：LOSO ----------
    pr("\n=== 第 2 部分：LOSO ===")
    V = torch.tensor(np.nan_to_num(EV, nan=.5), device=dev)        # (57, NC, NT, 3)；NaN 视为机会水平
    nq = V.shape[0]
    b = V[..., 0]
    tot = b.sum(0)
    loo = (tot[None] - b) / (nq - 1)                                # (57, NC, NT)：其余 56 人平均
    sel = loo.argmax(2)                                             # (57, NC)
    A = V.gather(2, sel[..., None, None].expand(-1, -1, 1, 3))[:, :, 0]   # (57, NC, 3)
    mac = A.mean(0)
    n90 = (A[..., 0] >= TH).sum(0)
    mnv = A[..., 0].min(0).values
    order = torch.argsort(mac[:, 0], descending=True).cpu().numpy()
    top10 = []
    for c in order[:10]:
        thr_modes = np.bincount(sel[:, c].cpu().numpy(), minlength=NT)
        top10.append(dict(cfg=NAMES[c], bacc=round(float(mac[c, 0]), 4), auc=round(float(mac[c, 1]), 4),
                          f1=round(float(mac[c, 2]), 4), n90=int(n90[c]), min=round(float(mnv[c]), 4),
                          thr_mode=E.THR2[int(thr_modes.argmax())][0]))
        pr(f"  {NAMES[c]} | {top10[-1]['thr_mode']} | bACC {mac[c, 0]:.4f} AUC {mac[c, 1]:.4f} "
           f"F1 {mac[c, 2]:.4f} n90 {int(n90[c])} min {mnv[c]:.4f}")
    on90 = torch.argsort(n90 * 10 + mac[:, 0], descending=True)[:5].cpu().numpy()
    top_n90 = [dict(cfg=NAMES[c], bacc=round(float(mac[c, 0]), 4), auc=round(float(mac[c, 1]), 4),
                    f1=round(float(mac[c, 2]), 4), n90=int(n90[c]), min=round(float(mnv[c]), 4)) for c in on90]
    pr("  n90 最高 top5：" + "; ".join(f"{d['cfg']} n90 {d['n90']} bACC {d['bacc']:.4f}" for d in top_n90))
    anchors = {}
    for nm in ("b|pool|w300|a0.7|tinf", "b|pool|w600|a0.5|tinf", "b|pool|w300|a0.5|tinf"):
        c = NAMES.index(nm)
        anchors[nm] = dict(bacc=round(float(mac[c, 0]), 4), auc=round(float(mac[c, 1]), 4),
                           f1=round(float(mac[c, 2]), 4), n90=int(n90[c]), min=round(float(mnv[c]), 4))
        pr(f"  锚点 {nm}: {anchors[nm]}")

    # 嵌套 (a)：外层留 i，在其余 56 人上联合选 (配置×阈值) 最大平均 bACC
    flat = loo.reshape(nq, -1)
    ka = flat.argmax(1)
    Va = V.reshape(nq, -1, 3)[torch.arange(nq, device=dev), ka]
    # 嵌套 (b)：外层留 i；在其余 56 人中对每配置再做内层 LOSO 选阈值，取内层 LOSO 平均 bACC 最大的配置；
    #           阈值对 i 用其余 56 人平均最大者
    Vb, kb = [], []
    for i in range(nq):
        keep = torch.tensor([j for j in range(nq) if j != i], device=dev)
        bi = b[keep]                                    # (56, NC, NT)
        ti = bi.sum(0)
        inner = ((ti[None] - bi) / (nq - 2)).argmax(2)  # (56, NC)
        sc = bi.gather(2, inner[..., None])[..., 0].mean(0)   # (NC,)
        c = int(sc.argmax())
        t = int(sel[i, c])
        kb.append((c, t))
        Vb.append(V[i, c, t])
    Vb = torch.stack(Vb)

    def summ(X):
        X = X.cpu().numpy()
        return dict(bacc=round(float(X[:, 0].mean()), 4), auc=round(float(X[:, 1].mean()), 4),
                    f1=round(float(X[:, 2].mean()), 4), n90=int((X[:, 0] >= TH).sum()),
                    min=round(float(X[:, 0].min()), 4))

    from collections import Counter
    na = Counter(f"{NAMES[int(k) // NT]}|{E.THR2[int(k) % NT][0]}" for k in ka.cpu().numpy())
    nb = Counter(f"{NAMES[c]}|{E.THR2[t][0]}" for c, t in kb)
    res["part2"] = dict(top10=top10, top_n90=top_n90, anchors=anchors,
                        nested_joint=dict(**summ(Va), picks=dict(na.most_common(5))),
                        nested_inner=dict(**summ(Vb), picks=dict(nb.most_common(5))),
                        baselines={"E28 b|pool": dict(bacc=.821, n90=18), "E24 Ward k=3": dict(bacc=.825, n90=16)})
    pr(f"  嵌套(a) 联合选 配置×阈值：{res['part2']['nested_joint']}")
    pr(f"  嵌套(b) 内层 LOSO 选配置：{res['part2']['nested_inner']}")
    json.dump(res, open(HERE / "e31b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pr(f"total {time.time() - t00:.0f}s")
    LOG.close()


if __name__ == "__main__":
    main()
