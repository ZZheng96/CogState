"""P3：把"范式块结构"当作唯一可用的定向约束，研究单通道跨被试解码的可达上限。

背景（见 理论框架.md / 整理与研究入口.md）：
  * 严格未知被试、6 s 单窗、无历史：AUC≈0.67 / bACC≈0.63。
  * 跨被试判别方向几乎正交（cos .03–.08），约 1/3 被试反号；无标签数据里没有任何统计量能定符号
    （label switching，Stephens 2000）——除非引入外部结构信息。
  * 57/57 被试都是固定 6 块 R-T-R-T-R-T，块长 218–382 s（均值 300 s），首块必为 relax。

本脚本检验的假设 H-P：标签可以由"块序号奇偶"决定，EEG 只需要用来在线定位 5 个块边界。
若边界可被因果检测，则 bACC/AUC 主要受边界定位误差支配，而不是受单窗可分性支配。

方法（全部只读真实顺序 realorder.npz）：
  A single      : 跨被试 HistGB 单窗分数（无协议）——基线
  B position    : 只用"已到达窗数/已到达秒数"训练 HistGB——位置捷径对照（使用测试流位置，无 EEG）
  C dp_mag      : 离线 MAP 分段（块长先验 + 交替对比 |Σ a_k mean_k|），输出硬奇偶标签——不可部署上界
  D hsmm        : 因果 duration-HMM 前向滤波（2 个符号假设 σ=±1 按后验混合），只用当前及过去
  E hsmm_hybrid : D 的后验 × 单窗证据（把 EEG 分数以弱权重并进去）

注意：C 用整段记录（含未来），只能作为"边界可检测性"的诊断上界；D/E 严格因果。
所有指标都在 subject-disjoint 5 折上评估（测试被试的真标签只用于打分）。

运行：py -3.11 test/10_projects/protocol_20261102/protocol_segment.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
T0 = time.perf_counter()

NBLK = 6
SIGN = np.array([-1, 1, -1, 1, -1, 1], float)  # relax / task / relax / task / relax / task


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


# ------------------------------------------------------------------ metrics
def metrics(y, p, g, thr=0.5):
    yr = np.asarray(p, float)
    T = np.broadcast_to(np.asarray(thr, float), yr.shape)
    yh = (yr >= T).astype(int)
    out = {
        "AUC": float(roc_auc_score(y, yr)),
        "bACC": float(balanced_accuracy_score(y, yh)),
        "F1_macro": float(f1_score(y, yh, average="macro")),
        "F1_task": float(f1_score(y, yh, average="binary", pos_label=1, zero_division=0)),
    }
    per = {}
    for s in np.unique(g):
        q = g == s
        if q.sum() == 0:
            continue
        yy, hh, pp = y[q], yh[q], yr[q]
        per[s] = {
            "bACC": float(balanced_accuracy_score(yy, hh)),
            "AUC": float(roc_auc_score(yy, pp)) if yy.min() != yy.max() else float("nan"),
        }
    b = [v["bACC"] for v in per.values()]
    a = [v["AUC"] for v in per.values() if np.isfinite(v["AUC"])]
    out["subject_macro_bACC"] = float(np.mean(b)) if b else float("nan")
    out["subject_macro_AUC"] = float(np.mean(a)) if a else float("nan")
    out["n_subj"] = len(per)
    out["n_ge_0.9"] = int(sum(v >= 0.9 for v in b))
    out["per_subject"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in per.items()}
    return out


# ------------------------------------------------------------------ segmentation primitives
def len_prior(counts):
    """训练被试的块长（窗数）经验分布 -> 平滑离散先验 p[len]（len 以窗为单位）。"""
    lo, hi = 24, 90
    h = np.zeros(hi - lo + 1)
    for c in counts:
        c = int(np.clip(c, lo, hi))
        h[c - lo] += 1
    h = h + 0.5  # Laplace
    p = h / h.sum()
    return lo, p


def dp_segments(s, lo, plen, w_prior=1.0, lmin=24, lmax=90):
    """离线 MAP：6 段、符号交替，最大化 Σ_k a_k*mean(seg k) - w*Σ_k (-log p[len_k])。
    对 SIGN 与 -SIGN 各跑一次取目标更大者：被试方向未知时，奇偶标签与全局符号无关，
    所以不能假定分数方向正确。返回 (每窗段号, 边界, 对比度)。"""
    n = len(s)
    if n < NBLK * lmin:
        return None
    cs = np.concatenate([[0.0], np.cumsum(s)])
    hi = lo + len(plen) - 1
    neg = -1e18

    def pen(ln):
        if ln < lo or ln > hi:
            return -1e9
        return -np.log(plen[int(ln) - lo])

    def run(sign):
        f = np.full((NBLK + 1, n + 1), neg)
        bk = np.zeros((NBLK + 1, n + 1), int)
        f[0, 0] = 0.0
        for k in range(1, NBLK + 1):
            a = sign[k - 1]
            for j in range(1, n + 1):
                i0 = max((k - 1) * lmin, j - lmax)
                i1 = j - lmin
                if i1 < i0:
                    continue
                best, bi = neg, -1
                for i in range(i0, i1 + 1):
                    if f[k - 1, i] <= neg / 2:
                        continue
                    v = f[k - 1, i] + a * (cs[j] - cs[i]) / (j - i) + w_prior * pen(j - i)
                    if v > best:
                        best, bi = v, i
                f[k, j], bk[k, j] = best, bi
        if f[NBLK, n] <= neg / 2:
            return None
        cuts, j = [n], n
        for k in range(NBLK, 0, -1):
            j = bk[k, j]
            cuts.append(j)
        cuts = cuts[::-1]
        seg = np.zeros(n, int)
        for k in range(NBLK):
            seg[cuts[k]:cuts[k + 1]] = k
        con = sum(sign[k] * (cs[cuts[k + 1]] - cs[cuts[k]]) / (cuts[k + 1] - cuts[k]) for k in range(NBLK))
        return f[NBLK, n], seg, cuts, con

    a = run(SIGN)
    b = run(-SIGN)
    cand = [r for r in (a, b) if r is not None]
    if not cand:
        return None
    best = max(cand, key=lambda r: r[0])
    return best[1], best[2], best[3]


def hsmm_causal(s, lo, plen, sigma_mode=True, lmin=24, lmax=90):
    """因果 duration-HMM 前向滤波。状态 (k, d=块内已过窗数)；发射 LLR = a_k * sigma * s_t。
    sigma_mode=True 时并列跑 sigma=+1/-1 两条链，按各自边际似然加权混合——这对应
    label switching 的两个模式；首块必为 relax 的先验使两条链在观测到块内对比后自行分开。
    只使用 t 及之前的 s。"""
    n = len(s)
    D = lmax
    pi = np.zeros(D + 2)
    pi[lo:lo + len(plen)] = plen
    S = np.array([pi[d:].sum() for d in range(D + 2)])
    haz = np.where(S > 1e-12, pi / np.maximum(S, 1e-12), 0.0)
    haz = np.clip(haz, 0.0, 1.0)
    haz[D] = 1.0                       # 到达最大块长强制切换
    lc = np.log(np.clip(1.0 - haz[1:D], 1e-12, 1.0))
    ls = np.log(np.clip(haz[1:D + 1], 1e-12, 1.0))

    def run(sig):
        A = np.full((NBLK, D + 1), -1e18)
        A[0, 1] = 0.0
        ll = 0.0
        out = np.empty(n)
        for ti in range(n):
            if ti > 0:
                B = np.full((NBLK, D + 1), -1e18)
                for k in range(NBLK):
                    B[k, 2:] = A[k, 1:D] + lc
                    if k + 1 < NBLK:
                        B[k + 1, 1] = np.logaddexp(B[k + 1, 1],
                                                   np.logaddexp.reduce(A[k, 1:D + 1] + ls) if D else -1e18)
                A = B
            A = A + (SIGN[:, None] * sig * s[ti])
            mx = A.max()
            if not np.isfinite(mx):
                return None
            A -= mx
            ll += mx
            tot = np.logaddexp.reduce(A.ravel())
            out[ti] = np.exp(A[1::2] - tot).sum()
        return out, ll

    r = run(1.0)
    if r is None:
        return None
    if not sigma_mode:
        return r[0]
    r2 = run(-1.0)
    if r2 is None:
        return r[0]
    w = 1.0 / (1.0 + np.exp(np.clip(r2[1] - r[1], -50, 50)))
    return w * r[0] + (1 - w) * r2[0]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--pos-offset", type=int, default=0,
                    help="仅在测试被试的位置特征上加偏移（窗数），模拟真实记录开头的 ≤60 s 预/后无标签段；"
                         "0 表示假设流恰好从第一块开始")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    D = np.load(HERE / "realorder.npz")
    F, y, g, t, blk, pos = D["F"], D["y"].astype(int), D["g"], D["t"], D["blk"], D["pos"]
    subs = sorted(np.unique(g))
    F = np.where(np.isfinite(F), F, np.nan)
    print(f"[data] rows={len(y)} subs={len(subs)} dim={F.shape[1]} pos_windows={len(y)} {el()}", flush=True)

    fold = np.empty(len(y), int)
    sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
        fold[te] = k
    for s in subs:
        assert np.unique(fold[g == s]).size == 1, f"subject {s} spans folds"
    print(f"[folds] " + " ".join(f"f{k}:{len(np.unique(g[fold == k]))}sub" for k in range(5)), flush=True)

    # 训练被试的块长分布（窗数），用于先验；按折只用训练折估计
    blk_len = {}
    for s in subs:
        q = g == s
        cnt = np.bincount(blk[q], minlength=NBLK)
        blk_len[s] = cnt

    res = {"meta": {"protocol": "real-order, subject-disjoint 5-fold (SGKF shuffle rs0)",
                    "n_subjects": len(subs), "n_windows": int(len(y)), "window_s": 6.0,
                    "blocks": "R-T-R-T-R-T, 首块=relax, 块长 218-382 s",
                    "note": "C=离线(用整段记录, 不可部署); D/E=严格因果"},
           "folds": {}, "methods": {}, "seconds": 0.0, "checks": {}}

    folds = [0] if args.smoke else list(range(5))
    P = {k: {} for k in folds}
    THR = {}
    for k in folds:
        tk = time.perf_counter()
        tr = np.flatnonzero(fold != k)
        te = np.flatnonzero(fold == k)
        med = np.nanmedian(F[tr], 0)
        med = np.where(np.isfinite(med), med, 0.0)
        Ftr = np.where(np.isfinite(F[tr]), F[tr], med)
        Fte = np.where(np.isfinite(F[te]), F[te], med)

        # A 单窗基线
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=0).fit(Ftr, y[tr])
        s_te = logit(m.predict_proba(Fte)[:, 1])

        # B 位置捷径（对照，无 EEG）
        Pos = np.column_stack([pos[tr], t[tr] / 60.0])
        mpos = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=3,
                                              min_samples_leaf=40, random_state=0).fit(Pos, y[tr])
        p_pos = mpos.predict_proba(np.column_stack([pos[te] + args.pos_offset,
                                                   (t[te] + 6.0 * args.pos_offset) / 60.0]))[:, 1]

        # B2 位置先验 + 未知起始偏移（真实记录开头最多 60 s 无标签段 = 10 窗）：对偏移后验均匀边缘化
        pu = []
        for u in range(0, 11):
            pu.append(mpos.predict_proba(np.column_stack([pos[te] + u, (t[te] + 6.0 * u) / 60.0]))[:, 1])
        p_posm = np.mean(pu, 0)
        # 阈值必须在训练折上定（不能用测试标签）：同样做偏移边缘化后最大化训练 macro bACC
        ptu = []
        for u in range(0, 11):
            ptu.append(mpos.predict_proba(np.column_stack([pos[tr] + u, (t[tr] + 6.0 * u) / 60.0]))[:, 1])
        p_tr_marg = np.mean(ptu, 0)
        cand = np.unique(np.quantile(p_tr_marg, np.linspace(0.01, 0.99, 197)))
        sc = [balanced_accuracy_score(y[tr], (p_tr_marg >= th).astype(int)) for th in cand]
        thr_marg = float(cand[int(np.argmax(sc))])
        THR[k] = thr_marg

        # 训练折块长先验
        cnt = [c for s in subs if fold[g == s][0] != k for c in blk_len[s]]
        lo, plen = len_prior(cnt)

        # 逐测试被试：离线 DP / 因果 HSMM / 过去块对比方向
        p_dp = np.full(len(te), np.nan)
        p_hs = np.full(len(te), np.nan)
        p_bd = np.full(len(te), np.nan)
        seg_err, cut_err = {}, {}
        mu_tr, sd_tr = Ftr.mean(0), np.where(Ftr.std(0) > 1e-12, Ftr.std(0), 1.0)
        for s in np.unique(g[te]):
            loc = te[g[te] == s]
            loc = loc[np.argsort(t[loc], kind="stable")]
            idx = np.searchsorted(te, loc)          # 全局索引 -> 折内位置
            z = s_te[idx]
            yb = y[loc]
            blk_ids = blk[loc]
            Zs = (np.where(np.isfinite(F[loc]), F[loc], med) - mu_tr) / sd_tr
            # 离线诊断：用整段记录的尺度做 z（仅 C 使用，C 本身就是离线上界）
            zz = (z - z.mean()) / max(z.std(), 1e-6)
            r = dp_segments(zz, lo, plen)
            if r is not None:
                seg, cuts, _c = r
                p_dp[idx] = np.where(seg % 2 == 1, 0.98, 0.02)
                tb = np.cumsum(np.bincount(blk_ids, minlength=NBLK))[:-1]
                tb = tb[tb < len(loc)]
                cut_err[s] = {"bounds": [int(c) for c in cuts[1:-1]], "true": tb.tolist(),
                              "abs_err": [int(min(abs(c - q) for q in tb)) for c in cuts[1:-1]]}
                seg_err[s] = float(np.mean(seg % 2 == yb))
            # 因果方法只能使用过去样本做尺度：前 min(10,n) 窗的中位数/MAD
            kk = min(10, len(z))
            med0 = float(np.median(z[:kk]))
            mad0 = float(1.4826 * np.median(np.abs(z[:kk] - med0)))
            zc = (z - med0) / max(mad0, 1e-6)
            p_hs[idx] = hsmm_causal(zc, lo, plen)

            # F 用"已完成块"的交替对比估计本人方向（用真实块边界，只检验方向估计本身）：
            #   d_b = mean(过去所有 task 块) - mean(过去所有 relax 块) ∝ 本人方向
            #   单对相邻块（b-2,b-1）会被块间慢漂移主导，必须累计多对才能把漂移平均掉。
            #   b=0/1 尚无可用的交替对比，回退跨被试分数。
            m = {b: Zs[blk_ids == b].mean(0) for b in range(NBLK) if (blk_ids == b).any()}
            sc = np.full(len(loc), np.nan)
            sc[blk_ids <= 1] = z[blk_ids <= 1]
            for b in range(2, NBLK):
                done = [j for j in range(b) if j in m]
                tsk = [j for j in done if SIGN[j] > 0]
                rlx = [j for j in done if SIGN[j] < 0]
                if not tsk or not rlx:
                    sc[blk_ids == b] = z[blk_ids == b]
                    continue
                d = np.mean([m[j] for j in tsk], 0) - np.mean([m[j] for j in rlx], 0)
                nd = float(np.linalg.norm(d))
                sc[blk_ids == b] = z[blk_ids == b] if nd < 1e-12 else (Zs[blk_ids == b] @ d) / nd
            ok = np.isfinite(sc)
            sc[ok] /= max(float(sc[ok].std()), 1e-6)
            p_bd[idx] = sc

        P[k] = {"single": s_te, "position": p_pos, "position_marg": p_posm, "dp_mag": p_dp,
                "hsmm": p_hs, "blockdir": p_bd, "te": te}
        bm = float(np.mean([v for v in seg_err.values()])) if seg_err else float("nan")
        print(f"[fold {k}] singleAUC={roc_auc_score(y[te], s_te):.4f} "
              f"posAUC={roc_auc_score(y[te], p_pos):.4f} "
              f"dpACC={bm:.4f} hsmmAUC={roc_auc_score(y[te], np.nan_to_num(p_hs, nan=0.5)):.4f} "
              f"s={time.perf_counter() - tk:.1f}s {el()}", flush=True)
        res["folds"][str(k)] = {"cut_err": cut_err,
                                "dp_seg_acc": bm,
                                "thr_position_marg": thr_marg,
                                "n_test_subjects": int(len(np.unique(g[te])))}

    def gather(name):
        p = np.full(len(y), np.nan)
        for k in folds:
            p[P[k]["te"]] = P[k][name]
        return p

    thr_arr = np.full(len(y), 0.5)
    for k in folds:
        thr_arr[P[k]["te"]] = THR[k]

    for name in ("single", "position", "position_marg", "dp_mag", "hsmm", "blockdir"):
        p = gather(name)
        ev = np.isfinite(p)
        thr = thr_arr[ev] if name == "position_marg" else 0.5
        m = metrics(y[ev], p[ev], g[ev], thr=thr)
        ev2 = ev & (blk >= 2)
        ev4 = ev & (blk >= 4)
        m["subset_blk_ge2"] = metrics(y[ev2], p[ev2], g[ev2],
                                      thr=(thr_arr[ev2] if name == "position_marg" else 0.5))
        m["subset_blk_ge4"] = metrics(y[ev4], p[ev4], g[ev4],
                                      thr=(thr_arr[ev4] if name == "position_marg" else 0.5))
        m["evaluated_windows"] = int(ev.sum())
        res["methods"][name] = m
        print(f"[result] {name:9s} AUC={m['AUC']:.4f} bACC={m['bACC']:.4f} F1m={m['F1_macro']:.4f} "
              f"F1task={m['F1_task']:.4f} macro_bACC={m['subject_macro_bACC']:.4f} "
              f"n>=.9={m['n_ge_0.9']}/{m['n_subj']} n={m['evaluated_windows']} | "
              f"blk>=2 AUC={m['subset_blk_ge2']['AUC']:.4f} bACC={m['subset_blk_ge2']['bACC']:.4f} | "
              f"blk>=4 AUC={m['subset_blk_ge4']['AUC']:.4f} bACC={m['subset_blk_ge4']['bACC']:.4f}", flush=True)

    # 诊断：边界定位误差
    ce = [e for f in res["folds"].values() for e in f["cut_err"].values()]
    if ce:
        ae = [d for e in ce for d in e["abs_err"]]
        res["checks"]["boundary_abs_err_windows"] = {
            "mean": float(np.mean(ae)), "median": float(np.median(ae)),
            "p90": float(np.percentile(ae, 90)), "n": len(ae)}

    res["seconds"] = time.perf_counter() - T0
    stem = "protocol_segment" + ("_smoke" if args.smoke else "") + (f"_off{args.pos_offset}" if args.pos_offset else "")
    res["meta"]["pos_offset_windows"] = args.pos_offset
    (HERE / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] -> {stem}.json {el()}", flush=True)


if __name__ == "__main__":
    main()
