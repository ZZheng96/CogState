"""P4：块边界能不能定到 ±2 窗？——方向估计与边界检测的耦合诊断。

动机（P3 结论）：协议时间表本身能给出 0.986，但真实记录前后各有 ≤60 s 无标签段，
"起点对齐"未知时同一模型只有 0.798。要把 0.798 拉回 0.95，唯一的技术点是**把 5 个块切点定位到 ±2 窗**，
因为切点一准，"块序号奇偶 = 标签"就不再需要时钟。

本脚本回答三个问题（全部在真实顺序、subject-disjoint 5 折上）：
  A. 如果方向已知（用真实块估 d），DP 分段能把切点定到多准？           —— 边界检测本身的上限
  B. 只用跨被试分数（方向未知）时有多准？                              —— P3 的 11 窗复核
  C. 用"粗略时钟初始化 -> 估方向 -> 重检边界 -> 再估方向"的迭代能否收敛？ —— 可部署方案

产物：boundary_refine.json / .log
运行：py -3.11 test/10_projects/protocol_20261102/boundary_refine.py [--smoke]
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
SIGN = np.array([-1, 1, -1, 1, -1, 1], float)


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def seg_from_cuts(cuts, n):
    seg = np.zeros(n, int)
    for k in range(len(cuts) - 1):
        seg[cuts[k]:cuts[k + 1]] = k
    return seg


def dp_alternating(s, plen, lo, lmin=24, lmax=90, w_prior=1.0):
    """6 段交替 MAP。返回 (cuts, objective)。cuts 长度 7，首 0 尾 n。"""
    n = len(s)
    if n < NBLK * lmin:
        return None, -np.inf
    cs = np.concatenate([[0.0], np.cumsum(s)])
    neg = -1e18

    def pen(ln):
        if ln < lo or ln >= lo + len(plen):
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
        return cuts[::-1], f[NBLK, n]

    a = run(SIGN)
    b = run(-SIGN)
    cand = [r for r in (a, b) if r[0] is not None]
    if not cand:
        return None, -np.inf
    best = max(cand, key=lambda r: r[1])
    return best[0], best[1]


def dir_from_seg(Z, seg):
    m = {k: Z[seg == k].mean(0) for k in range(NBLK) if (seg == k).any()}
    tsk = [m[k] for k in m if SIGN[k] > 0]
    rlx = [m[k] for k in m if SIGN[k] < 0]
    if not tsk or not rlx:
        return None
    d = np.mean(tsk, 0) - np.mean(rlx, 0)
    n = np.linalg.norm(d)
    return None if n < 1e-12 else d / n


def metrics(y, p, g):
    yh = (p >= 0.5).astype(int)
    per = {}
    for s in np.unique(g):
        q = g == s
        if q.sum() == 0:
            continue
        per[s] = float(balanced_accuracy_score(y[q], yh[q]))
    return {"AUC": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "F1_task": float(f1_score(y, yh, average="binary", pos_label=1, zero_division=0)),
            "subject_macro_bACC": float(np.mean(list(per.values()))),
            "n_ge_0.9": int(sum(v >= 0.9 for v in per.values())), "n_subj": len(per)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    D = np.load(HERE / "realorder.npz")
    F, y, g, t, blk = D["F"], D["y"].astype(int), D["g"], D["t"], D["blk"]
    subs = sorted(np.unique(g))
    fold = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        fold[te] = k
    folds = [0] if args.smoke else list(range(5))
    print(f"[data] rows={len(y)} dim={F.shape[1]} subs={len(subs)} folds={folds} {el()}", flush=True)

    # 块长先验（训练折被试）
    def prior_for(k):
        cnt = []
        for s in subs:
            if fold[g == s][0] == k:
                continue
            q = g == s
            cnt.extend(np.bincount(blk[q], minlength=NBLK).tolist())
        lo, hi = 24, 90
        h = np.zeros(hi - lo + 1) + 0.5
        for c in cnt:
            h[int(np.clip(c, lo, hi)) - lo] += 1
        return lo, h / h.sum()

    P = {nm: np.full(len(y), np.nan) for nm in
         ("oracle_dir", "cross_score", "iter_clock", "iter_uniform", "single")}
    diag = {nm: [] for nm in ("oracle_dir", "cross_score", "iter_clock", "iter_uniform")}
    trial_abs = {nm: [] for nm in ("oracle_dir", "cross_score", "iter_clock", "iter_uniform")}

    for k in folds:
        tk = time.perf_counter()
        tr = np.flatnonzero(fold != k)
        te = np.flatnonzero(fold == k)
        med = np.nanmedian(F[tr], 0)
        med = np.where(np.isfinite(med), med, 0.0)
        Ftr = np.where(np.isfinite(F[tr]), F[tr], med)
        Fte = np.where(np.isfinite(F[te]), F[te], med)
        mu, sd = Ftr.mean(0), np.where(Ftr.std(0) > 1e-12, Ftr.std(0), 1.0)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                           min_samples_leaf=40, l2_regularization=1.0,
                                           random_state=0).fit(Ftr, y[tr])
        s_all = logit(m.predict_proba(Fte)[:, 1])
        lo, plen = prior_for(k)

        for s in np.unique(g[te]):
            loc = te[g[te] == s]
            loc = loc[np.argsort(t[loc], kind="stable")]
            idx = np.searchsorted(te, loc)
            n = len(loc)
            Z = (np.where(np.isfinite(F[loc]), F[loc], med) - mu) / sd
            yy = y[loc]
            bb = blk[loc]
            true_cuts = np.concatenate([[0], np.cumsum(np.bincount(bb, minlength=NBLK))])
            true_cuts[-1] = n
            cuts_ok = len(true_cuts) == NBLK + 1 and np.all(np.diff(true_cuts) > 0)

            # ---- A 方向已知（用真实块）
            if cuts_ok:
                d_or = dir_from_seg(Z, seg_from_cuts(true_cuts, n))
            else:
                d_or = None
            if d_or is not None:
                sco = Z @ d_or
                c, _ = dp_alternating(sco, plen, lo)
                if c is not None:
                    seg = seg_from_cuts(c, n)
                    P["oracle_dir"][loc] = np.where(seg % 2 == 1, 0.99, 0.01)
                    diag["oracle_dir"].append(float(np.mean(seg % 2 == yy)))
                    trial_abs["oracle_dir"].extend([int(min(abs(c[j] - true_cuts[j]) for j in range(1, NBLK)))
                                                    for j in range(1, NBLK)])

            # ---- B 跨被试分数
            z = s_all[idx]
            zz = (z - z.mean()) / max(z.std(), 1e-6)
            c, _ = dp_alternating(zz, plen, lo)
            if c is not None:
                seg = seg_from_cuts(c, n)
                P["cross_score"][loc] = np.where(seg % 2 == 1, 0.99, 0.01)
                diag["cross_score"].append(float(np.mean(seg % 2 == yy)))
                if cuts_ok:
                    trial_abs["cross_score"].extend([int(min(abs(c[j] - true_cuts[j]) for j in range(1, NBLK)))
                                                     for j in range(1, NBLK)])

            # ---- C 迭代：粗略时钟初始化 -> 估方向 -> 重检边界
            def iterate(cuts0, tag):
                seg = seg_from_cuts(cuts0, n)
                for _ in range(args.iters):
                    d = dir_from_seg(Z, seg)
                    if d is None:
                        return None
                    c, _ = dp_alternating(Z @ d, plen, lo)
                    if c is None:
                        return None
                    new = seg_from_cuts(c, n)
                    if np.array_equal(new, seg):
                        break
                    seg = new
                P[tag][loc] = np.where(seg % 2 == 1, 0.99, 0.01)
                diag[tag].append(float(np.mean(seg % 2 == yy)))
                if cuts_ok:
                    trial_abs[tag].extend([int(min(abs(c[j] - true_cuts[j]) for j in range(1, NBLK)))
                                           for j in range(1, NBLK)])
                return seg

            e = n // NBLK
            iterate(np.array([round(i * n / NBLK) for i in range(NBLK)] + [n]), "iter_clock")
            iterate(np.array([0] + [round(n * (i + .5) / NBLK) for i in range(1, NBLK)] + [n]), "iter_uniform")

            P["single"][loc] = z

        print(f"[fold {k}] {el()} s={time.perf_counter() - tk:.1f}", flush=True)

    out = {"meta": {"protocol": "real-order SGKF(5,rs0) subject-disjoint", "n_windows": int(len(y)),
                    "n_subjects": len(subs), "iters": args.iters,
                    "note": "oracle_dir 用真实块估方向，仅诊断；iter_* 不用任何真实块信息"},
           "methods": {}, "boundary": {}, "seconds": 0.0}
    for nm in P:
        p = P[nm]
        ev = np.isfinite(p)
        out["methods"][nm] = metrics(y[ev], p[ev], g[ev])
        out["methods"][nm]["evaluated_windows"] = int(ev.sum())
        q = out["methods"][nm]
        print(f"[result] {nm:12s} AUC={q['AUC']:.4f} bACC={q['bACC']:.4f} F1m={q['F1_macro']:.4f} "
              f"macro={q['subject_macro_bACC']:.4f} n>=.9={q['n_ge_0.9']}/{q['n_subj']}", flush=True)
    for nm in diag:
        if diag[nm]:
            a = np.array(trial_abs[nm], float)
            out["boundary"][nm] = {"parity_acc_mean": float(np.mean(diag[nm])),
                                   "cut_abs_err_mean": float(a.mean()), "cut_abs_err_median": float(np.median(a)),
                                   "cut_abs_err_p90": float(np.percentile(a, 90)), "n_cuts": int(len(a))}
            print(f"[boundary] {nm:12s} parity={np.mean(diag[nm]):.4f} |err| mean={a.mean():.2f} "
                  f"med={np.median(a):.1f} p90={np.percentile(a, 90):.1f}", flush=True)
    out["seconds"] = time.perf_counter() - T0
    stem = "boundary_refine" + ("_smoke" if args.smoke else "")
    (HERE / f"{stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] -> {stem}.json {el()}", flush=True)


if __name__ == "__main__":
    main()
