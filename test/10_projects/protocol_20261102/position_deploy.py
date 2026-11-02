"""P5：把"起点偏移"当作部署约束来评估协议时间表模型。

P3 发现：只用"已到达窗数"的模型在流起点恰好对齐第一块时 bACC=0.9858；真实 BDF 前面有 ≤60 s 无标签段。
本脚本从 BDF 直接量出每个被试真实的 pre-roll（trial_start 相对文件首样本的秒数），
然后在 5 折上评估不同"假定偏移"下的真实表现：

  assume0   : 假设流从第一块开始（P3 的假设）
  assume9/10: 假设前导段是 54–60 s（训练集里 48/57 被试正好是 60.0 s）
  prior     : 用训练折被试的 u 经验分布做边缘化
  oracle    : 每个被试用自己的真实 u（不可部署，只作上限）

所有阈值/先验只用训练折被试估计；测试被试的真标签只用于打分。
输出：position_deploy.json / .log
运行：py -3.11 test/10_projects/protocol_20261102/position_deploy.py
"""
from __future__ import annotations

import argparse
import glob
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
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
T0 = time.perf_counter()


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def metrics(y, p, g, thr=0.5):
    yh = (p >= thr).astype(int)
    per = {}
    for s in np.unique(g):
        q = g == s
        per[s] = float(balanced_accuracy_score(y[q], yh[q])) if q.sum() else float("nan")
    return {"AUC": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "F1_task": float(f1_score(y, yh, average="binary", pos_label=1, zero_division=0)),
            "subject_macro_bACC": float(np.nanmean(list(per.values()))),
            "n_ge_0.9": int(np.sum(np.array(list(per.values())) >= 0.9)),
            "n_subj": int(np.sum(np.isfinite(list(per.values())))),
            "per_subject": {k: float(v) for k, v in per.items()}}


def measure_pre_roll():
    """真实 pre-roll（秒）：trial_start 相对 BDF 首样本。"""
    import mne

    from cogstate import load_trials
    out = {}
    for csv in sorted(glob.glob(str(ROOT / "EEG_dataset" / "*" / "*_events.csv"))):
        sid = Path(csv).parent.name
        bdf = next(Path(csv).parent.glob("*.bdf"))
        tr = load_trials(csv, str(bdf))
        out[sid] = float(tr[0].start)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    D = np.load(HERE / "realorder.npz")
    F, y, g, t, pos = D["F"], D["y"].astype(int), D["g"], D["t"], D["pos"]
    subs = sorted(np.unique(g))
    pre = measure_pre_roll()
    u_true = {s: int(round(pre[s] / 6.0)) for s in subs}
    print(f"[pre-roll] n={len(pre)} mean={np.mean(list(pre.values())):.1f}s "
          f"u histogram=" + ",".join(f"{k}:{v}" for k, v in
                                     sorted({u: sum(1 for x in u_true.values() if x == u) for u in set(u_true.values())}.items())),
          flush=True)

    fold = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        fold[te] = k
    folds = [0] if args.smoke else list(range(5))

    SHIFT_MODE = ("assume0", "assume9", "assume10", "prior", "oracle")
    P = {m: np.full(len(y), np.nan) for m in SHIFT_MODE}
    THR = {m: {} for m in SHIFT_MODE}
    U = np.array([u_true[s] for s in g])

    for k in folds:
        tr = np.flatnonzero(fold != k)
        te = np.flatnonzero(fold == k)
        med = np.nanmedian(F[tr], 0)
        med = np.where(np.isfinite(med), med, 0.0)
        Ftr = np.where(np.isfinite(F[tr]), F[tr], med)
        Fte = np.where(np.isfinite(F[te]), F[te], med)
        # 位置特征一律以"假定第 1 块起点"为 0：feature = 真实块内位置 + (u_true - u_assume)。
        # 模型只在 u_assume = u_true 的"对齐"特征上训练（pos, t），测试时套用不同假定。
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=3,
                                           min_samples_leaf=40, random_state=0).fit(
            np.column_stack([pos[tr], t[tr] / 60.0]), y[tr])

        def apply(assume):
            a_tr = np.full(len(tr), assume) if np.isscalar(assume) else assume[tr]
            a_te = np.full(len(te), assume) if np.isscalar(assume) else assume[te]
            Xtr = np.column_stack([pos[tr] + (U[tr] - a_tr), (t[tr] + 6.0 * (U[tr] - a_tr)) / 60.0])
            Xte = np.column_stack([pos[te] + (U[te] - a_te), (t[te] + 6.0 * (U[te] - a_te)) / 60.0])
            return m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]

        def pick_thr(p_tr):
            cand = np.unique(np.quantile(p_tr, np.linspace(0.01, 0.99, 197)))
            sc = [balanced_accuracy_score(y[tr], (p_tr >= th).astype(int)) for th in cand]
            return float(cand[int(np.argmax(sc))])

        for name, assume in (("assume0", 0), ("assume9", 9), ("assume10", 10), ("oracle", U)):
            p_tr, p_te = apply(assume)
            P[name][te] = p_te
            THR[name][k] = pick_thr(p_tr)
        # prior：用训练折被试的 u 分布做边缘化
        us = np.array(sorted(set(u_true[s] for s in subs if fold[g == s][0] != k)))
        w = np.array([np.mean([u_true[s] == u for s in subs if fold[g == s][0] != k]) for u in us])
        w = w / w.sum()
        acc_tr, acc_te = [], []
        for u, wi in zip(us, w):
            a, b = apply(u)
            acc_tr.append(wi * a)
            acc_te.append(wi * b)
        p_tr = np.sum(acc_tr, 0)
        P["prior"][te] = np.sum(acc_te, 0)
        THR["prior"][k] = pick_thr(p_tr)
        print(f"[fold {k}] u grid={list(us)} w={np.round(w, 3).tolist()} {el()}", flush=True)

    out = {"meta": {"protocol": "real-order SGKF(5,rs0) subject-disjoint", "n_subjects": len(subs),
                    "pre_roll_s": {k: round(v, 2) for k, v in pre.items()},
                    "u_true": {k: int(v) for k, v in u_true.items()},
                    "n_pre60": int(sum(1 for v in pre.values() if v > 59.9)),
                    "n_pre_lt60": int(sum(1 for v in pre.values() if v <= 59.9)),
                    "note": "u 单位=6 s 窗；assume9/10 对应'文件前导段 54–60 s'；prior 用训练折 u 经验分布"},
           "methods": {}, "subgroups": {}, "seconds": 0.0}

    for name in SHIFT_MODE:
        p = P[name]
        ev = np.isfinite(p)
        thr = np.full(len(y), 0.5)
        for k in folds:
            thr[fold == k] = THR[name][k]
        m = metrics(y[ev], p[ev], g[ev], thr[ev])
        out["methods"][name] = m
        print(f"[result] {name:9s} AUC={m['AUC']:.4f} bACC={m['bACC']:.4f} F1m={m['F1_macro']:.4f} "
              f"macro={m['subject_macro_bACC']:.4f} n>=.9={m['n_ge_0.9']}/{m['n_subj']}", flush=True)
        for tag, sel in (("pre60", np.abs(np.array([pre[s] for s in g])) > 59.9),
                         ("pre_lt60", np.abs(np.array([pre[s] for s in g])) <= 59.9)):
            q = ev & sel
            if q.sum():
                mm = metrics(y[q], p[q], g[q], thr[q])
                out["subgroups"][f"{name}|{tag}"] = {k2: mm[k2] for k2 in
                                                    ("AUC", "bACC", "F1_macro", "subject_macro_bACC", "n_ge_0.9", "n_subj")}
                print(f"   [sub] {tag:9s} AUC={mm['AUC']:.4f} bACC={mm['bACC']:.4f} F1m={mm['F1_macro']:.4f} "
                      f"n>=.9={mm['n_ge_0.9']}/{mm['n_subj']}", flush=True)

    out["seconds"] = time.perf_counter() - T0
    stem = "position_deploy" + ("_smoke" if args.smoke else "")
    (HERE / f"{stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] -> {stem}.json {el()}", flush=True)


if __name__ == "__main__":
    main()
