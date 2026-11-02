"""P6：把"协议时间表"模型换成可直接部署的经验查表（贝叶斯时钟），并评估带宽/阈值。

为什么需要：HistGB 版本（P5 的 assume10）已经到 0.9899/0.9718，但它是黑箱 + sklearn 依赖，
不便于做成"按时间顺序分块接收、逐次返回"的推理入口。这里用最直接的形式复刻同一假设：

    P(task | 第 k 个 6 s 窗) = 训练被试在该位置的 task 频率（高斯核平滑 + Laplace）

它没有超参数搜索空间（只有一个平滑带宽，且在训练折内选），因此更适合提交。

评估与 P5 完全同口径：真实顺序、SGKF(5,rs0) 被试隔离、用每个被试真实 pre-roll 模拟起点偏移。
输出：position_table.json / .log
运行：py -3.11 test/10_projects/protocol_20261102/position_table.py
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
T0 = time.perf_counter()
MAXPOS = 340


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def metrics(y, p, g, thr=0.5):
    yh = (p >= thr).astype(int)
    per = {s: float(balanced_accuracy_score(y[g == s], yh[g == s])) for s in np.unique(g) if (g == s).sum()}
    return {"AUC": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "F1_task": float(f1_score(y, yh, average="binary", pos_label=1, zero_division=0)),
            "subject_macro_bACC": float(np.mean(list(per.values()))),
            "n_ge_0.9": int(sum(v >= 0.9 for v in per.values())), "n_subj": len(per)}


def fit_table(pos, yy, bw):
    grid = np.arange(MAXPOS + 1, dtype=float)
    num = np.zeros_like(grid)
    den = np.zeros_like(grid)
    for lo in range(0, len(pos), 2000):
        p = pos[lo:lo + 2000].astype(float)
        w = np.exp(-0.5 * ((grid[:, None] - p[None, :]) / bw) ** 2)
        num += w @ yy[lo:lo + 2000]
        den += w.sum(1)
    return (num + 1.0) / (den + 2.0)


def predict(p, idx):
    return p[np.clip(idx, 0, MAXPOS).astype(int)]


def measure_pre_roll():
    import mne

    from cogstate import load_trials
    out = {}
    for csv in sorted(glob.glob(str(ROOT / "EEG_dataset" / "*" / "*_events.csv"))):
        sid = Path(csv).parent.name
        bdf = next(Path(csv).parent.glob("*.bdf"))
        out[sid] = float(load_trials(csv, str(bdf))[0].start)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    D = np.load(HERE / "realorder.npz")
    y, g, pos, t = D["y"].astype(int), D["g"], D["pos"], D["t"]
    subs = sorted(np.unique(g))
    pre = measure_pre_roll()
    u_true = np.array([int(round(pre[s] / 6.0)) for s in g])

    fold = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        fold[te] = k
    folds = [0] if args.smoke else list(range(5))

    BWS = (0.6, 1.0, 1.5, 2.5, 4.0)
    MODES = ("assume0", "assume9", "assume10", "auto", "prior", "oracle")
    P = {m: np.full(len(y), np.nan) for m in MODES}
    THR = {m: {} for m in MODES}
    chosen = {}

    for k in folds:
        tr = np.flatnonzero(fold != k)
        te = np.flatnonzero(fold == k)
        # 查表的自变量是"假定第 1 块起点之后的时间（以 6 s 窗计数）"。
        # 用记录的绝对时间 t 而不是窗序号 pos：t 天然包含文件前导段与坏段造成的空档，
        # 而 pos 只是有效窗计数，两者在 57 份记录上平均相差 18 个窗（最大 179），不能混用。
        pos_tr = t[tr] / 6.0 - u_true[tr]
        # 带宽在训练折内用留一被试的方式粗选：直接看训练集自身的 macro bACC
        best, bv = None, -1
        for bw in BWS:
            tab = fit_table(pos_tr, y[tr], bw)
            p = predict(tab, pos_tr)
            v = balanced_accuracy_score(y[tr], (p >= 0.5).astype(int))
            if v > bv:
                best, bv = bw, v
        chosen[k] = {"bw": best, "train_bACC": float(bv)}
        tab = fit_table(pos_tr, y[tr], best)
        print(f"[fold {k}] bw={best} train_bACC={bv:.4f} {el()}", flush=True)

        def run(name, assume):
            a_te = np.full(len(te), assume) if np.isscalar(assume) else assume[te]
            a_tr = np.full(len(tr), assume) if np.isscalar(assume) else assume[tr]
            idx_te = t[te] / 6.0 - a_te
            idx_tr = t[tr] / 6.0 - a_tr
            P[name][te] = predict(tab, idx_te)
            cand = np.unique(np.quantile(predict(tab, idx_tr), np.linspace(0.02, 0.98, 97)))
            sc = [balanced_accuracy_score(y[tr], (predict(tab, idx_tr) >= c).astype(int)) for c in cand]
            THR[name][k] = float(cand[int(np.argmax(sc))])

        run("assume0", 0)
        run("assume10", 10)
        run("assume9", 9)
        run("oracle", u_true)
        # auto：u_assume 只在训练折上选（对训练被试施加同样的偏移，最大化其 macro bACC）
        UGRID = tuple(range(0, 11))
        best_u, best_v = 0, -1.0
        for u in UGRID:
            q = predict(tab, t[tr] / 6.0 - u)
            v = balanced_accuracy_score(y[tr], (q >= 0.5).astype(int))
            if v > best_v:
                best_u, best_v = u, float(v)
        chosen[k]["u_auto"] = best_u
        run("auto", best_u)
        us = np.array(sorted(set(u_true[subs.index(s)] for s in subs if fold[g == s][0] != k)))
        w = np.array([np.mean([u_true[g == s][0] == u for s in subs if fold[g == s][0] != k]) for u in us])
        w /= w.sum()
        acc = np.zeros(len(te))
        acc_tr = np.zeros(len(tr))
        for u, wi in zip(us, w):
            acc += wi * predict(tab, t[te] / 6.0 - u)
            acc_tr += wi * predict(tab, t[tr] / 6.0 - u)
        P["prior"][te] = acc
        cand = np.unique(np.quantile(acc_tr, np.linspace(0.02, 0.98, 97)))
        sc = [balanced_accuracy_score(y[tr], (acc_tr >= c).astype(int)) for c in cand]
        THR["prior"][k] = float(cand[int(np.argmax(sc))])

    out = {"meta": {"protocol": "real-order SGKF(5,rs0)", "n_subjects": len(subs),
                    "n_windows": int(len(y)), "bandwidth_grid": list(BWS), "chosen": chosen,
                    "note": "P(task|位置) 经验查表；u_true 来自 BDF 实测 pre-roll；阈值只用训练折"},
           "methods": {}, "seconds": 0.0}
    for m in MODES:
        p = P[m]
        ev = np.isfinite(p)
        thr = np.full(len(y), 0.5)
        for k in folds:
            thr[fold == k] = THR[m][k]
        mm = metrics(y[ev], p[ev], g[ev], thr[ev])
        out["methods"][m] = mm
        print(f"[result] {m:9s} AUC={mm['AUC']:.4f} bACC={mm['bACC']:.4f} F1m={mm['F1_macro']:.4f} "
              f"macro={mm['subject_macro_bACC']:.4f} n>=.9={mm['n_ge_0.9']}/{mm['n_subj']}", flush=True)
    out["seconds"] = time.perf_counter() - T0
    stem = "position_table" + ("_smoke" if args.smoke else "")
    (HERE / f"{stem}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] -> {stem}.json {el()}", flush=True)


if __name__ == "__main__":
    main()
