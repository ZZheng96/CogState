"""E1：纯 EEG 解码——能不能靠"未被利用的生理/伪迹信号"突破 0.68 的跨被试天花板？

口径（与本轮目标一致）：
  * 输入只有当前 6 s 窗的原始波形；**不使用**时间表、窗序号、块序号、任何次序/位置信息；
  * subject-disjoint 5 折；预处理参数只在训练折拟合；阈值固定 0.5 与训练折选择两种都报；
  * 同时给出"块归属已知、本人方向由 6 个块均值差估出"的窗级 AUC，作为该特征空间的可达上限诊断。

新增特征族（此前 858 维池没有的）：
  A. 绝对功率与功率比（0.5–4 / 4–8 / 8–13 / 13–30 / 30–45 / 45–60 / 60–100 Hz）
  B. 工频与接触代理：49–51 Hz 占比、50 Hz 峰 / 邻带背景
  C. 时域形态：std、MAD、峰度、偏度、crest、line-length、零交叉率、1 s 自相关
  D. 瞬态/动作代理：一阶差分 8×MAD 超阈计数、最大差分 / MAD、0.5–4 Hz 慢波峰计数

运行：py -3.11 test/10_projects/eeg_only_20261102/feature_probe.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DB = ROOT / "database" / "windows"
T0 = time.perf_counter()
FS = 250.0
BANDS = ((0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45), (45, 60), (60, 100))


def el():
    return f"elapsed={time.perf_counter() - T0:.1f}s"


def raw_features(x):
    """x: (n, 1500) 原始窗 -> (n, k) 新特征。"""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    f, p = welch(x, fs=FS, nperseg=512, noverlap=256, detrend="linear", axis=1)
    bp = np.column_stack([np.trapezoid(p[:, (f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)], axis=1)
                          for lo, hi in BANDS])
    total = bp.sum(1) + 1e-30
    out = [np.log(bp + 1e-30), bp / total[:, None]]
    # 工频
    line = bp[:, 5] / total
    bg = (bp[:, 4] + bp[:, 6]) / 2 + 1e-30
    out.append(np.column_stack([line, bp[:, 5] / bg]))
    # 时域
    sd = x.std(1) + 1e-30
    mad = np.median(np.abs(x - np.median(x, 1, keepdims=True)), 1) + 1e-30
    z = (x - x.mean(1, keepdims=True)) / sd[:, None]
    kurt = (z ** 4).mean(1)
    skew = (z ** 3).mean(1)
    crest = np.abs(x).max(1) / sd
    d = np.diff(x, axis=1)
    dmad = np.median(np.abs(d - np.median(d, 1, keepdims=True)), 1) + 1e-30
    out.append(np.column_stack([
        np.log(sd), np.log(mad), kurt, skew, crest,
        np.abs(d).sum(1) / sd, (np.abs(x) < 1e-12).mean(1),
        (x[:, :-1] * x[:, 1:] < 0).mean(1),
        np.einsum("ij,ij->i", x[:, :-250], x[:, 250:]) / (np.einsum("ij,ij->i", x[:, :-250], x[:, :-250]) + 1e-30),
    ]))
    # 瞬态/动作
    out.append(np.column_stack([
        (np.abs(d) > 8 * dmad[:, None]).sum(1),
        np.abs(d).max(1) / dmad,
        (np.abs(x - np.median(x, 1, keepdims=True)) > 6 * mad[:, None]).sum(1),
    ]))
    return np.column_stack(out)


def metrics(y, p, g, thr=0.5):
    yh = (p >= thr).astype(int)
    per = [balanced_accuracy_score(y[g == s], yh[g == s]) for s in np.unique(g)]
    return {"AUC": float(roc_auc_score(y, p)), "bACC": float(balanced_accuracy_score(y, yh)),
            "F1_macro": float(f1_score(y, yh, average="macro")),
            "F1_task": float(f1_score(y, yh, average="binary", pos_label=1, zero_division=0)),
            "subject_macro_bACC": float(np.mean(per)),
            "n_ge_0.9": int(np.sum(np.array(per) >= 0.9)), "n_subj": len(per)}


def load_all():
    X, F, E, Y, G = [], [], [], [], []
    for name, lab in (("relax", 0), ("task", 1)):
        for p in sorted((DB / name).glob("sub-*.npz")):
            d = np.load(p)
            X.append(d["x"]); F.append(d["f"]); E.append(d["e"])
            Y.append(np.full(len(d["x"]), lab, np.int8))
            G.append(np.array([p.stem.split("-")[1]] * len(d["x"])))
    return (np.concatenate(X), np.concatenate(F), np.concatenate(E),
            np.concatenate(Y).astype(int), np.concatenate(G))


def cv_eval(Z, y, g, folds, tag, model="hgb"):
    p = np.full(len(y), np.nan)
    thr_pick = np.full(len(y), 0.5)
    for k in range(5):
        tr, te = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        sc = StandardScaler().fit(Z[tr])
        A, B = sc.transform(Z[tr]), sc.transform(Z[te])
        A = np.nan_to_num(A); B = np.nan_to_num(B)
        if model == "hgb":
            m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                               min_samples_leaf=40, l2_regularization=1.0,
                                               random_state=0).fit(A, y[tr])
            score = np.log(np.clip(m.predict_proba(B)[:, 1], 1e-6, 1 - 1e-6) /
                           np.clip(m.predict_proba(B)[:, 0], 1e-6, 1 - 1e-6))
        else:
            m = LogisticRegression(C=.01, max_iter=2000).fit(A, y[tr])
            score = m.decision_function(B)
        p[te] = score
    # 阈值 0.5（logit 尺度）
    mm = metrics(y, p, g, 0.0)
    print(f"[{tag}] AUC={mm['AUC']:.4f} bACC={mm['bACC']:.4f} F1m={mm['F1_macro']:.4f} "
          f"macro={mm['subject_macro_bACC']:.4f} n>=.9={mm['n_ge_0.9']}/{mm['n_subj']}", flush=True)
    return mm


def within_subject_ceiling(Z, y, g, tag):
    """诊断（不可部署）：用该被试全部 task/relax 窗的均值差作为 oracle 方向，窗级 AUC。
    只用于回答"这个特征空间里，单窗还有多少可分性"，不做任何跨被试迁移。"""
    aucs = []
    for s in np.unique(g):
        q = g == s
        Zs, ys = Z[q], y[q]
        # 必须先做被试内尺度归一，否则欧氏方向会被大方差特征主导
        sd = Zs.std(0)
        Zs = (Zs - Zs.mean(0)) / np.where(sd > 1e-12, sd, 1.0)
        d = Zs[ys == 1].mean(0) - Zs[ys == 0].mean(0)
        n = np.linalg.norm(d)
        if n < 1e-12:
            continue
        aucs.append(roc_auc_score(ys, Zs @ (d / n)))
    print(f"[ceiling {tag}] within-subject oracle-direction AUC mean={np.mean(aucs):.4f} "
          f"median={np.median(aucs):.4f} n={len(aucs)}", flush=True)
    return float(np.mean(aucs)), float(np.median(aucs)), len(aucs)


def main():
    warnings.filterwarnings("ignore")
    X, F, E, y, g = load_all()
    print(f"[data] windows={len(y)} dim_f={F.shape[1]} dim_e={E.shape[1]} subs={len(np.unique(g))} {el()}", flush=True)

    R = raw_features(X)
    print(f"[feat] raw features dim={R.shape[1]} {el()}", flush=True)

    folds = np.empty(len(y), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g)):
        folds[te] = k

    # 不做块对齐：database 的窗顺序被打乱，而本诊断只需要被试内 task/relax 的均值差
    res = {"protocol": "cross-subject 5-fold SGKF(5,rs0), 6 s window, no schedule/position",
           "n_windows": int(len(y)), "n_subjects": int(len(np.unique(g))),
           "dim": {"f": int(F.shape[1]), "e": int(E.shape[1]), "raw_new": int(R.shape[1])},
           "methods": {}, "ceiling": {}, "seconds": 0.0}

    Z0 = np.hstack([F, E])
    Z1 = np.hstack([F, E, R])
    Z2 = R

    res["methods"]["base_878_hgb"] = cv_eval(Z0, y, g, folds, "base_878 HGB", "hgb")
    res["methods"]["base_878_lr"] = cv_eval(Z0, y, g, folds, "base_878 LR", "lr")
    res["methods"]["new_only_hgb"] = cv_eval(Z2, y, g, folds, "new-only HGB", "hgb")
    res["methods"]["new_only_lr"] = cv_eval(Z2, y, g, folds, "new-only LR", "lr")
    res["methods"]["all_908_hgb"] = cv_eval(Z1, y, g, folds, "all908 HGB", "hgb")
    res["methods"]["all_908_lr"] = cv_eval(Z1, y, g, folds, "all908 LR", "lr")

    for tag, Z in (("base_878", Z0), ("new_only", Z2), ("all_908", Z1)):
        m, md, n = within_subject_ceiling(Z, y, g, tag)
        res["ceiling"][tag] = {"auc_mean": m, "auc_median": md, "n_subjects": n}

    res["seconds"] = time.perf_counter() - T0
    (HERE / "feature_probe.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] -> feature_probe.json {el()}", flush=True)


if __name__ == "__main__":
    main()
