"""黑盒评估：方案只提供 (t, score)，标签一律由 truth 生成。

指标（逐被试 + 宏平均）：bACC（阈值 0，score>0 判 task）、AUC、F1、bACC≥0.9 人数。
覆盖率 = 1 s 参考网格的可评估时刻中，在 ±0.5 s 内有方案输出的比例；缺失帧不计入指标。
"""
from __future__ import annotations

import numpy as np

from . import truth


def metrics(y: np.ndarray, s: np.ndarray, thr: float = 0.0) -> dict | None:
    from sklearn.metrics import f1_score, roc_auc_score
    m = (y >= 0) & np.isfinite(s)
    y, s = y[m].astype(int), s[m]
    if len(y) == 0 or y.min() == y.max():
        return None
    p = (s > thr).astype(int)
    return {"bacc": float(0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())),
            "auc": float(roc_auc_score(y, s)), "f1": float(f1_score(y, p)), "n": int(len(y))}


def coverage(source: str, sid: str, t: np.ndarray) -> float:
    ref, _ = truth.reference_grid(source, sid)
    if len(ref) == 0:
        return float("nan")
    t = np.sort(np.asarray(t, dtype=np.float64))
    if len(t) == 0:
        return 0.0
    i = np.searchsorted(t, ref)
    lo, hi = t[np.clip(i - 1, 0, len(t) - 1)], t[np.clip(i, 0, len(t) - 1)]
    d = np.minimum(np.abs(lo - ref), np.abs(hi - ref))
    return float(np.mean(d <= 0.5 + 1e-6))


def subject(source: str, sid: str, t: np.ndarray, score: np.ndarray) -> dict:
    y = truth.labels(source, sid, t)
    r = {"subject": sid, "coverage": coverage(source, sid, t)}
    m = metrics(y, np.asarray(score, dtype=np.float64))
    if m:
        r.update(m)
    return r


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if "bacc" in r]
    if not ok:
        return {}
    out = {k: float(np.mean([r[k] for r in ok])) for k in ("bacc", "auc", "f1", "coverage")}
    out["n_ge_09"] = int(sum(r["bacc"] >= 0.9 for r in ok))
    out["n_subjects"] = len(ok)
    return out
