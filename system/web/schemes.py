"""查看器的算法方案注册表。

方案直接对应 system.runner.RUNNABLE 的键（current / e17_raw / e76_gated）。逐帧分数（每 1 s 一帧，
窗口 [t-6, t]，分数 > 0 判为 task）由 runner.get_or_run 用系统内部算法 system/algorithms/ 计算，
缓存于 system/cache/runs/<key>/builtin/<sid>.npz（带版本指纹）。不读取任何外部缓存或文档。

口径：模型用全部 57 人训练，在训练集上推理（训练集内 / in-sample）。不含 LOSO / 交叉验证方案。
标签由 system.truth 按 BDF+events 生成，指标与覆盖率由 system.evaluate 计算。
"""
from __future__ import annotations

import model_eval as ME
from system import evaluate as EV, runner as RN, truth as TR

GROUP = "57人全量训练（训练集内）"
REGISTRY: dict[str, dict] = {}

_DESC = {
    "current": "E17 多尺度 LR+3MLP（57 人全量训练）；因果 EMA 去趋势(hl=1200 s) + 前向 HMM(dwell=300 s)。",
    "e17_raw": "同一 E17 bundle 的逐帧原始分数，不做跨帧平滑。",
    "e76_gated": "E17 的 LR/MLP 分量 + E76 因果门控（门控 LR 在 57 人训练集内拟合，"
                 "system/algorithms/fit_e76_gate.py），0.5×门控 + 0.5×(lr+mlp)，再 EMA1200 + HMM300。",
}


def register(key: str, *, label: str, desc: str, field: str = "score", group: str = GROUP) -> None:
    REGISTRY[key] = {"key": key, "label": label, "desc": desc, "field": field, "group": group}


for _k in ("current", "e17_raw", "e76_gated"):
    register(_k, label=RN.RUNNABLE[_k]["label"], desc=_DESC[_k] + "被评估被试在训练集中，指标偏乐观。")


def info() -> list[dict]:
    return [dict(s) for s in REGISTRY.values()]


def get(key: str) -> dict:
    if key not in REGISTRY:
        raise ValueError(f"未知方案：{key}")
    return REGISTRY[key]


def load(key: str, sid: str, compute_missing: bool = False) -> dict | None:
    s = get(key)
    d = ME.load(key, sid, compute_missing=compute_missing)
    if d is None:
        return None
    t = d["t"].astype("float64")
    return {"t": t, "y": TR.labels("builtin", sid, t), "score": d[s["field"]]}


def subject_metrics(key: str, sid: str) -> dict | None:
    d = load(key, sid)
    return None if d is None else EV.subject("builtin", sid, d["t"], d["score"])


def summary(key: str, sids: list[str]) -> dict:
    rows = [r for r in (subject_metrics(key, s) for s in sids) if r and "bacc" in r]
    out = {"scheme": {k: v for k, v in get(key).items() if k in ("key", "label", "desc")},
           "n_done": len(rows), "n_total": len(sids), "subjects": rows}
    if rows:
        out["mean"] = EV.aggregate(rows)
    return out
