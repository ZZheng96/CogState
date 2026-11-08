"""查看器的算法方案注册表。

每个方案 = 一组逐帧决策分数（每 1 s 一帧，窗口 [t-6, t]，分数 > 0 判为 task），按被试缓存为 npz
（字段 t / y / 若干分数列）。注册方式：

    register("key", label="...", desc="...", file=lambda sid: Path(...), field="post",
             compute=lambda sid: ...)   # compute 可选：缓存缺失时按需生成 npz

指标为黑盒统一口径：npz 里的 y 字段被忽略，标签由 system.truth 按 BDF+events 生成，
指标与覆盖率由 system.evaluate 计算。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

import model_eval as ME

if str(ME.ROOT) not in sys.path:
    sys.path.insert(0, str(ME.ROOT))
from system import evaluate as EV, truth as TR  # noqa: E402

REGISTRY: dict[str, dict] = {}


def register(key: str, *, label: str, desc: str, file: Callable[[str], Path], field: str,
             compute: Callable[[str], object] | None = None, group: str = "") -> None:
    REGISTRY[key] = {"key": key, "label": label, "desc": desc, "file": file, "field": field,
                     "compute": compute, "group": group}


def info() -> list[dict]:
    return [{k: v for k, v in s.items() if k in ("key", "label", "desc", "field", "group")} for s in REGISTRY.values()]


def get(key: str) -> dict:
    if key not in REGISTRY:
        raise ValueError(f"未知方案：{key}")
    return REGISTRY[key]


def load(key: str, sid: str, compute_missing: bool = False) -> dict | None:
    s = get(key)
    f = s["file"](sid)
    if not f.exists():
        if not (compute_missing and s["compute"]):
            return None
        s["compute"](sid)
    z = np.load(f)
    t = z["t"].astype(np.float64)
    # 黑盒统一口径：方案只提供 (t, score)，y 一律由 system.truth 生成
    return {"t": t, "y": TR.labels("builtin", sid, t), "score": z[s["field"]]}


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


# ---------------------------------------------------------------- 已注册方案
_insample = lambda sid: ME.CACHE / f"pred_{sid}.npz"  # noqa: E731
_cv = lambda sid: ME.CACHE / "e17_cv" / f"pred_{sid}.npz"  # noqa: E731
_POST = "因果 EMA 去趋势(hl=1200 s) + 前向 HMM(dwell=300 s)"

register("current", group="训练集内", file=_insample, field="post", compute=ME.compute,
         label="当前方案 src/infer.py（E17 + 后处理）",
         desc=f"E17 多尺度 LR+MLP，全部 57 人训练；{_POST}。被评估被试在训练集中，指标偏乐观。")
register("e17_raw", group="训练集内", file=_insample, field="raw", compute=ME.compute,
         label="E17 原始分数（无后处理）",
         desc="同一 model_bundle.pt 的逐帧原始分数，不做跨帧平滑。被评估被试在训练集中。")
register("e17_cv_post", group="跨被试", file=_cv, field="post",
         label="E17 跨被试 5 折 + 后处理",
         desc=f"每个被试由不含它的 4 折模型预测（system/web/build_oof.py）；{_POST}。")
register("e17_cv_raw", group="跨被试", file=_cv, field="raw",
         label="E17 跨被试 5 折（无后处理）",
         desc="每个被试由不含它的 4 折模型预测，逐帧原始分数。最接近未见被试的真实成绩。")

_e44 = lambda sid: ME.CACHE / "e44" / f"pred_{sid}.npz"  # noqa: E731
register("e44_dt1200_hmm", group="跨被试 LOSO", file=_e44, field="dt1200_hmm",
         label="E44 最佳：E42 + EMA 去趋势 + HMM",
         desc="E42 的 57 人 LOSO 分数，先减去因果 EMA 均值（hl=1200 帧），再做前向 HMM（dwell=300）。"
              "来源 test/44_late_drift，由 system/web/build_e44.py 生成。")
register("e42_hmm", group="跨被试 LOSO", file=_e44, field="hmm",
         label="E42 基线：LOSO + HMM",
         desc="E17 特征 LR+3MLP 的 57 人 LOSO 分数，只做前向 HMM（dwell=300），不去趋势。")
register("e42_raw", group="跨被试 LOSO", file=_e44, field="raw",
         label="E42 LOSO 原始分数",
         desc="E42 的 57 人 LOSO 逐帧融合分数，不做跨帧后处理。")

_E76_NESTED = ME.ROOT / "test" / "76_conditioned_fusion" / "nested_loso" / "cache"
_e76n = lambda sid: next(iter(sorted(_E76_NESTED.glob(f"fold_*_{sid}.npz"))), _E76_NESTED / f"fold_00_{sid}.npz")  # noqa: E731
register("e76_nested_shrink", group="跨被试 LOSO", file=_e76n, field="score_conditioned_shrink",
         label="E76 嵌套LOSO conditioned_shrink（0.8417/0.7748/0.7887）",
         desc="57 折嵌套 LOSO：外折被试完全不参与 LR/MLP/门控训练；0.5×门控 + 0.5×(lr+mlp)，"
              "再做 EMA 去趋势(1200) + HMM(300)。来源 test/76_conditioned_fusion/nested_loso/nested_loso.py。")
