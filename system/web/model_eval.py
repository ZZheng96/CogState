"""查看器的方案计算入口：全部委托给 system.runner（系统内部算法 system/algorithms/）。

每个方案按 runner.get_or_run 从记录首样本起流式推理（每 1 s 一个决策，窗口 = 过去 6 s），
结果缓存在 system/cache/runs/<key>/builtin/<sid>.npz（带版本指纹，模型文件变化即重算）。
口径：模型用全部 57 人训练，在训练集上推理（训练集内 / in-sample）。

用法：
  python system/web/model_eval.py --all [--scheme current]   # 预计算全部被试
  python system/web/model_eval.py --subject 018
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # 让 system 包可导入（只加仓库根，不加 src/test）
    sys.path.insert(0, str(ROOT))

from system import dataset, evaluate, runner  # noqa: E402

BUNDLE_PATH = runner.E17_BUNDLE


def compute(key: str, sid: str, force: bool = False) -> dict:
    d, _ = runner.get_or_run(key, "builtin", sid, force=force)
    return d


def load(key: str, sid: str, compute_missing: bool = False) -> dict | None:
    d = runner.load_cache(key, "builtin", sid)
    if d is None and compute_missing:
        d = compute(key, sid)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scheme", default="current", choices=[k for k in runner.RUNNABLE if k != "demo_noncausal"])
    a = ap.parse_args()
    sids = [a.subject] if a.subject else dataset.subjects("builtin")
    rows = []
    for sid in sids:
        d = compute(a.scheme, sid)
        r = evaluate.subject("builtin", sid, d["t"], d["score"])
        rows.append(r)
        print(sid, json.dumps({k: r.get(k) for k in ("auc", "f1", "bacc")}), flush=True)
    print(json.dumps(evaluate.aggregate(rows), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
