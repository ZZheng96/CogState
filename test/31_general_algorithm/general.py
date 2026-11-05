"""E31：固定配置的通用池化认知状态解码器。

方法：E27 cat 多窗因果特征 → 300 帧因果去均值 → 留一被试池化加权 LR
      → 校准段分数 z 标准化 → 过去 300 帧均值阈值。
所有被试共用同一超参数；034 缺少静息校准标签，自动退化为纯池化模型。

缓存验证：.venv\Scripts\python.exe test/31_general_algorithm/general.py
重新训练：.venv\Scripts\python.exe test/31_general_algorithm/general.py --recompute
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402


def _import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E26 = _import("e26lib", TESTS / "26_merge2" / "e26lib.py")
E27 = _import("e27", TESTS / "27_common_individual" / "e27.py")

DEV = "cuda"
FINAL_NAME = "b|pool|w300|a0.75"
FINAL_THRESHOLD = "m300"
CENTER_W = 300
THRESHOLD_W = 300
ALPHA = 0.75
LAM = 0.01
SINGLE_CLASS_CAL = "034"


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "bacc": float(np.nanmean(values[:, 0])),
        "auc": float(np.nanmean(values[:, 1])),
        "f1": float(np.nanmean(values[:, 2])),
        "n90": int(np.nansum(values[:, 0] >= 0.9)),
        "min_bacc": float(np.nanmin(values[:, 0])),
    }


def per_subject(subs: list[str], values: np.ndarray) -> dict:
    return {
        q: {"bacc": round(float(v[0]), 6), "auc": round(float(v[1]), 6), "f1": round(float(v[2]), 6)}
        for q, v in zip(subs, values)
    }


def validate_cache() -> dict:
    """配置和阈值均只对其余 56 人择优，再评价被留出的被试。"""
    data = np.load(HERE.parent / "28_block_invariant" / "e28_ev.npz")
    ev = data["EV"]
    names = data["names"].tolist()
    subs = data["subs"].tolist()
    pool = [i for i, name in enumerate(names) if name.startswith("b|pool|")]
    pool_bacc = ev[pool, :, :, 0]
    nested = []
    picks = []
    for j, q in enumerate(subs):
        other_mean = np.nanmean(np.delete(pool_bacc, j, axis=1), axis=1)
        config_i, threshold_i = np.unravel_index(np.nanargmax(other_mean), other_mean.shape)
        nested.append(ev[pool[config_i], j, threshold_i])
        picks.append((names[pool[config_i]], E26.THR2[threshold_i][0]))
    nested = np.asarray(nested)
    threshold_i = [name for name, _, _ in E26.THR2].index(FINAL_THRESHOLD)
    fixed = ev[names.index(FINAL_NAME), :, threshold_i]
    result = {
        "protocol": "LOSO；池化模型排除目标被试后段标签；本人仅用块0-3校准；配置和阈值由其余56人选择",
        "final": {"name": FINAL_NAME, "threshold": FINAL_THRESHOLD},
        "nested_pool": {
            **summary(nested),
            "per_subject": per_subject(subs, nested),
            "config_counts": dict(Counter(name for name, _ in picks)),
            "threshold_counts": dict(Counter(name for _, name in picks)),
        },
        "fixed_final": {**summary(fixed), "per_subject": per_subject(subs, fixed)},
    }
    (HERE / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def causal_mean(score: torch.Tensor, window: int) -> torch.Tensor:
    """过去 window 帧（含当前帧）的均值。"""
    index = torch.arange(1, len(score) + 1, device=score.device)
    low = (index - window).clamp(min=0)
    cumulative = torch.nn.functional.pad(score.cumsum(0), (1, 0))
    count = (index - low).float()
    return (cumulative[index] - cumulative[low]) / count


def prepare_features():
    """重建 E28 的 cat 特征和 300 帧因果去均值，数据始终驻留 GPU。"""
    subs, raw = L.load()
    features = E27.build_set("cat", raw, subs)
    centered = {}
    for q in subs:
        mean, _ = L.run_stats_t(features[q], CENTER_W)
        centered[q] = features[q] - mean
    return subs, raw, centered


def prepare_pool(subs: list[str], raw: dict, features: dict) -> dict:
    lengths = [len(raw[q][1]) for q in subs]
    return {
        "x": torch.cat([features[q] for q in subs]),
        "y": np.concatenate([raw[q][1] for q in subs]),
        "sid": torch.tensor(np.concatenate([[i] * n for i, n in enumerate(lengths)]), device=DEV),
        "offsets": np.concatenate([[0], np.cumsum(lengths)]),
        "base_weight": torch.cat([E27.bal_w(raw[q][1]) for q in subs]),
    }


def fit_target(target: str, subs: list[str], raw: dict, features: dict, pool: dict):
    """用其余人全部块和本人校准块训练；返回目标被试的连续分数。"""
    target_i = subs.index(target)
    others = (pool["sid"] != target_i).float()
    other_weight = pool["base_weight"] * others / (len(subs) - 1)
    if target != SINGLE_CLASS_CAL:
        calibration = torch.tensor(raw[target][2] <= 3, device=DEV)
        own_weight = torch.zeros_like(other_weight)
        start, end = int(pool["offsets"][target_i]), int(pool["offsets"][target_i + 1])
        own_weight[start:end] = E27.bal_w(raw[target][1], calibration)
        sample_weight = (1 - ALPHA) * other_weight + ALPHA * own_weight
    else:
        calibration = torch.tensor(raw[target][2] == 3, device=DEV)
        sample_weight = other_weight
    model = E27.lbfgs_lr(
        pool["x"], pool["y"], sample_weight, torch.tensor([LAM], device=DEV), iters=150
    )
    return E27.zc(model(features[target])[:, 0], calibration)


def recompute() -> dict:
    start_time = time.time()
    subs, raw, features = prepare_features()
    pool = prepare_pool(subs, raw, features)
    rows = []
    for i, q in enumerate(subs):
        score = fit_target(q, subs, raw, features, pool)
        block = raw[q][2]
        test = torch.tensor(block >= 4, device=DEV)
        y = torch.tensor(raw[q][1][block >= 4], device=DEV).bool()
        adjusted = score - causal_mean(score, THRESHOLD_W)
        metric = E26.metrics(adjusted[None], torch.where(test)[0], y, qws=())[0, 0]
        rows.append(metric.cpu().numpy())
        print(f"[{i + 1:02d}/{len(subs)}] {q} bACC={rows[-1][0]:.3f} AUC={rows[-1][1]:.3f} F1={rows[-1][2]:.3f}", flush=True)
        del score
        torch.cuda.empty_cache()
    values = np.asarray(rows)
    result = {
        "config": {"center_w": CENTER_W, "threshold_w": THRESHOLD_W, "alpha": ALPHA, "lambda": LAM},
        **summary(values),
        "per_subject": per_subject(subs, values),
        "elapsed_s": round(time.time() - start_time, 1),
    }
    (HERE / "recomputed.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="E31 固定配置通用池化解码器")
    parser.add_argument("--recompute", action="store_true", help="从缓存特征重新训练全部57个留一被试模型")
    args = parser.parse_args()
    cached = validate_cache()
    for name in ("nested_pool", "fixed_final"):
        value = cached[name]
        print(f"{name}: bACC={value['bacc']:.3f} AUC={value['auc']:.3f} F1={value['f1']:.3f} "
              f"n>=.9={value['n90']} min={value['min_bacc']:.3f}")
    if args.recompute:
        value = recompute()
        print(f"recomputed: bACC={value['bacc']:.3f} AUC={value['auc']:.3f} F1={value['f1']:.3f} "
              f"n>=.9={value['n90']} min={value['min_bacc']:.3f}")


if __name__ == "__main__":
    main()
