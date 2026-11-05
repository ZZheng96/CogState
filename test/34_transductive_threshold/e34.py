"""E34：仅用无标签测试段分布的转导阈值实验。

候选分数来自 E28（各模型训练时不使用目标被试测试标签）。阈值直接由完整测试段的
分数分布计算，因此是非因果方法；所有阈值仍不使用测试标签。配置与阈值方法的折外
评价采用 LOSO：每个目标被试均由其余 56 人的平均测试 bACC 选择。

运行：.venv\Scripts\python.exe test/34_transductive_threshold/e34.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "17_compliant"))
import loso as L  # noqa: E402

DEV = "cuda"
QS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
METHODS = ["mean", "kmeans", "block_mean", "block_median"] + [f"q{q:g}" for q in QS]
ADJS = ("raw", "m300", "m600", "m1200")


def causal_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    index = torch.arange(1, x.shape[1] + 1, device=x.device)
    low = (index - window).clamp(min=0)
    cumulative = torch.nn.functional.pad(x.cumsum(1), (1, 0))
    return (cumulative[:, index] - cumulative[:, low]) / (index - low).float()


def summary(a: np.ndarray) -> dict:
    return {
        "bacc": float(np.nanmean(a[:, 0])),
        "auc": float(np.nanmean(a[:, 1])),
        "f1": float(np.nanmean(a[:, 2])),
        "n90": int(np.nansum(a[:, 0] >= 0.9)),
        "min_bacc": float(np.nanmin(a[:, 0])),
    }


def kmeans_threshold(x: torch.Tensor) -> torch.Tensor:
    """逐行一维二聚类，返回两个聚类中心的中点。"""
    lo = x.quantile(0.25, dim=1)
    hi = x.quantile(0.75, dim=1)
    for _ in range(30):
        d = torch.stack([(x - lo[:, None]).abs(), (x - hi[:, None]).abs()], dim=2)
        near_hi = d[..., 1] < d[..., 0]
        count = near_hi.sum(1).float()
        new_hi = (x * near_hi).sum(1) / count.clamp(min=1)
        new_lo = (x * ~near_hi).sum(1) / (~near_hi).sum(1).float().clamp(min=1)
        if torch.max((new_lo - lo).abs() + (new_hi - hi).abs()) < 1e-5:
            lo, hi = new_lo, new_hi
            break
        lo, hi = new_lo, new_hi
    return (lo + hi) / 2


def metric(x: torch.Tensor, threshold: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    pred = x > threshold[:, None]
    tp = (pred & y).sum(1).float()
    fp = (pred & ~y).sum(1).float()
    npos, nneg = y.sum().float(), (~y).sum().float()
    bacc = 0.5 * (tp / npos + 1 - fp / nneg)
    f1 = 2 * tp / (2 * tp + fp + npos - tp)
    rank = x.argsort(1).argsort(1).float() + 1
    auc = (rank[:, y].sum(1) - npos * (npos + 1) / 2) / (npos * nneg)
    out = torch.stack([bacc, auc, f1], 1)
    out[torch.isnan(x).any(1)] = float("nan")
    return out.cpu().numpy()


def evaluate() -> tuple[np.ndarray, list[str]]:
    subs, raw = L.load()
    archive = np.load(HERE.parent / "28_block_invariant" / "e28_scores.npz")
    names = archive["names"].tolist()
    nq, nr = len(subs), len(names)
    out = np.full((nq, nr, len(ADJS), len(METHODS), 3), np.nan, dtype=np.float32)
    for i, q in enumerate(subs):
        y_np, block = raw[q][1], raw[q][2]
        test = block >= 4
        full = torch.tensor(archive[f"s{q}"], device=DEV, dtype=torch.float32)
        y = torch.tensor(y_np[test], device=DEV).bool()
        b4 = torch.tensor(block[test] == 4, device=DEV)
        b5 = ~b4
        for a, adjustment in enumerate(ADJS):
            adjusted = full if adjustment == "raw" else full - causal_mean(full, int(adjustment[1:]))
            x = adjusted[:, test]
            thresholds = {
                "mean": x.mean(1),
                "kmeans": kmeans_threshold(x),
                "block_mean": (x[:, b4].mean(1) + x[:, b5].mean(1)) / 2,
                "block_median": (x[:, b4].median(1).values + x[:, b5].median(1).values) / 2,
            }
            for quantile in QS:
                thresholds[f"q{quantile:g}"] = x.quantile(quantile, dim=1)
            for j, method in enumerate(METHODS):
                out[i, :, a, j] = metric(x, thresholds[method], y)
        print(f"[{i + 1:02d}/{nq}] {q}", flush=True)
    return out, subs


def main() -> None:
    ev, subs = evaluate()
    names = np.load(HERE.parent / "28_block_invariant" / "e28_scores.npz")["names"].tolist()
    np.savez_compressed(HERE / "e34_ev.npz", EV=ev, subs=subs, names=names, methods=METHODS, adjustments=ADJS)
    nq = len(subs)
    result = {"protocol": "因果去均值后，用无标签完整测试段二次定阈（非因果转导）；LOSO 选择候选、调整与阈值方法",
              "methods": METHODS, "adjustments": ADJS}
    # 所有人共用同一配置；此处用于展示固定配置上限，含模型选择偏差。
    mean_bacc = np.nanmean(ev[..., 0], axis=0)
    r, a, m = np.unravel_index(int(np.nanargmax(mean_bacc)), mean_bacc.shape)
    fixed = ev[:, r, a, m]
    result["fixed_best_in_sample_selection"] = {
        "name": names[r], "adjustment": ADJS[a], "method": METHODS[m], **summary(fixed),
        "per_subject": {q: fixed[i].tolist() for i, q in enumerate(subs)},
    }
    # 严格折外选择：目标被试完全不参与配置/阈值选择。
    nested = []
    picks = {}
    for i, q in enumerate(subs):
        objective = np.nanmean(np.delete(ev[..., 0], i, axis=0), axis=0)
        r, a, m = np.unravel_index(int(np.nanargmax(objective)), objective.shape)
        nested.append(ev[i, r, a, m])
        picks[q] = {"name": names[r], "adjustment": ADJS[a], "method": METHODS[m]}
    nested = np.asarray(nested)
    result["nested_loso"] = {**summary(nested), "per_subject": {q: nested[i].tolist() for i, q in enumerate(subs)}}
    result["picks"] = picks
    # 测试标签 oracle 只作候选集上限，不可作为通用结果。
    oracle_i = np.nanargmax(np.where(np.isnan(ev[..., 0]), -np.inf, ev[..., 0]).reshape(nq, -1), axis=1)
    oracle = ev.reshape(nq, -1, 3)[np.arange(nq), oracle_i]
    result["test_oracle"] = summary(oracle)
    (HERE / "e34.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for key in ("fixed_best_in_sample_selection", "nested_loso", "test_oracle"):
        v = result[key]
        print(f"{key}: bACC={v['bacc']:.4f} AUC={v['auc']:.4f} F1={v['f1']:.4f} n90={v['n90']} min={v['min_bacc']:.4f}")
    v = result["fixed_best_in_sample_selection"]
    print("fixed:", v["name"], v["adjustment"], v["method"])


if __name__ == "__main__":
    main()
