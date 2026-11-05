"""E2 / E47：LR(w300+w600) 与 GPU XGBoost 的因果秩平均。"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
sys.path.insert(0, str(ROOT / "test" / "46_causal_online_norm"))
import loso as L  # noqa: E402
import e46  # noqa: E402


def causal_rank_logit(x):
    """当前帧在截至当前的流分布中的秩；无未来值、无目标标签。"""
    s = torch.as_tensor(x, dtype=torch.float32, device=L.dev)
    n = len(s)
    less = s[None, :] < s[:, None]
    equal = s[None, :] == s[:, None]
    mask = torch.tril(torch.ones((n, n), dtype=torch.bool, device=s.device))
    rank = ((less & mask).sum(1).float() + 0.5 * (equal & mask).sum(1).float()) / torch.arange(
        1, n + 1, device=s.device)
    return torch.logit(rank.clamp(1e-4, 1 - 1e-4)).cpu().numpy()


def fit_xgb(x, y):
    train = xgb.QuantileDMatrix(x, label=y, max_bin=64)
    model = xgb.train({
        "tree_method": "gpu_hist", "predictor": "gpu_predictor", "max_bin": 64,
        "max_depth": 5, "eta": 0.08, "subsample": 0.8, "colsample_bytree": 0.35,
        "min_child_weight": 30, "objective": "binary:logistic", "eval_metric": "auc",
        "seed": 20261106, "verbosity": 0,
    }, train, num_boost_round=160)
    del train
    return model


def write_report(result):
    lines = [
        "# E2 / E47 异构秩集成", "", "## 结果", "",
        "57 人真 LOSO；LR 使用因果 w300+w600 拼接特征，XGBoost 使用同一特征矩阵。秩平均在每个目标被试内按截至当前的扩展窗计算，不读取未来分数。",
        "", "| 模型 | 全段 bACC | 全段 AUC | 全段 F1 | 晚段 bACC | 晚段 AUC | 晚段 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in result["models"].items():
        m, l = r["macro"], r["macro_late45"]
        lines.append(f"| {name} | {m['bACC']:.4f} | {m['AUC']:.4f} | {m['F1']:.4f} | {l['bACC']:.4f} | {l['AUC']:.4f} | {l['F1']:.4f} |")
    ens = result["models"]["rank_mean"]["macro"]
    xgb_result = result["models"]["xgb"]["macro"]
    lines += [
        "", f"因果秩集成的全段结果为 **bACC {ens['bACC']:.4f} / AUC {ens['AUC']:.4f} / F1 {ens['F1']:.4f}**。",
        f"该集成未超过单一 XGBoost（bACC {xgb_result['bACC']:.4f} / AUC {xgb_result['AUC']:.4f}），因此本数据上不支持秩平均增益。",
        "", "XGBoost 固定 160 轮；未使用目标被试做早停或超参选择。原始模型与集成均报告，避免只保留有利结果。",
        "", "## 复现", "", "```powershell",
        "& .\\.venv\\Scripts\\python.exe test\\47_rank_ensemble\\e47.py", "```", "",
        "结果：`e47.json`；逐人分数：`e47_scores.npz`。",
    ]
    (HERE / "报告_E47_异构秩集成.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start = time.time()
    subs, D = L.load()
    e46.SUBS = subs
    f300 = L.feats(D, subs, 300)
    f600 = L.feats(D, subs, 600)
    F = {q: torch.cat([f300[q], f600[q]], 1) for q in subs}
    del f300, f600
    Fcpu = {q: F[q].cpu().numpy() for q in subs}
    score_path = HERE / "e47_scores.npz"
    if score_path.exists():
        previous = np.load(score_path)
        scores = {k: previous[k] for k in previous.files}
        print(f"resume with {sum(k.startswith('lr__') for k in scores)} completed folds", flush=True)
    else:
        scores = {"subjects": np.asarray(subs)}
    result = {"experiment": "E2/E47", "protocol": "57-subject LOSO", "models": {},
              "xgb": {"rounds": 160, "max_depth": 5, "eta": 0.08, "colsample_bytree": 0.35}}
    for i, q in enumerate(subs):
        if all(f"{k}__{q}" in scores for k in ("lr", "xgb", "rank_mean")):
            print(f"skip completed {i + 1}/57 {q}", flush=True)
            continue
        train = [s for s in subs if s != q]
        xtr = np.concatenate([Fcpu[s] for s in train])
        ytr = np.concatenate([D[s][1] for s in train])
        lr = L.fit_lr(torch.cat([F[s] for s in train]), ytr, C=0.001, iters=150)
        slr = lr(F[q]).cpu().numpy()
        booster = fit_xgb(xtr, ytr)
        p = booster.predict(xgb.DMatrix(Fcpu[q]))
        sxgb = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
        scores[f"lr__{q}"] = slr
        scores[f"xgb__{q}"] = sxgb
        scores[f"rank_mean__{q}"] = 0.5 * (causal_rank_logit(slr) + causal_rank_logit(sxgb))
        del xtr, ytr, lr, booster
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{i + 1}/57 {q} elapsed={time.time() - start:.1f}s", flush=True)
        np.savez_compressed(HERE / "e47_scores.npz", **scores)
    for key in ("lr", "xgb", "rank_mean"):
        result["models"][key] = e46.evaluate_scores(D, scores, key)
        print(key, json.dumps(result["models"][key]["macro"]), flush=True)
    result["elapsed_s"] = time.time() - start
    (HERE / "e47.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)


if __name__ == "__main__":
    main()
