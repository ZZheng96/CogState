"""E52：严格因果、无标签个体方向自适应。

目标：验证是否可仅使用目标被试当前及过去已到达的无标签 EEG 特征，
对群体模型进行受限的个体方向适配。禁止使用目标被试标签、未来帧或整段统计量。

每个外层折：
1. 仅用其余被试标签训练 LR + 3-MLP 种子集成；
2. 目标被试按时间顺序得到 seed score；
3. 每 R 帧，仅用 seed 的历史高/低置信度帧生成伪标签，拟合岭方向；
4. 后续 R 帧使用 seed + alpha * individual_direction；
5. 固定配置，不用外层测试被试选参。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
import loso as L
from multiscale import fit_mlp

HERE = Path(__file__).resolve().parent
OUT = HERE / "result.json"
DEV = L.dev


def ridge_direction(X, pseudo, lam=1.0):
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Z = (X - mu) / sd
    eye = torch.eye(Z.shape[1], device=DEV)
    A = Z.T @ Z + lam * len(pseudo) * eye
    w = torch.linalg.solve(A, Z.T @ (pseudo - pseudo.mean()))
    return mu, sd, w


def adapt(F, seed, R=30, top=0.2, lam=1.0, alpha=0.5, warm=120):
    """严格前缀因果：第 t 帧只用 seed[:t] 和 F[:t]。"""
    n = len(seed)
    out = seed.clone()
    for t0 in range(warm, n, R):
        hist = seed[:t0]
        k = max(10, int(top * t0))
        hi = torch.topk(hist, k).indices
        lo = torch.topk(-hist, k).indices
        idx = torch.cat([hi, lo])
        p = torch.cat([torch.ones(k, device=DEV), -torch.ones(k, device=DEV)])
        mu, sd, w = ridge_direction(F[idx], p, lam)
        past = ((F[:t0] - mu) / sd) @ w
        current = ((F[t0:t0 + R] - mu) / sd) @ w
        current = (current - past.median()) / (past.std() + 1e-6)
        seed_now = seed[t0:t0 + R] / (hist.std() + 1e-6)
        out[t0:t0 + R] = seed_now + alpha * current
    return out


def score_metrics(y, s):
    return L.subj_metrics(y, s, torch.zeros_like(s))


def main():
    subs, D = L.load()
    rng = np.random.RandomState(0)
    perm = list(rng.permutation(subs))
    outer = [perm[k::5] for k in range(5)]
    # c300 是预先确定的低维适配特征；all 作为无标签自适应的对照。
    d0 = 878
    ws = (60, 120, 300, 600, 99999)
    Fw = {w: L.feats(D, subs, w) for w in ws}
    configs = {
        "c300_R30_top20_lam1_a05": (300, 30, 0.20, 1.0, 0.5),
        "c300_R60_top20_lam1_a05": (300, 60, 0.20, 1.0, 0.5),
        "all_R30_top20_lam1_a05": (None, 30, 0.20, 1.0, 0.5),
        "c300_R30_top35_lam1_a05": (300, 30, 0.35, 1.0, 0.5),
    }
    rows = {name: {} for name in configs}
    baseline = {}
    for fold, test in enumerate(outer):
        train = [q for q in subs if q not in test]
        X = torch.cat([Fw[300][q] for q in train])
        y = np.concatenate([D[q][1] for q in train])
        flr = L.fit_lr(X, y, 0.001)
        fm = [fit_mlp(X, y, seed=s) for s in range(3)]
        del X
        for q in test:
            seed_lr = flr(Fw[300][q])
            seed_m = torch.stack([f(Fw[300][q]) for f in fm]).mean(0)
            seed = seed_lr / seed_lr.std() + seed_m / seed_m.std()
            baseline[q] = score_metrics(D[q][1], seed)
            for name, (scale, R, top, lam, alpha) in configs.items():
                F = Fw[300][q] if scale == 300 else torch.cat([Fw[w][q] for w in ws], 1)
                rows[name][q] = score_metrics(D[q][1], adapt(F, seed, R, top, lam, alpha))
        print(f"fold {fold} done", flush=True)
        del flr, fm
        torch.cuda.empty_cache()
    def macro(d):
        return {k: float(np.mean([v[k] for v in d.values()])) for k in ("bacc", "auc", "f1")}
    result = {"protocol": "5-fold OOF, fixed config, strict prefix-causal, no target labels", "n_subjects": len(subs),
              "baseline": macro(baseline), "configs": {k: macro(v) for k, v in rows.items()},
              "per_subject": {"baseline": baseline, **rows}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"baseline": result["baseline"], "configs": result["configs"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
