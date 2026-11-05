"""E42：按真实流式口径复核 E17，并测试严格前向 HMM。

旧 E17 离线融合用目标整段 score.std() 平衡 LR/MLP；这里改为仅由训练折
估计 std_lr/std_mlp，与 streaming.py 的实际部署一致。不使用目标标签、块号或
完整目标记录统计。先运行固定 5 折 OOF；--outer 57 可执行逐人 LOSO。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402
from multiscale import fit_mlp  # noqa: E402


def import_e41():
    spec = importlib.util.spec_from_file_location("e41lib", TESTS / "41_causal_adaptation" / "e41.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E41 = import_e41()
DEV = "cuda"
SCALES = (60, 120, 300, 600, 99999)
SCORES = HERE / "e42_scores.npz"
RESULT = HERE / "e42.json"


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score > 0
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    npos, nneg = int(np.sum(y == 1)), int(np.sum(y == 0))
    if not npos or not nneg:
        return dict(bacc=float("nan"), auc=float("nan"), f1=float("nan"))
    bacc = 0.5 * (tp / npos + 1 - fp / nneg)
    f1 = 2 * tp / (2 * tp + fp + npos - tp)
    ranks = np.argsort(np.argsort(score)) + 1
    auc = (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    return dict(bacc=float(bacc), auc=float(auc), f1=float(f1))


def summary(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in ("bacc", "auc", "f1")} | {
        "n90": int(sum(r["bacc"] >= 0.9 for r in rows)),
        "min": float(np.nanmin([r["bacc"] for r in rows])),
        "auc_below_0.5": int(sum(r["auc"] < 0.5 for r in rows)),
    }


def build_features(subs, D):
    F = None
    for w in SCALES:
        Fw = L.feats(D, subs, w)
        F = Fw if F is None else {q: torch.cat([F[q], Fw[q]], 1) for q in subs}
        del Fw
        torch.cuda.empty_cache()
    return F


def train_scores(outer_n):
    subs, D = L.load()
    F = build_features(subs, D)
    perm = list(np.random.RandomState(0).permutation(subs))
    outer = [[q] for q in subs] if outer_n == 57 else [perm[k::outer_n] for k in range(outer_n)]
    scores = {}
    for fold, te in enumerate(outer):
        tr = [q for q in subs if q not in te]
        X = torch.cat([F[q] for q in tr])
        y = np.concatenate([D[q][1] for q in tr])
        flr = L.fit_lr(X, y, 0.001)
        fms = [fit_mlp(X, y, seed=s) for s in range(3)]
        # 只用训练折估计融合尺度，匹配 streaming.py，而非目标整段 std。
        lr_parts, mlp_parts = [], []
        for q in tr:
            a = flr(F[q])
            m = torch.stack([f(F[q]) for f in fms]).mean(0)
            lr_parts.append(a)
            mlp_parts.append(m)
        std_lr = float(torch.cat(lr_parts).std())
        std_mlp = float(torch.cat(mlp_parts).std())
        del X, lr_parts, mlp_parts
        torch.cuda.empty_cache()
        for q in te:
            a = flr(F[q])
            m = torch.stack([f(F[q]) for f in fms]).mean(0)
            scores[q] = (a / std_lr + m / std_mlp).cpu().numpy()
            print(f"[fold {fold + 1}/{len(outer)}] {q} {metrics(D[q][1], scores[q])}", flush=True)
        del flr, fms
        torch.cuda.empty_cache()
    np.savez_compressed(SCORES, subjects=np.array(subs), **scores)
    return subs, D, scores


def evaluate(subs, D, scores):
    base, hmm_fixed, hmm_loso = [], [], []
    all_ev = []
    banks = {}
    for q in subs:
        bank = E41.hmm_forward_bank(torch.tensor(scores[q], device=DEV)).cpu().numpy()
        banks[q] = bank
        y = D[q][1]
        base.append(metrics(y, scores[q]))
        ev = [metrics(y, row) for row in bank]
        all_ev.append(ev)
        fixed_i = E41.HMM_PARAMS.index((300, 1.0))
        hmm_fixed.append(ev[fixed_i])
    all_ev = np.asarray([[list(r.values()) for r in rows] for rows in all_ev])
    for i, q in enumerate(subs):
        other = np.nanmean(np.delete(all_ev[:, :, 0], i, axis=0), axis=0)
        chosen = int(np.nanargmax(other))
        hmm_loso.append(dict(zip(("bacc", "auc", "f1"), all_ev[i, chosen])))
    warm_base = [metrics(D[q][1][:600], scores[q][:600]) for q in subs]
    fixed_i = E41.HMM_PARAMS.index((300, 1.0))
    warm_hmm = [metrics(D[q][1][:600], banks[q][fixed_i, :600]) for q in subs]
    late_base = [metrics(D[q][1][D[q][2] >= 4], scores[q][D[q][2] >= 4]) for q in subs]
    late_hmm = [metrics(D[q][1][D[q][2] >= 4], banks[q][fixed_i][D[q][2] >= 4]) for q in subs]
    result = {
        "protocol": "固定E17 LR+3MLP；融合尺度仅由训练折估计；5折OOF（--outer 57时为LOSO）",
        "base": summary(base),
        "hmm_fixed_d300_t1": summary(hmm_fixed),
        "hmm_loso": summary(hmm_loso),
        "warmup600_base": summary(warm_base),
        "warmup600_hmm_fixed": summary(warm_hmm),
        "late4_5_base": summary(late_base),
        "late4_5_hmm_fixed": summary(late_hmm),
        "base_per_subject": {q: r for q, r in zip(subs, base)},
        "hmm_fixed_per_subject": {q: r for q, r in zip(subs, hmm_fixed)},
    }
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("base", "hmm_fixed_d300_t1", "hmm_loso", "warmup600_base", "warmup600_hmm_fixed"):
        print(name, result[name], flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer", type=int, default=5, choices=(5, 57))
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    start = time.time()
    subs, D = L.load()
    if args.reuse and SCORES.exists():
        data = np.load(SCORES)
        scores = {q: data[q] for q in subs}
    else:
        subs, D, scores = train_scores(args.outer)
    result = evaluate(subs, D, scores)
    result["outer"] = args.outer
    result["elapsed_s"] = round(time.time() - start, 1)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
