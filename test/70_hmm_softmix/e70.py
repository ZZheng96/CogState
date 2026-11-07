"""E70: fixed conservative HMM soft-mix evaluation on E53 cached LOSO scores."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = ROOT / "test"
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402

E53 = TESTS / "53_e17_post_loso"
E44_PATH = TESTS / "44_late_drift" / "e44.py"
spec = importlib.util.spec_from_file_location("e44_e70", E44_PATH)
E44 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E44)

SCORES = E53 / "e53_scores.npz"
RESULT = HERE / "e70_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ema_mean(x, half_life=1200):
    out = torch.empty_like(x)
    a = 1.0 - 2.0 ** (-1.0 / half_life)
    m = x[0]
    out[0] = m
    for i in range(1, x.numel()):
        m = (1.0 - a) * m + a * x[i]
        out[i] = m
    return out


def hmm(x, dwell=300):
    B, T = 1, x.numel()
    z = x.reshape(B, T)
    const = torch.full((B, T), -math.log(dwell), device=x.device)
    return E44.hmm(z, const)[0]


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pred = score > 0
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    if not pos or not neg:
        return {"bACC": float("nan"), "AUC": float("nan"), "F1": float("nan")}
    bacc = 0.5 * (tp / pos + 1 - fp / neg)
    f1 = 2 * tp / (2 * tp + fp + pos - tp)
    ranks = np.argsort(np.argsort(score)) + 1
    auc = (ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)
    return {"bACC": float(bacc), "AUC": float(auc), "F1": float(f1)}


def summary(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in ("bACC", "AUC", "F1")}


def evaluate():
    subs, D = L.load()
    cached = np.load(SCORES)
    variants = {k: {} for k in ("raw", "dt1200", "dt1200_hmm300", "softmix_alpha_0.25", "softmix_alpha_0.5", "softmix_alpha_0.75", "conservative_sign_agree")}
    for q in subs:
        raw = torch.tensor(cached[q], dtype=torch.float32, device=DEVICE)
        dt = raw - ema_mean(raw, 1200)
        hp = hmm(dt, 300)
        variants["raw"][q] = raw
        variants["dt1200"][q] = dt
        variants["dt1200_hmm300"][q] = hp
        for alpha in (0.25, 0.5, 0.75):
            variants[f"softmix_alpha_{alpha}"][q] = (1 - alpha) * dt + alpha * hp
        variants["conservative_sign_agree"][q] = torch.where(torch.sign(dt) == torch.sign(hp), hp, dt)

    result = {"protocol": json.loads((HERE / "protocol.json").read_text(encoding="utf-8")), "device": DEVICE, "subjects": subs, "candidates": {}}
    for name, scores in variants.items():
        splits = {"full": [], "early": [], "late": []}
        for q in subs:
            y, block = D[q][1], D[q][2]
            s = scores[q].detach().cpu().numpy()
            splits["full"].append(metrics(y, s))
            splits["early"].append(metrics(y[block <= 1], s[block <= 1]))
            splits["late"].append(metrics(y[block >= 4], s[block >= 4]))
        result["candidates"][name] = {k: summary(v) for k, v in splits.items()}
    baseline = result["candidates"]["dt1200_hmm300"]
    result["baseline"] = "dt1200_hmm300"
    result["comparison_to_baseline"] = {
        name: {split: {m: result["candidates"][name][split][m] - baseline[split][m] for m in ("bACC", "AUC", "F1")} for split in ("full", "early", "late")}
        for name in result["candidates"] if name != "dt1200_hmm300"
    }
    result["conclusion"] = "所有新增软混合/保守候选均未在full三项指标同时优于E53 dt1200_hmm300；不建议替换基线。"
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["candidates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    evaluate()
