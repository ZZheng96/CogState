"""诊断3：后处理候选（含方向自适应）在 57 人 LOSO 分数上的对比。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402

spec = importlib.util.spec_from_file_location("e42lib", TESTS / "42_e17_loso_hmm" / "e42.py")
E42 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E42)
spec = importlib.util.spec_from_file_location("e44lib", TESTS / "44_late_drift" / "e44.py")
E44 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E44)
metrics, summary = E42.metrics, E42.summary
DEV = "cuda"

subs, D = L.load()
data = np.load(HERE / "e53_scores.npz")
s42 = {q: data[q].astype(float) for q in subs}
S = E44.pad(subs, s42)
B, T = S.shape
lens = [len(s42[q]) for q in subs]
const = lambda d: torch.full((B, T), -np.log(d), device=DEV)  # noqa: E731


def ema(S, hl):
    return E44.ema_mean(S, hl)


def ema_sd(S, hl):
    m = ema(S, hl)
    return (ema(S ** 2, hl) - m ** 2).clamp_min(1e-6).sqrt()


def hmm(S, d=300):
    if S.dim() == 1:
        return E44.hmm(S[None], torch.full((1, S.shape[0]), -np.log(d), device=DEV))[0]
    return E44.hmm(S, const(d))


def causal_median_t(s, W):
    pad = torch.cat([torch.full((W - 1,), float("nan"), device=DEV), s])
    return pad.unfold(0, W, 1).nanmedian(1).values


def causal_quantile(s, W):
    T = s.shape[0]
    i = torch.arange(T, device=DEV)
    mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)
    less = ((s[None, :] < s[:, None]) & mask).sum(1).float()
    eq = ((s[None, :] == s[:, None]) & mask).sum(1).float()
    n = mask.sum(1).float()
    return torch.logit((less + 0.5 * eq + 0.5) / (n + 1))


def per_subject(fn):
    """fn(s: (T,)) -> (T,) 逐被试独立处理。"""
    out = torch.zeros_like(S)
    for i in range(B):
        out[i, :lens[i]] = fn(S[i, :lens[i]])
    return out


Z = S - ema(S, 1200)
V = {}
V["dt1200_hmm"] = hmm(Z)
# A) 方向自适应：用因果中位数/分位数中心化（对整体符号翻转不变）
for W in (300, 600, 1200):
    V[f"med{W}_hmm"] = per_subject(lambda s, W=W: hmm(s - causal_median_t(s, W)))
    V[f"q{W}_hmm"] = per_subject(lambda s, W=W: hmm(causal_quantile(s, W)))
# B) 方向自适应 + 去趋势
for W in (600, 1200):
    V[f"dt1200_med{W}_hmm"] = per_subject(lambda s, W=W: hmm((s - ema(s[None], 1200)[0]) - causal_median_t(s, W)))
# C) 尺度归一化（低组后段方差收缩）
V["dt1200_z1200_hmm"] = hmm(Z / ema_sd(Z, 1200))
V["dt1200_z600_hmm"] = hmm(Z / ema_sd(Z, 600))
# D) 去趋势 + 中位数中心化 + 尺度
V["dt1200_med600_z_hmm"] = per_subject(
    lambda s: hmm(((s - ema(s[None], 1200)[0]) - causal_median_t(s, 600)) / ema_sd(s[None], 600)[0]))
# E) 混合：原始 + 去趋势
V["mix0.5_dt1200_hmm"] = hmm(0.5 * (S + Z))
# F) 驻留时间
V["dt1200_hmm_d600"] = hmm(Z, 600)
V["dt1200_hmm_d150"] = hmm(Z, 150)

grp = E44.groups()
res = {}
for k, v in V.items():
    sc = {q: v[i, :lens[i]].cpu().numpy() for i, q in enumerate(subs)}
    full = [metrics(D[q][1], sc[q]) for q in subs]
    late = [metrics(D[q][1][D[q][2] >= 4], sc[q][D[q][2] >= 4]) for q in subs]
    f, l = summary(full), summary(late)
    low = summary([r for q, r in zip(subs, full) if grp[q] == "low"])
    res[k] = (f, l, low)
    print(f"{k:24s} full {f['bacc']:.4f}/{f['auc']:.4f}/{f['f1']:.4f} n90={f['n90']} "
          f"late {l['bacc']:.4f}/{l['auc']:.4f} L={low['bacc']:.3f} Llate={summary([r for q, r in zip(subs, late) if grp[q]=='low'])['bacc']:.3f}",
          flush=True)
