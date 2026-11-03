"""E26 公共：扩展因果阈值网格的 GPU 评估器。
阈值（全部因果、无标签，只用过去 w 帧本人分数）：
  t0；m{w} = 过去 w 帧均值；q{w}_{p} = 过去 w 帧 p 分位数（w≥1200 时窗内等间隔下采样到 ≤600 个点）。
  w ∈ {60,120,300,600,1200,2400}，p ∈ {.20,.25,...,.80}。共 1+6+6×13 = 85 种。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "24_cluster"))
from clib import dev  # noqa: E402
from common import calref, expand  # noqa: E402,F401

WS = (60, 120, 300, 600, 1200, 2400)
QS = tuple(round(.2 + .05 * i, 2) for i in range(13))
THR2 = [("t0", 0, None)] + [(f"m{w}", w, None) for w in WS] + [(f"q{w}_{p}", w, p) for w in WS for p in QS]


def _means(M, tst, w):
    """M (R,T) 含 NaN → 过去 w 帧（含当前）nanmean，取 tst 位置 → (R,nt)。"""
    v = ~torch.isnan(M)
    x = torch.where(v, M, torch.zeros_like(M))
    cs = torch.nn.functional.pad(x.cumsum(1), (1, 0))
    cn = torch.nn.functional.pad(v.float().cumsum(1), (1, 0))
    hi = tst + 1
    lo = (hi - w).clamp(min=0)
    return (cs[:, hi] - cs[:, lo]) / (cn[:, hi] - cn[:, lo]).clamp(min=1)


def _quants(M, tst, w):
    """→ (R,13,nt) 过去 w 帧分位数（线性插值，忽略 NaN）。"""
    st = max(1, w // 600)
    pad = torch.cat([torch.full((M.shape[0], w - 1), float("nan"), device=dev), M], 1)
    U = pad.unfold(1, w, 1)[..., ::st][:, tst]  # (R,nt,m)
    S = U.sort(2).values  # NaN 排到末尾
    n = (~torch.isnan(U)).sum(2).float().clamp(min=1)
    out = []
    for p in QS:
        pos = (n - 1) * p
        lo = pos.floor().long()
        hi = pos.ceil().long()
        a = S.gather(2, lo[..., None])[..., 0]
        b = S.gather(2, hi[..., None])[..., 0]
        out.append(a + (b - a) * (pos - lo.float()))
    return torch.stack(out, 1)


def thr_list(qws):
    return [0] + list(range(1, 1 + len(WS))) + [1 + len(WS) + WS.index(w) * len(QS) + i for w in qws
                                                 for i in range(len(QS))]


def thr_bank(M, tst, qws=WS):
    """M (R,T) → (R,K,nt) 阈值：t0 + 6 均值 + qws 窗口的 13 分位数（顺序与 thr_list(qws) 一致）。"""
    R = M.shape[0]
    out = [torch.zeros(R, 1, len(tst), device=dev)]
    out += [_means(M, tst, w)[:, None] for w in WS]
    out += [_quants(M, tst, w) for w in qws]
    return torch.cat(out, 1)


def metrics(M, tst, yy, qws=WS, ch=64, auc=True):
    """M (R,T) z 分数行；返回 (R,K,3) [bACC,AUC,F1]（测试帧），NaN 行 → NaN。"""
    npos, nneg = yy.sum().float(), (~yy).sum().float()
    res = []
    for a in range(0, M.shape[0], ch):
        Mb = M[a:a + ch]
        Th = thr_bank(Mb, tst, qws)
        St = Mb[:, tst][:, None]
        D = St - Th
        P = D > 0
        tp, fp = (P & yy).sum(2).float(), (P & ~yy).sum(2).float()
        b = 0.5 * (tp / npos + 1 - fp / nneg)
        f = 2 * tp / (2 * tp + fp + (npos - tp))
        if auc:
            r = D.argsort(2).argsort(2).float() + 1
            u = (r[..., yy].sum(2) - npos * (npos + 1) / 2) / (npos * nneg)
        else:
            u = torch.full_like(b, float("nan"))
        o = torch.stack([b, u, f], 2)
        o[torch.isnan(Mb).any(1)] = float("nan")
        res.append(o)
        del Th, D, P
    return torch.cat(res)


def subj_data(R, q):
    """→ (cands z 分数 (1653,T) GPU, cal_t, tst 索引, yy)。"""
    y, blk = R["D"][q][1], R["D"][q][2]
    cal_t = calref(q, R["D"])
    B = torch.tensor(R["SC"][q], device=dev, dtype=torch.float32)
    C = expand(B, cal_t)
    tst = torch.tensor(np.where(blk >= 4)[0], device=dev)
    return C, cal_t, tst, torch.tensor(y[blk >= 4], device=dev).bool()


def zpair(C, cal_t, ia, ib):
    M = C[ia] + C[ib]
    return (M - M[:, cal_t].mean(1, keepdim=True)) / (M[:, cal_t].std(1, keepdim=True) + 1e-6)
