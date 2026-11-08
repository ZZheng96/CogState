"""GPU 训练工具：因果滑动统计、LBFGS L2 LR、MLP。

复制自：
- test/17_compliant/loso.py: dev, run_stats_t, fit_lr
- test/17_compliant/multiscale.py: fit_mlp
"""
from __future__ import annotations

import torch

dev = "cuda"


def run_stats_t(X, w):
    """GPU 因果滑动均值/标准差（含当前帧）。X (n,d) tensor。"""
    n = X.shape[0]
    z = torch.zeros(1, X.shape[1], device=X.device, dtype=torch.float64)
    c1 = torch.cat([z, X.double().cumsum(0)])
    c2 = torch.cat([z, (X.double() ** 2).cumsum(0)])
    idx = torch.arange(1, n + 1, device=X.device)
    lo = torch.clamp(idx - w, min=0)
    cnt = (idx - lo)[:, None].double()
    m = (c1[idx] - c1[lo]) / cnt
    v = (c2[idx] - c2[lo]) / cnt - m ** 2
    return m.float(), v.clamp_min(1e-12).sqrt().float()


def fit_lr(X, y, C, iters=200):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    wv = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    yt = torch.tensor(y, device=dev, dtype=torch.float32)
    pw = (yt == 0).sum() / (yt == 1).sum()
    opt = torch.optim.LBFGS([wv, b], max_iter=iters, line_search_fn="strong_wolfe")
    lam = 1.0 / (C * len(y))

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ wv + b, yt, pos_weight=pw) \
            + 0.5 * lam * (wv ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    wv, b = wv.detach(), b.detach()
    return lambda X: ((X - mu) / sd) @ wv + b


def fit_mlp(X, y, hid=128, ep=30, wd=1e-3, drop=0.3, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(X.shape[1], hid), torch.nn.GELU(),
                              torch.nn.Dropout(drop), torch.nn.Linear(hid, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    yt = torch.tensor(y, device=dev, dtype=torch.float32)
    pw = (yt == 0).sum() / (yt == 1).sum()
    n = len(y)
    for _ in range(ep):
        for idx in torch.randperm(n, device=dev).split(512):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                net(Z[idx]).squeeze(1), yt[idx], pos_weight=pw).backward()
            opt.step()
    net.eval()

    def f(X):
        with torch.no_grad():
            return net((X - mu) / sd).squeeze(1)
    return f
