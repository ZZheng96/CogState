"""E17 训练与分块流式推理。

复制自 test/17_compliant/streaming.py: SCALES/WIN_S/C_LR/SEEDS, _closure, train_bundle,
_Scorer, StreamingInference（逐行一致；train_bundle 必须显式给出 files）。
特征 window_features 见 features.py；训练工具见 train_utils.py。
"""
from __future__ import annotations

import inspect
from collections import deque
from pathlib import Path

import numpy as np
import torch

from . import train_utils as L
from .features import window_features
from .train_utils import fit_mlp

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "models" / "e17_bundle.pt"
SCALES = (60, 120, 300, 600, 99999)
WIN_S = 6.0
C_LR = 0.001
SEEDS = (0, 1, 2)


# ---------------------------------------------------------------- 训练
def _closure(f):
    cv = inspect.getclosurevars(f).nonlocals
    return cv


def train_bundle(out_path, files):
    """files：{被试名: 特征 npz 路径}（F/y/t，1 s 步长滑窗）。"""
    dev = L.dev
    subs = list(files)
    Xs, ys = [], []
    for q in subs:
        S = np.load(files[q])
        o = np.argsort(S["t"], kind="stable")
        X = torch.tensor(np.nan_to_num(S["F"][o].astype(np.float32)), device=dev)
        parts = []
        for w in SCALES:
            m, sd = L.run_stats_t(X, w)
            parts.append(((X - m) / sd).clamp(-10, 10))
        Xs.append(torch.cat(parts, 1))
        ys.append(S["y"][o].astype(int))
        del X, parts
    X = torch.cat(Xs)
    del Xs
    y = np.concatenate(ys)
    print(f"[train] subjects={len(subs)} frames={len(y)} dim={X.shape[1]} task_frac={y.mean():.3f}", flush=True)
    flr = L.fit_lr(X, y, C_LR)
    c = _closure(flr)
    lr = dict(mu=c["mu"].cpu(), sd=c["sd"].cpu(), w=c["wv"].cpu(), b=c["b"].cpu())
    mlps, s_m = [], 0
    for s in SEEDS:
        fm = fit_mlp(X, y, seed=s)
        c = _closure(fm)
        net = c["net"]
        mlps.append(dict(mu=c["mu"].cpu(), sd=c["sd"].cpu(),
                         state={k: v.cpu() for k, v in net.state_dict().items()},
                         hid=net[1].out_features))
        s_m = s_m + torch.cat([fm(xb) for xb in X.split(8192)])
        print(f"[train] mlp seed {s} done", flush=True)
    s_m = s_m / len(SEEDS)
    s_lr = torch.cat([flr(xb) for xb in X.split(8192)])
    std_lr, std_mlp = float(s_lr.std()), float(s_m.std())
    s_e = s_lr / std_lr + s_m / std_mlp
    yt = torch.tensor(y, device=dev)
    pred = (s_e > 0).long()
    bacc = 0.5 * (float((pred[yt == 1] == 1).float().mean()) + float((pred[yt == 0] == 0).float().mean()))
    bundle = dict(version=1, scales=SCALES, win_s=WIN_S, fs=250.0, step_s=1.0, n_feat=X.shape[1] // len(SCALES),
                  lr=lr, mlps=mlps, std_lr=std_lr, std_mlp=std_mlp,
                  feat_mean=X.mean(0).cpu(), feat_var=X.var(0).cpu(), C=C_LR, subjects=subs,
                  train_bacc=bacc, score_direction="score>0 -> task(1)；score<=0 -> relax(0)")
    torch.save(bundle, out_path)
    print(f"[train] std_lr={std_lr:.4f} std_mlp={std_mlp:.4f} train_bACC={bacc:.4f} -> {out_path}", flush=True)
    return bundle


# ---------------------------------------------------------------- 推理
class _Scorer(torch.nn.Module):
    def __init__(self, b):
        super().__init__()
        d = b["n_feat"] * len(b["scales"])
        self.lr = b["lr"]
        self.nets = []
        for m in b["mlps"]:
            net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(d, m["hid"]), torch.nn.GELU(),
                                      torch.nn.Dropout(0.3), torch.nn.Linear(m["hid"], 1))
            net.load_state_dict(m["state"])
            net.eval()
            self.nets.append((m["mu"], m["sd"], net))
        self.std_lr, self.std_mlp = b["std_lr"], b["std_mlp"]

    def to(self, dev):
        self.lr = {k: v.to(dev) for k, v in self.lr.items()}
        self.nets = [(mu.to(dev), sd.to(dev), n.to(dev)) for mu, sd, n in self.nets]
        return self

    @torch.no_grad()
    def forward(self, X):
        lr = self.lr
        s_lr = ((X - lr["mu"]) / lr["sd"]) @ lr["w"] + lr["b"]
        s_m = torch.stack([n((X - mu) / sd).squeeze(1) for mu, sd, n in self.nets]).mean(0)
        return s_lr / self.std_lr + s_m / self.std_mlp


class StreamingInference:
    """分块流式推理。update(chunk) -> [(t_end 秒, score, state)]，state 1=task，0=relax。"""

    def __init__(self, bundle, fs=250, step=1.0, device=None, keep_debug=False):
        if isinstance(bundle, (str, Path)):
            bundle = torch.load(bundle, map_location="cpu", weights_only=False)
        self.b = bundle
        self.fs = float(fs)
        assert abs(self.fs - bundle["fs"]) < 1e-6, "模型按 250 Hz 训练"
        self.win = int(round(bundle["win_s"] * self.fs))
        self.step = int(round(step * self.fs))
        self.scales = tuple(bundle["scales"])
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scorer = _Scorer(bundle).to(self.dev)
        self.keep_debug = keep_debug
        self.reset()

    def reset(self):
        self.n_frames = 0
        d = self.b["n_feat"]
        self.c1 = np.zeros(d)
        self.c2 = np.zeros(d)
        self.hist = deque(maxlen=max(w for w in self.scales if w < 99999) + 1)
        self.hist.append((self.c1.copy(), self.c2.copy()))
        self.debug = []
        self.t_abs = 0
        self.mark_invalid()

    def mark_invalid(self, t_next=None):
        self.buf = np.empty(0)
        self.since = 0
        if t_next is not None:
            self.t_abs = int(round(float(t_next) * self.fs))

    def _frame(self, x):
        f = window_features(x, self.fs).astype(np.float64)
        self.c1 = self.c1 + f
        self.c2 = self.c2 + f ** 2
        self.hist.append((self.c1.copy(), self.c2.copy()))
        self.n_frames += 1
        n = self.n_frames
        parts = []
        zero = np.zeros_like(f)
        for w in self.scales:
            lo = max(n - w, 0)
            a1, a2 = (zero, zero) if lo == 0 else self.hist[-1 - (n - lo)]
            m = (self.c1 - a1) / (n - lo)
            v = (self.c2 - a2) / (n - lo) - m ** 2
            sd = np.sqrt(np.maximum(v, 1e-12))
            m32, sd32 = m.astype(np.float32), sd.astype(np.float32)
            parts.append(np.clip((f.astype(np.float32) - m32) / sd32, -10, 10))
        z = np.concatenate(parts)
        score = float(self.scorer(torch.tensor(z[None], device=self.dev))[0])
        if self.keep_debug:
            self.debug.append((f.astype(np.float32), z))
        return score

    def update(self, chunk):
        x = np.asarray(chunk, dtype=float).ravel()
        out = []
        i = 0
        while i < len(x):
            target = self.win if self.since < self.win else \
                self.win + ((self.since - self.win) // self.step + 1) * self.step
            seg = x[i:i + target - self.since]
            bad = ~np.isfinite(seg)
            if bad.any():
                k = int(np.nonzero(bad)[0][-1]) + 1
                self.buf = np.empty(0)
                seg_ok = seg[k:]
            else:
                seg_ok = seg
            self.buf = np.concatenate([self.buf, seg_ok])[-self.win:]
            self.since += len(seg)
            self.t_abs += len(seg)
            i += len(seg)
            if self.since >= self.win and (self.since - self.win) % self.step == 0 and len(self.buf) == self.win:
                s = self._frame(self.buf)
                out.append((self.t_abs / self.fs, s, int(s > 0)))
        return out
