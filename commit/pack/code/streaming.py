"""单通道 EEG 认知状态分块流式推理程序。

决策：每 1 s 输出一次，窗口 = 过去 6 s 原始单通道 EEG（250 Hz），严格因果。
特征：highdim_features(x,fs) ++ features_v2(x)，共 878 维；
      多尺度因果 running 标准化 w∈SCALES 帧（含当前帧，99999=全部过去），clamp[-10,10]，拼接 5×878。
模型：GPU LBFGS L2 LR（C=0.001）+ 3 种子 MLP；score = lr/std_lr + mlp/std_mlp（std 来自训练集），
      阈值 0，score 越大越偏 task（state=1 表示 task/认知负荷态，0 表示 relax/静息态）。

用法：
  python code/streaming.py --input <record.bdf> [--events <events.csv>] [--output <result.csv>]
                           [--chunk 1.0] [--device auto|cpu|cuda] [--model model/model_bundle.pt]

API：
  eng = StreamingInference(torch.load(bundle), fs=250, step=1.0)
  for t_end, score, state in eng.update(chunk): ...   # chunk 为任意长度的一维样本数组（伏特，同 BDF/mne）
  eng.mark_invalid()   # 缺失段/不连续处：清空 6 s 缓冲，保留 running 统计
  eng.reset()          # 换被试：清空全部状态
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SLID = ROOT / "training_features"   # 可选：仅用于重新训练
BUNDLE = ROOT / "model" / "model_bundle.pt"
GATE = ROOT / "model" / "deploy_gate.json"
SCALES = (60, 120, 300, 600, 99999)
WIN_S = 6.0
C_LR = 0.001
SEEDS = (0, 1, 2)
GATE_WINDOW = 120


def _paths():
    p = str(HERE)
    if p not in sys.path:
        sys.path.append(p)


_FEAT = None


def window_features(x, fs):
    """单个 6 s 窗的 878 维特征（与 build_sliding.py 完全一致）。"""
    global _FEAT
    if _FEAT is None:
        _paths()
        import scipy.signal  # noqa: F401
        from highdim_deep_ablation import highdim_features
        from emg_v2 import features_v2
        _FEAT = (highdim_features, features_v2)
    x = np.asarray(x, dtype=float)
    f = np.concatenate([_FEAT[0](x, fs), _FEAT[1](x)]).astype(np.float32)
    return np.nan_to_num(f)


# ---------------------------------------------------------------- 训练
def _closure(f):
    cv = inspect.getclosurevars(f).nonlocals
    return cv


def train_bundle(out_path=BUNDLE, subs=None, files=None):
    """files：可选 {被试名: 特征 npz 路径}（F/y/t，同 sliding1 格式），用于训练任意数据源。"""
    _paths()
    import loso as L
    from multiscale import fit_mlp
    dev = L.dev
    if files is None:
        subs = list(subs) if subs else sorted(p.stem[4:] for p in SLID.glob("sub-*.npz"))
        files = {q: SLID / f"sub-{q}.npz" for q in subs}
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
        self.gate = b.get("gate")
        self.gate_history = []
        self.ema_level = None
        self.hmm_lp = np.array([-math.log(2.0), -math.log(2.0)])

    def reset_history(self):
        self.gate_history.clear()

    def to(self, dev):
        self.lr = {k: v.to(dev) for k, v in self.lr.items()}
        self.nets = [(mu.to(dev), sd.to(dev), n.to(dev)) for mu, sd, n in self.nets]
        return self

    @torch.no_grad()
    def forward(self, X):
        lr = self.lr
        s_lr = ((X - lr["mu"]) / lr["sd"]) @ lr["w"] + lr["b"]
        s_m = torch.stack([n((X - mu) / sd).squeeze(1) for mu, sd, n in self.nets]).mean(0)
        lr_scaled = s_lr / self.std_lr
        mlp_scaled = s_m / self.std_mlp
        fixed = lr_scaled + mlp_scaled
        if self.gate is None:
            return fixed
        # Gate parameters are fitted on full-training OOF component scores.
        # Rolling statistics contain current and past outputs only.
        lr_v, mlp_v = float(lr_scaled[0]), float(mlp_scaled[0])
        d = abs(lr_v - mlp_v)
        self.gate_history.append((lr_v, mlp_v, d))
        a = np.asarray(self.gate_history, dtype=float)[-self.gate["window"]:]
        gm, gs = a[:, 0].mean(), a[:, 1].mean()
        dm, ds = a[:, 2].mean(), a[:, 2].std()
        x = np.array([lr_v, mlp_v, d, gm, gs, dm, a[:, 0].std(), a[:, 1].std(), ds])
        gate_score = float(np.dot(self.gate["coef"], x) + self.gate["intercept"])
        return torch.tensor([0.5 * gate_score + 0.5 * float(fixed[0])], device=X.device)


class StreamingInference:
    """分块流式推理。update(chunk) -> [(t_end 秒, score, state)]，state 1=task，0=relax。

    t_end 为窗口末端时刻（相对首个样本，秒）；mark_invalid(t_next) 可传入下一个样本的时刻，
    用于跳过缺失段后保持时间轴（不传则假设连续）。块内 NaN/inf 视为缺失：清空缓冲，保留 running 统计。
    """

    def __init__(self, bundle, fs=250, step=1.0, device=None, keep_debug=False, gate=None):
        if isinstance(bundle, (str, Path)):
            bundle = torch.load(bundle, map_location="cpu", weights_only=False)
        gate = Path(gate) if gate is not None else GATE
        if gate.exists():
            bundle = dict(bundle)
            bundle["gate"] = json.loads(gate.read_text(encoding="utf-8"))
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
        """换被试：清空缓冲与 running 统计。"""
        self.n_frames = 0
        d = self.b["n_feat"]
        self.c1 = np.zeros(d)            # 全部过去帧的累计和（float64）
        self.c2 = np.zeros(d)
        self.hist = deque(maxlen=max(w for w in self.scales if w < 99999) + 1)  # 累计和历史 (c1,c2)
        self.hist.append((self.c1.copy(), self.c2.copy()))
        self.debug = []
        self.scorer.reset_history()
        self.ema_level = None
        self.hmm_lp = np.array([-math.log(2.0), -math.log(2.0)])
        self.t_abs = 0                   # 下一个样本的绝对样本序号
        self.mark_invalid()

    def mark_invalid(self, t_next=None):
        """缺失段/不连续：清空 6 s 缓冲并重置步长相位，保留 running 统计。"""
        self.buf = np.empty(0)
        self.scorer.reset_history()
        self.since = 0                   # 自上次 mark_invalid 以来的样本数（决定 1 s 步长网格）
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
            lo = max(n - w, 0)       # 与 loso.run_stats_t 相同：累计和之差 / 帧数
            a1, a2 = (zero, zero) if lo == 0 else self.hist[-1 - (n - lo)]
            m = (self.c1 - a1) / (n - lo)
            v = (self.c2 - a2) / (n - lo) - m ** 2
            sd = np.sqrt(np.maximum(v, 1e-12))
            m32, sd32 = m.astype(np.float32), sd.astype(np.float32)
            parts.append(np.clip((f.astype(np.float32) - m32) / sd32, -10, 10))
        z = np.concatenate(parts)
        raw = float(self.scorer(torch.tensor(z[None], device=self.dev))[0])
        a = 1.0 - 0.5 ** (1.0 / 1200.0)
        if self.ema_level is None:
            self.ema_level = raw
        self.ema_level += a * (raw - self.ema_level)
        ema = raw - self.ema_level
        stay, switch = math.log1p(-1.0 / 300.0), -math.log(300.0)
        em = np.array([-np.logaddexp(0.0, ema), -np.logaddexp(0.0, -ema)])
        if self.n_frames > 1:
            self.hmm_lp = np.array([
                np.logaddexp(self.hmm_lp[0] + stay, self.hmm_lp[1] + switch),
                np.logaddexp(self.hmm_lp[0] + switch, self.hmm_lp[1] + stay),
            ])
        u = self.hmm_lp + em
        self.hmm_lp = u - np.logaddexp(u[0], u[1])
        score = float(self.hmm_lp[1] - self.hmm_lp[0])
        if self.keep_debug:
            self.debug.append((f.astype(np.float32), z))
        return score

    def update(self, chunk):
        x = np.asarray(chunk, dtype=float).ravel()
        out = []
        i = 0
        while i < len(x):
            # 决策点：自 mark_invalid 起第 win + k*step 个样本处
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


# ---------------------------------------------------------------- 命令行推理
def read_bdf(path):
    """读取单通道 BDF，BAD_ACQ_SKIP 区间置 NaN，返回 (x, fs)。"""
    _paths()
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    import mne
    from cogstate import bad_intervals
    raw = mne.io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    x = raw.get_data()[0].astype(float)
    for a, b in bad_intervals(raw):
        x[max(int(np.floor(a * fs)), 0):int(np.ceil(b * fs))] = np.nan
    raw.close()
    return x, fs


def metrics(y, s):
    from sklearn.metrics import f1_score, roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    p = (s > 0).astype(int)
    bacc = 0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())
    return dict(bacc=float(bacc), auc=float(roc_auc_score(y, s)), f1=float(f1_score(y, p)))


def run(input, events=None, output=None, chunk=1.0, device=None, model=BUNDLE):
    model = Path(model)
    eng = StreamingInference(model, device=device, gate=model.with_name("deploy_gate.json"))
    x, fs = read_bdf(input)
    if events:
        _paths()
        from cogstate import load_trials
        segs = [(int(round(t.start * fs)), min(int(np.floor((t.end + 1e-6) * fs)), len(x)), t.label)
                for t in load_trials(str(events), str(input))]
    else:
        segs = [(0, len(x), None)]
    n = max(1, int(round(chunk * fs)))
    res = []
    for s, e, label in segs:
        eng.mark_invalid(t_next=s / fs)
        for i in range(s, e, n):
            res.extend((t, sc, st, label) for t, sc, st in eng.update(x[i:min(i + n, e)]))
    print(f"[input]  {Path(input).name}  fs={fs:g} Hz  duration={len(x) / fs:.1f} s")
    print(f"[model]  {model.name}  device={eng.dev}  chunk={n / fs:g} s")
    print(f"[output] frames={len(res)}")
    head = "t_end_s,score,state" + (",label" if events else "")
    rows = [f"{t:.3f},{sc:.6f},{st}" + (f",{lb}" if events else "") for t, sc, st, lb in res]
    print("  " + head)
    for r in rows[:3]:
        print("  " + r)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text("\n".join([head] + rows) + "\n", encoding="utf-8")
        print(f"[saved]  {output}")
    if events and res:
        y = [r[3] for r in res]
        if len(set(y)) == 2:
            m = metrics(y, [r[1] for r in res])
            print(f"[eval]   bACC={m['bacc']:.4f}  AUC={m['auc']:.4f}  F1={m['f1']:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser(description="单通道 EEG 认知状态分块流式推理")
    ap.add_argument("--input", required=True, help="单通道 BDF 文件，采样率 250 Hz")
    ap.add_argument("--events", help="可选，事件 CSV；提供时仅推理 trial 区间并输出 label 与评估指标")
    ap.add_argument("--output", help="可选，结果 CSV 保存路径")
    ap.add_argument("--chunk", type=float, default=1.0, help="每次送入的数据块长度（秒，>0），默认 1.0")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="计算设备，默认 auto")
    ap.add_argument("--model", default=None, help="模型文件，默认 model/model_bundle.pt")
    a = ap.parse_args()
    if a.chunk <= 0:
        ap.error("--chunk 必须大于 0")
    run(a.input, a.events, a.output, a.chunk, None if a.device == "auto" else a.device, a.model or BUNDLE)


if __name__ == "__main__":
    main()


