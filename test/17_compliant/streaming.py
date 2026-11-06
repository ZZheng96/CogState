"""E17 分块流式推理程序（比赛提交形态）。

决策：每 1 s 输出一次，窗口 = 过去 6 s 原始单通道 EEG（250 Hz），严格因果。
特征：highdim_features(x,fs) ++ features_v2(x)，共 878 维；
      多尺度因果 running 标准化 w∈SCALES 帧（含当前帧，99999=全部过去），clamp[-10,10]，拼接 5×878。
模型：GPU LBFGS L2 LR（C=0.001）+ 3 种子 MLP；score = lr/std_lr + mlp/std_mlp（std 来自训练集），
      阈值 0，score 越大越偏 task（state=1 表示 task/认知负荷态，0 表示 relax/静息态）。

用法：
  .venv\\Scripts\\python.exe test/17_compliant/streaming.py train
  .venv\\Scripts\\python.exe test/17_compliant/streaming.py demo --subject 018 --chunk 0.37

API：
  eng = StreamingInference(torch.load(bundle), fs=250, step=1.0)
  for t_end, score, state in eng.update(chunk): ...   # chunk 为任意长度的一维样本数组（伏特，同 BDF/mne）
  eng.mark_invalid()   # 缺失段/不连续处：清空 6 s 缓冲，保留 running 统计
  eng.reset()          # 换被试：清空全部状态
"""
from __future__ import annotations

import argparse
import inspect
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLID = ROOT / "test" / "11_individual_20261102" / "sliding1"   # 1 s 步长离线滑窗（与 loso.py 一致）
BUNDLE = HERE / "model_bundle.pt"
SCALES = (60, 120, 300, 600, 99999)
WIN_S = 6.0
C_LR = 0.001
SEEDS = (0, 1, 2)


def _paths():
    for d in ("src", "test/03_deep", "test/08_diagnostics", "test/01_features", "test/17_compliant"):
        p = str(ROOT / d)
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


def train_bundle(out_path=BUNDLE, subs=None):
    _paths()
    import loso as L
    from multiscale import fit_mlp
    dev = L.dev
    subs = list(subs) if subs else sorted(p.stem[4:] for p in SLID.glob("sub-*.npz"))
    Xs, ys = [], []
    for q in subs:
        S = np.load(SLID / f"sub-{q}.npz")
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
    """分块流式推理。update(chunk) -> [(t_end 秒, score, state)]，state 1=task，0=relax。

    t_end 为窗口末端时刻（相对首个样本，秒）；mark_invalid(t_next) 可传入下一个样本的时刻，
    用于跳过缺失段后保持时间轴（不传则假设连续）。块内 NaN/inf 视为缺失：清空缓冲，保留 running 统计。
    """

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
        """换被试：清空缓冲与 running 统计。"""
        self.n_frames = 0
        d = self.b["n_feat"]
        self.c1 = np.zeros(d)            # 全部过去帧的累计和（float64）
        self.c2 = np.zeros(d)
        self.hist = deque(maxlen=max(w for w in self.scales if w < 99999) + 1)  # 累计和历史 (c1,c2)
        self.hist.append((self.c1.copy(), self.c2.copy()))
        self.debug = []
        self.t_abs = 0                   # 下一个样本的绝对样本序号
        self.mark_invalid()

    def mark_invalid(self, t_next=None):
        """缺失段/不连续：清空 6 s 缓冲并重置步长相位，保留 running 统计。"""
        self.buf = np.empty(0)
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
        score = float(self.scorer(torch.tensor(z[None], device=self.dev))[0])
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


# ---------------------------------------------------------------- demo / 共用数据读取
def load_subject(sid):
    """返回 (raw 数据 1D, fs, trials, 坏段已置 NaN)。"""
    _paths()
    if not hasattr(np, "in1d"):  # .venv 的 mne 1.4 在 numpy 2.x 下读 BDF 需要该别名
        np.in1d = np.isin
    import mne
    from cogstate import bad_intervals, load_trials
    sid = str(sid).zfill(3)
    csv = next((ROOT / "EEG_dataset").glob(f"{sid}/*_events.csv"))
    bdf = next(csv.parent.glob("*.bdf"))
    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    x = raw.get_data()[0].astype(float)
    for a, b in bad_intervals(raw):
        x[max(int(np.floor(a * fs)), 0):int(np.ceil(b * fs))] = np.nan
    trials = load_trials(str(csv), str(bdf))
    raw.close()
    return x, fs, trials


def stream_subject(eng, x, fs, trials, chunk_fn, max_s=None):
    """按 trial 喂入（每个 trial 前 mark_invalid），返回 [(t_end, score, state, label)]。

    max_s：只喂入从首个 trial 起的累计前 max_s 秒（用于前缀不变性测试）。
    """
    eng.reset()
    res, fed, budget = [], 0, None if max_s is None else int(round(max_s * fs))
    for tr in trials:
        s = int(round(tr.start * fs))
        e = min(int(np.floor((tr.end + 1e-6) * fs)), len(x))  # 窗末端样本序号 <= tr.end*fs（与离线 sliding 一致）
        eng.mark_invalid(t_next=s / fs)
        i = s
        while i < e:
            n = max(1, int(chunk_fn()))
            n = min(n, e - i)
            if budget is not None:
                n = min(n, budget - fed)
                if n <= 0:
                    return res
            for t, sc, st in eng.update(x[i:i + n]):
                res.append((t, sc, st, tr.label))
            i += n
            fed += n
    return res


def metrics(y, s):
    from sklearn.metrics import f1_score, roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    p = (s > 0).astype(int)
    bacc = 0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())
    return dict(bacc=float(bacc), auc=float(roc_auc_score(y, s)), f1=float(f1_score(y, p)))


def demo(sid, chunk, seed=0):
    eng = StreamingInference(BUNDLE)
    x, fs, trials = load_subject(sid)
    rng = np.random.RandomState(seed)
    mean_n = chunk * fs
    res = stream_subject(eng, x, fs, trials, lambda: rng.uniform(0.5, 1.5) * mean_n)
    t, s, st, y = map(np.asarray, zip(*res))
    m = metrics(y, s)
    print(f"[demo] subject {str(sid).zfill(3)} chunk~{chunk}s frames={len(res)} trials={len(trials)}")
    for r in res[:3]:
        print(f"  t_end={r[0]:.3f}s score={r[1]:+.4f} state={'task' if r[2] else 'relax'} label={r[3]}")
    print(f"[demo] bACC {m['bacc']:.4f} AUC {m['auc']:.4f} F1 {m['f1']:.4f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "demo"])
    ap.add_argument("--subject", default="018")
    ap.add_argument("--chunk", type=float, default=0.37)
    ap.add_argument("--out", default=str(BUNDLE))
    a = ap.parse_args()
    if a.cmd == "train":
        train_bundle(a.out)
    else:
        demo(a.subject, a.chunk)


if __name__ == "__main__":
    main()
