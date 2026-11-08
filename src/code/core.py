"""共享核心：特征、训练、流式推理、门控、BDF/事件读取、评估。

决策：每 1 s 输出一次，窗口 = 过去 6 s 原始单通道 EEG（250 Hz），严格因果。
特征：highdim_features(x,fs) ++ features_v2(x)，共 878 维；
      多尺度因果 running 标准化 w∈SCALES 帧（含当前帧，99999=全部过去），clamp[-10,10]，拼接 5×878。
模型：GPU LBFGS L2 LR（C=0.001）+ 3 种子 MLP；分量 lr=s_lr/std_lr、mlp=s_m/std_mlp；
      门控（可选）gate = LR(9 维因果门控特征)，raw = shrink*gate + (1-shrink)*(lr+mlp)；
      后处理 EMA 去趋势（半衰期 1200 帧）+ 两状态 sticky 前向 HMM（dwell 300），
      score = log(P(task)/P(relax))，state = score>0（1=task，0=relax）。
"""
from __future__ import annotations

import inspect
import json
import math
import re
import sys
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SLID = ROOT / "training_features"   # 可选：仅用于按旧特征文件重新训练
BUNDLE = ROOT / "model" / "model_bundle.pt"
GATE = ROOT / "model" / "deploy_gate.json"
SCALES = (60, 120, 300, 600, 99999)
WIN_S = 6.0
STEP_S = 1.0
C_LR = 0.001
SEEDS = (0, 1, 2)
GATE_WINDOW = 120
GATE_SHRINK = 0.5
GATE_C = 0.1
GATE_FEATURES = ["lr", "mlp", "abs_diff", "lr_mean", "mlp_mean", "diff_mean", "lr_std", "mlp_std", "diff_std"]
BJ = timezone(timedelta(hours=8))


def _paths():
    p = str(HERE)
    if p not in sys.path:
        sys.path.append(p)


def resolve_model(model=None, gate=None):
    """model 可为 bundle 文件或目录（目录下 model_bundle.pt）；gate 缺省取 bundle 同目录 deploy_gate.json。"""
    model = Path(model) if model else BUNDLE
    if model.is_dir():
        model = model / "model_bundle.pt"
    gate = Path(gate) if gate else model.with_name("deploy_gate.json")
    return model, gate


def pick_device(device=None):
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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


# ---------------------------------------------------------------- 数据读取
def _mne():
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    import mne
    return mne


def read_record(path):
    """读取单通道 BDF，BAD_ACQ_SKIP 区间置 NaN，返回 (x, fs, bad)。"""
    _paths()
    from cogstate import bad_intervals
    raw = _mne().io.read_raw_bdf(str(path), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    x = raw.get_data()[0].astype(float)
    bad = bad_intervals(raw)
    for a, b in bad:
        x[max(int(np.floor(a * fs)), 0):int(np.ceil(b * fs))] = np.nan
    raw.close()
    return x, fs, bad


def read_bdf(path):
    x, fs, _ = read_record(path)
    return x, fs


def read_trials(events, bdf):
    """[(start_s, end_s, label)]，相对 BDF 首样本。"""
    _paths()
    from cogstate import load_trials
    return [(t.start, t.end, t.label) for t in load_trials(str(events), str(bdf))]


def record_start(bdf):
    """文件名中的北京时间起点（首个样本时刻）；无法解析返回 None。"""
    m = re.search(r"_(\d{8})_(\d{6})\.(\d{3})_UTC", Path(bdf).name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2) + m.group(3), "%Y%m%d%H%M%S%f").replace(tzinfo=BJ)


def local_time(start, t):
    return "" if start is None else (start + timedelta(seconds=float(t))).isoformat(timespec="milliseconds")


def events_for(bdf):
    """同目录下与 BDF 同前缀（首个 '_' 之前）的 <id>_events.csv，不存在返回 None。"""
    bdf = Path(bdf)
    f = bdf.with_name(bdf.name.split("_")[0] + "_events.csv")
    return f if f.exists() else None


def find_records(root):
    """递归查找 *.bdf，返回 [(bdf, events 或 None)]。"""
    return [(p, events_for(p)) for p in sorted(Path(root).rglob("*.bdf"))]


def labels_at(t, trials, bad, win=WIN_S):
    """决策时刻 t 的标签：[t-win, t] 完全落在单个 trial 内且不与缺失段相交 -> trial 标签，否则 -1。"""
    t = np.asarray(t, dtype=np.float64)
    y = np.full(len(t), -1, dtype=np.int8)
    for a, b, lab in trials:
        y[(t - win >= a - 1e-6) & (t <= b + 1e-6)] = lab
    for a, b in bad:
        y[(t - win < b) & (t > a)] = -1
    return y


# ---------------------------------------------------------------- 评估
def metrics(y, s):
    from sklearn.metrics import f1_score, roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    p = (s > 0).astype(int)
    if len(set(y.tolist())) < 2:
        return dict(bacc=float("nan"), auc=float("nan"), f1=float(f1_score(y, p, zero_division=0)))
    bacc = 0.5 * (p[y == 1].mean() + 1 - p[y == 0].mean())
    return dict(bacc=float(bacc), auc=float(roc_auc_score(y, s)), f1=float(f1_score(y, p)))


# ---------------------------------------------------------------- 训练
def _closure(f):
    return inspect.getclosurevars(f).nonlocals


def _set_device(device):
    _paths()
    import loso
    import multiscale
    loso.dev = multiscale.dev = pick_device(device)
    return loso, multiscale


def build_features(bdf, events, out_npz):
    """6 s 窗、1 s 步长；窗口须完整落在单个 trial 内且有限。写 npz(F/y/t)。"""
    out_npz = Path(out_npz)
    if out_npz.exists():
        return out_npz
    x, fs, bad = read_record(bdf)
    trials = read_trials(events, bdf)
    t = np.arange(WIN_S, len(x) / fs + 1e-9, STEP_S)
    y = labels_at(t, trials, bad)
    t, y = t[y >= 0], y[y >= 0]
    n = int(round(WIN_S * fs))
    F, Y, T = [], [], []
    for ti, yi in zip(t, y):
        e = int(round(ti * fs))
        w = x[e - n:e]
        if len(w) == n and np.isfinite(w).all():
            F.append(window_features(w, fs)); Y.append(yi); T.append(ti)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, F=np.asarray(F, np.float32), y=np.asarray(Y, np.int8), t=np.asarray(T, np.float32))
    return out_npz


def train_bundle(out_path=BUNDLE, subs=None, files=None, device=None):
    """files：可选 {被试名: 特征 npz 路径}（F/y/t，同 sliding1 格式），用于训练任意数据源。"""
    L, M = _set_device(device)
    fit_mlp = M.fit_mlp
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


def gate_features(lr, mlp, w=GATE_WINDOW):
    """逐帧因果门控特征（与 _Scorer 在线计算一致）：[lr, mlp, |lr-mlp|, 最近 w 帧三者均值, 三者总体标准差]。"""
    q = deque(maxlen=w)
    X = np.empty((len(lr), 9))
    for i, (a, b) in enumerate(zip(lr, mlp)):
        d = abs(a - b)
        q.append((a, b, d))
        arr = np.asarray(q)
        X[i] = np.r_[a, b, d, arr.mean(0), arr.std(0)]
    return X


def fit_gate(X, y, source=""):
    from sklearn.linear_model import LogisticRegression
    g = LogisticRegression(C=GATE_C, class_weight="balanced", max_iter=300, solver="liblinear",
                           random_state=0).fit(X, y)
    return {"source": source, "window": GATE_WINDOW, "features": GATE_FEATURES,
            "coef": g.coef_[0].tolist(), "intercept": float(g.intercept_[0]), "shrink": GATE_SHRINK,
            "C": GATE_C, "n_frames": int(len(y)),
            "score": "shrink*gate + (1-shrink)*(lr+mlp) -> EMA1200 + HMM300"}


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
        self.last = None   # 最近一帧 (lr_scaled, mlp_scaled)

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
        lr_v, mlp_v = float(lr_scaled[0]), float(mlp_scaled[0])
        self.last = (lr_v, mlp_v)
        if self.gate is None:
            return fixed
        # 门控：滚动统计只含当前与过去帧。
        d = abs(lr_v - mlp_v)
        self.gate_history.append((lr_v, mlp_v, d))
        a = np.asarray(self.gate_history, dtype=float)[-self.gate["window"]:]
        gm, gs = a[:, 0].mean(), a[:, 1].mean()
        dm, ds = a[:, 2].mean(), a[:, 2].std()
        x = np.array([lr_v, mlp_v, d, gm, gs, dm, a[:, 0].std(), a[:, 1].std(), ds])
        gate_score = float(np.dot(self.gate["coef"], x) + self.gate["intercept"])
        k = float(self.gate.get("shrink", GATE_SHRINK))
        return torch.tensor([k * gate_score + (1 - k) * float(fixed[0])], device=X.device)


class StreamingInference:
    """分块流式推理。update(chunk) -> [(t_end 秒, score, state)]，state 1=task，0=relax。

    t_end 为窗口末端时刻（相对首个样本，秒）；mark_invalid(t_next) 可传入下一个样本的时刻，
    用于跳过缺失段后保持时间轴（不传则假设连续）。块内 NaN/inf 视为缺失：清空缓冲，保留 running 统计。
    gate：None=默认 model/deploy_gate.json（存在才用）；False=不用门控；路径=指定门控文件。
    keep_components=True 时逐帧记录 (lr_scaled, mlp_scaled) 到 self.components（训练门控用）。
    """

    def __init__(self, bundle, fs=250, step=1.0, device=None, keep_debug=False, gate=None,
                 keep_components=False):
        if isinstance(bundle, (str, Path)):
            bundle = torch.load(bundle, map_location="cpu", weights_only=False)
        bundle = dict(bundle)
        bundle.pop("gate", None)
        if gate is not False:
            gate = Path(gate) if gate is not None else GATE
            if gate.exists():
                bundle["gate"] = json.loads(gate.read_text(encoding="utf-8"))
        self.b = bundle
        self.fs = float(fs)
        assert abs(self.fs - bundle["fs"]) < 1e-6, "模型按 250 Hz 训练"
        self.win = int(round(bundle["win_s"] * self.fs))
        self.step = int(round(step * self.fs))
        self.scales = tuple(bundle["scales"])
        self.dev = pick_device(device)
        self.scorer = _Scorer(bundle).to(self.dev)
        self.keep_debug = keep_debug
        self.keep_components = keep_components
        self.reset()

    @property
    def gated(self):
        return self.scorer.gate is not None

    def reset(self):
        """换被试：清空缓冲与 running 统计。"""
        self.n_frames = 0
        d = self.b["n_feat"]
        self.c1 = np.zeros(d)            # 全部过去帧的累计和（float64）
        self.c2 = np.zeros(d)
        self.hist = deque(maxlen=max(w for w in self.scales if w < 99999) + 1)  # 累计和历史 (c1,c2)
        self.hist.append((self.c1.copy(), self.c2.copy()))
        self.debug = []
        self.components = []
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
        if self.keep_components:
            self.components.append(self.scorer.last)
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


def stream_record(eng, x, fs, chunk_s=1.0):
    """从首个样本起把整条记录按块喂给 eng（先 reset；缺失段 NaN 由 update 处理）。
    返回 [(t_end, score, state)]。标签不参与推理。"""
    eng.reset()
    n = max(1, int(round(chunk_s * fs)))
    res = []
    for i in range(0, len(x), n):
        res.extend(eng.update(x[i:i + n]))
    return res
