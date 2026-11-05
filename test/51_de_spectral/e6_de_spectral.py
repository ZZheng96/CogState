"""E6：原始 6 s 窗的细分频带 DE / 相对功率 / 谱比值，57 人真 LOSO。

数据仅取 database/windows 的 x 与既有 50 维 STFT 功率列。BDF 仅用于重放
切窗循环、恢复时间顺序并逐字节校验；不使用 realorder/sliding 的特征缓存。
新增频谱和累计因果归一化均在 GPU 上计算，分类器为固定超参 GPU LR。
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import mne
import numpy as np
import torch

if not hasattr(np, "in1d"):
    np.in1d = np.isin  # 旧版 MNE 与 NumPy 2 的局部兼容。
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cogstate import bad_intervals, load_trials, valid_window  # noqa: E402

FS = 250.0
WINDOW_S = 6.0
N_SAMPLES = 1500
NPERSEG = 250
HOP = 125
FINE_BANDS = (
    (1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13), (13, 15),
    (15, 18), (18, 22), (22, 26), (26, 30), (30, 35), (35, 40), (40, 45),
)
BROAD = {"delta": (0, 2), "theta": (2, 4), "alpha": (4, 6),
         "beta": (6, 11), "gamma": (11, 14)}
C = 0.001
LBFGS_ITERS = 150
CLIP = 10.0
VAR_EPS = 1e-8
FEATURES_PATH = HERE / "e6_features.npz"
SCORES_PATH = HERE / "e6_scores.npz"
RESULTS_PATH = HERE / "e6_results.json"
REPORT_PATH = HERE / "报告_E6_DE细分频谱.md"
VARIANTS = {"stft50": slice(0, 50), "spectral45": slice(50, 95),
            "combined95": slice(0, 95)}
DEV = None


def spectral_names() -> list[str]:
    names = [f"de_{lo}_{hi}Hz" for lo, hi in FINE_BANDS]
    names += [f"relative_{lo}_{hi}Hz" for lo, hi in FINE_BANDS]
    names += [f"log_ratio_{FINE_BANDS[i][0]}_{FINE_BANDS[i][1]}Hz__"
              f"{FINE_BANDS[i + 1][0]}_{FINE_BANDS[i + 1][1]}Hz"
              for i in range(len(FINE_BANDS) - 1)]
    names += ["log_ratio_theta_alpha", "log_ratio_alpha_beta",
              "log_ratio_beta_gamma", "log_ratio_delta_beta"]
    assert len(names) == 45
    return names


def stft50_indices() -> list[int]:
    manifest = json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names"]
    expected = [f"stft_{kind}_{lo}_{hi}_{stat}"
                for kind in ("log_power", "relative")
                for lo, hi in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
                for stat in ("mean", "std", "slope", "max", "min")]
    indices = [names.index(n) for n in expected]
    assert indices == list(range(23, 73)), indices
    return indices


def gpu_spectral_features(x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Welch 型 1 s STFT 频带方差，再计算 DE、相对功率与对数谱比值。"""
    if x.ndim != 2 or x.shape[1] != N_SAMPLES or not np.isfinite(x).all():
        raise ValueError(f"非法原始窗口批次 {x.shape}")
    window = torch.hann_window(NPERSEG, periodic=True, device=DEV)
    scale = FS * window.square().sum()
    freq = torch.fft.rfftfreq(NPERSEG, 1 / FS, device=DEV)
    masks = [(freq >= lo) & (freq < hi) for lo, hi in FINE_BANDS]
    tiny = torch.finfo(torch.float32).tiny
    rows = []
    for start in range(0, len(x), batch_size):
        a = torch.as_tensor(x[start:start + batch_size], device=DEV)
        # 11 个当前窗内部的 1 s 分段；每段去均值，不用窗外样本。
        frames = a.unfold(1, NPERSEG, HOP)
        frames = frames - frames.mean(dim=-1, keepdim=True)
        fft = torch.fft.rfft(frames * window, dim=-1)
        psd = fft.abs().square() / scale
        psd[..., 1:-1] *= 2.0
        psd = psd.mean(dim=1)
        # df=1 Hz；sigma_b^2 是带限方差（PSD 在频带内积分）。
        var = torch.stack([psd[:, m].sum(dim=1) for m in masks], dim=1)
        de = 0.5 * torch.log((2 * math.pi * math.e) * var.clamp_min(tiny))
        relative = var / var.sum(dim=1, keepdim=True).clamp_min(tiny)
        broad = {k: var[:, a:b].sum(dim=1) for k, (a, b) in BROAD.items()}
        pairs = [(var[:, i], var[:, i + 1]) for i in range(len(FINE_BANDS) - 1)]
        pairs += [(broad["theta"], broad["alpha"]), (broad["alpha"], broad["beta"]),
                  (broad["beta"], broad["gamma"]), (broad["delta"], broad["beta"])]
        ratios = torch.stack([torch.log(a.clamp_min(tiny) / b.clamp_min(tiny))
                              for a, b in pairs], dim=1)
        feat = torch.cat([de, relative, ratios], dim=1)
        if not bool(torch.isfinite(feat).all()):
            raise RuntimeError(f"非有限频谱特征，批次起点 {start}")
        rows.append(feat.cpu().numpy())
    out = np.concatenate(rows).astype(np.float32)
    assert out.shape == (len(x), 45)
    return out


def recover_subject(sid: str, stft_cols: list[int]):
    """重放原始切窗；以已知 shuffle 的逆置换还原 database 波形顺序。"""
    folder = ROOT / "EEG_dataset" / sid
    csv = next(folder.glob("*_events.csv"))
    bdf = next(folder.glob("*.bdf"))
    raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose="ERROR")
    fs = float(raw.info["sfreq"])
    if not np.isclose(fs, FS):
        raise ValueError(f"{sid}: 采样率 {fs}")
    bad = bad_intervals(raw)
    buckets = {0: [], 1: []}
    for tr in load_trials(str(csv), str(bdf)):
        end = tr.start + WINDOW_S
        while end <= tr.end + 1e-6:
            if valid_window(end - WINDOW_S, end, bad, raw.n_times / fs):
                x = raw.get_data(start=round((end - WINDOW_S) * fs),
                                 stop=round(end * fs))[0]
                if len(x) == N_SAMPLES and np.isfinite(x).all():
                    buckets[tr.label].append((x.astype(np.float32), end, tr.trial_id))
            end += WINDOW_S
    raw.close()

    ordered = []
    for label, name in ((0, "relax"), (1, "task")):
        with np.load(ROOT / "database" / "windows" / name / f"sub-{sid}.npz") as d:
            stored_x = d["x"].astype(np.float32)
            stored_f = d["f"][:, stft_cols].astype(np.float32)
            stored_y = d["label"].astype(np.int8)
        bucket = buckets[label]
        n = len(stored_x)
        if n != len(bucket) or not np.all(stored_y == label):
            raise RuntimeError(f"{sid}/{name}: 数量或标签不一致")
        perm = np.random.default_rng(0).permutation(n)
        replayed = np.stack([v[0] for v in bucket])
        if not np.array_equal(replayed[perm], stored_x):
            raise RuntimeError(f"{sid}/{name}: 原始波形逐字节校验失败")
        inverse = np.empty(n, dtype=int)
        inverse[perm] = np.arange(n)
        x_original = stored_x[inverse]
        f_original = stored_f[inverse]
        for j, (wave, end, trial) in enumerate(bucket):
            if not np.array_equal(x_original[j], wave):
                raise RuntimeError(f"{sid}/{name}: 逆置换校验失败")
            ordered.append((end, trial, label, x_original[j], f_original[j]))
    ordered.sort(key=lambda v: v[0])
    t = np.asarray([v[0] for v in ordered], dtype=np.float64)
    if len(t) == 0 or np.any(np.diff(t) <= 0):
        raise RuntimeError(f"{sid}: 时间顺序异常")
    return (np.stack([v[3] for v in ordered]).astype(np.float32),
            np.stack([v[4] for v in ordered]).astype(np.float32),
            np.asarray([v[2] for v in ordered], dtype=np.int8), t,
            np.asarray([v[1] for v in ordered], dtype=np.int16))


def build_features(subs: list[str]):
    cols = stft50_indices()
    all50, all45, ys, groups, times, blocks = [], [], [], [], [], []
    for i, sid in enumerate(subs, 1):
        x, f50, y, t, blk = recover_subject(sid, cols)
        f45 = gpu_spectral_features(x)
        all50.append(f50)
        all45.append(f45)
        ys.append(y)
        groups.append(np.full(len(y), sid))
        times.append(t)
        blocks.append(blk)
        print(f"[extract {i:02d}/57] sub-{sid} n={len(y)} "
              f"relax={int((y == 0).sum())} task={int((y == 1).sum())}", flush=True)
    arrays = {"stft50": np.concatenate(all50), "spectral45": np.concatenate(all45),
              "y": np.concatenate(ys), "groups": np.concatenate(groups),
              "t_end": np.concatenate(times), "trial": np.concatenate(blocks),
              "subjects": np.asarray(subs),
              "stft50_names": np.asarray(json.loads((ROOT / "database" / "manifest.json").read_text(encoding="utf-8"))["feature_names"][23:73]),
              "spectral45_names": np.asarray(spectral_names()),
              "fine_bands_hz": np.asarray(FINE_BANDS, dtype=np.int16)}
    np.savez_compressed(FEATURES_PATH, **arrays)
    return arrays


def load_or_build_features(subs: list[str]):
    if not FEATURES_PATH.exists():
        return build_features(subs)
    with np.load(FEATURES_PATH) as d:
        arrays = {k: d[k] for k in d.files}
    if arrays["subjects"].tolist() != subs or arrays["stft50"].shape[1] != 50 \
            or arrays["spectral45"].shape[1] != 45:
        raise RuntimeError("既有 e6_features.npz 与当前固定配置不一致")
    print(f"[cache] {FEATURES_PATH.name}: {len(arrays['y'])} windows", flush=True)
    return arrays


def causal_normalize(x: torch.Tensor) -> torch.Tensor:
    """每被试按原始顺序，使用当前及过去全部帧的累计均值/方差。"""
    xd = x.double()
    zero = torch.zeros((1, x.shape[1]), dtype=torch.float64, device=x.device)
    c1 = torch.cat([zero, xd.cumsum(0)])
    c2 = torch.cat([zero, xd.square().cumsum(0)])
    count = torch.arange(1, len(x) + 1, dtype=torch.float64, device=x.device)[:, None]
    mean = c1[1:] / count
    var = (c2[1:] / count - mean.square()).clamp_min(0)
    z = (xd - mean) / (var + VAR_EPS).sqrt()
    return z.float().clamp(-CLIP, CLIP)


def prepare_normalized(arrays, subs: list[str]):
    raw = np.concatenate([arrays["stft50"], arrays["spectral45"]], axis=1)
    normalized, labels, locations = {}, {}, {}
    for sid in subs:
        loc = np.flatnonzero(arrays["groups"] == sid)
        if np.any(np.diff(arrays["t_end"][loc]) <= 0):
            raise RuntimeError(f"{sid}: 缓存未按原始时间排序")
        x = torch.as_tensor(raw[loc], device=DEV)
        normalized[sid] = causal_normalize(x).contiguous()
        labels[sid] = torch.as_tensor(arrays["y"][loc], device=DEV)
        locations[sid] = loc
    changed = torch.as_tensor(raw[locations[subs[0]]], device=DEV).clone()
    original = torch.as_tensor(raw[locations[subs[0]]], device=DEV)
    changed[64:] = changed[64:] * 7 + 123
    delta = (causal_normalize(original)[:64] - causal_normalize(changed)[:64]).abs().max()
    if float(delta) != 0.0:
        raise RuntimeError(f"因果前缀核验失败: {float(delta)}")
    return normalized, labels, locations, float(delta)


def fit_lr(x: torch.Tensor, y: torch.Tensor):
    """固定 C 的类别平衡 ridge LR；目标被试不在 x/y 中。"""
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(1e-6)
    z = (x - mean) / scale
    w = torch.zeros(x.shape[1], device=DEV, requires_grad=True)
    b = torch.zeros(1, device=DEV, requires_grad=True)
    target = y.float()
    pos_weight = (target == 0).sum() / (target == 1).sum().clamp_min(1)
    lam = 1.0 / (C * len(y))
    opt = torch.optim.LBFGS([w, b], max_iter=LBFGS_ITERS,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            z @ w + b, target, pos_weight=pos_weight) + 0.5 * lam * w.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return mean, scale, w.detach(), b.detach()


def predict(model, x: torch.Tensor) -> torch.Tensor:
    mean, scale, w, b = model
    with torch.no_grad():
        return ((x - mean) / scale) @ w + b


def classification_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(score > 0, dtype=int)
    return {"bACC": float(balanced_accuracy_score(y, pred)),
            "AUC": float(roc_auc_score(y, score)),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "n": int(len(y)), "n_relax": int((y == 0).sum()),
            "n_task": int((y == 1).sum())}


def save_scores(arrays, subs, scores):
    out = {"subjects": np.asarray(subs)}
    for sid in subs:
        loc = np.flatnonzero(arrays["groups"] == sid)
        out[f"y__{sid}"] = arrays["y"][loc]
        out[f"t_end__{sid}"] = arrays["t_end"][loc]
        out[f"trial__{sid}"] = arrays["trial"][loc]
        for name in VARIANTS:
            if sid in scores[name]:
                out[f"{name}__{sid}"] = scores[name][sid]
    np.savez_compressed(SCORES_PATH, **out)


def evaluate(arrays, subs, scores):
    variants = {}
    for name in VARIANTS:
        rows = []
        for sid in subs:
            loc = np.flatnonzero(arrays["groups"] == sid)
            rows.append({"subject": sid,
                         **classification_metrics(arrays["y"][loc], scores[name][sid])})
        y = np.concatenate([arrays["y"][arrays["groups"] == sid] for sid in subs])
        s = np.concatenate([scores[name][sid] for sid in subs])
        macro = {m: float(np.mean([r[m] for r in rows])) for m in ("bACC", "AUC", "F1")}
        variants[name] = {"dim": {"stft50": 50, "spectral45": 45, "combined95": 95}[name],
                          "macro": macro, "pooled": classification_metrics(y, s),
                          "n_bACC_ge_0.9": int(sum(r["bACC"] >= 0.9 for r in rows)),
                          "per_subject": rows}
        print(f"[result {name:10s}] macro bACC={macro['bACC']:.6f} AUC={macro['AUC']:.6f} "
              f"F1={macro['F1']:.6f}; pooled bACC={variants[name]['pooled']['bACC']:.6f} "
              f"AUC={variants[name]['pooled']['AUC']:.6f} F1={variants[name]['pooled']['F1']:.6f}",
              flush=True)
    increment = {}
    for scope in ("macro", "pooled"):
        increment[scope] = {m: variants["combined95"][scope][m] - variants["stft50"][scope][m]
                            for m in ("bACC", "AUC", "F1")}
    return variants, increment


def write_report(result):
    lines = ["# E6：细分频带 DE、相对功率与谱比值", "",
             "## 全段结果（57 人 LOSO）", "",
             "主结果为逐被试宏平均；F1 以 task 为正类，logit > 0 判 task。所有有效标签帧均纳入，无晚段筛选。", "",
             "| 特征 | 维数 | 宏 bACC | 宏 AUC | 宏 F1 | 汇总 bACC | 汇总 AUC | 汇总 F1 | bACC≥.9 人数 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"stft50": "50 维 STFT 对照", "spectral45": "45 维新增频谱",
              "combined95": "50+45 维联合"}
    for name in VARIANTS:
        r = result["variants"][name]
        a, p = r["macro"], r["pooled"]
        lines.append(f"| {labels[name]} | {r['dim']} | {a['bACC']:.4f} | {a['AUC']:.4f} | "
                     f"{a['F1']:.4f} | {p['bACC']:.4f} | {p['AUC']:.4f} | {p['F1']:.4f} | "
                     f"{r['n_bACC_ge_0.9']}/57 |")
    d = result["increment_combined95_minus_stft50"]
    lines += ["", "### 联合特征相对 50 维对照的增量", "",
              "| 口径 | ΔbACC | ΔAUC | ΔF1 |", "|---|---:|---:|---:|"]
    for scope, cn in (("macro", "逐人宏平均"), ("pooled", "全帧汇总")):
        lines.append(f"| {cn} | {d[scope]['bACC']:+.4f} | {d[scope]['AUC']:+.4f} | {d[scope]['F1']:+.4f} |")
    lines += ["", "## 特征与因果协议", "",
              f"- 数据：`database/windows` 的 {result['protocol']['n_windows']} 个原始 6 s、250 Hz、单通道、非重叠窗，57 人。BDF 只用于重放切窗和恢复时间；114 个标签分片均通过原始波形逐字节校验及固定种子逆置换校验。",
              "- 对照：同一数据库 `f` 的第 23–72 列，共 50 维；定义同 `highdim_deep_ablation.py` 的基座 STFT：1–45 Hz 五宽频带，1 s Hann、0.5 s hop，log-power/relative 的 mean/std/slope/max/min。",
              "- 新增频谱：原始波形在 GPU 上分 11 个 1 s Hann 段（hop 0.5 s，逐段去均值），计算单边 PSD 并对 11 段取平均。14 个细频带为 1–2、2–4、4–6、6–8、8–10、10–13、13–15、15–18、18–22、22–26、26–30、30–35、35–40、40–45 Hz。",
              "- **DE 公式**：$DE_b=0.5\\log(2\\pi e\\,\\sigma_b^2)$，自然对数；$\\sigma_b^2=\\frac1{11}\\sum_t\\sum_{f_k\\in[l_b,u_b)}P_t[k]\\Delta f$ 是 Welch 型带限方差，$\\Delta f=1$ Hz。14 维 DE。",
              "- 相对功率 $r_b=\\sigma_b^2/\\sum_{j=1}^{14}\\sigma_j^2$，14 维；谱比值取 $\\log(\\sigma_a^2/\\sigma_b^2)$，含 13 个相邻细带比和 theta/alpha、alpha/beta、beta/gamma、delta/beta 四个宽带比，17 维。新增合计 45 维。",
              "- 每名被试独立按原始时间顺序做累计因果 z-score：第 $t$ 帧均值和方差只由第 $1..t$ 帧（含当前）计算；首帧为 0，固定裁剪到 [-10,10]。训练和测试被试使用完全相同的规则。前缀篡改核验最大变化为 0。",
              f"- 每折只用其余 56 人训练；训练折再拟合一个全局特征尺度。GPU LR 固定为类别平衡 BCE、C={C}、L2=$1/(C\\,N)$、LBFGS 最大 {LBFGS_ITERS} 次；不做目标被试训练、阈值调节或超参选择。",
              "", "## 逐人全段结果", "",
              "| 被试 | n | 50维 bACC | AUC | F1 | 45维 bACC | AUC | F1 | 95维 bACC | AUC | F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    per = {name: {r["subject"]: r for r in result["variants"][name]["per_subject"]}
           for name in VARIANTS}
    for sid in result["protocol"]["subjects"]:
        row = [f"sub-{sid}", str(per["stft50"][sid]["n"])]
        for name in VARIANTS:
            r = per[name][sid]
            row += [f"{r[m]:.4f}" for m in ("bACC", "AUC", "F1")]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 与 E1 的帧缓存差异", "",
              "本实验仅借用 review.md E1 的累计因果归一化思想，**不能直接比较数值**：E1/E46 的 `L.load()` 使用 `test/11_individual_20261102/sliding1` 的 878 维缓存，6 s 窗、1 s stride；本实验使用 `database/windows` 原始波形和其中 50 维 STFT 列，6 s 窗、6 s stride，并在 GPU 重新计算 45 维频谱。两者帧数、帧位置、特征维数均不同。",
              "", "## 复现与产物", "",
              "```powershell", "& .\\.venv\\Scripts\\python.exe test\\51_de_spectral\\e6_de_spectral.py", "```", "",
              "`e6_de_spectral.py` 为完整脚本；`e6_results.json` 含协议、逐人和总体指标；`e6_scores.npz` 含逐人标签、时间、trial 与三种模型 logit；`e6_features.npz` 为固定特征缓存；`e6_run.log` 为实际运行日志。",
              f"实际 GPU 峰值 reserved 显存：{result['runtime']['gpu_max_reserved_mib']:.1f} MiB（<4 GB）。总耗时 {result['runtime']['seconds']:.1f} s。", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    global DEV
    start = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("E6 要求 CUDA GPU")
    DEV = torch.device("cuda")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    subs = sorted(p.stem[4:] for p in (ROOT / "database" / "windows" / "relax").glob("sub-*.npz"))
    if len(subs) != 57:
        raise RuntimeError(f"预期57人，实际{len(subs)}")
    print(f"[setup] device={torch.cuda.get_device_name(DEV)} torch={torch.__version__} "
          f"subjects={len(subs)} C={C} iters={LBFGS_ITERS}", flush=True)
    arrays = load_or_build_features(subs)
    if len(arrays["y"]) != 16067 or not np.isfinite(arrays["stft50"]).all() \
            or not np.isfinite(arrays["spectral45"]).all():
        raise RuntimeError("特征数量或有限性核验失败")
    normalized, labels, _, prefix_delta = prepare_normalized(arrays, subs)
    scores = {name: {} for name in VARIANTS}
    for i, target in enumerate(subs, 1):
        train = [q for q in subs if q != target]
        assert target not in train and len(train) == 56
        xtrain = torch.cat([normalized[q] for q in train])
        ytrain = torch.cat([labels[q] for q in train])
        for name, columns in VARIANTS.items():
            model = fit_lr(xtrain[:, columns], ytrain)
            score = predict(model, normalized[target][:, columns]).float()
            if not bool(torch.isfinite(score).all()):
                raise RuntimeError(f"{target}/{name}: 非有限分数")
            scores[name][target] = score.cpu().numpy()
        del xtrain, ytrain
        msg = []
        for name in VARIANTS:
            m = classification_metrics(arrays["y"][arrays["groups"] == target], scores[name][target])
            msg.append(f"{name}:bACC={m['bACC']:.4f},AUC={m['AUC']:.4f},F1={m['F1']:.4f}")
        print(f"[LOSO {i:02d}/57] sub-{target} " + " | ".join(msg), flush=True)
        save_scores(arrays, subs, scores)
    if any(len(scores[name]) != 57 for name in VARIANTS):
        raise RuntimeError("LOSO 未完整覆盖57人")
    variants, increment = evaluate(arrays, subs, scores)
    result = {
        "experiment": "E6 DE and fine spectral features",
        "protocol": {"n_subjects": 57, "n_windows": int(len(arrays["y"])),
                     "subjects": subs, "window_seconds": 6, "stride_seconds": 6,
                     "source": "database/windows raw x + cached STFT50; BDF replay only for exact order recovery",
                     "order_check": "114/114 shards: stored_x == replayed_x[default_rng(0).permutation(n)] and inverse-permutation byte equality",
                     "normalization": "per-subject cumulative causal z using current and all past frames; float64 moments; var_eps=1e-8; clamp[-10,10]",
                     "causal_prefix_max_abs_change": prefix_delta,
                     "classifier": f"GPU logistic regression, balanced BCE, C={C}, lambda=1/(C*N), LBFGS max_iter={LBFGS_ITERS}, threshold=0",
                     "target_exclusion": "each target excluded from training, global scaling, hyperparameter and threshold selection"},
        "features": {"stft50": {"dim": 50, "source_columns": "f[:,23:73]"},
                     "spectral45": {"dim": 45, "DE": 14, "relative_power": 14,
                                    "log_spectral_ratios": 17, "bands_hz": FINE_BANDS,
                                    "DE_formula": "0.5*ln(2*pi*e*band_limited_variance)",
                                    "PSD": "1 s periodic Hann, 0.5 s hop, per-segment demean, one-sided density, mean of 11 frames, df=1 Hz"},
                     "combined95": {"dim": 95}},
        "device": {"name": torch.cuda.get_device_name(DEV), "torch": torch.__version__},
        "variants": variants, "increment_combined95_minus_stft50": increment,
    }
    result["runtime"] = {"seconds": time.time() - start,
                         "gpu_max_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                         "gpu_max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    if result["runtime"]["gpu_max_reserved_mib"] >= 4096:
        raise RuntimeError("GPU reserved 显存超过4GB")
    RESULTS_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    save_scores(arrays, subs, scores)
    write_report(result)
    print(f"[complete] seconds={result['runtime']['seconds']:.1f} "
          f"GPU_reserved={result['runtime']['gpu_max_reserved_mib']:.1f}MiB", flush=True)


if __name__ == "__main__":
    main()
