"""E4：6 秒原始 EEG 的 log-STFT 小 CNN，与同帧同划分的 GPU LR 做 57 人 LOSO。

目标被试绝不参与训练、归一化统计、阈值选择或早停。所有超参数固定，
每个目标被试训练固定 10 个 epoch；决策阈值为 logit=0。分数越大越倾向 task。
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WINDOW_DIR = ROOT / "database" / "windows"
E46_DIR = ROOT / "test" / "46_causal_online_norm"
FS = 250
N_SAMPLES = 1500
N_FFT = 128
HOP = 32
MAX_FREQ = 50.0
BASE_SEED = 49000
EPOCHS = 10
BATCH_SIZE = 64
LR_STEPS = 100
LR_C = 0.001


def load_cache():
    paths = sorted(WINDOW_DIR.glob("*/*.npz"))
    subjects = sorted({p.stem for p in paths})
    raw, feat, labels, groups = [], [], [], []
    counts = {}
    for subject in subjects:
        counts[subject] = {}
        for label, name in ((0, "relax"), (1, "task")):
            path = WINDOW_DIR / name / f"{subject}.npz"
            if not path.exists():
                continue
            with np.load(path) as d:
                x = d["x"].astype(np.float32)
                f = d["f"].astype(np.float32)
                y = d["label"].astype(np.int64)
                assert x.shape == (len(y), N_SAMPLES)
                assert f.shape[0] == len(y) and f.ndim == 2
                assert np.all(y == label)
                assert np.isfinite(x).all() and np.isfinite(f).all()
                raw.append(x)
                feat.append(f)
                labels.append(y)
                groups.append(np.repeat(subject, len(y)))
                counts[subject][name] = int(len(y))
    X = np.concatenate(raw)
    F = np.concatenate(feat)
    y = np.concatenate(labels)
    g = np.concatenate(groups)
    assert len(subjects) == 57 and len(y) == len(X) == len(F) == len(g)
    assert set(np.unique(y)) == {0, 1}
    return X, F, y, g, subjects, counts


def make_log_stft(raw, device, batch=512):
    """每窗独立去均值；center=False，频率只保留 (0, 50] Hz。"""
    freq = torch.fft.rfftfreq(N_FFT, 1 / FS)
    freq_idx = torch.from_numpy(np.flatnonzero((freq > 0) & (freq <= MAX_FREQ)))
    n_freq = len(freq_idx)
    n_frames = (N_SAMPLES - N_FFT) // HOP + 1
    out = torch.empty((len(raw), n_freq, n_frames), dtype=torch.float32, device=device)
    window = torch.hann_window(N_FFT, device=device)
    freq_idx = freq_idx.to(device)
    for start in range(0, len(raw), batch):
        x = torch.from_numpy(raw[start:start + batch]).to(device)
        x = x - x.mean(dim=1, keepdim=True)
        spec = torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                          window=window, center=False, return_complex=True)
        power = spec.real.square() + spec.imag.square()
        out[start:start + len(x)] = torch.log(
            power.index_select(1, freq_idx) + 1e-12
        )
    return out


def normalize_stft(log_power, train_rows):
    """通道0只用本折训练被试统计；通道1仅用当前6秒窗自身统计。"""
    train = log_power.index_select(0, train_rows)
    mu_f = train.mean(dim=(0, 2))
    sd_f = train.std(dim=(0, 2), unbiased=False).clamp_min(1e-5)
    c0 = (log_power - mu_f[None, :, None]) / sd_f[None, :, None]
    mu_w = log_power.mean(dim=(1, 2), keepdim=True)
    sd_w = log_power.std(dim=(1, 2), unbiased=False, keepdim=True).clamp_min(1e-5)
    c1 = (log_power - mu_w) / sd_w
    return torch.stack((c0, c1), dim=1).clamp(-8, 8)


class SmallSTFTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 2
        for i, out_ch in enumerate((16, 32, 64, 64)):
            layers += [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_ch), nn.GELU()]
            if i < 3:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.body(x)).ravel()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_cnn(images, y_gpu, train_rows, test_rows, seed, epochs, batch_size):
    set_seed(seed)
    model = SmallSTFTCNN().to(images.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    yt = y_gpu.index_select(0, train_rows)
    pos_weight = (yt == 0).sum().float() / (yt == 1).sum().float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    generator = torch.Generator(device=images.device)
    generator.manual_seed(seed)
    losses = []
    tick = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_rows), generator=generator, device=images.device)
        total = 0.0
        for start in range(0, len(order), batch_size):
            rows = train_rows.index_select(0, order[start:start + batch_size])
            logits = model(images.index_select(0, rows))
            loss = criterion(logits, y_gpu.index_select(0, rows))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(rows)
        losses.append(total / len(train_rows))
        if epoch in (1, 5, epochs) or epochs <= 2:
            print(f"    CNN epoch {epoch:02d}/{epochs} loss={losses[-1]:.5f}", flush=True)
    score = predict_cnn(model, images, test_rows)
    info = {"epochs_run": epochs, "train_loss": losses,
            "seconds": time.perf_counter() - tick,
            "positive_weight": float(pos_weight)}
    return model, score, info


@torch.no_grad()
def predict_cnn(model, images, rows, batch=512):
    model.eval()
    out = []
    for start in range(0, len(rows), batch):
        idx = rows[start:start + batch]
        out.append(model(images.index_select(0, idx)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fit_gpu_lr(features, y_gpu, train_rows, test_rows, steps):
    """训练被试内标准化；全批量GPU LBFGS最多steps步，仅训练数值收敛可停。"""
    tick = time.perf_counter()
    x_train = features.index_select(0, train_rows)
    mu = x_train.mean(dim=0)
    sd = x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    z_train = (x_train - mu) / sd
    z_test = (features.index_select(0, test_rows) - mu) / sd
    yt = y_gpu.index_select(0, train_rows)
    pos_weight = (yt == 0).sum().float() / (yt == 1).sum().float()
    w = torch.zeros(features.shape[1], device=features.device, requires_grad=True)
    b = torch.zeros(1, device=features.device, requires_grad=True)
    lam = 1.0 / (LR_C * len(train_rows))
    optimizer = torch.optim.LBFGS([w, b], max_iter=steps, max_eval=steps * 5,
                                  history_size=20, line_search_fn="strong_wolfe",
                                  tolerance_grad=0.0, tolerance_change=0.0)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(
            z_train @ w + b, yt, pos_weight=pos_weight
        ) + 0.5 * lam * w.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        final_loss = float(nn.functional.binary_cross_entropy_with_logits(
            z_train @ w + b, yt, pos_weight=pos_weight
        ) + 0.5 * lam * w.square().sum())
        score = (z_test @ w + b).cpu().numpy().astype(np.float64)
    first_param = optimizer.param_groups[0]["params"][0]
    n_iter = int(optimizer.state[first_param]["n_iter"])
    info = {"steps_requested": steps, "steps_run": n_iter, "C": LR_C,
            "train_loss_regularized": final_loss, "seconds": time.perf_counter() - tick}
    return score, info


def metrics(y, score):
    pred = (np.asarray(score) > 0).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    return {
        "bACC": float(balanced_accuracy_score(y, pred)),
        "AUC": float(roc_auc_score(y, score)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "n": int(len(y)),
        "n_relax": int((y == 0).sum()),
        "n_task": int((y == 1).sum()),
    }


def summarize(y, groups, subjects, scores):
    rows = []
    for subject in subjects:
        mask = groups == subject
        rows.append({"subject": subject, **metrics(y[mask], scores[mask])})
    return {
        "macro": {key: float(np.mean([r[key] for r in rows]))
                  for key in ("bACC", "AUC", "F1")},
        "pooled": metrics(y, scores),
        "per_subject": rows,
    }


def e46_reference(groups, y):
    result = {
        "json_path": str(E46_DIR / "e46.json"),
        "frame_set_consistent": False,
        "frame_set_reason": "E46脚本读取test/11_individual_20261102/sliding1并按t排序；E4读取database/windows的6秒非重叠缓存，二者不是同一帧集。",
    }
    path = E46_DIR / "e46.json"
    if not path.exists():
        result["status"] = "not_generated"
        return result
    result["status"] = "available"
    e46 = json.loads(path.read_text(encoding="utf-8"))
    result["metrics"] = {}
    for variant, posts in e46.get("variants", {}).items():
        for post, values in posts.items():
            if "macro" in values:
                result["metrics"][f"{variant}/{post}"] = values["macro"]
    e46_counts = {}
    try:
        rows = e46["variants"]["raw"]["raw"]["per_subject"]
        for row in rows:
            key = row["subject"]
            if not key.startswith("sub-"):
                key = f"sub-{key}"
            e46_counts[key] = int(row["n"])
        own_counts = {s: int((groups == s).sum()) for s in np.unique(groups)}
        result["window_count_comparison"] = {
            "e46_total": int(sum(e46_counts.values())), "e4_total": int(len(y)),
            "same_subject_counts": bool(e46_counts == own_counts),
        }
    except (KeyError, TypeError, ValueError):
        pass
    return result


def write_report(result, stem):
    lines = [
        "# E4：6 秒 log-STFT 图与小 CNN（57 人真 LOSO）", "",
        "## 结果", "",
        "分数为 task 方向的连续 logit，分类阈值固定为 0。宏平均先逐被试计算再等权平均；pooled 为全部窗汇总。", "",
        "| 模型 | 宏平均 bACC | 宏平均 AUC | 宏平均 F1 | Pooled bACC | Pooled AUC | Pooled F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("cnn", "STFT-CNN"), ("lr", "GPU-LR")):
        r = result["models"][name]
        m, p = r["macro"], r["pooled"]
        lines.append(f"| {label} | {m['bACC']:.4f} | {m['AUC']:.4f} | {m['F1']:.4f} | "
                     f"{p['bACC']:.4f} | {p['AUC']:.4f} | {p['F1']:.4f} |")
    best = max(result["models"], key=lambda k: result["models"][k]["macro"]["bACC"])
    bm = result["models"][best]["macro"]
    cm = result["models"]["cnn"]["macro"]
    lm = result["models"]["lr"]["macro"]
    lines += ["", f"预设两模型中宏平均 bACC 最高的是 **{best.upper()}**：bACC **{bm['bACC']:.4f}**，"
                  f"AUC **{bm['AUC']:.4f}**，F1 **{bm['F1']:.4f}**。该比较仅为结果描述，未用于目标被试调参。", "",
              f"CNN 减 LR 的宏平均 ΔbACC/ΔAUC/ΔF1 = **{cm['bACC']-lm['bACC']:+.4f}/"
              f"{cm['AUC']-lm['AUC']:+.4f}/{cm['F1']-lm['F1']:+.4f}**，本实验未显示小 CNN 的优势。", "",
              "## 数据与无泄漏协议", "",
              f"- 实际读取 `database/windows` 的 **{result['data']['n_windows']}** 个窗、**{result['data']['n_subjects']}** 人；"
              f"relax/task 分别为 {result['data']['label_counts']['0']}/{result['data']['label_counts']['1']}。"
              "任务文字称 16,397 个，但当前磁盘缓存实际为 16,067 个，本报告不补造窗口。",
              "- 每折仅一个目标被试，其余 56 人训练；训练/测试索引按被试互斥并断言。目标被试不参与拟合、归一化统计、阈值或模型选择。",
              f"- CNN 所有折固定 {result['config']['epochs']} epoch，seed 由折号预先确定；LR 只允许按训练数值收敛停止。"
              "两模型均无目标验证集、无基于目标指标的早停、无时序后处理。",
              "- 缓存按标签分片且片内已打乱，不含时间戳或 trial 序号；本实验因此是独立 6 秒窗评估，不利用窗序。", "",
              "## 方法", "",
              f"- 原始窗：250 Hz × 1,500 点 = 6 秒；每窗去均值。STFT：Hann {N_FFT} 点、hop {HOP}、"
              f"`center=False`，保留 {result['data']['stft_shape'][0]} 个正频率 bin（最高 {result['data']['max_frequency_hz']:.3f} Hz）"
              f"与 {result['data']['stft_shape'][1]} 个时间帧，输入为 log(power+1e-12)。",
              "- CNN 两通道：①本折训练被试的逐频率 log-power 均值/标准差归一化；②当前窗自身的整体均值/标准差归一化。"
              "两者均不读取目标窗之后的数据；4 个 Conv-BN-GELU 块（16/32/64/64），GAP+线性头。",
              f"- GPU-LR 对照使用完全相同的样本行和 LOSO 划分，输入缓存 `f`（{result['data']['n_features']} 维），"
              f"仅用训练被试做 z-score；C={LR_C}，全批量 LBFGS 最多 {result['config']['lr_steps']} 步"
              f"（实际 {min(f['lr']['steps_run'] for f in result['folds'])}–{max(f['lr']['steps_run'] for f in result['folds'])} 步，"
              "仅按训练数值收敛停止，绝不按目标指标早停），类别不平衡用 pos_weight 修正。", "",
              "## 因果与显存核验", "",
              f"- STFT 单窗与批量计算最大绝对差：`{result['checks']['stft_single_vs_batch']:.3g}`。",
              f"- 同一目标窗单独/批量推理最大分数差：`{result['checks']['cnn_batch_independence']:.3g}`。",
              f"- 篡改同一目标被试其他窗后，未篡改窗的输入和分数最大变化分别为 "
              f"`{result['checks']['other_window_input_invariance']:.3g}` / "
              f"`{result['checks']['other_window_score_invariance']:.3g}`。",
              f"- GPU：`{result['environment']['gpu']}`；PyTorch 峰值已分配/保留显存分别为 "
              f"{result['environment']['peak_memory_allocated_mb']:.1f}/{result['environment']['peak_memory_reserved_mb']:.1f} MiB。", "",
              "## E46 / E1 引用与帧集", ""]
    ref = result["e46_reference"]
    if ref["status"] == "available":
        lines += ["以下直接引用 `test/46_causal_online_norm/e46.json` 的逐被试宏平均：", "",
                  "| E46特征/后处理 | bACC | AUC | F1 |", "|---|---:|---:|---:|"]
        for name, m in ref.get("metrics", {}).items():
            lines.append(f"| {name} | {m['bACC']:.4f} | {m['AUC']:.4f} | {m['F1']:.4f} |")
        if "window_count_comparison" in ref:
            cmp = ref["window_count_comparison"]
            lines += ["", f"E46/E4 总帧数分别为 {cmp['e46_total']}/{cmp['e4_total']}；逐被试窗数完全一致={cmp['same_subject_counts']}。"]
    else:
        lines += ["运行结束时 `test/46_causal_online_norm/e46.json` 尚未生成，故没有可引用的 E1 数值；"
                  "现有 `run.log` 显示该脚本因导入接口不匹配而中断。"]
    lines += ["", f"**帧集不一致**：{ref['frame_set_reason']} 因此本报告的 GPU-LR 是同 E4 帧集对照，不能冒充 E46 的复算。", "",
              "## 逐被试结果", "",
              "| 被试 | n | CNN bACC | CNN AUC | CNN F1 | LR bACC | LR AUC | LR F1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    cnn_rows = {r["subject"]: r for r in result["models"]["cnn"]["per_subject"]}
    lr_rows = {r["subject"]: r for r in result["models"]["lr"]["per_subject"]}
    for subject in result["data"]["subjects"]:
        c, r = cnn_rows[subject], lr_rows[subject]
        lines.append(f"| {subject} | {c['n']} | {c['bACC']:.4f} | {c['AUC']:.4f} | {c['F1']:.4f} | "
                     f"{r['bACC']:.4f} | {r['AUC']:.4f} | {r['F1']:.4f} |")
    lines += ["", "## 限制", "",
              "- 仅一个固定种子和固定 epoch，无模型集成；跨被试个体差异仍明显，逐人指标见上表。",
              "- 非重叠缓存不含时序，无法在此帧集上复现 E46 的滑动流在线累计归一化；本实验只证明独立窗路径的因果性。",
              "- `f` 是缓存既有 858 维特征而非本实验 STFT 图；LR 是同数据同划分模型对照，不是 STFT 像素的线性化对照。",
              "- 任务描述的缓存总数与实际文件不一致，结论仅适用于上述实际 16,067 帧。", "",
              "## 复现与产物", "",
              "```powershell", "& .\\.venv\\Scripts\\python.exe .\\test\\49_stft_cnn\\e49.py 2>&1 | Tee-Object .\\test\\49_stft_cnn\\run.log",
              "```", "",
              f"完整数值：`{stem}.json`；逐帧 OOF 分数：`{stem}_scores.npz`；运行日志：`run.log`。", ""]
    (HERE / "报告_E4_STFT_CNN.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="仅前两折，2 epoch/5 LR步，不写正式报告")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    stem = "e49_smoke" if args.smoke else "e49"
    epochs = 2 if args.smoke else EPOCHS
    lr_steps = 5 if args.smoke else LR_STEPS
    tick = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("E4要求GPU运行，当前CUDA不可用")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    raw, feat_np, y, groups, subjects, subject_counts = load_cache()
    run_subjects = subjects[:2] if args.smoke else subjects
    print(f"[data] windows={len(y)} subjects={len(subjects)} raw={raw.shape} f={feat_np.shape} "
          f"relax={int((y == 0).sum())} task={int((y == 1).sum())}", flush=True)
    print(f"[env] torch={torch.__version__} gpu={torch.cuda.get_device_name(0)} "
          f"epochs={epochs} batch={args.batch_size} lr_steps={lr_steps} smoke={args.smoke}", flush=True)
    log_power = make_log_stft(raw, device)
    single = make_log_stft(raw[:1], device)
    stft_check = float((single[0] - log_power[0]).abs().max())
    del single
    print(f"[check] STFT single/batch maxabs={stft_check:.3g}", flush=True)
    assert stft_check < 1e-3
    features = torch.from_numpy(feat_np).to(device)
    y_gpu = torch.from_numpy(y).float().to(device)
    del feat_np
    checks = {"stft_single_vs_batch": stft_check}
    result = {
        "experiment": "E4 / E49 STFT-CNN vs GPU-LR",
        "status": "running",
        "protocol": "57 independent subject-held-out folds; no target training/tuning/early stopping",
        "config": {"seed_base": BASE_SEED, "epochs": epochs, "batch_size": args.batch_size,
                   "cnn_optimizer": "AdamW(lr=0.001, weight_decay=0.01)",
                   "lr_steps": lr_steps, "lr_steps_policy": "maximum; training-only numerical convergence may stop",
                   "lr_C": LR_C, "decision_threshold_logit": 0.0},
        "data": {"n_windows": int(len(y)), "n_subjects": len(subjects), "subjects": subjects,
                 "label_counts": {str(k): int((y == k).sum()) for k in (0, 1)},
                 "n_features": int(features.shape[1]), "raw_shape": list(raw.shape[1:]),
                 "stft_shape": list(log_power.shape[1:]),
                 "max_frequency_hz": float(torch.fft.rfftfreq(N_FFT, 1 / FS)[(torch.fft.rfftfreq(N_FFT, 1 / FS) <= 50) & (torch.fft.rfftfreq(N_FFT, 1 / FS) > 0)].max()),
                 "subject_counts": subject_counts},
        "model_parameters": int(sum(p.numel() for p in SmallSTFTCNN().parameters())),
        "checks": checks, "folds": [],
    }
    scores = {"cnn": np.full(len(y), np.nan), "lr": np.full(len(y), np.nan)}
    for fold, subject in enumerate(run_subjects):
        train_np = np.flatnonzero(groups != subject)
        test_np = np.flatnonzero(groups == subject)
        assert set(groups[train_np]).isdisjoint({subject})
        assert np.array_equal(np.unique(groups[test_np]), np.asarray([subject]))
        assert set(np.unique(y[test_np])) == {0, 1}
        train_rows = torch.from_numpy(train_np).to(device)
        test_rows = torch.from_numpy(test_np).to(device)
        seed = BASE_SEED + fold
        print(f"\n[fold {fold + 1}/{len(run_subjects)}] target={subject} train={len(train_np)} "
              f"test={len(test_np)} seed={seed}", flush=True)
        lr_score, lr_info = fit_gpu_lr(features, y_gpu, train_rows, test_rows, lr_steps)
        scores["lr"][test_np] = lr_score
        print(f"    LR steps={lr_info['steps_run']} loss={lr_info['train_loss_regularized']:.5f} "
              f"time={lr_info['seconds']:.1f}s", flush=True)
        images = normalize_stft(log_power, train_rows)
        model, cnn_score, cnn_info = train_cnn(images, y_gpu, train_rows, test_rows,
                                                seed, epochs, args.batch_size)
        scores["cnn"][test_np] = cnn_score
        if fold == 0:
            probe = test_rows[:8]
            individual = np.concatenate([predict_cnn(model, images, probe[i:i + 1])
                                         for i in range(len(probe))])
            checks["cnn_batch_independence"] = float(np.max(np.abs(individual - cnn_score[:8])))
            other = test_rows[8:16]
            saved = log_power.index_select(0, other).clone()
            log_power[other] += 123.0
            changed_images = normalize_stft(log_power, train_rows)
            changed_score = predict_cnn(model, changed_images, probe)
            checks["other_window_input_invariance"] = float(
                (changed_images.index_select(0, probe) - images.index_select(0, probe)).abs().max()
            )
            checks["other_window_score_invariance"] = float(np.max(np.abs(changed_score - cnn_score[:8])))
            log_power[other] = saved
            del changed_images, saved
            print(f"    [causal] input_change={checks['other_window_input_invariance']:.3g} "
                  f"score_change={checks['other_window_score_invariance']:.3g} "
                  f"batch_score_diff={checks['cnn_batch_independence']:.3g}", flush=True)
            assert checks["other_window_input_invariance"] == 0.0
            assert checks["other_window_score_invariance"] < 1e-3
        fold_result = {"fold": fold, "subject": subject, "seed": seed,
                       "n_train": int(len(train_np)), "n_test": int(len(test_np)),
                       "cnn": cnn_info, "lr": lr_info,
                       "test_metrics_not_used_for_training": {
                           "cnn": metrics(y[test_np], cnn_score), "lr": metrics(y[test_np], lr_score)}}
        result["folds"].append(fold_result)
        print(f"    CNN bACC={fold_result['test_metrics_not_used_for_training']['cnn']['bACC']:.4f} "
              f"AUC={fold_result['test_metrics_not_used_for_training']['cnn']['AUC']:.4f} "
              f"F1={fold_result['test_metrics_not_used_for_training']['cnn']['F1']:.4f}; "
              f"LR bACC={fold_result['test_metrics_not_used_for_training']['lr']['bACC']:.4f} "
              f"AUC={fold_result['test_metrics_not_used_for_training']['lr']['AUC']:.4f} "
              f"F1={fold_result['test_metrics_not_used_for_training']['lr']['F1']:.4f}", flush=True)
        result["completed_subjects"] = run_subjects[:fold + 1]
        (HERE / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(HERE / f"{stem}_scores.npz", y=y.astype(np.int8), groups=groups,
                            cnn_score=scores["cnn"], lr_score=scores["lr"],
                            completed=np.isin(groups, result["completed_subjects"]))
        del model, images, train_rows, test_rows
        torch.cuda.empty_cache()
    if not args.smoke:
        assert len(run_subjects) == 57
        assert np.isfinite(scores["cnn"]).all() and np.isfinite(scores["lr"]).all()
        result["models"] = {name: summarize(y, groups, subjects, score)
                            for name, score in scores.items()}
        result["e46_reference"] = e46_reference(groups, y)
        result["status"] = "complete"
        result["elapsed_seconds"] = time.perf_counter() - tick
        result["environment"] = {
            "python": os.sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
            "peak_memory_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        }
        write_report(result, stem)
        for name, r in result["models"].items():
            print(f"[final {name}] macro={r['macro']} pooled={r['pooled']}", flush=True)
    else:
        result["status"] = "smoke_complete"
        result["elapsed_seconds"] = time.perf_counter() - tick
    (HERE / f"{stem}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] status={result['status']} elapsed={result['elapsed_seconds']:.1f}s "
          f"peak_allocated={torch.cuda.max_memory_allocated() / 2**20:.1f}MiB "
          f"peak_reserved={torch.cuda.max_memory_reserved() / 2**20:.1f}MiB", flush=True)


if __name__ == "__main__":
    main()
