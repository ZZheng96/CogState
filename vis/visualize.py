"""单通道 EEG 波形与特征的描述性可视化分析。

默认复用 test/highdim_deep_features.npz，输出波形、频谱、特征分布和
受试者内单特征 AUC 图。所有统计仅用于数据诊断，不作为模型泛化评估。
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft, welch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cogstate import BANDS  # noqa: E402
from preprocessing import PreprocessingConfig, causal_filter  # noqa: E402

FS = 250.0
CLASS_NAMES = {0: "relax", 1: "task"}
CLASS_COLORS = {0: "#2878B5", 1: "#D1495B"}
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.2,
})


def load_data(data_path: Path, manifest_path: Path):
    """读取并校验波形、特征、标签及受试者对齐关系。"""
    if not data_path.exists():
        raise FileNotFoundError(f"数据缓存不存在：{data_path}")
    with np.load(data_path, allow_pickle=False) as data:
        missing = {"raw", "X", "y", "groups"}.difference(data.files)
        if missing:
            raise ValueError(f"缓存缺少字段：{sorted(missing)}")
        raw = data["raw"]
        features = data["X"].astype(np.float64, copy=False)
        labels = data["y"].astype(np.int64, copy=False)
        groups = data["groups"].astype(str, copy=False)

    if raw.ndim != 2 or features.ndim != 2:
        raise ValueError("raw 和 X 必须为二维数组")
    if not (len(raw) == len(features) == len(labels) == len(groups)):
        raise ValueError("波形、特征、标签和受试者数量不一致")
    if raw.shape[1] != round(6 * FS):
        raise ValueError(f"当前脚本要求 250 Hz、6 秒窗口，实际为 {raw.shape}")
    if not np.isfinite(raw).all() or not np.isfinite(features).all():
        raise ValueError("数据包含 NaN 或无穷值")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("标签只能为 0(relax) 或 1(task)")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names = np.asarray(manifest["feature_names"], dtype=str)
    if len(names) != features.shape[1]:
        raise ValueError(f"特征名数量 {len(names)} 与特征维数 {features.shape[1]} 不一致")
    return raw, features, labels, groups, names, manifest


def representative_windows(raw, labels, groups, subject):
    """为每个类别选取原始标准差接近中位数的窗口，避免极端示例。"""
    subject_ids = np.flatnonzero(groups == subject)
    if not len(subject_ids):
        available = ", ".join(sorted(set(groups)))
        raise ValueError(f"受试者 {subject} 不存在；可选：{available}")
    selected = {}
    for label in (0, 1):
        ids = subject_ids[labels[subject_ids] == label]
        if not len(ids):
            raise ValueError(f"受试者 {subject} 缺少 {CLASS_NAMES[label]} 窗口")
        scales = raw[ids].std(axis=1, dtype=np.float64)
        ordered = ids[np.argsort(scales, kind="stable")]
        selected[label] = int(ordered[len(ordered) // 2])
    return selected


def save_figure(fig, path: Path, dpi: int):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已保存 {path}", flush=True)


def plot_waveforms(raw, labels, groups, subject, output_dir: Path, dpi: int):
    """绘制代表性窗口的原始波形、因果滤波波形和窗内 STFT。"""
    selected = representative_windows(raw, labels, groups, subject)
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    images, spectrograms = [], []
    time = np.arange(raw.shape[1]) / FS
    config = PreprocessingConfig(sfreq=FS)

    for column, label in enumerate((0, 1)):
        index = selected[label]
        signal = raw[index].astype(np.float64)
        filtered = causal_filter(signal, config)
        class_name = CLASS_NAMES[label]

        axes[0, column].plot(time, signal * 1e6, color=CLASS_COLORS[label], lw=0.65)
        axes[0, column].set_title(f"{class_name}｜原始波形（窗口 {index}）")
        axes[0, column].set_ylabel("幅值（µV）")

        axes[1, column].plot(time, filtered * 1e6, color=CLASS_COLORS[label], lw=0.8)
        axes[1, column].set_title(f"{class_name}｜因果 1–45 Hz 滤波")
        axes[1, column].set_ylabel("幅值（µV）")

        freq, frame_time, z = stft(
            filtered, fs=FS, window="hann", nperseg=round(FS),
            noverlap=round(FS / 2), detrend="constant", boundary=None, padded=False,
        )
        power_db = 10 * np.log10(np.maximum(np.abs(z) ** 2 * 1e12, np.finfo(float).tiny))
        image = axes[2, column].pcolormesh(frame_time, freq, power_db, shading="auto", cmap="magma")
        axes[2, column].set(ylim=(1, 45), title=f"{class_name}｜时频功率", xlabel="窗内时间（s）", ylabel="频率（Hz）")
        colorbar = fig.colorbar(image, ax=axes[2, column], pad=0.02)
        colorbar.set_label("功率（dB re 1 µV²）")
        images.append(image)
        spectrograms.append(power_db.ravel())

    limits = np.percentile(np.concatenate(spectrograms), [5, 99])
    for image in images:
        image.set_clim(*limits)
    fig.suptitle(f"受试者 {subject}：6 秒窗口波形与时频结构", fontsize=16, y=1.01)
    fig.text(
        0.5, -0.01,
        "示例按各类别原始标准差中位数选取；原始直流偏移使零初始状态滤波出现启动瞬态。STFT 仅用当前窗口，1 s Hann、0.5 s 步长。",
        ha="center", fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "waveform_analysis.png", dpi)
    return selected


def subject_mean_psd(raw, labels, groups, max_windows_per_class: int):
    """先在受试者×类别内平均，再跨受试者汇总，避免窗口数多者占权。"""
    subjects = np.array(sorted(set(groups)))
    frequencies = None
    rows = []
    for subject in subjects:
        subject_rows = []
        for label in (0, 1):
            ids = np.flatnonzero((groups == subject) & (labels == label))
            if not len(ids):
                raise ValueError(f"受试者 {subject} 缺少类别 {label}")
            if len(ids) > max_windows_per_class:
                positions = np.linspace(0, len(ids) - 1, max_windows_per_class).round().astype(int)
                ids = ids[np.unique(positions)]
            frequencies, power = welch(
                raw[ids].astype(np.float64), fs=FS, nperseg=min(round(2 * FS), raw.shape[1]),
                noverlap=0, detrend="constant", axis=-1,
            )
            subject_rows.append(power.mean(axis=0))
        rows.append(subject_rows)
    return frequencies, np.asarray(rows), subjects


def plot_spectrum(raw, labels, groups, output_dir: Path, dpi: int, max_windows_per_class: int):
    """绘制受试者宏平均 PSD 与相对频带功率。"""
    frequencies, psd, subjects = subject_mean_psd(raw, labels, groups, max_windows_per_class)
    psd_db = 10 * np.log10(np.maximum(psd * 1e12, np.finfo(float).tiny))
    band_power = np.zeros((len(subjects), 2, len(BANDS)), dtype=np.float64)
    total_mask = (frequencies >= 1) & (frequencies < 45)
    total = np.trapezoid(psd[:, :, total_mask], frequencies[total_mask], axis=-1)
    for band_index, (low, high) in enumerate(BANDS):
        mask = (frequencies >= low) & (frequencies < high)
        power = np.trapezoid(psd[:, :, mask], frequencies[mask], axis=-1)
        band_power[:, :, band_index] = power / np.maximum(total, np.finfo(float).tiny)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    frequency_mask = (frequencies >= 1) & (frequencies <= 45)
    for label in (0, 1):
        values = psd_db[:, label, :]
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(subjects))
        color = CLASS_COLORS[label]
        axes[0].plot(frequencies[frequency_mask], mean[frequency_mask], color=color, lw=2, label=CLASS_NAMES[label])
        axes[0].fill_between(
            frequencies[frequency_mask], mean[frequency_mask] - sem[frequency_mask],
            mean[frequency_mask] + sem[frequency_mask], color=color, alpha=0.18,
        )
    for low, high in BANDS:
        axes[0].axvspan(low, high, color="gray", alpha=0.035)
    axes[0].set(
        title="受试者宏平均功率谱密度（阴影：跨受试者 SEM）",
        xlabel="频率（Hz）", ylabel="PSD（dB µV²/Hz）", xlim=(1, 45),
    )
    axes[0].legend()

    positions = np.arange(len(BANDS))
    width = 0.36
    summary = {}
    for offset, label in ((-width / 2, 0), (width / 2, 1)):
        values = band_power[:, label, :]
        mean = values.mean(axis=0)
        sem = values.std(axis=0, ddof=1) / np.sqrt(len(subjects))
        axes[1].bar(
            positions + offset, mean, width, yerr=sem, capsize=4,
            color=CLASS_COLORS[label], alpha=0.88, label=CLASS_NAMES[label],
        )
        summary[CLASS_NAMES[label]] = {
            BAND_NAMES[i]: float(mean[i]) for i in range(len(BANDS))
        }
    axes[1].set(
        title="1–45 Hz 内相对频带功率（误差棒：跨受试者 SEM）",
        xlabel="频带", ylabel="相对功率", xticks=positions, xticklabels=BAND_NAMES,
    )
    axes[1].legend()
    fig.suptitle(f"频谱对比：{len(subjects)} 名受试者，每类最多抽取 {max_windows_per_class} 窗/人", fontsize=15)
    fig.tight_layout()
    save_figure(fig, output_dir / "spectral_analysis.png", dpi)
    return summary


def within_subject_auc(features, labels, groups):
    """逐受试者计算 Mann-Whitney AUC，保留判别方向以暴露个体差异。"""
    subjects = np.array(sorted(set(groups)))
    scores = np.full((len(subjects), features.shape[1]), np.nan, dtype=np.float32)
    for index, subject in enumerate(subjects, 1):
        ids = np.flatnonzero(groups == subject)
        positive = labels[ids] == 1
        n_positive = int(positive.sum())
        n_negative = len(ids) - n_positive
        if n_positive and n_negative:
            ranks = rankdata(features[ids], axis=0)
            scores[index - 1] = (
                (ranks[positive].sum(axis=0) - n_positive * (n_positive + 1) / 2)
                / (n_positive * n_negative)
            )
        if index % 10 == 0 or index == len(subjects):
            print(f"  单特征 AUC：{index}/{len(subjects)} 名受试者", flush=True)
    return subjects, scores


def feature_statistics(scores, names):
    valid = np.isfinite(scores)
    count = valid.sum(axis=0)
    safe_scores = np.where(valid, scores, np.nan)
    mean_auc = np.nanmean(safe_scores, axis=0)
    median_auc = np.nanmedian(safe_scores, axis=0)
    task_higher = np.sum(valid & (scores > 0.5), axis=0) / np.maximum(count, 1)
    consistency = np.maximum(task_higher, 1 - task_higher)
    median_separation = np.nanmedian(2 * np.abs(safe_scores - 0.5), axis=0)
    ranking_score = median_separation * consistency
    order = np.argsort(-ranking_score, kind="stable")
    table = pd.DataFrame({
        "feature": names,
        "within_auc_mean": mean_auc,
        "within_auc_median": median_auc,
        "median_2x_abs_auc_minus_0_5": median_separation,
        "task_higher_fraction": task_higher,
        "direction_consistency": consistency,
        "ranking_score": ranking_score,
        "valid_subjects": count,
    })
    return table, order


def within_subject_z(values, groups):
    result = np.empty(len(values), dtype=np.float64)
    for subject in sorted(set(groups)):
        mask = groups == subject
        current = values[mask]
        scale = current.std()
        result[mask] = (current - current.mean()) / scale if scale > 1e-12 else 0.0
    return result


def plot_feature_distributions(features, labels, groups, table, order, output_dir: Path, dpi: int, top: int):
    """展示排序靠前特征的受试者内标准化类别分布。"""
    selected = order[:top]
    columns = min(3, top)
    rows = int(np.ceil(top / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.2 * rows), squeeze=False)
    bins = np.linspace(-5, 5, 61)
    for plot_index, feature_index in enumerate(selected):
        ax = axes.flat[plot_index]
        normalized = within_subject_z(features[:, feature_index], groups)
        normalized = np.clip(normalized, -5, 5)
        for label in (0, 1):
            ax.hist(
                normalized[labels == label], bins=bins, density=True, histtype="stepfilled",
                alpha=0.35, color=CLASS_COLORS[label], label=CLASS_NAMES[label],
            )
            ax.hist(
                normalized[labels == label], bins=bins, density=True, histtype="step",
                lw=1.2, color=CLASS_COLORS[label],
            )
        row = table.iloc[feature_index]
        ax.set_title(
            f"{textwrap.fill(row['feature'], 32)}\n"
            f"中位 AUC={row['within_auc_median']:.3f}｜方向一致性={row['direction_consistency']:.2f}",
            fontsize=10,
        )
        ax.set(xlabel="受试者内 z-score（截断至 ±5）", ylabel="密度")
    for plot_index in range(top, rows * columns):
        axes.flat[plot_index].axis("off")
    axes.flat[0].legend()
    fig.suptitle("类别特征分布：先在各受试者内部标准化", fontsize=15)
    fig.text(
        0.5, -0.01,
        "排序依据：受试者内 |2×AUC−1| 中位数 × 方向一致性；这是使用全数据标签的描述性诊断，不是独立测试或特征选择验证。",
        ha="center", fontsize=10,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "feature_distributions.png", dpi)


def plot_subject_effects(subjects, scores, table, order, output_dir: Path, dpi: int, top: int = 20):
    """用 AUC 热图显示特征判别方向在不同受试者间是否一致。"""
    selected = order[:min(top, len(order))]
    values = scores[:, selected]
    fig, ax = plt.subplots(figsize=(16, 12))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=0, vmax=1, interpolation="nearest")
    labels = [textwrap.fill(table.iloc[index]["feature"], 24) for index in selected]
    ax.set(
        title="单特征受试者内 AUC：0.5 为无区分，红/蓝表示相反判别方向",
        xlabel="特征（按描述性区分度排序）", ylabel="受试者",
        xticks=np.arange(len(selected)), xticklabels=labels,
        yticks=np.arange(len(subjects)), yticklabels=subjects,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("受试者内 AUC")
    fig.tight_layout()
    save_figure(fig, output_dir / "subject_feature_effects.png", dpi)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "test" / "highdim_deep_features.npz", help="波形与特征缓存")
    parser.add_argument("--manifest", type=Path, default=ROOT / "database" / "manifest.json", help="特征名称清单")
    parser.add_argument("--subject", default="001", help="波形示例受试者，默认 001")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "vis" / "output", help="输出目录")
    parser.add_argument("--top", type=int, default=6, help="类别分布图展示的特征数")
    parser.add_argument("--psd-windows-per-class", type=int, default=30, help="PSD 每名受试者每类最多窗口数")
    parser.add_argument("--dpi", type=int, default=160, help="输出图片 DPI")
    args = parser.parse_args()
    if args.top < 1 or args.psd_windows_per_class < 1 or args.dpi < 72:
        parser.error("--top 和 --psd-windows-per-class 必须为正数，--dpi 至少为 72")
    return args


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("[1/5] 读取并校验数据", flush=True)
    raw, features, labels, groups, names, _ = load_data(args.data, args.manifest)
    print(
        f"  窗口={len(labels)}，受试者={len(set(groups))}，特征={features.shape[1]}，"
        f"relax={(labels == 0).sum()}，task={(labels == 1).sum()}", flush=True,
    )

    print("[2/5] 生成波形与时频图", flush=True)
    selected = plot_waveforms(raw, labels, groups, args.subject, args.output_dir, args.dpi)
    print("[3/5] 生成频谱与频带功率图", flush=True)
    band_summary = plot_spectrum(raw, labels, groups, args.output_dir, args.dpi, args.psd_windows_per_class)

    print("[4/5] 计算受试者内单特征统计", flush=True)
    subjects, scores = within_subject_auc(features, labels, groups)
    table, order = feature_statistics(scores, names)
    table.iloc[order].to_csv(args.output_dir / "feature_statistics.csv", index=False, encoding="utf-8-sig")
    plot_feature_distributions(features, labels, groups, table, order, args.output_dir, args.dpi, args.top)

    print("[5/5] 生成受试者特征效应热图与摘要", flush=True)
    plot_subject_effects(subjects, scores, table, order, args.output_dir, args.dpi)
    top_features = [
        {
            "feature": str(table.iloc[index]["feature"]),
            "within_auc_mean": float(table.iloc[index]["within_auc_mean"]),
            "within_auc_median": float(table.iloc[index]["within_auc_median"]),
            "direction_consistency": float(table.iloc[index]["direction_consistency"]),
        }
        for index in order[:10]
    ]
    summary = {
        "data": {
            "windows": int(len(labels)), "subjects": int(len(set(groups))),
            "features": int(features.shape[1]), "waveform_subject": args.subject,
            "waveform_indices": selected,
        },
        "relative_bandpower_subject_macro": band_summary,
        "top_features_descriptive": top_features,
        "caveat": "描述性诊断使用全部窗口标签，不是交叉验证、独立测试或部署性能。",
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print("  描述性区分度前 5 特征：", flush=True)
    for rank, row in enumerate(top_features[:5], 1):
        print(
            f"    {rank}. {row['feature']}：受试者内 AUC={row['within_auc_mean']:.3f}，"
            f"方向一致性={row['direction_consistency']:.2f}", flush=True,
        )
    print(f"完成，结果位于 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
