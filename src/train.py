from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import eeg_decoder
from eeg_decoder import (
    BANDS,
    FEATURE_VERSION,
    DecoderConfig,
    build_feature_dataset,
    causal_smooth_scores,
    discover_records,
    feature_names,
    fuse_protocol_scores,
    learn_protocol_template,
    model_scores,
)


def candidates(seed: int) -> dict:
    return {
        "logistic_C0.01": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.01, class_weight="balanced", max_iter=1000, random_state=seed
                    ),
                ),
            ]
        ),
        "logistic_C0.1": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.1, class_weight="balanced", max_iter=1000, random_state=seed
                    ),
                ),
            ]
        ),
        "logistic_C1": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0, class_weight="balanced", max_iter=1000, random_state=seed
                    ),
                ),
            ]
        ),
        "shrinkage_LDA": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LinearDiscriminantAnalysis(
                        solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]
                    ),
                ),
            ]
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=20,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=50,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=False,
            random_state=seed,
        ),
    }


def feature_names_for_mode(mode: str) -> list[str]:
    base = feature_names()
    if mode == "raw":
        return base
    if mode == "delta":
        return [f"baseline_delta_{name}" for name in base]
    if mode == "raw_delta":
        return base + [f"baseline_delta_{name}" for name in base]
    raise ValueError(f"未知 feature_mode: {mode}")


def apply_feature_mode(dataset: dict, mode: str) -> dict:
    transformed = dict(dataset)
    raw = dataset["X"]
    if mode == "delta":
        transformed["X"] = raw - dataset["baseline"]
    elif mode == "raw_delta":
        transformed["X"] = np.concatenate([raw, raw - dataset["baseline"]], axis=1)
    elif mode != "raw":
        raise ValueError(f"未知 feature_mode: {mode}")
    return transformed


def make_folds(groups: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    records = np.unique(groups)
    np.random.default_rng(seed).shuffle(records)
    return [fold for fold in np.array_split(records, n_splits) if len(fold)]


def classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    pred = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "bacc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, scores)),
        "relax_recall": float(tn / (tn + fp)),
        "task_recall": float(tp / (tp + fn)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def cross_validate(
    dataset: dict,
    folds: list[np.ndarray],
    seed: int,
    postprocess: str = "none",
    stride_samples: int = 250,
    protocol_templates: list[dict] | None = None,
    protocol_weight: float = 0.1,
    fs: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = dataset["y"]
    groups = dataset["groups"]
    pure = dataset["pure"]
    rows = []
    fold_rows = []
    all_scores = {}
    for name, estimator in candidates(seed).items():
        scores = np.full(len(y), np.nan, dtype=np.float64)
        for fold_index, test_records in enumerate(folds, start=1):
            context_mask = np.isin(groups, test_records)
            test_mask = context_mask & (y >= 0)
            train_mask = (~context_mask) & pure
            model = clone(estimator)
            model.fit(dataset["X"][train_mask], y[train_mask])
            context_scores = model_scores(model, dataset["X"][context_mask])
            smoothed_context = causal_smooth_scores(
                context_scores,
                groups[context_mask],
                dataset["endpoints"][context_mask],
                stride_samples,
                postprocess,
            )
            if protocol_templates is not None:
                smoothed_context = fuse_protocol_scores(
                    smoothed_context,
                    dataset["endpoints"][context_mask],
                    fs,
                    protocol_templates[fold_index - 1],
                    protocol_weight,
                )
            scores[test_mask] = smoothed_context[test_mask[context_mask]]
            metrics = classification_metrics(y[test_mask], scores[test_mask])
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold_index,
                    "test_records": ",".join(test_records),
                    "train_windows": int(np.sum(train_mask)),
                    "test_windows": int(np.sum(test_mask)),
                    **metrics,
                }
            )
            print(f"{name} fold {fold_index}: bACC={metrics['bacc']:.4f}, AUC={metrics['auc']:.4f}", flush=True)
        labeled = y >= 0
        if not np.all(np.isfinite(scores[labeled])):
            raise RuntimeError(f"{name} 的 OOF 预测不完整")
        all_scores[name] = scores
        metrics = classification_metrics(y[labeled], scores[labeled])
        model_folds = [row for row in fold_rows if row["model"] == name]
        rows.append(
            {
                "model": name,
                **metrics,
                "fold_bacc_mean": float(np.mean([row["bacc"] for row in model_folds])),
                "fold_bacc_std": float(np.std([row["bacc"] for row in model_folds])),
                "fold_auc_mean": float(np.mean([row["auc"] for row in model_folds])),
                "fold_auc_std": float(np.std([row["auc"] for row in model_folds])),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["selection_score"] = (comparison["bacc"] + comparison["auc"]) / 2
    comparison = comparison.sort_values("selection_score", ascending=False).reset_index(drop=True)
    return comparison, pd.DataFrame(fold_rows), all_scores


def best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y, scores)
    values = (tpr + 1 - fpr) / 2
    valid = np.isfinite(thresholds) & (thresholds > 0) & (thresholds < 1)
    best_value = np.max(values[valid])
    candidates_mask = valid & np.isclose(values, best_value)
    threshold = float(thresholds[candidates_mask][np.argmin(np.abs(thresholds[candidates_mask] - 0.5))])
    return threshold, float(best_value)


def save_data_overview(summaries: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    x = np.arange(len(summaries))
    axes[0].bar(x, summaries["relax_s"] / 60, label="relax", color="#4C78A8")
    axes[0].bar(x, summaries["task_s"] / 60, bottom=summaries["relax_s"] / 60, label="task", color="#F58518")
    axes[0].set_ylabel("Labeled duration (min)")
    axes[0].legend(loc="upper right")
    axes[0].set_title("Record-level label and quality audit")
    axes[1].bar(x, summaries["bad_s"], color="#E45756")
    axes[1].set_ylabel("BAD_ACQ_SKIP (s)")
    axes[1].set_xlabel("Record")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summaries["record"], rotation=90, fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_model_comparison(comparison: pd.DataFrame, path: Path) -> None:
    plot = comparison.iloc[::-1]
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.barh(y - 0.18, plot["bacc"], height=0.36, label="bACC @ 0.5", color="#4C78A8")
    ax.barh(y + 0.18, plot["auc"], height=0.36, label="AUC", color="#F58518")
    ax.set_yticks(y)
    ax.set_yticklabels(plot["model"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Out-of-fold score")
    ax.set_title("Group-wise cross-validation (held-out records)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_roc(y: np.ndarray, all_scores: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for name, scores in all_scores.items():
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Out-of-fold ROC")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def feature_importance(model, names: list[str]) -> np.ndarray | None:
    estimator = model.named_steps["classifier"] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "coef_"):
        return np.asarray(estimator.coef_).ravel()
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_).ravel()
    return None


def save_best_diagnostics(
    dataset: dict,
    scores: np.ndarray,
    threshold: float,
    final_model,
    output_dir: Path,
    model_feature_names: list[str],
) -> pd.DataFrame:
    y = dataset["y"]
    groups = dataset["groups"]
    pred = scores >= 0.5
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im = axes[0].imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        axes[0].text(j, i, f"{value:,}", ha="center", va="center")
    axes[0].set_xticks([0, 1], ["relax", "task"])
    axes[0].set_yticks([0, 1], ["relax", "task"])
    axes[0].set(xlabel="Predicted", ylabel="True", title="OOF confusion matrix @ 0.5")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    bins = np.linspace(0, 1, 60)
    axes[1].hist(scores[y == 0], bins=bins, density=True, alpha=0.6, label="relax", color="#4C78A8")
    axes[1].hist(scores[y == 1], bins=bins, density=True, alpha=0.6, label="task", color="#F58518")
    axes[1].axvline(threshold, color="black", ls="--", label=f"final threshold={threshold:.3f}")
    axes[1].set(xlabel="P(task)", ylabel="Density", title="OOF score distributions")
    axes[1].legend()
    fig.savefig(output_dir / "best_score_diagnostics.png", dpi=180)
    plt.close(fig)

    importance = feature_importance(final_model, model_feature_names)
    if importance is None:
        rng = np.random.default_rng(42)
        candidates = np.flatnonzero(dataset["pure"])
        selected = rng.choice(candidates, size=min(5000, len(candidates)), replace=False)
        importance = permutation_importance(
            final_model,
            dataset["X"][selected],
            y[selected],
            scoring="roc_auc",
            n_repeats=2,
            random_state=42,
            n_jobs=-1,
        ).importances_mean
    if importance is not None:
        order = np.argsort(np.abs(importance))[-20:][::-1]
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        names = np.array(model_feature_names)[order]
        values = importance[order]
        ax.barh(np.arange(len(order))[::-1], values, color=np.where(values >= 0, "#F58518", "#4C78A8"))
        ax.set_yticks(np.arange(len(order))[::-1], names)
        ax.set(xlabel="Coefficient or importance", title="Final model: top 20 features")
        ax.grid(axis="x", alpha=0.25)
        fig.savefig(output_dir / "feature_importance.png", dpi=180)
        plt.close(fig)

    rows = []
    for record in np.unique(groups):
        mask = groups == record
        rows.append({"record": record, **classification_metrics(y[mask], scores[mask])})
    record_metrics = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(15, 5), constrained_layout=True)
    x = np.arange(len(record_metrics))
    ax.bar(x - 0.2, record_metrics["bacc"], width=0.4, label="bACC @ 0.5", color="#4C78A8")
    ax.bar(x + 0.2, record_metrics["auc"], width=0.4, label="AUC", color="#F58518")
    ax.axhline(0.5, color="black", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(record_metrics["record"], rotation=90, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("OOF score")
    ax.set_title("Performance by held-out record")
    ax.legend(loc="lower right")
    fig.savefig(output_dir / "record_performance.png", dpi=180)
    plt.close(fig)
    return record_metrics


def save_example_timeline(dataset: dict, scores: np.ndarray, record_metrics: pd.DataFrame, threshold: float, path: Path) -> str:
    median_auc = record_metrics["auc"].median()
    record = record_metrics.iloc[(record_metrics["auc"] - median_auc).abs().argmin()]["record"]
    mask = dataset["groups"] == record
    time_min = dataset["endpoints"][mask] / 250 / 60
    y = dataset["y"][mask]
    record_scores = scores[mask]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time_min, record_scores, lw=1, color="#4C78A8")
    axes[0].axhline(threshold, color="black", ls="--", label="final threshold")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("P(task)")
    axes[0].set_title(f"Causal OOF timeline — record {record} (median-AUC example)")
    axes[0].legend()
    axes[1].step(time_min, y, where="post", color="#F58518")
    axes[1].set_yticks([0, 1], ["relax", "task"])
    axes[1].set(xlabel="Time from BDF start (min)", ylabel="True phase")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(record)


def write_report(
    path: Path,
    config: DecoderConfig,
    feature_mode: str,
    postprocess: str,
    protocol_prior: bool,
    protocol_weight: float,
    summaries: pd.DataFrame,
    comparison: pd.DataFrame,
    best_name: str,
    threshold: float,
    tuned_bacc: float,
    dataset: dict,
    folds: int,
    example_record: str,
) -> None:
    best = comparison.iloc[0]
    table_rows = []
    for _, row in comparison.iterrows():
        table_rows.append(
            f"| {row['model']} | {row['bacc']:.4f} | {row['auc']:.4f} | "
            f"{row['fold_bacc_mean']:.4f} ± {row['fold_bacc_std']:.4f} | {row['selection_score']:.4f} |"
        )
    text = f"""# 工业场景 EEG 认知状态解码实验报告

## 1. 实验概况

本实验使用 {len(summaries)} 条真实记录，按**整条记录留出**进行 {folds} 折分组交叉验证。所有验证窗口均来自训练阶段未见过的记录，避免相邻高相关窗口泄漏。共得到 {len(dataset['y']):,} 个有效带标签判断窗口，其中 {int(np.sum(dataset['pure'])):,} 个窗口内部标签完全一致，仅后者用于拟合分类器；全部有效窗口用于验证，包括状态切换附近。

最终内部选择结果为 **{best_name}**。OOF bACC（固定 0.5 阈值）为 **{best['bacc']:.4f}**，OOF AUC 为 **{best['auc']:.4f}**。最终部署阈值 {threshold:.6f} 由全部 OOF 预测选择，对应调阈值 OOF bACC 为 {tuned_bacc:.4f}；该调阈值指标与阈值选择使用同一批 OOF 预测，不能视为独立测试成绩。

## 2. 数据审计

- 标签时长中位数：relax {summaries['relax_s'].median() / 60:.2f} min，task {summaries['task_s'].median() / 60:.2f} min。
- 含 BAD_ACQ_SKIP 的记录数：{int(np.sum(summaries['bad_intervals'] > 0))}；坏段总时长 {summaries['bad_s'].sum():.2f} s。
- 记录总时长范围：{summaries['duration_s'].min() / 60:.2f}–{summaries['duration_s'].max() / 60:.2f} min。
- 详细逐记录结果见 `test/results/record_summary.csv`。

![数据审计](test/results/data_overview.png)

## 3. 合规因果流程

1. 从 BDF 文件名解析毫秒级开始时间，以实际事件时间配对 trial 并生成样本标签。
2. 将 BAD_ACQ_SKIP 掩码传递到窗口级质量判断，含坏段的窗口不参与训练或指标计算。
3. 每个判断仅使用过去 {config.window_s:g} 秒（{config.window_samples} 点），每 {config.stride_s:g} 秒判断一次。
4. 当前窗口去均值后计算 {config.welch_s:g} 秒 Welch 频谱、{len(BANDS)} 个细分频带的绝对/相对对数功率，以及 Hjorth、过零率、频谱熵等 11 个补充特征；48–52 Hz 工频区间不构造频带特征。
5. 特征模式为 `{feature_mode}`，模型输入共 {len(feature_names_for_mode(feature_mode))} 维。`delta` 使用当前特征减去开头标称 60 秒对应的有效窗口基线中位数，`raw_delta` 同时保留原始与差分特征；坏段自动顺延，校准未完成时使用扩展中位数，全程不使用标签。
6. 训练时排除跨标签窗口；推理时按真实时间顺序输出，不借用未来标签或未来样本。
7. 流式类的原始信号缓存只有 {config.window_s:g} 秒，不使用全记录标准化、双向滤波或未来平滑；仅在线自适应模式保存标称 60 秒（坏段顺延）的无标签特征统计。
8. 后处理为 `{postprocess}`。`mean3` 仅平均当前及前两次有效分数，4 秒窗口下总感受野为 6 秒；遇到坏段或时间间隙即重置。
9. 协议时间先验启用状态为 `{protocol_prior}`。正式结果仅在关闭该先验时作为纯 EEG 成绩；启用时会根据训练事件的相对录制时刻预测阶段，属于额外时间信息，不能解释为纯 EEG 解码提升。

`score_task` 为越大越倾向认知负荷态的连续分数；无效窗口输出 `valid=false` 和中性分数 0.5。

## 4. 模型对比

内部选择分为 `0.5 × (OOF bACC + OOF AUC)`，只用于在候选模型间排序，不代表主办方未公开的综合分公式。bACC 均在固定 0.5 阈值下计算，以便公平比较。

| 模型 | OOF bACC | OOF AUC | 折 bACC 均值 ± 标准差 | 选择分 |
|---|---:|---:|---:|---:|
{chr(10).join(table_rows)}

![模型对比](test/results/model_comparison.png)

![ROC](test/results/roc_curves.png)

## 5. 最终模型诊断

![分数与混淆矩阵](test/results/best_score_diagnostics.png)

![逐记录表现](test/results/record_performance.png)

![特征重要性](test/results/feature_importance.png)

下图为记录级 AUC 接近中位数的记录 {example_record}，不是挑选的最佳示例。概率来自该记录完全留出时的因果预测。

![因果时间线](test/results/example_timeline.png)

## 6. 结果解释与限制

- 交叉验证估计的是对保留记录内新记录的泛化，不等同于决赛新被试、新设备状态或新测试集成绩。
- 多模型对比和最终阈值均使用同一数据集的 OOF 预测，仍存在选择偏差；最终模型需以主办方保留测试集验证。
- 窗口跨状态边界时，历史 EEG 与末端标签可能暂时不一致，这是严格因果任务的真实难度，未通过未来信息修正。
- 当前没有可靠证据把编号记录映射到具体被试，因此按记录分组；若确认同一被试有多条记录，应改为按被试分组重新验证。
- 概率分布可能在不同被试间漂移，AUC 与 bACC 应同时关注，不能仅报告总体准确率。
- 协议时间先验不是纯 EEG 解码证据，且赛题未明确允许根据录制时刻推断阶段；正式模型默认关闭该功能，协议实验结果不计入合规优化指标。

## 7. 复现入口

```powershell
python -m pip install -r requirements.txt
python src/train.py --data EEG_dataset --output test/results --model-out src/model.joblib
python src/infer.py --model src/model.joblib --input EEG_dataset/001/001_20261017_170137.984_UTC+0800.bdf --output test/results/001_predictions.csv --chunk-size 250
python -m pytest test -q
```

模型文件为 `src/model.joblib`，包含预处理和分类器、阈值、窗口配置、特征定义版本、训练记录清单及环境版本。推理代码和训练代码共用 `extract_features`，避免离线与部署特征不一致。
"""
    path.write_text(text, encoding="utf-8")


def cache_fingerprint(data_dir: Path) -> str:
    files = []
    for record_id, bdf_path, event_path in discover_records(data_dir):
        for path in (bdf_path, event_path):
            stat = path.stat()
            files.append([record_id, path.name, stat.st_size, stat.st_mtime_ns])
    payload = {
        "data_dir": str(data_dir.resolve()),
        "files": files,
        "feature_version": FEATURE_VERSION,
        "feature_source_sha256": hashlib.sha256(Path(eeg_decoder.__file__).read_bytes()).hexdigest(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_or_build_features(data_dir: Path, output_dir: Path, config: DecoderConfig, rebuild: bool) -> tuple[dict, pd.DataFrame]:
    cache_dir = output_dir.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"w{config.window_s:g}_s{config.stride_s:g}_welch{config.welch_s:g}_fs{config.fs}_{cache_fingerprint(data_dir)}"
    cache_path = cache_dir / f"features_{key}.npz"
    summary_path = cache_dir / f"summaries_{key}.json"
    if cache_path.exists() and summary_path.exists() and not rebuild:
        loaded = np.load(cache_path, allow_pickle=False)
        dataset = {name: loaded[name] for name in loaded.files}
        summaries = pd.DataFrame(json.loads(summary_path.read_text(encoding="utf-8")))
        print(f"Loaded cached features: {cache_path}", flush=True)
        return dataset, summaries
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset, summary_rows = build_feature_dataset(data_dir, config)
    np.savez_compressed(cache_path, **dataset)
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset, pd.DataFrame(summary_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="按记录分组验证并训练因果 EEG 解码器")
    parser.add_argument("--data", type=Path, default=root / "EEG_dataset")
    parser.add_argument("--output", type=Path, default=root / "test" / "results")
    parser.add_argument("--model-out", type=Path, default=root / "src" / "model.joblib")
    parser.add_argument("--report-out", type=Path, default=root / "实验报告.md")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-s", type=float, default=4.0)
    parser.add_argument("--stride-s", type=float, default=1.0)
    parser.add_argument("--feature-mode", choices=["raw", "delta", "raw_delta"], default="raw")
    parser.add_argument("--postprocess", choices=["none", "mean3"], default="mean3")
    parser.add_argument("--protocol-prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--protocol-weight", type=float, default=0.1)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    config = DecoderConfig(window_s=args.window_s, stride_s=args.stride_s)
    if args.postprocess == "mean3" and config.window_s + 2 * config.stride_s > 6:
        raise ValueError("mean3 与当前窗口组合会超过 6 秒有效感受野")
    dataset, summaries = load_or_build_features(args.data, args.output, config, args.rebuild)
    dataset = apply_feature_mode(dataset, args.feature_mode)
    model_feature_names = feature_names_for_mode(args.feature_mode)
    summaries.to_csv(args.output / "record_summary.csv", index=False)
    save_data_overview(summaries, args.output / "data_overview.png")

    folds = make_folds(dataset["groups"], args.folds, args.seed)
    protocol_templates = None
    if args.protocol_prior:
        all_records = set(np.unique(dataset["groups"]).tolist())
        protocol_templates = [
            learn_protocol_template(args.data, all_records - set(test_records.tolist()))
            for test_records in folds
        ]
    comparison, fold_metrics, all_scores = cross_validate(
        dataset,
        folds,
        args.seed,
        args.postprocess,
        config.stride_samples,
        protocol_templates,
        args.protocol_weight,
        config.fs,
    )
    labeled = dataset["y"] >= 0
    dataset = {name: values[labeled] for name, values in dataset.items()}
    all_scores = {name: values[labeled] for name, values in all_scores.items()}
    comparison.to_csv(args.output / "model_comparison.csv", index=False)
    fold_metrics.to_csv(args.output / "fold_metrics.csv", index=False)
    save_model_comparison(comparison, args.output / "model_comparison.png")
    save_roc(dataset["y"], all_scores, args.output / "roc_curves.png")

    best_name = str(comparison.iloc[0]["model"])
    scores = all_scores[best_name]
    threshold, tuned_bacc = best_threshold(dataset["y"], scores)
    final_model = clone(candidates(args.seed)[best_name])
    final_model.fit(dataset["X"][dataset["pure"]], dataset["y"][dataset["pure"]])
    record_metrics = save_best_diagnostics(
        dataset, scores, threshold, final_model, args.output, model_feature_names
    )
    record_metrics.to_csv(args.output / "record_metrics.csv", index=False)
    example_record = save_example_timeline(
        dataset, scores, record_metrics, threshold, args.output / "example_timeline.png"
    )
    pd.DataFrame(
        {
            "record": dataset["groups"],
            "endpoint": dataset["endpoints"],
            "y": dataset["y"],
            "pure_window": dataset["pure"],
            "score_task": scores,
        }
    ).to_csv(args.output / "oof_predictions.csv", index=False)

    protocol = learn_protocol_template(args.data) if args.protocol_prior else None
    artifact = {
        "format_version": 1,
        "feature_version": FEATURE_VERSION,
        "model_name": best_name,
        "model": final_model,
        "threshold": threshold,
        "config": config.to_dict(),
        "feature_mode": args.feature_mode,
        "baseline_s": 60.0,
        "postprocess": args.postprocess,
        "protocol": protocol,
        "protocol_weight": args.protocol_weight,
        "feature_names": model_feature_names,
        "input_unit": "uV",
        "score_direction": "higher means task",
        "train_records": sorted(np.unique(dataset["groups"]).tolist()),
        "validation": {
            "scheme": f"{args.folds}-fold grouped by record; fixed threshold 0.5; protocol prior={args.protocol_prior}",
            "oof_bacc_at_0.5": float(comparison.iloc[0]["bacc"]),
            "oof_auc": float(comparison.iloc[0]["auc"]),
            "oof_bacc_tuned_optimistic": tuned_bacc,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "mne": mne.__version__,
            "joblib": joblib.__version__,
        },
    }
    joblib.dump(artifact, args.model_out, compress=3)
    write_report(
        args.report_out,
        config,
        args.feature_mode,
        args.postprocess,
        args.protocol_prior,
        args.protocol_weight,
        summaries,
        comparison,
        best_name,
        threshold,
        tuned_bacc,
        dataset,
        args.folds,
        example_record,
    )
    print(json.dumps({"best_model": best_name, "threshold": threshold, "model": str(args.model_out), **artifact["validation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
