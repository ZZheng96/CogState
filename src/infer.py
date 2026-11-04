from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib

from eeg_decoder import (
    FEATURE_VERSION,
    CausalStreamDecoder,
    DecoderConfig,
    load_bdf,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="对 BDF 执行严格因果的分块 EEG 推理")
    parser.add_argument("--model", type=Path, default=root / "src" / "model.joblib")
    parser.add_argument("--input", type=Path, required=True, help="输入 BDF 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出预测 CSV")
    parser.add_argument("--chunk-size", type=int, default=250, help="每次输入的采样点数")
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size 必须为正整数")

    artifact = joblib.load(args.model)
    if artifact.get("format_version") != 1:
        raise ValueError("不支持的模型格式版本")
    if artifact.get("feature_version") != FEATURE_VERSION:
        raise ValueError("模型特征版本与当前代码不一致，请重新训练")
    expected_features = int(artifact["model"].n_features_in_)
    if len(artifact["feature_names"]) != expected_features:
        raise ValueError("模型特征清单与分类器输入维度不一致")
    if artifact.get("input_unit") != "uV":
        raise ValueError("模型输入单位必须为 uV")
    if artifact.get("protocol") is not None:
        raise ValueError("正式推理仅支持纯 EEG 模型，请使用 --no-protocol-prior 重新训练")
    decoder = CausalStreamDecoder(
        model=artifact["model"],
        threshold=artifact["threshold"],
        config=DecoderConfig.from_dict(artifact["config"]),
        feature_mode=artifact["feature_mode"],
        baseline_s=artifact["baseline_s"],
        postprocess=artifact["postprocess"],
    )
    record = load_bdf(args.input)
    if record["fs"] != decoder.config.fs:
        raise ValueError(f"输入采样率 {record['fs']} Hz 与模型要求的 {decoder.config.fs} Hz 不一致")
    outputs = []
    for start in range(0, record["n_samples"], args.chunk_size):
        end = min(start + args.chunk_size, record["n_samples"])
        outputs.extend(
            decoder.push(
                record["signal_uv"][start:end], record["valid"][start:end]
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "time_s", "score_task", "prediction", "valid"],
        )
        writer.writeheader()
        writer.writerows(outputs)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "model": str(args.model),
        "chunk_size": args.chunk_size,
        "input_unit": "uV",
        "score_direction": "higher means task",
        "windows": len(outputs),
        "valid_windows": sum(item["valid"] for item in outputs),
        "task_windows": sum(item["prediction"] == "task" for item in outputs),
        "threshold": artifact["threshold"],
        "window_s": decoder.config.window_s,
        "feature_mode": decoder.feature_mode,
        "postprocess": decoder.postprocess,
        "protocol_prior": decoder.protocol is not None,
        "protocol_weight": decoder.protocol_weight if decoder.protocol is not None else None,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
