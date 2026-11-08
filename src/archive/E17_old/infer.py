from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from eeg_decoder import load_bdf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
from streaming import StreamingInference  # noqa: E402


class ForwardHMM:
    """两状态对称 sticky HMM；只递推当前发射分数和上一帧后验。"""

    def __init__(self, dwell: int = 300, temperature: float = 1.0):
        self.log_stay = math.log1p(-1 / dwell)
        self.log_switch = -math.log(dwell)
        self.temperature = temperature
        self.logpost = np.array([-math.log(2), -math.log(2)])
        self.started = False

    def update(self, score: float) -> float:
        z = score / self.temperature
        emission = np.array([-np.logaddexp(0, z), -np.logaddexp(0, -z)])
        if self.started:
            predicted = np.array([
                np.logaddexp(self.logpost[0] + self.log_stay, self.logpost[1] + self.log_switch),
                np.logaddexp(self.logpost[0] + self.log_switch, self.logpost[1] + self.log_stay),
            ])
        else:
            predicted = self.logpost
        unnormalized = predicted + emission
        self.logpost = unnormalized - np.logaddexp(unnormalized[0], unnormalized[1])
        self.started = True
        return float(self.logpost[1] - self.logpost[0])


class CausalDetrend:
    """因果 EMA 去趋势（E44 dt1200）：返回 score - m_t，m_t 含当前帧，首帧 m_0 = score_0。

    与 test/44_late_drift/e44.py 的 ema_mean 逐行一致：a = 1 - 0.5**(1/halflife)，
    m_t = m_{t-1} + a*(s_t - m_{t-1})；减均值后不做尺度处理。
    """

    def __init__(self, halflife: float = 1200):
        self.alpha = 1 - 0.5 ** (1 / halflife)
        self.mean = None

    def update(self, score: float) -> float:
        if self.mean is None:
            self.mean = score
        else:
            self.mean = self.mean + self.alpha * (score - self.mean)
        return score - self.mean


class PostProcessor:
    """融合分数 -> [可选 dt 去趋势] -> ForwardHMM。halflife=None 退回 E42 的纯 HMM。"""

    def __init__(self, detrend_halflife: float | None = 1200, dwell: int = 300, temperature: float = 1.0):
        self.detrend = CausalDetrend(detrend_halflife) if detrend_halflife else None
        self.hmm = ForwardHMM(dwell=dwell, temperature=temperature)

    def update(self, score: float) -> float:
        if self.detrend is not None:
            score = self.detrend.update(score)
        return self.hmm.update(score)


def stream_outputs(decoder, post: PostProcessor, signal_v: np.ndarray, fs: float, chunk_size: int) -> list[dict]:
    outputs = []
    n_samples = len(signal_v)
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        for time_s, score, _ in decoder.update(signal_v[start:end]):
            hmm_score = post.update(score)
            outputs.append({
                "sample_index": int(round(time_s * fs)) - 1,
                "time_s": round(time_s, 6),
                "score_task": hmm_score,
                "prediction": "task" if hmm_score > 0 else "relax",
                "valid": 1,
            })
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="对 BDF 执行严格因果的分块 EEG 推理")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "test" / "17_compliant" / "model_bundle.pt"
    )
    parser.add_argument("--input", type=Path, required=True, help="输入 BDF 文件")
    parser.add_argument("--output", type=Path, required=True, help="输出预测 CSV")
    parser.add_argument("--chunk-size", type=int, default=250, help="每次输入的采样点数")
    parser.add_argument(
        "--detrend-halflife", type=float, default=1200,
        help="HMM 前因果 EMA 去趋势半衰期（帧，1 帧=1 s）；<=0 关闭，退回纯 HMM",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size 必须为正整数")

    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    if bundle.get("version") != 1:
        raise ValueError("不支持的模型格式版本")
    if float(bundle["win_s"]) > 6:
        raise ValueError("模型原始 EEG 窗口超过 6 秒")
    record = load_bdf(args.input)
    decoder = StreamingInference(bundle, fs=record["fs"], step=1.0)
    halflife = args.detrend_halflife if args.detrend_halflife > 0 else None
    post = PostProcessor(detrend_halflife=halflife, dwell=300, temperature=1.0)

    # E17 特征函数使用 BDF 原始伏特单位；无效采样以 NaN 触发因果缓冲重置。
    signal_v = np.asarray(record["signal_uv"], dtype=float) * 1e-6
    signal_v[~record["valid"]] = np.nan
    outputs = stream_outputs(decoder, post, signal_v, record["fs"], args.chunk_size)

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
        "engine": "E17_multiscale_LR_MLP",
        "chunk_size": args.chunk_size,
        "input_unit": "uV converted to V for feature extraction",
        "score_direction": "higher means task",
        "windows": len(outputs),
        "valid_windows": sum(item["valid"] for item in outputs),
        "task_windows": sum(item["prediction"] == "task" for item in outputs),
        "threshold": 0.0,
        "window_s": float(bundle["win_s"]),
        "feature_mode": "878_features_x_5_causal_scales",
        "postprocess": (f"causal_ema_detrend_hl{halflife:g}+forward_hmm_d300_t1" if halflife
                        else "forward_hmm_d300_t1"),
        "protocol_prior": False,
        "protocol_weight": None,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
