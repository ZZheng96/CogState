"""E3 / E48：E2 分数 + R/T 交替与块长先验的严格因果前向 HSMM。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
sys.path.insert(0, str(ROOT / "test" / "46_causal_online_norm"))
import loso as L  # noqa: E402
import e46  # noqa: E402

SIGN = np.array([-1, 1, -1, 1, -1, 1], dtype=float)


def causal_scale(s):
    s = np.asarray(s, dtype=float)
    n0 = min(10, len(s))
    med = np.median(s[:n0])
    scale = np.median(np.abs(s[:n0] - med)) * 1.4826 + 0.1
    return (s - med) / scale


def duration_priors(D, train, lo=120, hi=420):
    priors = []
    for k in range(6):
        h = np.zeros(hi - lo + 1)
        for q in train:
            count = int(np.sum(D[q][2] == k))
            if count > 0:
                h[int(np.clip(count, lo, hi)) - lo] += 1
        h += 0.5
        priors.append((lo, h / h.sum()))
    return priors


def hsmm_forward(eeg, schedule, priors, initial_duration=1, eeg_weight=1.0, schedule_weight=0.0):
    """六块显式时长 HSMM。d 为当前块已持续的帧数，使用生存/结束概率递推。"""
    eeg = np.asarray(eeg, dtype=float)
    schedule = np.asarray(schedule, dtype=float)
    n = len(eeg)
    m = 6
    lmax = max(lo + len(plen) - 1 for lo, plen in priors)
    log_survive = np.full((m, lmax + 1), -np.inf)
    log_end = np.full((m, lmax + 1), -np.inf)
    for k, (lo, plen) in enumerate(priors):
        lengths = np.arange(lo, lo + len(plen))
        for d in range(1, lmax + 1):
            denom = plen[lengths >= d].sum()
            if denom <= 0:
                continue
            p_end = plen[lengths == d].sum() / denom
            p_stay = plen[lengths >= d + 1].sum() / denom
            if p_end > 0:
                log_end[k, d] = np.log(p_end)
            if p_stay > 0:
                log_survive[k, d] = np.log(p_stay)
    A = np.full((m, lmax + 1), -np.inf)
    A[0, int(initial_duration)] = 0.0
    out = np.empty(n)
    for t in range(n):
        if t:
            prev = A
            A = np.full((m, lmax + 1), -np.inf)
            for k in range(m):
                # 同块持续：d -> d+1；若到达经验最大块长则必然结束。
                A[k, 2:] = prev[k, 1:-1] + log_survive[k, 1:-1]
                if k < m - 1:
                    # 当前块结束，下一块从 d=1 开始；禁止末块循环回首块。
                    A[k + 1, 1] = logsumexp(prev[k, 1:] + log_end[k, 1:])
        emission = SIGN * (eeg_weight * eeg[t] + schedule_weight * schedule[t])
        A += emission[:, None]
        pos = logsumexp(A[SIGN > 0])
        neg = logsumexp(A[SIGN < 0])
        out[t] = np.clip(pos - neg, -30, 30)
    return out


def protocol_table(D, train, maxlen, bandwidth=0.6):
    """训练被试的 P(task | 相对首个带标签窗的帧序)；目标被试完全不参与。"""
    y = np.full((len(train), maxlen), np.nan)
    for i, q in enumerate(train):
        n = min(maxlen, len(D[q][1]))
        y[i, :n] = D[q][1][:n]
    mask = np.isfinite(y).astype(float)
    num = gaussian_filter1d(np.nan_to_num(y).sum(0), bandwidth, mode="nearest")
    den = gaussian_filter1d(mask.sum(0), bandwidth, mode="nearest")
    p = (num + 0.5) / (den + 1.0)
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def write_report(result):
    lines = [
        "# E3 / E48 协议先验因果 HSMM", "", "## 结果", "",
        "全部为 57 人 LOSO。E2 发射分数、块长分布和协议表均排除目标被试；前向递推只读当前及过去帧。",
        "", "| 模型 | 全段 bACC | 全段 AUC | 全段 F1 | 晚段 bACC | 晚段 AUC | 晚段 F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in result["models"].items():
        m, l = r["macro"], r["macro_late45"]
        lines.append(f"| {name} | {m['bACC']:.4f} | {m['AUC']:.4f} | {m['F1']:.4f} | {l['bACC']:.4f} | {l['AUC']:.4f} | {l['F1']:.4f} |")
    fused = result["models"]["e2_schedule_hsmm"]["macro"]
    pure = result["models"]["e2_hsmm"]["macro"]
    lines += [
        "", f"纯 EEG 发射 + 块结构的 HSMM 为 **bACC {pure['bACC']:.4f} / AUC {pure['AUC']:.4f} / F1 {pure['F1']:.4f}**。",
        f"加入协议表后的融合结果为 **bACC {fused['bACC']:.4f} / AUC {fused['AUC']:.4f} / F1 {fused['F1']:.4f}**。",
        "", "## 解释边界", "",
        "`schedule_only` 与 `e2_schedule_hsmm` 的高分主要来自 R/T 交替时间表，不应解释为纯 EEG 生理解码。协议表按首个带标签窗的相对位置对齐；若实际文件含未知前导段，须先估计起点或边缘化偏移。若块数、顺序或首块相位改变，该分支会失效。",
        "", "固定融合权重：EEG 1.0、协议 LLR 1.0；未根据目标标签选权重。",
        "", "## 复现", "", "```powershell",
        "& .\\.venv\\Scripts\\python.exe test\\48_protocol_hsmm\\e48.py", "```", "",
        "输入为 `test/47_rank_ensemble/e47_scores.npz`；结果见 `e48.json` 与 `e48_scores.npz`。",
    ]
    (HERE / "报告_E48_协议先验HSMM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    start = time.time()
    subs, D = L.load()
    e46.SUBS = subs
    source = np.load(HERE.parent / "47_rank_ensemble" / "e47_scores.npz")
    scores = {"subjects": np.asarray(subs)}
    result = {"experiment": "E3/E48", "protocol": "57-subject LOSO", "models": {},
              "assumptions": {"blocks": "R-T-R-T-R-T", "duration_lo_hi_seconds": [120, 420],
                              "eeg_weight": 1.0, "schedule_weight": 1.0}}
    probe = np.linspace(-1, 1, 80)
    probe_changed = probe.copy()
    probe_changed[40:] += 1000
    pri = [(24, np.ones(97) / 97) for _ in range(6)]
    result["causal_prefix_max_abs"] = float(np.max(np.abs(
        hsmm_forward(probe, np.zeros(80), pri)[:40]
        - hsmm_forward(probe_changed, np.zeros(80), pri)[:40])))
    maxlen = max(len(D[q][1]) for q in subs)
    for i, q in enumerate(subs):
        train = [s for s in subs if s != q]
        z = causal_scale(source[f"rank_mean__{q}"])
        table = protocol_table(D, train, maxlen)
        sched = table[:len(z)]
        scores[f"e2__{q}"] = z
        scores[f"schedule_only__{q}"] = sched
        priors = duration_priors(D, train)
        scores[f"e2_hsmm__{q}"] = hsmm_forward(z, sched, priors, eeg_weight=1.0, schedule_weight=0.0)
        scores[f"e2_schedule_hsmm__{q}"] = hsmm_forward(z, sched, priors, eeg_weight=1.0, schedule_weight=1.0)
        np.savez_compressed(HERE / "e48_scores.npz", **scores)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/57 elapsed={time.time() - start:.1f}s", flush=True)
    for key in ("e2", "schedule_only", "e2_hsmm", "e2_schedule_hsmm"):
        result["models"][key] = e46.evaluate_scores(D, scores, key)
        print(key, json.dumps(result["models"][key]["macro"]), flush=True)
    result["elapsed_s"] = time.time() - start
    np.savez_compressed(HERE / "e48_scores.npz", **scores)
    (HERE / "e48.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(result)


if __name__ == "__main__":
    main()
