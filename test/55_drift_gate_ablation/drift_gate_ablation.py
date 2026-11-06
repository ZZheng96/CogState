"""E55：严格因果分数层漂移检测门控消融。

只读取 E53 的 LOSO 分数；所有门控仅使用当前及过去分数，阈值为预先固定值，
不使用目标被试标签。评价中的前600/块4-5/分组仅用于报告。
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = TESTS.parent
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L  # noqa: E402

E53_SPEC = importlib.util.spec_from_file_location("e53lib", TESTS / "53_e17_post_loso" / "e53.py")
E53 = importlib.util.module_from_spec(E53_SPEC)
E53_SPEC.loader.exec_module(E53)
E44 = E53.E44
metrics, summary = E53.metrics, E53.summary
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REV = tuple(E44.REV)
OUT_JSON = HERE / "drift_gate_ablation.json"
OUT_MD = HERE / "报告_E55_严格因果漂移门控消融.md"


def groups():
    with open(TESTS / "43_group_analysis" / "group_members.csv", encoding="utf-8-sig") as f:
        return {r["subject"]: r["group"] for r in csv.DictReader(f)}


def ema(x, hl):
    a = 1.0 - 0.5 ** (1.0 / hl)
    out = torch.empty_like(x)
    m = x[:, 0]
    out[:, 0] = m
    for t in range(1, x.shape[1]):
        m = m + a * (x[:, t] - m)
        out[:, t] = m
    return out


def rolling_flip(x, w=120):
    sign = (x > 0).float()
    flip = torch.zeros_like(sign)
    flip[:, 1:] = (sign[:, 1:] != sign[:, :-1]).float()
    cs = torch.cat([torch.zeros((x.shape[0], 1), device=x.device), flip.cumsum(1)], 1)
    idx = torch.arange(x.shape[1], device=x.device)
    lo = (idx - w + 1).clamp_min(0)
    n = idx - lo + 1
    return (cs[:, idx + 1] - cs[:, lo]) / n


def causal_var(x, hl):
    m = ema(x, hl)
    m2 = ema(x * x, hl)
    return (m2 - m * m).clamp_min(1e-6)


def build_candidates(subs, scores):
    # padding is only for batched computation; padded frames are removed below.
    T = max(len(scores[q]) for q in subs)
    S = torch.zeros((len(subs), T), device=DEV)
    lens = []
    for i, q in enumerate(subs):
        lens.append(len(scores[q]))
        S[i, : len(scores[q])] = torch.as_tensor(scores[q], device=DEV)
    base = E44.hmm(S - ema(S, 1200), torch.full_like(S, -np.log(300.0)))
    raw = S
    short, long = ema(S, 30), ema(S, 600)
    drift_z = (short - long).abs() / causal_var(S, 600).sqrt()
    var_ratio = causal_var(S, 30) / causal_var(S, 600)
    V = {"baseline_dt1200_hmm_d300": base}
    # 固定阈值：漂移 z>1.0 时降低驻留到30或600；不由本批标签选择。
    for thr, dwell in ((1.0, 30), (1.5, 30), (1.0, 600)):
        fire = drift_z > thr
        sw = torch.where(fire, torch.full_like(S, -np.log(dwell)), torch.full_like(S, -np.log(300)))
        V[f"ema_drift_z{thr:g}_d{dwell}"] = E44.hmm(S - ema(S, 1200), sw)
    flip = rolling_flip(S, 120)
    for thr in (0.20, 0.35):
        V[f"flip120_raw_if_gt{thr:g}"] = torch.where(flip > thr, raw, base)
        V[f"flip120_hmm_if_gt{thr:g}"] = torch.where(flip > thr, E44.hmm(S, torch.full_like(S, -np.log(300))), base)
    # 方差变化只改变 HMM 驻留，不切换标签方向；阈值为固定 2x/0.5x。
    var_change = (var_ratio > 2.0) | (var_ratio < 0.5)
    for dwell in (30, 600):
        sw = torch.where(var_change, torch.full_like(S, -np.log(dwell)), torch.full_like(S, -np.log(300)))
        V[f"variance_ratio_gate_d{dwell}"] = E44.hmm(S - ema(S, 1200), sw)
    return {k: {q: V[k][i, : lens[i]].detach().cpu().numpy() for i, q in enumerate(subs)} for k in V}, {
        "ema_drift_z1": {q: float((drift_z[i, : lens[i]] > 1.0).float().mean().cpu()) for i, q in enumerate(subs)},
        "ema_drift_z1_5": {q: float((drift_z[i, : lens[i]] > 1.5).float().mean().cpu()) for i, q in enumerate(subs)},
        "flip_gt20": {q: float((flip[i, : lens[i]] > 0.20).float().mean().cpu()) for i, q in enumerate(subs)},
        "flip_gt35": {q: float((flip[i, : lens[i]] > 0.35).float().mean().cpu()) for i, q in enumerate(subs)},
        "variance_change": {q: float(var_change[i, : lens[i]].float().mean().cpu()) for i, q in enumerate(subs)},
    }


def split_rows(subs, D, sc, grp):
    sections = {
        "full": lambda y, b: np.ones(len(y), dtype=bool),
        "front600": lambda y, b: np.arange(len(y)) < min(600, len(y)),
        "block45": lambda y, b: b >= 4,
    }
    out = {}
    for name, keep_fn in sections.items():
        rows = []
        for q in subs:
            y, b = D[q][1], D[q][2]
            keep = keep_fn(y, b)
            rows.append(metrics(y[keep], sc[q][keep]))
        dec = {"all": summary(rows)}
        for g, fn in (("high", lambda q: grp[q] == "high"), ("low", lambda q: grp[q] == "low"), ("rev4", lambda q: q in REV)):
            dec[g] = summary([r for q, r in zip(subs, rows) if fn(q)])
        dec["rev4_subjects"] = {q: rows[subs.index(q)] for q in REV}
        out[name] = dec
    return out


def main():
    subs, D = L.load()
    data = np.load(TESTS / "53_e17_post_loso" / "e53_scores.npz")
    scores = {q: data[q] for q in subs}
    candidates, trigger = build_candidates(subs, scores)
    grp = groups()
    result = {"protocol": {"source": "test/53_e17_post_loso/e53_scores.npz", "causal": True,
                            "thresholds_fixed": {"drift_z": [1.0, 1.5], "flip120": [0.20, 0.35], "variance_ratio": [0.5, 2.0]},
                            "target_labels_used_for_gate": False}, "candidates": {}}
    base = candidates["baseline_dt1200_hmm_d300"]
    for name, sc in candidates.items():
        result["candidates"][name] = {"metrics": split_rows(subs, D, sc, grp)}
        if name == "baseline_dt1200_hmm_d300":
            continue
        # 每个被试的触发率及宏平均触发率。
        if name.startswith("ema_drift_z"):
            key = "ema_drift_z1_5" if "z1.5" in name else "ema_drift_z1"
        elif name.startswith("flip120"):
            key = "flip_gt35" if "0.35" in name else "flip_gt20"
        else:
            key = "variance_change"
        result["candidates"][name]["trigger_rate"] = {"mean": float(np.mean(list(trigger[key].values()))), "per_subject": trigger[key]}
        m = result["candidates"][name]["metrics"]
        b = result["candidates"]["baseline_dt1200_hmm_d300"]["metrics"]
        ok = (m["full"]["all"]["bacc"] >= b["full"]["all"]["bacc"] - 0.005 and
              m["block45"]["all"]["bacc"] >= b["block45"]["all"]["bacc"] + 0.01 and
              m["full"]["high"]["bacc"] >= b["full"]["high"]["bacc"] - 0.01)
        result["candidates"][name]["retain"] = bool(ok)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(result)
    print(json.dumps({k: {s: result["candidates"][k]["metrics"][s]["all"] for s in ("full", "front600", "block45")} for k in result["candidates"]}, ensure_ascii=False))


def write_report(result):
    c = result["candidates"]
    b = c["baseline_dt1200_hmm_d300"]["metrics"]
    lines = ["# E55：严格因果漂移检测门控消融", "", "## 结论", ""]
    kept = [k for k, v in c.items() if v.get("retain")]
    lines.append("可保留候选：" + (", ".join(kept) if kept else "无；所有候选明确放弃。"))
    lines += ["", "## 当前基线", "", "基线为 `dt1200 + HMM(dwell=300)`，仅分数层因果后处理。", "",
              f"|区段|bACC|AUC|F1|", "|---|---:|---:|---:|"]
    for s, label in (("full", "全段"), ("front600", "前600帧"), ("block45", "块4–5")):
        x = b[s]["all"]
        lines.append(f"|{label}|{x['bacc']:.4f}|{x['auc']:.4f}|{x['f1']:.4f}|")
    lines += ["", "## 候选结果", "", "触发率为全体帧中门控条件成立的比例；阈值预先固定，未使用目标标签。", "",
              "|候选|全段 bACC/AUC/F1|前600 bACC/AUC/F1|块4–5 bACC/AUC/F1|高组全段 bACC|低组全段 bACC|反转4人块4–5 bACC|触发率|判定|",
              "|---|---|---|---|---:|---:|---:|---:|"]
    for k, v in c.items():
        m = v["metrics"]
        vals = [f"{m[s]['all']['bacc']:.4f}/{m[s]['all']['auc']:.4f}/{m[s]['all']['f1']:.4f}" for s in ("full", "front600", "block45")]
        tr = v.get("trigger_rate", {}).get("mean", 0.0)
        lines.append(f"|{k}|{'|'.join(vals)}|{m['full']['high']['bacc']:.4f}|{m['full']['low']['bacc']:.4f}|{m['block45']['rev4']['bacc']:.4f}|{tr:.4f}|{'可保留' if v.get('retain') else '放弃'}|")
    lines += ["", "## 4 个反转被试（块4–5）", "", "|候选|004|009|010|050|", "|---|---:|---:|---:|---:|"]
    for k, v in c.items():
        r = v["metrics"]["block45"]["rev4_subjects"]
        lines.append(f"|{k}|" + "|".join(f"{r[q]['bacc']:.4f}" for q in REV) + "|")
    lines += ["", "## 严格因果口径", "", "短/长 EMA、120 帧符号翻转率与 EMA 方差均只看当前及历史分数；切换仅改变 HMM 驻留或选择原始/HMM 输出。评价分组和块号不进入门控。"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
