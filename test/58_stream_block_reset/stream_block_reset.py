from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "test" / "17_compliant"))
import infer  # noqa: E402
from cogstate import load_trials  # noqa: E402
from streaming import BUNDLE, StreamingInference  # noqa: E402

E53 = ROOT / "test" / "53_e17_post_loso"
E57 = ROOT / "test" / "57_post_reset_ablation"
OUT_JSON = HERE / "results.json"
OUT_CSV = HERE / "results.csv"
OUT_MD = HERE / "报告_58_真实BDF流式块边界重置.md"


def metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    p = score > 0
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, score))
    return {"bacc": float(.5 * (p[y == 1].mean() + (1 - p[y == 0]).mean())),
            "auc": auc, "f1": float(f1_score(y, p, zero_division=0))}


class ResetPost:
    def __init__(self, mode):
        self.mode = mode
        self.dt = infer.CausalDetrend(1200)
        self.hmm = infer.ForwardHMM(300, 1.0)

    def reset(self):
        if self.mode in ("dt_reset_hmm_keep", "dt_reset_hmm_reset"):
            self.dt = infer.CausalDetrend(1200)
        if self.mode == "dt_reset_hmm_reset":
            self.hmm = infer.ForwardHMM(300, 1.0)

    def update(self, score):
        z = self.dt.update(score)
        return self.hmm.update(z)


def block_at(t, trials):
    for i, tr in enumerate(trials):
        if tr.start <= t <= tr.end + 1e-9:
            return i
    return None


def stream_subject(sid, chunk, mode, do_prefix=False):
    bdf = next((ROOT / "EEG_dataset" / sid).glob("*.bdf"))
    csv_path = ROOT / "EEG_dataset" / sid / f"{sid}_events.csv"
    rec = infer.load_bdf(bdf)
    fs = rec["fs"]
    signal = np.asarray(rec["signal_uv"], dtype=float) * 1e-6
    signal[~rec["valid"]] = np.nan
    trials = load_trials(str(csv_path), str(bdf))
    bundle = __import__("torch").load(BUNDLE, map_location="cpu", weights_only=False)

    def run(x, c, prefix=False):
        dec = StreamingInference(bundle, fs=fs, step=1.0)
        post = ResetPost(mode)
        out = []
        prev = None
        for start in range(0, len(x), c):
            end = min(start + c, len(x))
            for time_s, score, _ in dec.update(x[start:end]):
                b = block_at(time_s, trials)
                if b is not None and prev is not None and b != prev:
                    post.reset()
                z = post.update(float(score))
                out.append({"time_s": float(time_s), "score": z, "block": b})
                prev = b
        return out

    out = run(signal, chunk)
    y, s, b, t = [], [], [], []
    for r in out:
        tr = next((q for q in trials if q.start < r["time_s"] <= q.end + 1e-9), None)
        if tr is None or r["block"] is None:
            continue
        y.append(tr.label); s.append(r["score"]); b.append(r["block"] + 1); t.append(r["time_s"])
    y, s, b, t = map(np.asarray, (y, s, b, t))
    parts = {"full": np.ones(len(y), bool), "first600": t < 600, "block45": b >= 4}
    result = {k: metrics(y[m], s[m]) for k, m in parts.items() if m.any()}
    result["n_windows"] = int(len(y))
    result["block_boundaries_s"] = [float(x.start) for x in trials]
    result["n_blocks"] = len(trials)
    if do_prefix:
        baseline = run(signal, len(signal))
        pref = run(signal[: int(1500 * fs)], 250)
        result["prefix"] = {"n": len(pref), "identical": pref == baseline[:len(pref)],
                             "max_abs": float(max((abs(a["score"] - z["score"]) for a, z in zip(pref, baseline)), default=0))}
        result["chunks"] = {}
        base_scores = np.array([z["score"] for z in baseline])
        for name, c in (("250pt", 250), ("137pt", 137), ("92pt", 92), ("1250pt", 1250)):
            q = run(signal, c)
            qs = np.array([z["score"] for z in q])
            result["chunks"][name] = {"n": len(q), "identical": q == baseline,
                                      "max_abs": float(np.max(np.abs(qs - base_scores))) if len(q) == len(baseline) else None}
    return result


def aggregate(per, subjects, part):
    vals = [per[s][part] for s in subjects if part in per[s]]
    return {k: float(np.nanmean([v[k] for v in vals])) for k in ("bacc", "auc", "f1")} if vals else {}


def fallback_verify():
    import importlib.util
    spec = importlib.util.spec_from_file_location("e57", E57 / "ablation.py")
    e57 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e57)
    subs, D = e57.loso.load()
    data = np.load(E53 / "e53_scores.npz")
    groups = e57.group_map()
    modes = {"dt_reset_hmm_keep": "每真实block重置detrend，保留HMM",
             "dt_reset_hmm_reset": "每真实block重置detrend+HMM"}
    out, rows = {"protocol": "fallback verify data; explicit six labeled blocks from LOSO cache", "source": str(E53 / "e53_scores.npz"), "modes": {}}, []
    for mode, label in modes.items():
        per = {}
        for q in subs:
            y, b = D[q][1], D[q][2]
            raw = data[q].astype(float); z = np.empty(len(raw)); post = ResetPost(mode); prev = None
            for i, (x, bi) in enumerate(zip(raw, b)):
                if prev is not None and bi != prev: post.reset()
                z[i] = post.update(x); prev = bi
            t = np.arange(len(y)); per[q] = {k: metrics(y[m], z[m]) for k, m in {"full": np.ones(len(y), bool), "first600": t < 600, "block45": b >= 4}.items()}
        agg = {}
        for scope, ss in (("all", subs), ("high", [q for q in subs if groups[q] == "high"]), ("low", [q for q in subs if groups[q] == "low"])):
            agg[scope] = {p: aggregate(per, ss, p) for p in ("full", "first600", "block45")}
            for p, m in agg[scope].items(): rows.append({"mode": mode, "label": label, "scope": scope, "part": p, **m})
        out["modes"][mode] = {"label": label, "aggregate": agg, "per_subject": per}
    out["comparison"] = {"e53": json.loads((E53 / "e53.json").read_text(encoding="utf-8")), "e57": json.loads((E57 / "results.json").read_text(encoding="utf-8"))}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "label", "scope", "part", "bacc", "auc", "f1"]); w.writeheader(); w.writerows(rows)
    lines = ["# E58 真实BDF流式 block 重置核验", "", "真实BDF全量运行因E17特征提取在当前环境耗时过长，故按规则退回现有 verify 数据：E53 LOSO 分数、明确六个真实标签 block 边界；不冒充真实BDF验证。", "", "|方案|范围|片段|bACC|AUC|F1|", "|---|---|---|---:|---:|---:|"]
    lines += [f"|{r['label']}|{r['scope']}|{r['part']}|{r['bacc']:.4f}|{r['auc']:.4f}|{r['f1']:.4f}|" for r in rows]
    lines += ["", "## 结论", "", "该最小复现实验仅验证真实 block 边界触发的因果后处理，不足以证明端到端真实BDF收益；因此不建议据此纳入正式部署。正式部署前需在可接受时限内完成真实BDF全体核验并验证所有chunk一致性。"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({m: v["aggregate"]["all"] for m, v in out["modes"].items()}, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma-separated IDs; default all BDF subjects")
    ap.add_argument("--chunk", type=int, default=137)
    ap.add_argument("--no-prefix", action="store_true")
    ap.add_argument("--fallback-verify", action="store_true")
    args = ap.parse_args()
    if args.fallback_verify:
        fallback_verify(); return
    subjects = [p.name for p in sorted((ROOT / "EEG_dataset").iterdir()) if (p / f"{p.name}_events.csv").exists() and list(p.glob("*.bdf"))]
    if args.subjects:
        subjects = [x.zfill(3) for x in args.subjects.split(",")]
    modes = {"dt_reset_hmm_keep": "每真实block重置detrend，保留HMM",
             "dt_reset_hmm_reset": "每真实block重置detrend+HMM"}
    all_results = {"protocol": "真实BDF strict-causal StreamingInference; reset only when block boundary has arrived",
                   "chunk": args.chunk, "subjects": subjects, "modes": {}}
    rows = []
    for mode, label in modes.items():
        per = {}
        for i, sid in enumerate(subjects, 1):
            print(f"[{mode}] {i}/{len(subjects)} {sid}", flush=True)
            per[sid] = stream_subject(sid, args.chunk, mode, do_prefix=(i == 1 and not args.no_prefix))
        groups = {}
        gm = {}
        with (ROOT / "test" / "43_group_analysis" / "group_members.csv").open(encoding="utf-8-sig") as f:
            for r in csv.DictReader(f): gm[r["subject"]] = r["group"]
        for scope, ss in (("all", subjects), ("high", [s for s in subjects if gm.get(s) == "high"]), ("low", [s for s in subjects if gm.get(s) == "low"])):
            groups[scope] = {part: aggregate(per, ss, part) for part in ("full", "first600", "block45")}
            for part, m in groups[scope].items():
                rows.append({"mode": mode, "label": label, "scope": scope, "part": part, **m})
        all_results["modes"][mode] = {"label": label, "aggregate": groups, "per_subject": per}
    all_results["comparison"] = {"e53": json.loads((E53 / "e53.json").read_text(encoding="utf-8")),
                                  "e57": json.loads((E57 / "results.json").read_text(encoding="utf-8"))}
    OUT_JSON.write_text(json.dumps(all_results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "label", "scope", "part", "bacc", "auc", "f1"])
        w.writeheader(); w.writerows(rows)
    lines = ["# E58 真实BDF流式 block 重置核验", "", f"输入：真实 BDF + 事件 CSV，chunk={args.chunk}；模型与推理引擎沿用 `{BUNDLE}`。", "严格因果：仅在当前输出时间到达真实 trial 边界时重置；窗口、特征、EMA、HMM均不读取未来。", "", "## 真实BDF结果", "", "|方案|范围|片段|bACC|AUC|F1|", "|---|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['label']}|{r['scope']}|{r['part']}|{r['bacc']:.4f}|{r['auc']:.4f}|{r['f1']:.4f}|")
    lines += ["", "## 与E53/E57", "", "E53为离线分数基线，E57为离线分数重置消融；二者仅作为参考，不与真实BDF窗口数量混合计算。", "", "## 前缀与分块一致性"]
    for mode, x in all_results["modes"].items():
        p = next(iter(x["per_subject"].values())).get("prefix")
        lines.append(f"- {x['label']}：前缀 identical={p.get('identical') if p else '未运行'}，max_abs={p.get('max_abs') if p else 'NA'}；不同chunk结果见 JSON。")
    lines += ["", "## 部署建议", "", "真实BDF验证确认代码路径可因果执行，但是否纳入正式部署以全体真实BDF指标及跨chunk一致性为准。若任一chunk不一致或块重置未提升块4-5且全段下降不超过0.005，则不建议纳入。", "", f"JSON：`{OUT_JSON}`", f"CSV：`{OUT_CSV}`"]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({m: v["aggregate"]["all"] for m, v in all_results["modes"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
