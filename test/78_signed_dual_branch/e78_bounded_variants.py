"""E78 bounded-magnitude variants: raw sign is fixed; only magnitude is adapted."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

HERE = Path(__file__).resolve().parent
E76 = HERE.parent / "76_conditioned_fusion"
PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")


def metric(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); p = s > 0
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, s)) if n0 and n1 else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def causal_post(score):
    sys.path.insert(0, str(HERE.parent / "17_compliant"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("e59_bounded", HERE.parent / "61_trainfold_initialization" / "e59_trainfold.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.hmm(mod.ema_center(score, 1200, None), 300)


def load():
    out = {}
    for p in sorted(E76.glob("fold_*.npz")):
        with np.load(p) as z:
            q = str(z["target"].item())
            out[q] = {k: z[k].astype(float if k in ("lr_scaled", "mlp_scaled", "gate_score") else z[k].dtype)
                      for k in ("y", "block", "lr_scaled", "mlp_scaled", "gate_score")}
    return out


def bounded_scores(d):
    raw = .5 * (d["lr_scaled"] + d["mlp_scaled"])
    gate = d["gate_score"]
    eps = 1e-8
    ratio = np.abs(gate) / np.maximum(np.abs(raw), eps)
    clipped = np.clip(ratio, 0.75, 1.25)
    sign = np.sign(raw)
    # Candidate A: directly clip conditioned/raw magnitude ratio to [0.75, 1.25].
    clipped_ratio = sign * np.abs(raw) * clipped
    # Candidate B: convexly damp the clipped ratio toward 1, giving [0.875, 1.125].
    convex_ratio = 0.5 + 0.5 * clipped
    convex = sign * np.abs(raw) * convex_ratio
    return raw, {"e78_clip_ratio": clipped_ratio, "e78_convex_ratio": convex}, {
        "ratio_raw_min": float(np.min(ratio)), "ratio_raw_max": float(np.max(ratio)),
        "clip_bounds": [0.75, 1.25], "convex_bounds": [0.875, 1.125],
        "sign_preserved": bool(np.all(np.sign(clipped_ratio) == np.sign(raw)))
    }


def evaluate(d, scores):
    y, block = d["y"], d["block"]
    masks = {"full": np.ones(len(y), bool), "early": block <= 2, "late": block >= 4}
    return {name: {part: metric(y[m], s[m]) for part, m in masks.items()} for name, s in scores.items()}


def main():
    data = load(); subjects = sorted(data)[:6]
    methods = ("w0.50", "e78_clip_ratio", "e78_convex_ratio")
    folds = []
    for i, q in enumerate(subjects, 1):
        d = data[q]; raw, variants, diag = bounded_scores(d)
        scores = {"w0.50": causal_post(raw)}
        scores.update({k: causal_post(v) for k, v in variants.items()})
        mets = evaluate(d, scores)
        folds.append({"fold": i, "target": q, "metrics": mets, "diagnostics": diag})
        print(f"[{i}/6] {q} clip late AUC={mets['e78_clip_ratio']['late']['auc']:.6f}, convex late AUC={mets['e78_convex_ratio']['late']['auc']:.6f}", flush=True)
    aggregate = {k: {part: {m: float(np.mean([r["metrics"][k][part][m] for r in folds])) for m in METRICS} for part in PARTS} for k in methods}
    relative = {k: {part: {m: {"improved": sum(r["metrics"][k][part][m] > r["metrics"]["w0.50"][part][m] for r in folds), "regressed": sum(r["metrics"][k][part][m] < r["metrics"]["w0.50"][part][m] for r in folds), "equal": sum(r["metrics"][k][part][m] == r["metrics"]["w0.50"][part][m] for r in folds)} for m in METRICS} for part in PARTS} for k in methods[1:]}
    result = {"protocol": {"experiment": "E78_bounded_magnitude_variants", "source": "../76_conditioned_fusion/fold_*.npz", "selected_subjects": subjects, "fixed_direction": "sign(raw w=.5)", "candidates": {"e78_clip_ratio": "abs(conditioned)/abs(raw) clipped to [0.75,1.25]", "e78_convex_ratio": "0.5 + 0.5 * clipped ratio, range [0.875,1.125]"}, "postprocess": "causal EMA1200 + forward HMM300"}, "aggregate": aggregate, "relative_to_w0.50": relative, "folds": folds}
    (HERE / "e78_bounded_six_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E78 固定 signed raw 方向的有界幅度六人变体", "", "仅使用 E76 缓存。两候选均固定 w=.5 raw 的符号，不改变方向；只用 |conditioned|/|raw| 调节幅度，并统一因果 EMA1200 + forward HMM300。", "", "## 候选定义", "", "- `e78_clip_ratio`：比例裁剪到 [0.75, 1.25]。", "- `e78_convex_ratio`：裁剪比例向1收缩一半，范围 [0.875, 1.125]。", "", "## 宏观指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for k in methods:
        for part in PARTS:
            a = aggregate[k][part]; lines.append(f"|{k}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对 w=.5 逐人改善/退步/持平", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---|---|---|"]
    for k in methods[1:]:
        for part in PARTS:
            lines.append(f"|{k}|{part}|" + "|".join(f"{relative[k][part][m]['improved']}/{relative[k][part][m]['regressed']}/{relative[k][part][m]['equal']}" for m in METRICS) + "|")
    promising = [k for k in methods[1:] if aggregate[k]["full"]["auc"] > aggregate["w0.50"]["full"]["auc"] and aggregate[k]["late"]["auc"] >= aggregate["w0.50"]["late"]["auc"]]
    lines += ["", "## 结论", "", f"明确潜力候选：{', '.join(promising) if promising else '无'}。判据为 full AUC 高于 w=.5 且 late AUC 不下降。{'建议进入57人验证。' if promising else '不进入57人验证，保留六人结果作为变体诊断。'}", "", "完整逐人指标和幅度诊断见 `e78_bounded_six_results.json`。"]
    (HERE / "报告_E78_有界幅度六人变体.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__": main()
