"""E78 full 57-fold validation from E76 cached fold outputs."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

HERE = Path(__file__).resolve().parent
E76 = HERE.parent / "76_conditioned_fusion"
PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")
METHODS = ("w0.50", "e78_clip_ratio", "e78_convex_ratio")


def metric(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); p = s > 0
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, s)) if n0 and n1 else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def causal_post(score):
    sys.path.insert(0, str(HERE.parent / "17_compliant"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("e59_e78_57", HERE.parent / "61_trainfold_initialization" / "e59_trainfold.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.hmm(mod.ema_center(score, 1200, None), 300)


def load():
    out = {}
    for p in sorted(E76.glob("fold_*.npz")):
        with np.load(p) as z:
            q = str(z["target"].item())
            out[q] = {k: z[k].astype(float if k in ("lr_scaled", "mlp_scaled", "gate_score") else z[k].dtype)
                      for k in ("t", "y", "block", "lr_scaled", "mlp_scaled", "gate_score")}
    return out


def variants(d):
    raw = .5 * (d["lr_scaled"] + d["mlp_scaled"])
    ratio = np.abs(d["gate_score"]) / np.maximum(np.abs(raw), 1e-8)
    clipped = np.clip(ratio, 0.75, 1.25)
    sign = np.sign(raw)
    return raw, {
        "e78_clip_ratio": sign * np.abs(raw) * clipped,
        "e78_convex_ratio": sign * np.abs(raw) * (0.5 + 0.5 * clipped),
    }, {"ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
        "clip_bounds": [0.75, 1.25], "convex_bounds": [0.875, 1.125]}


def evaluate(d, scores):
    y, block = d["y"], d["block"]
    masks = {"full": np.ones(len(y), bool), "early": block <= 2, "late": block >= 4}
    return {k: {part: metric(y[m], s[m]) for part, m in masks.items()} for k, s in scores.items()}


def main():
    data = load(); subjects = sorted(data)
    if len(subjects) != 57:
        raise RuntimeError(f"E76缓存应为57折，实际为{len(subjects)}折")
    folds = []
    for i, q in enumerate(subjects, 1):
        d = data[q]; raw, vals, diag = variants(d)
        score = {"w0.50": causal_post(raw)}
        score.update({k: causal_post(v) for k, v in vals.items()})
        metrics = evaluate(d, score)
        np.savez_compressed(HERE / f"e78_bounded_57v2_fold_{i:02d}_{q}.npz", target=q, t=d["t"], y=d["y"], block=d["block"], **score)
        folds.append({"fold": i, "target": q, "metrics": metrics, "diagnostics": diag})
        print(f"[{i}/57] {q} clip full/late AUC={metrics['e78_clip_ratio']['full']['auc']:.6f}/{metrics['e78_clip_ratio']['late']['auc']:.6f}", flush=True)
    aggregate = {k: {part: {m: float(np.mean([r["metrics"][k][part][m] for r in folds if r["metrics"][k][part][m] is not None])) for m in METRICS} for part in PARTS} for k in METHODS}
    relative = {k: {part: {m: {"improved": sum(r["metrics"][k][part][m] > r["metrics"]["w0.50"][part][m] for r in folds if r["metrics"][k][part][m] is not None and r["metrics"]["w0.50"][part][m] is not None), "regressed": sum(r["metrics"][k][part][m] < r["metrics"]["w0.50"][part][m] for r in folds if r["metrics"][k][part][m] is not None and r["metrics"]["w0.50"][part][m] is not None), "equal": sum(r["metrics"][k][part][m] == r["metrics"]["w0.50"][part][m] for r in folds if r["metrics"][k][part][m] is not None and r["metrics"]["w0.50"][part][m] is not None)} for m in METRICS} for part in PARTS} for k in METHODS[1:]}
    protocol = {"experiment": "E78_bounded_magnitude_57fold", "source": "../76_conditioned_fusion/fold_*.npz", "folds": 57, "subjects": subjects, "direction": "fixed sign(raw w=.5)", "clip_ratio": "abs(conditioned)/max(abs(raw),eps), clipped [0.75,1.25]", "convex_ratio": "0.5+0.5*clip_ratio, range [0.875,1.125]", "postprocess": "causal EMA1200 + forward HMM300", "outputs": "one compressed NPZ per fold; aggregate JSON and Chinese report", "preserves": ["e78_six_results.json", "e78_bounded_six_results.json", "报告_E78_六人诊断.md", "报告_E78_有界幅度六人变体.md"]}
    result = {"protocol": protocol, "aggregate": aggregate, "relative_to_w0.50": relative, "folds": folds}
    (HERE / "e78_bounded_57_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E78 固定 signed raw 方向有界幅度 57 人验证", "", "严格复用 E76 的 57 个逐折缓存；每个目标被试独立计算，统一因果 EMA1200 + forward HMM300。方向固定为 w=.5 raw，仅调节幅度。", "", "## 协议", "", "- `e78_clip_ratio`：`|conditioned|/|raw|` 裁剪到 `[0.75, 1.25]`。", "- `e78_convex_ratio`：裁剪比例向 1 收缩一半，范围 `[0.875, 1.125]`。", "- 每折预测及标签保存为独立压缩 NPZ，未覆盖六人结果。", "", "## 57人宏观指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for k in METHODS:
        for part in PARTS:
            a = aggregate[k][part]; lines.append(f"|{k}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对 w=.5 逐被试改善/退步/持平", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---|---|---|"]
    for k in METHODS[1:]:
        for part in PARTS:
            lines.append(f"|{k}|{part}|" + "|".join(f"{relative[k][part][m]['improved']}/{relative[k][part][m]['regressed']}/{relative[k][part][m]['equal']}" for m in METRICS) + "|")
    best = max(METHODS[1:], key=lambda k: aggregate[k]["late"]["auc"])
    lines += ["", "## 结论", "", f"57人验证中，late AUC最高的候选为 `{best}`。clip ratio 的 full AUC/F1/bACC = {aggregate['e78_clip_ratio']['full']['auc']:.6f}/{aggregate['e78_clip_ratio']['full']['f1']:.6f}/{aggregate['e78_clip_ratio']['full']['bacc']:.6f}；late AUC/F1/bACC = {aggregate['e78_clip_ratio']['late']['auc']:.6f}/{aggregate['e78_clip_ratio']['late']['f1']:.6f}/{aggregate['e78_clip_ratio']['late']['bacc']:.6f}。", "", "逐折NPZ、逐折指标和诊断见本目录的 `e78_bounded_fold_*.npz` 与 `e78_bounded_57_results.json`。"]
    (HERE / "报告_E78_有界幅度57人验证.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__": main()
