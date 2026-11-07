"""E78 early-only clipped magnitude variant on E76 cached folds."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

HERE = Path(__file__).resolve().parent
E76 = HERE.parent / "76_conditioned_fusion"
PARTS = ("full", "early", "late")
METHODS = ("w0.50", "e78_early_only_clip")
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
    spec = importlib.util.spec_from_file_location("e59_early_only", HERE.parent / "61_trainfold_initialization" / "e59_trainfold.py")
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
        d = data[q]
        raw = .5 * (d["lr_scaled"] + d["mlp_scaled"])
        ratio = np.clip(np.abs(d["gate_score"]) / np.maximum(np.abs(raw), 1e-8), 0.75, 1.25)
        early = d["block"] <= 2
        adapted = raw.copy()
        adapted[early] = np.sign(raw[early]) * np.abs(raw[early]) * ratio[early]
        scores = {"w0.50": causal_post(raw), "e78_early_only_clip": causal_post(adapted)}
        metrics = evaluate(d, scores)
        np.savez_compressed(HERE / f"e78_early_only_57v1_fold_{i:02d}_{q}.npz", target=q, t=d["t"], y=d["y"], block=d["block"], **scores)
        folds.append({"fold": i, "target": q, "metrics": metrics, "diagnostics": {"ratio_bounds": [0.75, 1.25], "adapted_blocks": "<=2", "preserved_blocks": ">=3 (including late >=4)"}})
        vals = [metrics['e78_early_only_clip'][part]['auc'] for part in ('full', 'early', 'late')]
        fmt = lambda x: 'NA' if x is None else f'{x:.6f}'
        print(f"[{i}/57] {q} early-only full/early/late AUC=" + '/'.join(fmt(x) for x in vals), flush=True)
    aggregate = {k: {part: {m: float(np.mean([r["metrics"][k][part][m] for r in folds if r["metrics"][k][part][m] is not None])) for m in METRICS} for part in PARTS} for k in METHODS}
    relative = {part: {m: {"improved": sum(r["metrics"][METHODS[1]][part][m] > r["metrics"][METHODS[0]][part][m] for r in folds if r["metrics"][METHODS[1]][part][m] is not None and r["metrics"][METHODS[0]][part][m] is not None), "regressed": sum(r["metrics"][METHODS[1]][part][m] < r["metrics"][METHODS[0]][part][m] for r in folds if r["metrics"][METHODS[1]][part][m] is not None and r["metrics"][METHODS[0]][part][m] is not None), "equal": sum(r["metrics"][METHODS[1]][part][m] == r["metrics"][METHODS[0]][part][m] for r in folds if r["metrics"][METHODS[1]][part][m] is not None and r["metrics"][METHODS[0]][part][m] is not None)} for m in METRICS} for part in PARTS}
    protocol = {"experiment": "E78_early_only_clip_57fold", "source": "../76_conditioned_fusion/fold_*.npz", "folds": 57, "direction": "fixed sign(w=.5 raw)", "adaptation": "clip abs(conditioned)/abs(raw) to [0.75,1.25] only where block<=2", "preservation": "block>=3 unchanged, especially late block>=4", "postprocess": "causal EMA1200 + forward HMM300", "outputs": "one NPZ per fold, JSON, Chinese report", "preserves_existing": True}
    result = {"protocol": protocol, "aggregate": aggregate, "relative_to_w0.50": relative, "folds": folds}
    (HERE / "e78_early_only_57_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E78 early-only clip ratio 57人验证", "", "严格复用 E76 逐折缓存。仅对 early block<=2 使用 clip ratio 幅度调节；late block>=4 及其他 block 保留 w=.5 原始 score，统一因果 EMA1200 + forward HMM300。", "", "## 协议", "", "- raw = `0.5 * (lr_scaled + mlp_scaled)`，全程固定 raw 符号。", "- early block<=2：`|conditioned|/|raw|` 裁剪到 `[0.75,1.25]`。", "- block>=3 不调节，特别是 late block>=4 严格保留基线。", "", "## 宏观指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for k in METHODS:
        for part in PARTS:
            a = aggregate[k][part]; lines.append(f"|{k}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对 w=.5 逐被试改善/退步/持平", "", "|阶段|AUC|F1|bACC|", "|---|---|---|---|"]
    for part in PARTS:
        lines.append(f"|{part}|" + "|".join(f"{relative[part][m]['improved']}/{relative[part][m]['regressed']}/{relative[part][m]['equal']}" for m in METRICS) + "|")
    lines += ["", "## 结论", "", f"early-only clip 的 full AUC/F1/bACC = {aggregate[METHODS[1]]['full']['auc']:.6f}/{aggregate[METHODS[1]]['full']['f1']:.6f}/{aggregate[METHODS[1]]['full']['bacc']:.6f}；late AUC/F1/bACC = {aggregate[METHODS[1]]['late']['auc']:.6f}/{aggregate[METHODS[1]]['late']['f1']:.6f}/{aggregate[METHODS[1]]['late']['bacc']:.6f}。", "由于 late 区间逐点保持 raw score，理论上其未经过分块调节；最终指标仍统一经过同一因果后处理。", "", "逐折产物见 `e78_early_only_57v1_fold_*.npz`，完整JSON见 `e78_early_only_57_results.json`。"]
    (HERE / "报告_E78_early_only_clip_57人.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__": main()
