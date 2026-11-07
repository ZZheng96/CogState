"""E78: signed + magnitude dual-branch fusion diagnostic."""
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
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, s)) if n0 and n1 else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def causal_post(score):
    sys.path.insert(0, str(HERE.parent / "17_compliant"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("e59", HERE.parent / "61_trainfold_initialization" / "e59_trainfold.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.hmm(mod.ema_center(score, 1200, None), 300)


def load():
    out = {}
    for p in sorted(E76.glob("fold_*.npz")):
        with np.load(p) as z:
            q = str(z["target"].item())
            out[q] = {k: z[k].astype(float if k in ("lr_scaled", "mlp_scaled", "gate_score") else z[k].dtype)
                      for k in ("t", "y", "block", "lr_scaled", "mlp_scaled", "gate_score", "score_w0.50")}
    return out


def dual_branch(d):
    raw = .5 * (d["lr_scaled"] + d["mlp_scaled"])
    gate = d["gate_score"]
    # Signed branch keeps gate direction, but rejects a direction conflict using raw direction.
    sign = np.where(np.sign(gate) == np.sign(raw), np.sign(gate), np.sign(raw))
    # Magnitude branch is conservative geometric mean, with epsilon-safe zero handling.
    magnitude = np.sqrt(np.maximum(np.abs(gate), 0.) * np.maximum(np.abs(raw), 0.))
    return sign * magnitude


def evaluate(d, scores):
    y, block = d["y"], d["block"]
    masks = {"full": np.ones(len(y), bool), "early": block <= 2, "late": block >= 4}
    return {name: {part: metric(y[m], s[m]) for part, m in masks.items()} for name, s in scores.items()}


def main():
    data = load(); subjects = sorted(data); selected = subjects[:6]
    folds = []
    for i, q in enumerate(selected, 1):
        d = data[q]
        raw = .5 * (d["lr_scaled"] + d["mlp_scaled"])
        dual = dual_branch(d)
        scores = {"w0.50": causal_post(raw), "conditioned": causal_post(d["gate_score"]), "e78_dual": causal_post(dual)}
        mets = evaluate(d, scores)
        folds.append({"fold": i, "target": q, "metrics": mets,
                      "diagnostics": {"raw_abs_mean": float(np.mean(np.abs(raw))),
                                      "gate_abs_mean": float(np.mean(np.abs(d["gate_score"]))),
                                      "dual_abs_mean": float(np.mean(np.abs(dual))),
                                      "sign_conflict_rate": float(np.mean(np.sign(raw) != np.sign(d["gate_score"])))}})
        print(f"[{i}/6] {q} dual late AUC={mets['e78_dual']['late']['auc']:.6f}", flush=True)
    methods = ("w0.50", "conditioned", "e78_dual")
    aggregate = {k: {part: {m: float(np.mean([r["metrics"][k][part][m] for r in folds])) for m in METRICS}
                            for part in PARTS} for k in methods}
    relative = {}
    for k in methods[1:]:
        relative[k] = {part: {m: {"improved": sum(r["metrics"][k][part][m] > r["metrics"]["w0.50"][part][m] for r in folds),
                                  "regressed": sum(r["metrics"][k][part][m] < r["metrics"]["w0.50"][part][m] for r in folds),
                                  "equal": sum(r["metrics"][k][part][m] == r["metrics"]["w0.50"][part][m] for r in folds)} for m in METRICS} for part in PARTS}
    result = {"protocol": json.loads((HERE / "protocol.json").read_text(encoding="utf-8")),
              "selected_subjects": selected, "aggregate": aggregate, "relative_to_w0.50": relative, "folds": folds}
    (HERE / "e78_six_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E78 signed+幅度双分支六人诊断", "", "严格使用E76逐折缓存；六人目标被试为数据集排序后的前六人。分支保留门控符号并以固定w=.5幅度作几何均值，统一因果EMA1200+forward HMM300。", "", "## 宏观指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for k in methods:
        for part in PARTS:
            a = aggregate[k][part]; lines.append(f"|{k}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对固定w=.5逐人改善/退步/持平", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---|---|---|"]
    for k in methods[1:]:
        for part in PARTS:
            lines.append(f"|{k}|{part}|" + "|".join(f"{relative[k][part][m]['improved']}/{relative[k][part][m]['regressed']}/{relative[k][part][m]['equal']}" for m in METRICS) + "|")
    gain = aggregate["e78_dual"]["full"]["auc"] > aggregate["w0.50"]["full"]["auc"] and aggregate["e78_dual"]["late"]["auc"] >= aggregate["w0.50"]["late"]["auc"]
    lines += ["", "## 结论", "", f"六人诊断明确潜力：{'是，满足full AUC提升且late AUC不下降，可进入57人验证。' if gain else '否，未同时满足full AUC提升与late AUC不下降，不扩展57人。'}", "", f"当前E78双分支 full AUC/F1/bACC = {aggregate['e78_dual']['full']['auc']:.6f}/{aggregate['e78_dual']['full']['f1']:.6f}/{aggregate['e78_dual']['full']['bacc']:.6f}；late AUC/F1/bACC = {aggregate['e78_dual']['late']['auc']:.6f}/{aggregate['e78_dual']['late']['f1']:.6f}/{aggregate['e78_dual']['late']['bacc']:.6f}。完整逐人诊断见 `e78_six_results.json`。"]
    (HERE / "报告_E78_六人诊断.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__": main()
