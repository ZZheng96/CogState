"""E79: strict LOSO late orientation-invariant absolute-feature Logistic expert."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

HERE = Path(__file__).resolve().parent
E76 = HERE.parent / "76_conditioned_fusion"
PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")


def metric(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); p = s > 0
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    return {"n": int(len(y)), "n_negative": n0, "n_positive": n1,
            "auc": float(roc_auc_score(y, s)) if n0 and n1 else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def evaluate(y, block, s):
    masks = {"full": np.ones(len(y), bool), "early": block <= 2, "late": block >= 4}
    return {k: metric(y[v], s[v]) for k, v in masks.items()}


def stage(raw):
    import importlib.util
    sys.path.insert(0, str(HERE.parent / "17_compliant"))
    spec = importlib.util.spec_from_file_location("e59_trainfold", HERE.parent / "61_trainfold_initialization" / "e59_trainfold.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.hmm(mod.ema_center(np.asarray(raw, float), 1200, None), 300)


def load():
    out = {}
    for p in sorted(E76.glob("fold_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            q = str(z["target"].item())
            required = ("t", "y", "block", "lr_scaled", "mlp_scaled", "score_w0.50")
            out[q] = {k: z[k].astype(float if k in ("lr_scaled", "mlp_scaled", "score_w0.50") else z[k].dtype) for k in required}
    return out


def abs_features(lr, mlp):
    lr = np.asarray(lr, float); mlp = np.asarray(mlp, float)
    return np.column_stack((np.abs(lr), np.abs(mlp), np.abs(lr - mlp))).astype(np.float64)


def main():
    data = load(); subjects = sorted(data)
    assert len(subjects) == 57, f"expected 57 E76 folds, got {len(subjects)}"
    folds = []
    for i, q in enumerate(subjects, 1):
        started = time.time(); d = data[q]
        train_subjects = [p for p in subjects if p != q]
        train_masked = [(data[p]["block"] >= 4) for p in train_subjects]
        X_parts = [abs_features(data[p]["lr_scaled"], data[p]["mlp_scaled"])[m] for p, m in zip(train_subjects, train_masked)]
        y_parts = [data[p]["y"][m] for p, m in zip(train_subjects, train_masked)]
        X = np.concatenate(X_parts); y = np.concatenate(y_parts)
        mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
        clf = LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear", max_iter=300, random_state=0)
        clf.fit((X - mu) / sd, y)
        xt = abs_features(d["lr_scaled"], d["mlp_scaled"])
        u_abs_scaled = clf.decision_function((xt - mu) / sd)
        fixed = d["score_w0.50"]
        # Baseline is the exact E76 cached postprocessed score, never restaged.
        raw_fixed = d["lr_scaled"] + d["mlp_scaled"]
        raw_late_mlp = 2.0 * d["mlp_scaled"]
        raw_late_abs = u_abs_scaled
        raw_late_blend = 0.75 * raw_fixed + 0.25 * u_abs_scaled
        active = d["block"] >= 4
        late_mlp = fixed.copy(); late_mlp[active] = stage(raw_late_mlp)[active]
        late_abs = fixed.copy(); late_abs[active] = stage(raw_late_abs)[active]
        late_blend = fixed.copy(); late_blend[active] = stage(raw_late_blend)[active]
        scores = {"fixed_w05": fixed, "late_mlp": late_mlp, "late_abs_only": late_abs, "late_abs_blend025": late_blend}
        metrics = {k: evaluate(d["y"], d["block"], v) for k, v in scores.items()}
        # Causal prefix check: suffix perturbation must not alter prefix candidate output.
        prefix_n = max(1, len(fixed) // 2); pert = raw_late_abs.copy(); pert[prefix_n:] += 7.0
        prefix_diff = float(np.max(np.abs(stage(raw_late_abs)[:prefix_n] - stage(pert)[:prefix_n])))
        np.savez_compressed(HERE / f"fold_{i:02d}_{q}.npz", target=np.array(q), train_subjects=np.array(train_subjects),
                            t=d["t"], y=d["y"], block=d["block"], lr_scaled=d["lr_scaled"], mlp_scaled=d["mlp_scaled"],
                            u_abs_scaled=u_abs_scaled, score_fixed_w05=fixed, score_late_mlp=late_mlp,
                            score_late_abs_only=late_abs, score_late_abs_blend025=late_blend,
                            train_mu=mu, train_sd=sd)
        fold = {"fold": i, "target": q, "train_subjects": train_subjects,
                "late_train_frames": int(len(y)), "metrics": metrics,
                "train_mu": mu.tolist(), "train_sd": sd.tolist(),
                "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                "u_abs_signed_corr": float(np.corrcoef(u_abs_scaled, d["lr_scaled"] + d["mlp_scaled"])[0, 1]),
                "causal_prefix_max_abs_diff": prefix_diff, "elapsed_s": time.time() - started}
        (HERE / f"fold_{i:02d}_{q}.json").write_text(json.dumps(fold, ensure_ascii=False, indent=2), encoding="utf-8")
        folds.append(fold)
        print(f"[{i}/57] {q} late_abs={metrics['late_abs_only']['late']['auc']:.6f} fixed={metrics['fixed_w05']['late']['auc']:.6f}", flush=True)
    methods = ("fixed_w05", "late_mlp", "late_abs_only", "late_abs_blend025")
    aggregate = {k: {p: {m: float(np.mean([r["metrics"][k][p][m] for r in folds if r["metrics"][k][p][m] is not None])) if any(r["metrics"][k][p][m] is not None for r in folds) else None for m in METRICS} for p in PARTS} for k in methods}
    relative = {}
    for k in methods[1:]:
        relative[k] = {p: {m: {"improved": sum(a is not None and b is not None and a > b for a, b in ((r["metrics"][k][p][m], r["metrics"]["fixed_w05"][p][m]) for r in folds)),
                              "regressed": sum(a is not None and b is not None and a < b for a, b in ((r["metrics"][k][p][m], r["metrics"]["fixed_w05"][p][m]) for r in folds)),
                              "equal": sum(a is not None and b is not None and a == b for a, b in ((r["metrics"][k][p][m], r["metrics"]["fixed_w05"][p][m]) for r in folds))} for m in METRICS} for p in PARTS}
    protocol = {"experiment": "E79_late_orientation_invariant", "status": "complete", "n_folds": 57,
                "source": "E76 fold NPZ cache only; no src/raw data/old experiments modified",
                "outer_split": "strict 57-fold LOSO; target excluded from all training frames",
                "training_frames": "train subjects only, block>=4",
                "features": "[abs(lr_scaled), abs(mlp_scaled), abs(lr_scaled-mlp_scaled)]",
                "standardization": "training-fold mean/std only; zero std replaced by 1",
                "logistic": "C=0.1, class_weight=balanced, solver=liblinear, max_iter=300, random_state=0",
                "baseline": "exact E76 cached score_w0.50; never restaged",
                "comparators": ["fixed_w05", "late_mlp", "late_abs_only", "late_abs_blend025"],
                "activation": "block<=2 and block==3 fixed baseline; block>=4 expert candidate",
                "postprocessing": "causal EMA1200 + forward HMM300, no target labels/future data",
                "parts": {"full": "all", "early": "block<=2", "late": "block>=4"},
                "causal_check": "suffix perturbation prefix invariance recorded per fold"}
    result = {"protocol": protocol, "aggregate": aggregate, "relative_to_fixed_w05": relative, "folds": folds}
    (HERE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "e79_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E79 后段方向不变专家", "", "严格57折LOSO；复用E76逐折缓存。训练仅使用其余56人且仅取block≥4帧，特征为三个绝对方向不变特征，标准化参数仅来自训练折。基线严格使用E76缓存中的 `score_w0.50`，未对基线重复后处理。候选仅在block≥4启用，统一使用因果EMA1200+前向HMM300。", "", "## 宏观指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for k in methods:
        for p in PARTS:
            a = aggregate[k][p]; lines.append(f"|{k}|{p}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对fixed_w05逐人比较（改善/退步/持平）", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---|---|---|"]
    for k in methods[1:]:
        for p in PARTS:
            c = relative[k][p]; lines.append(f"|{k}|{p}|{c['auc']['improved']}/{c['auc']['regressed']}/{c['auc']['equal']}|{c['f1']['improved']}/{c['f1']['regressed']}/{c['f1']['equal']}|{c['bacc']['improved']}/{c['bacc']['regressed']}/{c['bacc']['equal']}|")
    lines += ["", "## 重点被试与因果核对", "", "每折JSON记录训练被试、训练帧数、训练均值/标准差、Logistic参数、与signed组件和的相关性及后缀扰动前缀最大差异；逐折NPZ保存预测与标准化参数。", "", "## 结论", "", "按预注册方案报告结果；是否保留方向不变专家以late与full宏观指标、逐人改善/退步和因果检查共同判断。"]
    (HERE / "报告_E79_后段方向不变专家.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
