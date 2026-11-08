"""E87：57人全量训练 + 逐被试推理（非LOSO，训练集含被评被试）。

用途：回答“57人全部训练后，每个个体表现如何”。这是训练集内评估，
指标偏乐观，不能作为泛化成绩；与 nested_loso 的严格留出结果对照使用。

流程：
  1. 用全部57人训练 LR(C=.001) 与 MLP(seed 0/1/2 均值)；
  2. 融合尺度 std_lr/std_mlp 由全部57人训练输出估计；
  3. 门控 Logistic(C=.1, balanced) 用全部57人的组件分数训练；
  4. 对每个被试推理，EMA1200 + 前向HMM300，阈值0；
  5. 输出逐被试 full/early/late AUC/F1/bACC。

运行：.venv\\Scripts\\python.exe test/87_full_train_inference/e87.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L
from multiscale import fit_mlp


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E42 = imp("e42_e87", TESTS / "42_e17_loso_hmm" / "e42.py")
E61 = imp("e61_e87", TESTS / "61_trainfold_initialization" / "e59_trainfold.py")

PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")
METHODS = ("w0.50", "conditioned", "conditioned_shrink", "mlp")
CONFIG = {"lr_C": .001, "mlp_seeds": [0, 1, 2], "gate_C": .1,
          "gate_class_weight": "balanced", "gate_random_state": 0,
          "rolling_window": 120, "postprocessing": "EMA1200+forward_HMM300", "threshold": 0}


def causal_stats(x, w=120):
    x = np.asarray(x, float)
    cs = np.cumsum(np.r_[0., x]); cs2 = np.cumsum(np.r_[0., x * x])
    n = np.arange(1, len(x) + 1); lo = np.maximum(0, n - w); cnt = n - lo
    mean = (cs[n] - cs[lo]) / cnt
    var = np.maximum((cs2[n] - cs2[lo]) / cnt - mean * mean, 0.)
    return mean, np.sqrt(var)


def gate_features(lr, mlp):
    d = np.abs(lr - mlp)
    lm, ls = causal_stats(lr); mm, ms = causal_stats(mlp); dm, ds = causal_stats(d)
    return np.column_stack([lr, mlp, d, lm, mm, dm, ls, ms, ds]).astype(np.float32)


def metric(y, s):
    y = np.asarray(y, int); s = np.asarray(s, float); p = s > 0
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {"n": len(y), "n_negative": n0, "n_positive": n1,
            "auc": float(roc_auc_score(y, s)) if n0 and n1 else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def evaluate(y, b, s):
    masks = {"full": np.ones(len(y), bool), "early": b <= 2, "late": b >= 4}
    return {k: metric(y[v], s[v]) for k, v in masks.items()}


def stage(raw):
    return E61.hmm(E61.ema_center(np.asarray(raw, float), 1200, None), 300)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    t0 = time.time()
    subs, D = L.load(); assert len(subs) == 57
    F = E42.build_features(subs, D)

    X = torch.cat([F[q] for q in subs]); y = np.concatenate([D[q][1] for q in subs])
    flr = L.fit_lr(X, y, CONFIG["lr_C"])
    fms = [fit_mlp(X, y, seed=s) for s in CONFIG["mlp_seeds"]]
    lr_train = torch.cat([flr(F[q]) for q in subs])
    mlp_train = torch.cat([torch.stack([f(F[q]) for f in fms]).mean(0) for q in subs])
    sd_lr, sd_mlp = float(lr_train.std()), float(mlp_train.std())
    assert sd_lr > 0 and sd_mlp > 0
    del X, lr_train, mlp_train; torch.cuda.empty_cache()

    comp = {}
    for q in subs:
        lr = (flr(F[q]) / sd_lr).detach().cpu().numpy()
        mlp = (torch.stack([f(F[q]) for f in fms]).mean(0) / sd_mlp).detach().cpu().numpy()
        comp[q] = (lr, mlp)

    Xg = np.concatenate([gate_features(*comp[q]) for q in subs])
    yg = np.concatenate([D[q][1] for q in subs])
    gate = LogisticRegression(C=CONFIG["gate_C"], class_weight=CONFIG["gate_class_weight"],
                              max_iter=300, solver="liblinear", random_state=0).fit(Xg, yg)
    del Xg, yg

    rows = []
    for i, q in enumerate(subs, 1):
        lr, mlp = comp[q]
        gate_score = gate.decision_function(gate_features(lr, mlp))
        raws = {"w0.50": lr + mlp, "conditioned": gate_score,
                "conditioned_shrink": .5 * gate_score + .5 * (lr + mlp), "mlp": 2 * mlp}
        scores = {k: stage(v) for k, v in raws.items()}
        metrics = {k: evaluate(D[q][1], D[q][2], v) for k, v in scores.items()}
        rows.append({"subject": q, "metrics": metrics})
        np.savez_compressed(HERE / f"sub_{q}.npz", subject=np.array(q), y=D[q][1], block=D[q][2],
                            lr_scaled=lr, mlp_scaled=mlp, gate_score=gate_score,
                            **{f"score_{k}": v for k, v in scores.items()})
        print(f"[{i}/57] {q} shrink={metrics['conditioned_shrink']['full']}", flush=True)

    aggregate = {m: {p: {k: float(np.mean([r["metrics"][m][p][k] for r in rows if r["metrics"][m][p][k] is not None]))
                        for k in METRICS} for p in PARTS} for m in METHODS}
    result = {"protocol": "E87_full57_train_then_per_subject_inference", "status": "complete",
              "note": "训练集内评估：57人全部参与训练，指标偏乐观，不是泛化成绩",
              "n_subjects": 57, "config": CONFIG, "aggregate": aggregate, "subjects": rows,
              "elapsed_s": round(time.time() - t0, 2)}
    (HERE / "e87_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# E87 57人全量训练后逐被试推理（非LOSO）", "",
             "57人全部参与训练，再对每个被试推理。**这是训练集内评估，指标偏乐观，不能作为泛化成绩**；"
             "泛化结论以 nested_loso 的严格留出结果为准。", "",
             "## 聚合指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for m in METHODS:
        for p in PARTS:
            a = aggregate[m][p]
            lines.append(f"|{m}|{p}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 逐被试（conditioned_shrink）", "",
              "|被试|AUC|F1|bACC|", "|---|---:|---:|---:|"]
    for r in rows:
        a = r["metrics"]["conditioned_shrink"]["full"]
        lines.append(f"|{r['subject']}|{a['auc']:.4f}|{a['f1']:.4f}|{a['bacc']:.4f}|")
    lines += ["", "## 达标人数（conditioned_shrink）", "", "|阶段|bACC≥0.9|bACC≥0.8|", "|---|---:|---:|"]
    for p in PARTS:
        vals = [r["metrics"]["conditioned_shrink"][p]["bacc"] for r in rows
                if r["metrics"]["conditioned_shrink"][p]["bacc"] is not None]
        lines.append(f"|{p}|{sum(v >= .9 for v in vals)}|{sum(v >= .8 for v in vals)}|")
    (HERE / "报告_E87_全量训练逐被试.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
