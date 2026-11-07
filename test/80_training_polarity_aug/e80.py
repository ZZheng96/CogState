"""E80: strict LOSO training polarity augmentation diagnostic.

Only the training fold is augmented: each training sequence is duplicated with
its complete E42 feature vector multiplied by -1 and keeps the original label.
The target fold is never augmented or used for parameter selection.  Baseline
and augmented LR/MLP fusion use identical train-fold output scaling and E74
causal post-processing.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = TESTS.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L
from multiscale import fit_mlp


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
E42 = imp("e42_e80", TESTS / "42_e17_loso_hmm" / "e42.py")
E61 = imp("e61_e80", TESTS / "61_trainfold_initialization" / "e59_trainfold.py")
TARGETS = ("001", "002", "003", "004", "005", "006")
PARTS = ("full", "early", "late")
METHODS = ("baseline", "polarity_aug")


def metric(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float); p = s > 0
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {"n": len(y), "n_negative": n0, "n_positive": n1,
            "auc": float(roc_auc_score(y, s)) if n0 and n1 else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def evaluate(y, b, s):
    masks = {"full": np.ones(len(y), bool), "early": b <= 2, "late": b >= 4}
    return {part: metric(y[m], s[m]) for part, m in masks.items()}


def stage(raw):
    return E61.hmm(E61.ema_center(raw, 1200, None), 300)


def checks(subs, D):
    out = {}
    for q in subs:
        p = L.SLID / f"sub-{q}.npz"
        with np.load(p) as z:
            order = np.argsort(z["t"]); t, y, b = z["t"][order], z["y"][order].astype(int), z["blk"][order].astype(int)
        out[q] = {"source": str(p), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                  "n": int(len(t)), "sorted_time_strictly_increasing": bool(np.all(np.diff(t) > 0)),
                  "label_order_equal_L_load": bool(np.array_equal(y, D[q][1])),
                  "block_order_equal_L_load": bool(np.array_equal(b, D[q][2]))}
        assert out[q]["sorted_time_strictly_increasing"] and out[q]["label_order_equal_L_load"] and out[q]["block_order_equal_L_load"]
    return out


def fit_components(F, D, train, augmented):
    X = torch.cat([F[p] for p in train]); y = np.concatenate([D[p][1] for p in train])
    # The augmented arm is a dual-polarity expert: a second model sees -X
    # with unchanged labels. At inference its input is also polarity-reversed.
    signs = (1, -1) if augmented else (1,)
    experts = []
    for sign in signs:
        Xs = sign * X
        flr = L.fit_lr(Xs, y, 0.001)
        fms = [fit_mlp(Xs, y, seed=s) for s in range(3)]
        experts.append((sign, flr, fms))
    lr_train = torch.cat([sum((flr(sign * F[p]) for sign, flr, _ in experts)) / len(experts) for p in train])
    mlp_train = torch.cat([sum((torch.stack([f(sign * F[p]) for f in fms]).mean(0) for sign, _, fms in experts)) / len(experts) for p in train])
    std_lr, std_mlp = float(lr_train.std()), float(mlp_train.std())
    assert std_lr > 0 and std_mlp > 0
    return experts, std_lr, std_mlp


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; refusing CPU execution")
    all_subs, D = L.load(); assert len(all_subs) == 57
    data_checks = checks(TARGETS, D)
    F = E42.build_features(all_subs, D)
    folds = []
    for i, q in enumerate(TARGETS, 1):
        t0 = time.time(); tr = [p for p in all_subs if p != q]
        row = {"fold": i, "target": q, "train_subjects": tr, "metrics": {}}
        with np.load(data_checks[q]["source"]) as source:
            t_sorted = source["t"][np.argsort(source["t"])]
        for method in METHODS:
            experts, std_lr, std_mlp = fit_components(F, D, tr, method == "polarity_aug")
            lr_t = sum((flr(sign * F[q]) for sign, flr, _ in experts)) / len(experts)
            mlp_t = sum((torch.stack([f(sign * F[q]) for f in fms]).mean(0) for sign, _, fms in experts)) / len(experts)
            lr = (lr_t / std_lr).detach().cpu().numpy()
            mlp = (mlp_t / std_mlp).detach().cpu().numpy()
            raw = 2 * .5 * (lr + mlp)
            score = stage(raw)
            row["metrics"][method] = evaluate(D[q][1], D[q][2], score)
            if method == "baseline":
                row["baseline_raw_std"] = {"lr": std_lr, "mlp": std_mlp}
            else:
                row["augmented_raw_std"] = {"lr": std_lr, "mlp": std_mlp}
            np.savez_compressed(HERE / f"fold_{i:02d}_{q}_{method}.npz", target=np.array(q), train_subjects=np.array(tr), t=t_sorted,
                                y=D[q][1], block=D[q][2], method=np.array(method), lr_scaled=lr, mlp_scaled=mlp,
                                score=score, std_lr=std_lr, std_mlp=std_mlp)
            del experts, lr_t, mlp_t, lr, mlp, score
            torch.cuda.empty_cache()
        folds.append(row); row["elapsed_s"] = round(time.time() - t0, 2)
        (HERE / f"fold_{i:02d}_{q}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/6] {q} baseline={row['metrics']['baseline']['full']} aug={row['metrics']['polarity_aug']['full']}", flush=True)
    aggregate = {m: {part: {k: float(np.mean([r["metrics"][m][part][k] for r in folds if r["metrics"][m][part][k] is not None]))
                              if any(r["metrics"][m][part][k] is not None for r in folds) else None for k in ("auc", "f1", "bacc")}
                       for part in PARTS} for m in METHODS}
    comparisons = {part: {k: {"improved": 0, "regressed": 0, "equal": 0, "undefined": 0} for k in ("auc", "f1", "bacc")} for part in PARTS}
    for r in folds:
        for part in PARTS:
            for k in comparisons[part]:
                a, b = r["metrics"]["polarity_aug"][part][k], r["metrics"]["baseline"][part][k]
                z = "undefined" if a is None or b is None else ("improved" if a > b else "regressed" if a < b else "equal")
                comparisons[part][k][z] += 1
    protocol = {"experiment": "E80_training_polarity_aug", "status": "complete", "diagnostic": True,
        "subjects_evaluated": list(TARGETS), "n_folds": 6, "all_dataset_subjects": 57,
        "features": "E42 fixed multiscale features, scales=(60,120,300,600,99999)",
        "baseline": "E74 LR C=0.001 + mean of MLP seeds [0,1,2]; train-fold output std; fixed wLR=.50",
        "augmentation": "training fold only: dual polarity experts fit separately on X and -X with unchanged labels; target inference averages both experts; no target labels/statistics used",
        "postprocessing": "E61 ema_center(raw,1200,None) then forward hmm(300)", "threshold": 0,
        "parts": {"full": "all frames", "early": "block<=2", "late": "block>=4"},
        "expansion_rule": "expand to 57 only if six-fold diagnostic shows clear potential; no score post-processing gate",
        "device": "cuda", "data_checks": data_checks}
    (HERE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"protocol": protocol, "aggregate": aggregate, "polarity_aug_vs_baseline": comparisons, "folds": folds}
    (HERE / "e80_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E80 训练阶段 signed polarity augmentation（六人严格 LOSO 诊断）", "", "训练折内将每个 E42 特征序列复制为 `X` 与 `-X`，标签保持不变；目标折不增强、不参与任何参数估计或选择。baseline 与增强组均为 LR C=.001 + 三个 MLP seed 均值，尺度仅由训练折输出估计，统一 E74 的 EMA1200+前向HMM300。未做 score 后处理门控。", "", "## 宏观指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for m in METHODS:
        for part in PARTS:
            a = aggregate[m][part]; lines.append(f"|{m}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 增强相对 baseline 逐人比较", "", "|阶段|指标|改善/退步/持平|", "|---|---|---|"]
    for part in PARTS:
        for k, c in comparisons[part].items(): lines.append(f"|{part}|{k}|{c['improved']}/{c['regressed']}/{c['equal']}|")
    lines += ["", "## 结论", "", "六人诊断结果见上表及 `e80_results.json`；是否扩展57人依据增强对 baseline 的 full/late AUC、F1、bACC 是否形成一致且明确的改善。", "", "逐折训练折、数据校验、尺度与连续分数保存在本目录。"]
    (HERE / "报告_E80_训练阶段极性增强.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
