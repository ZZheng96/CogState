"""E74: fixed three-weight LR/MLP fusion under the E53/E67 LOSO protocol."""
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
E42 = imp("e42_e74", TESTS / "42_e17_loso_hmm" / "e42.py")
E61 = imp("e61_e74", TESTS / "61_trainfold_initialization" / "e59_trainfold.py")
SCORES = TESTS / "53_e17_post_loso" / "e53_scores.npz"
WEIGHTS = (0.5, 0.25, 0.0)
PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")


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


def protocol(subs, data_checks):
    return {"experiment": "E74_full_fusion", "status": "running", "subjects": subs, "n_folds": 57,
            "features": "E42 fixed multiscale features, scales=(60,120,300,600,99999)",
            "training": "LOSO: each fold trains on 56 subjects; LR C=0.001; fit_mlp defaults (30 epoch), seeds [0,1,2] mean",
            "scaling": "std_lr/std_mlp are pooled outputs over training subjects only",
            "components": "save LR_scaled, MLP_scaled and raw for each fixed wLR",
            "fusion": "raw=2*(wLR*LR_scaled+(1-wLR)*MLP_scaled)", "weights_lr": list(WEIGHTS),
            "weight_selection": "fixed before target-label scoring; no target-label-based selection",
            "postprocessing": "E61 ema_center(raw,1200,None) then forward hmm(300), continuous sequence",
            "threshold": 0, "positive_label": 1,
            "parts": {"full": "all frames", "early": "block<=2", "late": "block>=4"},
            "macro": "unweighted subject mean; null AUC/bACC for single-class slices; F1 zero_division=0",
            "device": "cuda", "data_checks": data_checks}


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; refusing CPU execution")
    subs, D = L.load(); assert len(subs) == 57
    data_checks = checks(subs, D)
    proto = protocol(subs, data_checks)
    (HERE / "protocol.json").write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8")
    F = E42.build_features(subs, D)
    fold_rows = []
    for i, q in enumerate(subs, 1):
        t0 = time.time(); tr = [p for p in subs if p != q]
        X = torch.cat([F[p] for p in tr]); ytr = np.concatenate([D[p][1] for p in tr])
        flr = L.fit_lr(X, ytr, 0.001); fms = [fit_mlp(X, ytr, seed=s) for s in range(3)]
        lr_train = torch.cat([flr(F[p]) for p in tr])
        mlp_train = torch.cat([torch.stack([f(F[p]) for f in fms]).mean(0) for p in tr])
        std_lr, std_mlp = float(lr_train.std()), float(mlp_train.std())
        assert std_lr > 0 and std_mlp > 0
        lr = (flr(F[q]) / std_lr).detach().cpu().numpy()
        mlp = (torch.stack([f(F[q]) for f in fms]).mean(0) / std_mlp).detach().cpu().numpy()
        raws = {f"w{w:.2f}": 2 * (w * lr + (1 - w) * mlp) for w in WEIGHTS}
        scores = {name: stage(raw) for name, raw in raws.items()}
        y, b = D[q][1], D[q][2]
        metrics = {name: evaluate(y, b, score) for name, score in scores.items()}
        with np.load(data_checks[q]["source"]) as source:
            t_sorted = source["t"][np.argsort(source["t"])]
        np.savez_compressed(HERE / f"fold_{i:02d}_{q}.npz", target=np.array(q), train_subjects=np.array(tr),
                            t=t_sorted, y=y, block=b,
                            lr_scaled=lr, mlp_scaled=mlp, std_lr=std_lr, std_mlp=std_mlp,
                            **{f"raw_{n}": v for n, v in raws.items()}, **{f"score_{n}": v for n, v in scores.items()})
        row = {"fold": i, "target": q, "train_subjects": tr, "std_lr": std_lr, "std_mlp": std_mlp,
               "metrics": metrics, "elapsed_s": round(time.time() - t0, 2)}
        fold_rows.append(row)
        (HERE / f"fold_{i:02d}_{q}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/57] {q} {metrics['w0.50']['full']} {time.time()-t0:.0f}s", flush=True)
        del X, flr, fms, lr_train, mlp_train
        torch.cuda.empty_cache()
    aggregate = {}
    for name in [f"w{w:.2f}" for w in WEIGHTS]:
        aggregate[name] = {part: {m: (float(np.mean([r["metrics"][name][part][m] for r in fold_rows if r["metrics"][name][part][m] is not None])) if any(r["metrics"][name][part][m] is not None for r in fold_rows) else None) for m in METRICS} for part in PARTS}
    baseline = "w0.50"
    comparisons = {}
    for name in aggregate:
        comparisons[name] = {}
        for part in PARTS:
            comparisons[name][part] = {m: {"improved": 0, "regressed": 0, "equal": 0, "undefined": 0} for m in METRICS}
            for r in fold_rows:
                for m in METRICS:
                    a, z = r["metrics"][name][part][m], r["metrics"][baseline][part][m]
                    k = "undefined" if a is None or z is None else ("improved" if a > z else "regressed" if a < z else "equal")
                    comparisons[name][part][m][k] += 1
    proto["status"] = "complete"
    (HERE / "protocol.json").write_text(json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"protocol": proto, "aggregate": aggregate, "relative_to_w0.50": comparisons, "folds": fold_rows}
    (HERE / "e74_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E74 全融合固定权重 LOSO", "", "严格复现E53/E67：E42特征、57折LOSO（每折训练56人）、LR C=.001、MLP默认30 epoch且seed 0/1/2取均值；std仅由训练折估计。固定 `wLR=0.50/0.25/0`，raw=`2*(wLR*LR_scaled+(1-wLR)*MLP_scaled)`，统一EMA1200+前向HMM300。未使用目标标签选择权重。", "", "## 宏观指标", "", "|wLR|阶段|AUC|F1|bACC|", "|---:|---|---:|---:|---:|"]
    for name in aggregate:
        for part in PARTS:
            a = aggregate[name][part]; lines.append(f"|{name[1:]}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对wLR=0.50逐人比较", "", "|wLR|阶段|AUC 改善/退步/持平|F1 改善/退步/持平|bACC 改善/退步/持平|", "|---:|---|---|---|---|"]
    for name in aggregate:
        if name == baseline: continue
        for part in PARTS:
            c = comparisons[name][part]; lines.append(f"|{name[1:]}|{part}|{c['auc']['improved']}/{c['auc']['regressed']}/{c['auc']['equal']}|{c['f1']['improved']}/{c['f1']['regressed']}/{c['f1']['equal']}|{c['bacc']['improved']}/{c['bacc']['regressed']}/{c['bacc']['equal']}|")
    lines += ["", "逐折组件与score保存在本目录 `fold_*.npz`，逐人JSON保留训练折、尺度和指标；数据源SHA-256及排序/标签/块顺序核验见 `protocol.json`。", ""]
    (HERE / "报告_E74_全融合固定权重.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False))

if __name__ == "__main__": main()
