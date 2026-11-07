"""E75: 方向不变特征筛选，仅固定六个诊断目标，严格目标无标签变换。"""
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
E42 = imp("e42_e75", TESTS / "42_e17_loso_hmm" / "e42.py")
E61 = imp("e61_e75", TESTS / "61_trainfold_initialization" / "e59_trainfold.py")
E74 = TESTS / "74_full_fusion" / "e74_results.json"
TARGETS = ("004", "009", "010", "018", "034", "050")
PARTS = ("full", "early", "late")
METHODS = ("A_e74_raw", "B_subject_z", "B_subject_robust", "C_center_abs")


def metric(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float); ok = np.isfinite(s); y, s = y[ok], s[ok]
    p = s > 0; n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    return {"n": int(len(y)), "n_negative": n0, "n_positive": n1,
            "auc": float(roc_auc_score(y, s)) if n0 and n1 else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[y == 1].mean() + (~p[y == 0]).mean())) if n0 and n1 else None}


def evaluate(y, b, s):
    masks = {"full": np.ones(len(y), bool), "early": b <= 2, "late": b >= 4}
    return {part: metric(y[m], s[m]) for part, m in masks.items()}


def stage(raw):
    return E61.hmm(E61.ema_center(raw, 1200, None), 300)


def transform(X, method):
    """每个被试独立只从X估计中心/尺度；不读取标签。"""
    if method == "A_e74_raw": return X
    center = X.mean(0)
    if method == "C_center_abs": return (X - center).abs().clamp(-10, 10)
    if method == "B_subject_z":
        scale = X.std(0).clamp_min(1e-6)
    elif method == "B_subject_robust":
        scale = (torch.quantile(X, .75, dim=0) - torch.quantile(X, .25, dim=0)).clamp_min(1e-6)
    else: raise ValueError(method)
    return ((X - center) / scale).clamp(-10, 10)


def metric_delta(a, b):
    return {part: {m: (a[part][m] - b[part][m]) if a[part][m] is not None and b[part][m] is not None else None
                   for m in ("auc", "f1", "bacc")} for part in PARTS}


def first_failure(y, b, s):
    """预先固定诊断：按时间顺序，首个完整block内AUC<.5；无则null。"""
    rows = []
    for block in sorted(set(map(int, b))):
        m = b == block; z = metric(y[m], s[m])
        if z["auc"] is not None: rows.append({"block": block, "auc": z["auc"], "f1": z["f1"], "bacc": z["bacc"]})
    bad = next((r for r in rows if r["auc"] < .5), None)
    return {"first_block_auc_below_0.5": bad, "by_block": rows}


def save_json(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")


def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; refusing CPU execution")
    subs, D = L.load(); assert len(subs) == 57 and set(TARGETS) <= set(subs)
    checks = {}
    for q in TARGETS:
        p = L.SLID / f"sub-{q}.npz"
        with np.load(p) as z:
            o = np.argsort(z["t"]); y, b = z["y"][o].astype(int), z["blk"][o].astype(int)
        checks[q] = {"source": str(p), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n": int(len(y)),
                     "label_order_equal_L_load": bool(np.array_equal(y, D[q][1])), "block_order_equal_L_load": bool(np.array_equal(b, D[q][2]))}
        assert checks[q]["label_order_equal_L_load"] and checks[q]["block_order_equal_L_load"]
    baseline = json.loads(E74.read_text(encoding="utf-8"))
    protocol = {"experiment": "E75_direction_invariant", "status": "running", "targets": TARGETS, "n_folds": 6,
                "features": "E42 fixed multiscale features, unchanged; no feature selected by target labels",
                "training": "each target is LOSO: 56 training subjects; LR C=.001; MLP default 30 epochs; seeds 0/1/2 mean",
                "A_e74_raw": "exact E74 wLR=.5 reference implementation",
                "B_subject_z": "each subject X-only mean/std, independently for train and target, then LR+MLP fusion wLR=.5",
                "B_subject_robust": "each subject X-only median/IQR, independently for train and target, then LR+MLP fusion wLR=.5",
                "C_center_abs": "each subject X-only mean-center then abs, then LR+MLP fusion wLR=.5",
                "fusion": "raw=2*(.5*LR_scaled+.5*MLP_scaled); output std pooled over transformed training subjects only",
                "postprocessing": "E61 EMA1200 then forward HMM300; threshold 0; positive label 1",
                "parts": {"full":"all frames", "early":"block<=2", "late":"block>=4"},
                "first_failure": "first chronological complete block with AUC<0.5, fixed before target scoring; not used for selection",
                "target_label_usage": "target labels only used after scoring for reporting; no candidate selection",
                "device": "cuda", "data_checks": checks}
    save_json(HERE / "protocol.json", protocol)
    F = E42.build_features(subs, D)
    folds = []
    for i, q in enumerate(TARGETS, 1):
        t0 = time.time(); tr = [p for p in subs if p != q]; yq, bq = D[q][1], D[q][2]
        models_out = {}
        for method in METHODS:
            Ft = {p: transform(F[p], method) for p in tr + [q]}
            X = torch.cat([Ft[p] for p in tr]); ytr = np.concatenate([D[p][1] for p in tr])
            flr = L.fit_lr(X, ytr, .001); fms = [fit_mlp(X, ytr, seed=s) for s in range(3)]
            with torch.no_grad():
                lr_tr = torch.cat([flr(Ft[p]) for p in tr]); mlp_tr = torch.cat([torch.stack([f(Ft[p]) for f in fms]).mean(0) for p in tr])
                slr, smlp = float(lr_tr.std()), float(mlp_tr.std())
                lr = (flr(Ft[q]) / slr).cpu().numpy(); mlp = (torch.stack([f(Ft[q]) for f in fms]).mean(0) / smlp).cpu().numpy()
            raw = 2 * (.5 * lr + .5 * mlp); score = stage(raw)
            models_out[method] = {"std_lr": slr, "std_mlp": smlp, "metrics": evaluate(yq, bq, score), "first_failure": first_failure(yq, bq, score), "raw": raw, "score": score}
            del Ft, X, flr, fms, lr_tr, mlp_tr
            torch.cuda.empty_cache()
        base = models_out["A_e74_raw"]["metrics"]
        for method in METHODS: models_out[method]["delta_vs_A_e74"] = metric_delta(models_out[method]["metrics"], base)
        with np.load(L.SLID / f"sub-{q}.npz") as z: t = z["t"][np.argsort(z["t"])]
        arrays = {"target":np.array(q), "y":yq, "block":bq, "t":t}
        for method, v in models_out.items(): arrays[f"score_{method}"] = v["score"]; arrays[f"raw_{method}"] = v["raw"]
        np.savez_compressed(HERE / f"fold_{q}.npz", **arrays)
        row = {"fold": i, "target": q, "train_subjects": tr, "methods": {m:{k:v for k,v in x.items() if k not in ("raw","score")} for m,x in models_out.items()}, "elapsed_s": round(time.time()-t0,2)}
        save_json(HERE / f"fold_{q}.json", row); folds.append(row)
        print(f"[{i}/6] {q} {row['methods']['A_e74_raw']['metrics']['full']}", flush=True)
    aggregate = {}
    for method in METHODS:
        aggregate[method] = {part: {m: float(np.mean([r["methods"][method]["metrics"][part][m] for r in folds if r["methods"][method]["metrics"][part][m] is not None])) for m in ("auc","f1","bacc")} for part in PARTS}
    baseagg = aggregate["A_e74_raw"]
    delta_aggregate = {method: metric_delta(aggregate[method], baseagg) for method in METHODS}
    protocol["status"] = "complete"; save_json(HERE / "protocol.json", protocol)
    save_json(HERE / "e75_results.json", {"protocol":protocol, "aggregate":aggregate, "delta_aggregate_vs_A_e74":delta_aggregate, "folds":folds})
    lines = ["# E75 方向不变特征筛选", "", "固定六个诊断目标 `004/009/010/018/034/050`、LOSO训练折、seed 0/1/2、MLP默认30 epoch、训练折std、EMA1200+前向HMM300。A为E74原始wLR=.5；B/C的变换严格逐被试只用无标签X估计，未使用y，也未按目标标签挑候选。", "", "## 六目标宏平均", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for method in METHODS:
        for part in PARTS:
            a = aggregate[method][part]; lines.append(f"|{method}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对A的宏平均差值", "", "|方法|阶段|ΔAUC|ΔF1|ΔbACC|", "|---|---|---:|---:|---:|"]
    for method in METHODS[1:]:
        for part in PARTS:
            d = delta_aggregate[method][part]; lines.append(f"|{method}|{part}|{d['auc']:+.6f}|{d['f1']:+.6f}|{d['bacc']:+.6f}|")
    lines += ["", "## 逐目标与首次失效", "", "逐目标完整指标、相对A差值和按时间首个AUC<0.5的完整block见 `fold_*.json`。`first_failure`仅为预先固定诊断，不参与候选选择。", "", "## 结论", "", "本实验只报告六目标诊断范围；是否存在方向不变收益须同时看full与late及多数目标的一致改善。未达到该条件前，不宣称方向反转问题已解决。"]
    (HERE / "报告_E75_方向不变特征筛选.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"aggregate":aggregate,"delta":delta_aggregate}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
