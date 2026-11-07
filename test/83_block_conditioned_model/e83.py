"""E83: shared model with causal-available block condition.

All training frames are used in one shared classifier; block one-hot is appended
inside each fold. The target labels and target statistics are never used.
"""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L
from multiscale import fit_mlp

def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
E42 = imp("e42_e83", TESTS / "42_e17_loso_hmm" / "e42.py")
E61 = imp("e61_e83", TESTS / "61_trainfold_initialization" / "e59_trainfold.py")
TARGETS = ("001", "002", "003", "004", "005", "006")
PARTS = ("full", "early", "late")
METHODS = ("baseline", "block_conditioned")
BLOCKS = 8

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
                  "block_order_equal_L_load": bool(np.array_equal(b, D[q][2])),
                  "block_values": sorted(set(map(int, b)))}
        assert out[q]["sorted_time_strictly_increasing"] and out[q]["label_order_equal_L_load"] and out[q]["block_order_equal_L_load"]
    return out

def condition(x, block):
    b = torch.as_tensor(block, device=x.device, dtype=torch.long)
    onehot = torch.nn.functional.one_hot(b.clamp(0, BLOCKS - 1), num_classes=BLOCKS).to(dtype=x.dtype)
    return torch.cat([x, onehot], dim=1)

def fit_model(F, D, train, conditioned):
    xs = [condition(F[p], D[p][2]) if conditioned else F[p] for p in train]
    X = torch.cat(xs); y = np.concatenate([D[p][1] for p in train])
    flr = L.fit_lr(X, y, 0.001); fms = [fit_mlp(X, y, seed=s) for s in range(3)]
    lr_parts = [flr(x) for x in xs]; mlp_parts = [torch.stack([f(x) for f in fms]).mean(0) for x in xs]
    std_lr, std_mlp = float(torch.cat(lr_parts).std()), float(torch.cat(mlp_parts).std())
    assert std_lr > 0 and std_mlp > 0
    return flr, fms, std_lr, std_mlp, int(len(y))

def predict(model, x, block, conditioned):
    flr, fms, std_lr, std_mlp, _ = model
    if conditioned: x = condition(x, block)
    lr = (flr(x) / std_lr).detach().cpu().numpy()
    mlp = (torch.stack([f(x) for f in fms]).mean(0) / std_mlp).detach().cpu().numpy()
    return lr, mlp

def main():
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; refusing CPU execution")
    all_subs, D = L.load(); assert len(all_subs) == 57
    data_checks = checks(TARGETS, D); F = E42.build_features(all_subs, D); folds = []
    for i, q in enumerate(TARGETS, 1):
        t0 = time.time(); tr = [p for p in all_subs if p != q]
        base_model = fit_model(F, D, tr, False); cond_model = fit_model(F, D, tr, True)
        lr_b, mlp_b = predict(base_model, F[q], D[q][2], False)
        lr_c, mlp_c = predict(cond_model, F[q], D[q][2], True)
        raw_b, raw_c = lr_b + mlp_b, lr_c + mlp_c
        score_b, score_c = stage(raw_b), stage(raw_c)
        metrics = {"baseline": evaluate(D[q][1], D[q][2], score_b), "block_conditioned": evaluate(D[q][1], D[q][2], score_c)}
        with np.load(data_checks[q]["source"]) as source: t_sorted = source["t"][np.argsort(source["t"])]
        row = {"fold": i, "target": q, "train_subjects": tr, "train_frames": cond_model[4],
               "baseline": {"std_lr": base_model[2], "std_mlp": base_model[3]},
               "block_conditioned": {"std_lr": cond_model[2], "std_mlp": cond_model[3]},
               "metrics": metrics, "elapsed_s": round(time.time()-t0, 2)}
        folds.append(row)
        np.savez_compressed(HERE / f"fold_{i:02d}_{q}.npz", target=np.array(q), train_subjects=np.array(tr), t=t_sorted,
                            y=D[q][1], block=D[q][2], baseline_raw=raw_b, conditioned_raw=raw_c,
                            baseline_score=score_b, conditioned_score=score_c, baseline_lr_scaled=lr_b,
                            baseline_mlp_scaled=mlp_b, conditioned_lr_scaled=lr_c, conditioned_mlp_scaled=mlp_c)
        (HERE / f"fold_{i:02d}_{q}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/6] {q} baseline={metrics['baseline']['full']} conditioned={metrics['block_conditioned']['full']}", flush=True)
        del base_model, cond_model; torch.cuda.empty_cache()
    aggregate = {m: {part: {k: float(np.mean([r["metrics"][m][part][k] for r in folds if r["metrics"][m][part][k] is not None])) if any(r["metrics"][m][part][k] is not None for r in folds) else None for k in ("auc", "f1", "bacc")} for part in PARTS} for m in METHODS}
    comparisons = {part: {k: {"improved": 0, "regressed": 0, "equal": 0, "undefined": 0} for k in ("auc", "f1", "bacc")} for part in PARTS}
    for r in folds:
        for part in PARTS:
            for k in comparisons[part]:
                a, b = r["metrics"]["block_conditioned"][part][k], r["metrics"]["baseline"][part][k]
                z = "undefined" if a is None or b is None else ("improved" if a > b else "regressed" if a < b else "equal")
                comparisons[part][k][z] += 1
    protocol = {"experiment": "E83_block_conditioned_model", "status": "complete", "diagnostic": True,
        "subjects_evaluated": list(TARGETS), "n_folds": 6, "all_dataset_subjects": 57,
        "features": "E42 fixed multiscale features, scales=(60,120,300,600,99999)",
        "baseline": "E74 LR C=0.001 + mean MLP seeds [0,1,2]; train-fold output std; fixed wLR=.50",
        "conditioned_model": "same shared LR/MLP trained on all training frames; append fixed 8-class block one-hot to E42 features; target block only supplies current-frame condition",
        "postprocessing": "E61 ema_center(raw,1200,None) then forward hmm(300)", "threshold": 0,
        "parts": {"full": "all frames", "early": "block<=2", "late": "block>=4"},
        "expansion_rule": "expand to 57 only if six-fold diagnostic shows clear common full and late potential; otherwise stop",
        "device": "cuda", "data_checks": data_checks}
    (HERE / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    result = {"protocol": protocol, "aggregate": aggregate, "block_conditioned_vs_baseline": comparisons, "folds": folds}
    (HERE / "e83_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E83 block-conditioned shared model（六人严格 LOSO CUDA 诊断）", "", "在共享模型中将当前帧 block 作为固定 8 类 one-hot 条件追加到 E42 特征；LR/MLP 均使用全部训练帧，不训练独立 late specialist。目标折只提供当前帧 block 条件，不使用目标标签或目标统计。两组统一 EMA1200+前向HMM300。", "", "## 宏观指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for m in METHODS:
        for part in PARTS:
            a = aggregate[m][part]; lines.append(f"|{m}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 相对baseline逐人比较", "", "|阶段|指标|改善/退步/持平|", "|---|---|---|"]
    for part in PARTS:
        for k, c in comparisons[part].items(): lines.append(f"|{part}|{k}|{c['improved']}/{c['regressed']}/{c['equal']}|")
    full_auc = aggregate["block_conditioned"]["full"]["auc"] - aggregate["baseline"]["full"]["auc"]
    late_auc = aggregate["block_conditioned"]["late"]["auc"] - aggregate["baseline"]["late"]["auc"]
    potential = comparisons["full"]["auc"]["improved"] > comparisons["full"]["auc"]["regressed"] and comparisons["late"]["auc"]["improved"] > comparisons["late"]["auc"]["regressed"]
    lines += ["", "## 结论", "", f"full AUC差值={full_auc:+.6f}，late AUC差值={late_auc:+.6f}。full/late均需共同明确潜力才扩展57人；本诊断判定：{'有潜力，允许扩展' if potential else '无共同明确潜力，不扩展57人'}。", "", "逐折训练样本数、尺度、连续raw/score与数据校验保存在本目录。"]
    (HERE / "报告_E83_block_conditioned_model.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "comparisons": comparisons}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
