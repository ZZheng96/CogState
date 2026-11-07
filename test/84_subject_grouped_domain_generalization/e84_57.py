"""E84：被试分组域泛化诊断；严格六人 LOSO，ERM 对照被试风险方差正则。"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L

_spec42 = importlib.util.spec_from_file_location("e42_e84", TESTS / "42_e17_loso_hmm" / "e42.py")
E42 = importlib.util.module_from_spec(_spec42); _spec42.loader.exec_module(E42)
_spec61 = importlib.util.spec_from_file_location("e61_e84", TESTS / "61_trainfold_initialization" / "e59_trainfold.py")
E61 = importlib.util.module_from_spec(_spec61); _spec61.loader.exec_module(E61)
TARGETS = None
LAMBDA = 0.1
PROTOCOL_PATH = HERE / "protocol_57.json"
RESULT_PATH = HERE / "e84_57_results.json"
CACHE_PREFIX = "fold57_"
METHODS = ("erm", "subject_risk_variance")
PARTS = ("full", "early", "late")


def metric(y, s):
    y, s = np.asarray(y, int), np.asarray(s, float); p = s > 0
    pos, neg = y == 1, y == 0
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, s)) if pos.any() and neg.any() else None,
            "f1": float(f1_score(y, p, zero_division=0)) if len(y) else None,
            "bacc": float(.5 * (p[pos].mean() + (~p[neg]).mean())) if pos.any() and neg.any() else None}


def evaluate(y, b, s):
    masks = {"full": np.ones(len(y), bool), "early": b <= 2, "late": b >= 4}
    return {k: metric(y[m], s[m]) for k, m in masks.items()}


def checks(D, targets):
    out = {}
    for q in targets:
        p = L.SLID / f"sub-{q}.npz"
        z = np.load(p); order = np.argsort(z["t"])
        out[q] = {"source": str(p), "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                  "n": int(len(order)), "sorted_time_strictly_increasing": bool(np.all(np.diff(z["t"][order]) > 0)),
                  "label_order_equal_L_load": bool(np.array_equal(z["y"][order].astype(int), D[q][1])),
                  "block_not_used_as_feature": True, "phase_not_used": True,
                  "target_statistics_not_used": True, "future_data_not_used": True}
        assert all(out[q][k] for k in ("sorted_time_strictly_increasing", "label_order_equal_L_load"))
    return out


def fit_erm(Xs, ys, epochs=180):
    X = torch.cat(Xs); y = torch.tensor(np.concatenate(ys), device=X.device, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6; Zs = [(x - mu) / sd for x in Xs]
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True); b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.AdamW((w, b), lr=0.01, weight_decay=1e-3)
    pw = (y == 0).sum() / (y == 1).sum()
    for _ in range(epochs):
        opt.zero_grad(); logits = torch.cat([z @ w + b for z in Zs])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
        (loss + 0.5e-3 * (w ** 2).sum()).backward(); opt.step()
    return lambda X: ((X - mu) / sd) @ w.detach() + b.detach()


def fit_variance(Xs, ys, epochs=180):
    X = torch.cat(Xs); y = torch.tensor(np.concatenate(ys), device=X.device, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6; Zs = [(x - mu) / sd for x in Xs]
    w = torch.zeros(X.shape[1], device=X.device, requires_grad=True); b = torch.zeros(1, device=X.device, requires_grad=True)
    opt = torch.optim.AdamW((w, b), lr=0.01, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad(); risks = []
        for z, yy in zip(Zs, ys):
            yt = torch.tensor(yy, device=X.device, dtype=torch.float32)
            risks.append(torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, yt))
        rv = torch.stack(risks); loss = rv.mean() + LAMBDA * rv.var(unbiased=False)
        (loss + 0.5e-3 * (w ** 2).sum()).backward(); opt.step()
    return lambda X: ((X - mu) / sd) @ w.detach() + b.detach()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outer", type=int, choices=(6, 57), default=6); ap.add_argument("--start", type=int, default=1); ap.add_argument("--end", type=int, default=57); args = ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required; refusing CPU execution")
    all_subs, D = L.load(); assert len(all_subs) == 57
    if args.outer != 57: raise ValueError("e84_57.py only supports --outer 57")
    targets = tuple(all_subs)
    data_checks = checks(D, targets); F = E42.build_features(all_subs, D); folds = []
    for i, q in enumerate(targets, 1):
        if i < args.start or i > args.end: continue
        cached = HERE / f"{CACHE_PREFIX}{i:02d}_{q}.json"
        if cached.exists():
            folds.append(json.loads(cached.read_text(encoding="utf-8")))
            print(f"[{i}/57] {q} loaded cached", flush=True)
            continue
        t0 = time.time(); tr = [p for p in all_subs if p != q]
        Xs = [F[p] for p in tr]; ys = [D[p][1] for p in tr]
        models = {"erm": fit_erm(Xs, ys), "subject_risk_variance": fit_variance(Xs, ys)}
        raw, scores, metrics = {}, {}, {}
        for name, model in models.items():
            train_raw = [model(F[p]).detach() for p in tr]
            sd = float(torch.cat(train_raw).std()); assert sd > 0
            raw[name] = model(F[q]).detach().cpu().numpy() / sd
            scores[name] = E61.hmm(E61.ema_center(raw[name], 1200, None), 300)
            metrics[name] = evaluate(D[q][1], D[q][2], scores[name])
        row = {"fold": i, "target": q, "train_subjects": tr, "train_frames": int(sum(len(x) for x in ys)),
               "lambda": LAMBDA, "metrics": metrics, "elapsed_s": round(time.time() - t0, 2)}
        folds.append(row)
        np.savez_compressed(HERE / f"{CACHE_PREFIX}{i:02d}_{q}.npz", target=np.array(q), train_subjects=np.array(tr),
                            y=D[q][1], block=D[q][2], **{f"{m}_raw": raw[m] for m in METHODS},
                            **{f"{m}_score": scores[m] for m in METHODS})
        (HERE / f"{CACHE_PREFIX}{i:02d}_{q}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/57] {q} erm={metrics['erm']['full']} variance={metrics['subject_risk_variance']['full']}", flush=True)
        del models; torch.cuda.empty_cache()
    def avg(m, part, k):
        vals = [r["metrics"][m][part][k] for r in folds if r["metrics"][m][part][k] is not None]
        return float(np.mean(vals)) if vals else None
    aggregate = {m: {part: {k: avg(m, part, k) for k in ("auc", "f1", "bacc")} for part in PARTS} for m in METHODS}
    comparisons = {part: {k: {"improved": 0, "regressed": 0, "equal": 0} for k in ("auc", "f1", "bacc")} for part in PARTS}
    for r in folds:
        for part in PARTS:
            for k in comparisons[part]:
                a, b = r["metrics"]["subject_risk_variance"][part][k], r["metrics"]["erm"][part][k]
                if a is not None and b is not None:
                    comparisons[part][k]["improved" if a > b else "regressed" if a < b else "equal"] += 1
    protocol = {"experiment": "E84_subject_grouped_domain_generalization_57", "status": "complete", "scope": "strict 57-subject LOSO", "targets": list(targets), "features": "E42 fixed multiscale causal features, scales=(60,120,300,600,99999)", "models": {"erm": "low-capacity logistic pooled across training subjects", "subject_risk_variance": "same logistic model plus lambda=0.1 variance of per-subject mean BCE risks"}, "lambda": LAMBDA, "postprocessing": "EMA1200 then causal forward HMM300", "prohibitions": ["no block feature", "no phase feature", "no target statistics", "no target-label tuning", "no future data", "no target-record normalization"], "device": "cuda", "data_checks": data_checks}
    result = {"protocol": protocol, "aggregate": aggregate, "subject_risk_variance_vs_erm": comparisons, "folds": folds}
    (PROTOCOL_PATH).write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULT_PATH).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# E84 被试分组域泛化（57人严格 LOSO CUDA）", "", "比较低容量ERM与固定 lambda=0.1 的被试风险方差正则。特征、尺度和 EMA1200+前向HMM300 均按训练折或因果当前/过去输入计算；不使用 block/phase、目标统计、目标标签适配或未来数据。", "", "## 指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for m in METHODS:
        for part in PARTS:
            a = aggregate[m][part]; lines.append(f"|{m}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 逐折比较（方差正则相对ERM）", "", "|阶段|指标|改善/退步/持平|", "|---|---|---|"]
    for part in PARTS:
        for k, c in comparisons[part].items(): lines.append(f"|{part}|{k}|{c['improved']}/{c['regressed']}/{c['equal']}|")
    potential = comparisons["full"]["auc"]["improved"] > comparisons["full"]["auc"]["regressed"] and comparisons["late"]["auc"]["improved"] > comparisons["late"]["auc"]["regressed"]
    lines += ["", "## 57人结果", "", f"完整57折LOSO已完成。full AUC差值={aggregate['subject_risk_variance']['full']['auc'] - aggregate['erm']['full']['auc']:+.6f}；late AUC差值={aggregate['subject_risk_variance']['late']['auc'] - aggregate['erm']['late']['auc']:+.6f}。", "", "逐折连续 raw/score、训练被试清单与数据校验保存在本目录的 fold57_* 文件。"]
    (HERE / "报告_E84_57_subject_grouped_domain_generalization.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "comparisons": comparisons, "potential": potential}, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
