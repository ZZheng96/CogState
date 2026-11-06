"""E59: 训练折中心初始化 EMA/HMM 与训练内固定阈值对照。

运行:
  .venv\\Scripts\\python.exe test/61_trainfold_initialization/e59_trainfold.py --folds 1
  .venv\\Scripts\\python.exe test/61_trainfold_initialization/e59_trainfold.py --folds 57
  --balanced 仅预留接口，暂不实现。
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TESTS = ROOT / "test"
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L
from multiscale import fit_mlp
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FOLD_DIR = HERE / "corrected" / "folds"
RESULT = HERE / "corrected" / "e61_results.json"
THRESHOLDS = np.round(np.arange(-4.0, 4.001, 0.1), 1)


def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
E42 = imp("e42_e59", TESTS / "42_e17_loso_hmm" / "e42.py")


def metric(y, s, threshold=0.0):
    y = np.asarray(y, dtype=int); s = np.asarray(s, dtype=float)
    p = s > threshold; pos = y == 1; neg = ~pos
    if not pos.any() or not neg.any(): return {"bacc": float("nan"), "auc": float("nan"), "f1": float("nan")}
    tp = int((p & pos).sum()); fp = int((p & neg).sum())
    bacc = .5 * (tp / pos.sum() + 1 - fp / neg.sum())
    f1 = f1_score(y, p, zero_division=0)
    auc = roc_auc_score(y, s)
    return {"bacc": float(bacc), "auc": float(auc), "f1": float(f1)}


def ema_center(raw, hl, center=None):
    a = 1.0 - 0.5 ** (1.0 / hl)
    out = np.empty_like(raw, dtype=float)
    m = float(raw[0] if center is None else center)
    for i, x in enumerate(raw):
        m += a * (float(x) - m)
        out[i] = float(x) - m
    return out


def hmm(x, dwell=300):
    stay, switch = math.log1p(-1.0 / dwell), -math.log(dwell)
    lp = np.array([-math.log(2.), -math.log(2.)]); out = np.empty(len(x), dtype=float)
    for i, z in enumerate(x):
        em = np.array([-np.logaddexp(0., z), -np.logaddexp(0., -z)])
        if i: lp = np.array([np.logaddexp(lp[0]+stay, lp[1]+switch), np.logaddexp(lp[0]+switch, lp[1]+stay)])
        u = lp + em; lp = u - np.logaddexp(u[0], u[1]); out[i] = lp[1] - lp[0]
    return out


def dual(raw, center=None):
    """E59 指定少量对照：h120, w=.50, cap=.50，严格因果。"""
    # 直接重建两条因果 EMA level；cap=.5 为绝对分数单位
    a = 1.0 - 0.5 ** (1.0 / 1200); m = float(raw[0] if center is None else center)
    lm = np.empty_like(raw, dtype=float)
    for i, x in enumerate(raw): m += a * (x - m); lm[i] = m
    a = 1.0 - 0.5 ** (1.0 / 120); m = float(raw[0] if center is None else center)
    sm = np.empty_like(raw, dtype=float)
    for i, x in enumerate(raw): m += a * (x - m); sm[i] = m
    z = raw - lm - .5 * np.clip(sm - lm, -.5, .5)
    return z


def post(raw, center, mode):
    z = ema_center(raw, 1200, center) if mode == "baseline" else dual(raw, center)
    return hmm(z)


def choose_threshold(train_scores, train_D):
    rows = []
    for q, scores in train_scores.items():
        y = train_D[q][1]
        pos = np.sort(scores[y == 1]); neg = np.sort(scores[y == 0])
        if not len(pos) or not len(neg): continue
        rows.append(.5 * (1 - np.searchsorted(pos, THRESHOLDS, side="right") / len(pos)
                            + np.searchsorted(neg, THRESHOLDS, side="right") / len(neg)))
    vals = np.mean(rows, axis=0)
    # ties prefer threshold nearest zero
    best = max(range(len(THRESHOLDS)), key=lambda i: (vals[i], -abs(THRESHOLDS[i])))
    return float(THRESHOLDS[best]), float(vals[best])


def summary(rows):
    rows = [r for r in rows if r is not None]
    if not rows: return {k: float("nan") for k in ("bacc", "auc", "f1")}, float("nan")
    return {k: float(np.nanmean([r[k] for r in rows])) for k in ("bacc", "auc", "f1")}, float(np.nanmin([r["bacc"] for r in rows]))


def load_fold(path):
    z = np.load(path, allow_pickle=True); return {q: z[q] for q in z.files}


def run_fold(fold, q, subs, D, F):
    t0 = time.time(); tr = [p for p in subs if p != q]
    X = torch.cat([F[p] for p in tr]); y = np.concatenate([D[p][1] for p in tr])
    flr = L.fit_lr(X, y, .001); fms = [fit_mlp(X, y, seed=s) for s in range(3)]
    lr_parts = [flr(F[p]) for p in tr]; mlp_parts = [torch.stack([f(F[p]) for f in fms]).mean(0) for p in tr]
    std_lr = float(torch.cat(lr_parts).std()); std_mlp = float(torch.cat(mlp_parts).std())
    center_lr = float(torch.cat(lr_parts).mean()); center_mlp = float(torch.cat(mlp_parts).mean())
    raw = {}
    for p in subs:
        a = flr(F[p]); m = torch.stack([f(F[p]) for f in fms]).mean(0)
        raw[p] = (a / std_lr + m / std_mlp).detach().cpu().numpy()
    np.savez_compressed(FOLD_DIR / f"fold_{fold:02d}_{q}.npz", **{"raw_" + p: raw[p] for p in raw}, train_subjects=np.array(tr), target=np.array(q), std_lr=std_lr, std_mlp=std_mlp, center_lr=center_lr, center_mlp=center_mlp, pooled_center=float(center_lr / std_lr + center_mlp / std_mlp))
    thresholds, scores_all = {}, {}
    for mode in ("baseline", "dual_h120_w50_c50"):
        for init in ("first", "train_center"):
            center = None if init == "first" else float(center_lr / std_lr + center_mlp / std_mlp)
            ts = {p: post(raw[p], center, "baseline" if mode == "baseline" else "dual") for p in tr}
            threshold, train_bacc = choose_threshold(ts, D)
            thresholds[f"{mode}_{init}"] = {"threshold": threshold, "train_bacc": train_bacc}
            scores_all[f"{mode}_{init}"] = post(raw[q], center, "baseline" if mode == "baseline" else "dual")
    target = {}
    for name, s in scores_all.items():
        yq, b = D[q][1], D[q][2]; t = thresholds[name]["threshold"]
        target[name] = {"full": metric(yq, s, t), "early12": None if not (b <= 2).any() else metric(yq[b <= 2], s[b <= 2], t), "late45": metric(yq[b >= 4], s[b >= 4], t), "full_threshold0": metric(yq, s, 0.0), "early12_threshold0": None if not (b <= 2).any() else metric(yq[b <= 2], s[b <= 2], 0.0), "late45_threshold0": metric(yq[b >= 4], s[b >= 4], 0.0)}
    out = {"fold": fold, "target": q, "thresholds": thresholds, "metrics": target, "elapsed_s": round(time.time()-t0, 1), "raw_file": f"fold_{fold:02d}_{q}.npz"}
    (FOLD_DIR / f"fold_{fold:02d}_{q}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{fold}/{len(subs)}] {q} {target['baseline_train_center']['full']} {out['elapsed_s']}s", flush=True)
    return out


def select_targets(subs, requested, folds):
    if not requested: return subs[:folds]
    targets = []
    for token in ",".join(requested).split(","):
        matches = [q for q in subs if q == token or q.removeprefix("sub-") == token]
        if len(matches) != 1: raise ValueError(f"Unknown or ambiguous target: {token}")
        if matches[0] not in targets: targets.append(matches[0])
    return targets


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, default=57, choices=(1,57)); ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--targets", nargs="+", help="Target IDs, space or comma separated")
    a = ap.parse_args()
    if a.balanced: raise NotImplementedError("--balanced 仅预留，暂不实现")
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    subs, D = L.load()
    targets = select_targets(subs, a.targets, a.folds)
    print(f"device={DEV}; targets={targets}", flush=True)
    F = E42.build_features(subs, D)
    # 已完成折直接复用，模型原始输出与中心均已落盘。
    results = []
    for q in targets:
        i = subs.index(q) + 1
        p = FOLD_DIR / f"fold_{i:02d}_{q}.json"
        results.append(json.loads(p.read_text(encoding="utf-8")) if p.exists() else run_fold(i, q, subs, D, F))
    agg = {}
    for name in results[0]["metrics"]:
        agg[name] = {}
        for part in ("full", "early12", "late45"):
            rows = [r["metrics"][name][part] for r in results]; m, worst = summary(rows); agg[name][part] = {**m, "worst_bacc": worst}
    payload = {"experiment":"E61_trainfold_initialization_corrected", "targets":targets, "scope":"diagnostic target folds only", "device":DEV, "n_folds":len(results), "threshold_grid":"-4..4 step .1 incl 0", "threshold_protocol":"训练被试逐人因果后处理后宏bACC选择；训练内拟合、非内层OOF，目标完全隔离但训练内乐观", "results":agg, "folds":results}
    RESULT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, parts in agg.items(): print(name, json.dumps(parts, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
