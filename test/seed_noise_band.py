"""随机种子噪声带实验：固定划分/超参/早停，只改变训练种子，量化 OOF 指标的种子间波动。

源特征 highdim_deep_features.npz 的当前 6 秒窗 858 维（剔除每折训练部分常量列后 856 维）。
推理只用当前单窗特征，无测试受试者信息、无历史、无窗口位置/时间表/R/T、无测试校准。
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test"
SRC = OUT / "highdim_deep_features.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 1, 2, 3, 4)
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261029
torch.set_num_threads(2)


class MLP(nn.Module):
    def __init__(self, dim, hidden=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.15),
                                 nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict(model, X):
    model.eval()
    with torch.no_grad():
        z = torch.from_numpy(X.astype("float32"))
        return torch.cat([torch.sigmoid(model(z[i:i + 512].to(DEVICE))).cpu()
                          for i in range(0, len(X), 512)]).numpy()


def metrics(y, p):
    return {"AUC": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, p >= 0.5))}


def fit_mlp(X, y, inner_train, inner_val, seed):
    """固定架构/优化器/早停规则，唯一变量是随机种子（初始化与批次顺序）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    posw = torch.tensor(float((y[inner_train] == 0).sum() / max((y[inner_train] == 1).sum(), 1)),
                        device=DEVICE)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=posw)
    rng = np.random.default_rng(seed)
    best, state, stale, best_epoch = float("inf"), None, 0, 0
    xt = torch.from_numpy(X[inner_train]).to(DEVICE)
    yt = torch.from_numpy(y[inner_train].astype("float32")).to(DEVICE)
    for epoch in range(1, 61):
        model.train()
        order = rng.permutation(len(xt))
        for start in range(0, len(order), 256):
            ix = order[start:start + 256]
            opt.zero_grad()
            loss = lossfn(model(xt[ix]), yt[ix])
            loss.backward()
            opt.step()
        p = np.clip(predict(model, X[inner_val]), 1e-6, 1 - 1e-6)
        val_loss = float(-np.mean(y[inner_val] * np.log(p) + (1 - y[inner_val]) * np.log(1 - p)))
        if val_loss < best - 1e-5:
            best, state, stale, best_epoch = val_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(state)
    return model, best_epoch


def subject_aucs(y, p, groups):
    out = {}
    for s in sorted(set(groups)):
        m = groups == s
        if len(np.unique(y[m])) < 2:
            continue
        out[s] = float(roc_auc_score(y[m], p[m]))
    return out


def subject_summary(aucs):
    v = np.array(list(aucs.values()))
    return {"n_subjects": int(len(v)), "median": float(np.median(v)),
            "q1": float(np.quantile(v, 0.25)), "q3": float(np.quantile(v, 0.75)),
            "min": float(v.min()), "max": float(v.max()),
            "frac_auc_below_0.5": float(np.mean(v < 0.5))}


def bootstrap(y, p, groups, subjects, rng):
    """受试者级有放回重采样，重算池化 OOF AUC 与 bACC。"""
    index = {s: np.where(groups == s)[0] for s in subjects}
    auc, bacc = [], []
    for _ in range(N_BOOTSTRAP):
        pick = rng.choice(subjects, size=len(subjects), replace=True)
        ids = np.concatenate([index[s] for s in pick])
        yy, pp = y[ids], p[ids]
        auc.append(roc_auc_score(yy, pp))
        bacc.append(balanced_accuracy_score(yy, pp >= 0.5))
    auc, bacc = np.array(auc), np.array(bacc)
    return {"AUC": {"lo": float(np.quantile(auc, 0.025)), "hi": float(np.quantile(auc, 0.975))},
            "bACC": {"lo": float(np.quantile(bacc, 0.025)), "hi": float(np.quantile(bacc, 0.975))},
            "n_bootstrap": N_BOOTSTRAP}


def main():
    targets = [OUT / "seed_noise_band_results.json", OUT / "seed_noise_band_oof.npz"]
    if any(p.exists() for p in targets):
        raise FileExistsError("Refusing to overwrite existing seed-noise artifacts")
    tick = time.perf_counter()
    data = np.load(SRC)
    X0 = data["X"].astype("float32")
    y = data["y"].astype(int)
    groups = data["groups"].astype(str)
    if X0.shape[1] != 858:
        raise ValueError(f"expected 858 source features, got {X0.shape[1]}")
    subjects = sorted(set(groups))
    print(f"[data] X={X0.shape} y={np.bincount(y).tolist()} subjects={len(subjects)} device={DEVICE}",
          flush=True)

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    splits = []
    for fold, (tr, te) in enumerate(cv.split(X0, y, groups), 1):
        assert not (set(groups[tr]) & set(groups[te]))
        keep = X0[tr].std(axis=0) > 1e-8
        if int(keep.sum()) != 856:
            raise ValueError(f"fold {fold}: expected 856 nonconstant features, got {int(keep.sum())}")
        mu = X0[tr][:, keep].mean(axis=0)
        sd = np.maximum(X0[tr][:, keep].std(axis=0), 1e-6)
        X = ((X0[:, keep] - mu) / sd).astype("float32")
        it, iv = next(StratifiedGroupKFold(5, shuffle=True, random_state=fold)
                      .split(X[tr], y[tr], groups[tr]))
        splits.append({"fold": fold, "tr": tr, "te": te, "X": X,
                       "inner_train": tr[it], "inner_val": tr[iv]})
        print(f"[split] fold={fold}/5 train_subjects={len(set(groups[tr]))} "
              f"test_subjects={len(set(groups[te]))} n_test={len(te)}", flush=True)

    results = {"protocol": {
        "source": SRC.name, "source_features": 858, "mlp_input_features": 856,
        "cv": "StratifiedGroupKFold(5, shuffle=True, random_state=0)",
        "group": "subject; test subjects fully disjoint from training subjects",
        "inner_split": "first split of StratifiedGroupKFold(5, shuffle=True, random_state=outer_fold); identical for all seeds",
        "scaler": "mean/std fit on outer training fold only; constant training columns dropped (858->856)",
        "model": "MLP 856-96-32-1, GELU, dropout 0.15",
        "training": "AdamW lr=1e-3 weight_decay=1e-3, BCEWithLogits pos_weight, batch 256, max 60 epochs, patience 7 on inner validation NLL",
        "seed_scope": "only model init and batch order vary; splits/hyperparameters/early stopping identical",
        "seeds": list(SEEDS), "bootstrap": f"subject-level resampling with replacement, {N_BOOTSTRAP} draws",
        "inference": "current single 6-second 856-d feature window only",
        "test_subject_isolation": True, "test_history": False,
        "window_position_schedule_r_t": False, "test_subject_calibration": False},
        "config": {"seeds": list(SEEDS), "n_bootstrap": N_BOOTSTRAP,
                   "bootstrap_seed": BOOTSTRAP_SEED, "device": DEVICE,
                   "torch_version": str(torch.__version__)},
        "seeds": {}}

    oof_by_seed = {}
    for seed in SEEDS:
        start = time.perf_counter()
        oof = np.full(len(y), np.nan)
        folds = []
        for part in splits:
            X, tr, te = part["X"], part["tr"], part["te"]
            model, best_epoch = fit_mlp(X, y, part["inner_train"], part["inner_val"], seed)
            p = predict(model, X[te])
            oof[te] = p
            r = metrics(y[te], p)
            folds.append({"fold": part["fold"], **r, "best_epoch": int(best_epoch),
                          "n_test": int(len(te)),
                          "test_subjects": sorted(set(groups[te]))})
            print(f"[seed {seed} fold {part['fold']}/5] AUC={r['AUC']:.5f} bACC={r['bACC']:.5f} "
                  f"epoch={best_epoch} n={len(te)}", flush=True)
        assert np.isfinite(oof).all()
        oof_by_seed[seed] = oof
        final = metrics(y, oof)
        aucs = subject_aucs(y, oof, groups)
        boot = bootstrap(y, oof, groups, subjects, np.random.default_rng(BOOTSTRAP_SEED + seed))
        results["seeds"][str(seed)] = {
            "folds": folds,
            "fold_mean": {m: float(np.mean([f[m] for f in folds])) for m in ("AUC", "bACC")},
            "oof": final,
            "bootstrap_95ci": boot,
            "per_subject_auc": {"summary": subject_summary(aucs), "values": aucs},
            "seconds": float(time.perf_counter() - start)}
        print(f"[seed {seed}] OOF AUC={final['AUC']:.5f} bACC={final['bACC']:.5f} "
              f"bootstrap95 AUC=[{boot['AUC']['lo']:.5f},{boot['AUC']['hi']:.5f}] "
              f"bACC=[{boot['bACC']['lo']:.5f},{boot['bACC']['hi']:.5f}]", flush=True)

    for metric in ("AUC", "bACC"):
        v = np.array([results["seeds"][str(s)]["oof"][metric] for s in SEEDS])
        results.setdefault("seed_summary", {})[metric] = {
            "mean": float(v.mean()), "std": float(v.std(ddof=1)),
            "min": float(v.min()), "max": float(v.max()), "values": v.tolist()}

    mean_oof = np.mean([oof_by_seed[s] for s in SEEDS], axis=0)
    mean_aucs = subject_aucs(y, mean_oof, groups)
    mean_boot = bootstrap(y, mean_oof, groups, subjects, np.random.default_rng(BOOTSTRAP_SEED + 100))
    results["seed_mean_prediction"] = {
        "oof": metrics(y, mean_oof), "bootstrap_95ci": mean_boot,
        "per_subject_auc": {"summary": subject_summary(mean_aucs), "values": mean_aucs}}
    print(f"[seed-mean] OOF AUC={results['seed_mean_prediction']['oof']['AUC']:.5f} "
          f"bACC={results['seed_mean_prediction']['oof']['bACC']:.5f}", flush=True)

    results["runtime_seconds"] = float(time.perf_counter() - tick)
    summaries = results["seed_summary"]
    print(f"[summary] OOF AUC mean={summaries['AUC']['mean']:.5f} "
          f"sd={summaries['AUC']['std']:.5f} range=[{summaries['AUC']['min']:.5f},{summaries['AUC']['max']:.5f}]",
          flush=True)
    print(f"[summary] OOF bACC mean={summaries['bACC']['mean']:.5f} "
          f"sd={summaries['bACC']['std']:.5f} range=[{summaries['bACC']['min']:.5f},{summaries['bACC']['max']:.5f}]",
          flush=True)

    fold_id = np.full(len(y), -1, dtype="int8")
    for part in splits:
        fold_id[part["te"]] = part["fold"]
    target = targets[1].open("xb")
    with target:
        np.savez_compressed(target, y=y, groups=groups, fold_id=fold_id,
                            oof_seed_mean=mean_oof.astype("float32"),
                            **{f"oof_seed{s}": oof_by_seed[s].astype("float32") for s in SEEDS})
    targets[0].write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] elapsed={results['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
