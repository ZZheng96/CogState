"""严格 6 秒单窗类别条件受试者域对齐实验。"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test"
SEED = 0
LAMBDAS = (0.0, 0.05, 0.2)
EPOCHS = 18
BATCH_SIZE = 256
torch.set_num_threads(2)


class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 128), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(128, 32), nn.GELU())
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(1), z


def predict(net, x):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            logits, _ = net(torch.from_numpy(x[i:i + 1024]).float())
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)


def alignment_loss(z, y, groups):
    """Mean squared distance of same-class subject means to same-class grand means."""
    terms = []
    for cls in (0, 1):
        subjects = np.unique(groups[y == cls])
        means = []
        for subject in subjects:
            mask = (y == cls) & (groups == subject)
            if mask.any():
                means.append(z[torch.from_numpy(mask)].mean(dim=0))
        if means:
            means = torch.stack(means)
            terms.append(((means - means.mean(dim=0, keepdim=True)) ** 2).mean())
    return torch.stack(terms).mean() if terms else z.new_tensor(0.0)


def train_fold(xtr, ytr, gtr, xte, fold, lam):
    torch.manual_seed(SEED + fold)
    net = Net(xtr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    posw = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    bce = nn.BCEWithLogitsLoss(pos_weight=posw)
    xt = torch.from_numpy(xtr).float()
    yt = torch.from_numpy(ytr.astype(np.float32))
    rng = np.random.default_rng(SEED + 1000 * fold + int(lam * 100))
    n = len(ytr)
    for epoch in range(1, EPOCHS + 1):
        net.train()
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, BATCH_SIZE):
            ix = order[start:start + BATCH_SIZE]
            opt.zero_grad()
            logits, z = net(xt[ix])
            loss = bce(logits, yt[ix])
            if lam:
                loss = loss + lam * alignment_loss(z, ytr[ix], gtr[ix])
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        print(f"[fold {fold}/5] lambda={lam:g} epoch={epoch}/{EPOCHS} "
              f"loss={np.mean(losses):.5f}", flush=True)
    return predict(net, xte)


def metrics(y, p):
    return {"AUC": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, p >= 0.5))}


def paired_bootstrap(y, groups, predictions, baseline, n_boot=2000):
    subjects = np.unique(groups)
    by_subject = {s: np.flatnonzero(groups == s) for s in subjects}
    rng = np.random.default_rng(20261029)
    da, db = [], []
    for _ in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        ix = np.concatenate([by_subject[s] for s in sampled])
        da.append(roc_auc_score(y[ix], predictions[ix]) - roc_auc_score(y[ix], baseline[ix]))
        db.append(balanced_accuracy_score(y[ix], predictions[ix] >= 0.5) -
                  balanced_accuracy_score(y[ix], baseline[ix] >= 0.5))
    def summary(values):
        return {"estimate": float(np.mean(values)),
                "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
                "p_positive": float(np.mean(np.asarray(values) > 0))}
    return {"n_bootstrap": n_boot, "delta_AUC": summary(da), "delta_bACC": summary(db)}


def main():
    tick = time.perf_counter()
    data = np.load(OUT / "highdim_deep_features.npz")
    emg = np.load(OUT / "emg_features_v2.npz")["E"]
    base = data["X"].astype(np.float32)
    x0 = np.hstack([base, emg.astype(np.float32)])
    y = data["y"].astype(int)
    groups = data["groups"]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    splits = list(cv.split(x0, y, groups))
    print(f"[data] X={x0.shape} (highdim=858 + emg=20), subjects={len(np.unique(groups))}; "
          "current 6-second window only", flush=True)
    oofs = {str(lam): np.full(len(y), np.nan, dtype=np.float32) for lam in LAMBDAS}
    folds = {str(lam): [] for lam in LAMBDAS}
    for fold, (tr, te) in enumerate(splits, 1):
        assert not (set(groups[tr]) & set(groups[te]))
        keep = x0[tr].std(axis=0) > 1e-8
        mu = x0[tr][:, keep].mean(axis=0)
        sd = np.maximum(x0[tr][:, keep].std(axis=0), 1e-6)
        x = ((x0[:, keep] - mu) / sd).astype(np.float32)
        print(f"[fold {fold}/5] train_windows={len(tr)} test_windows={len(te)} "
              f"train_subjects={len(np.unique(groups[tr]))} features={int(keep.sum())}", flush=True)
        for lam in LAMBDAS:
            p = train_fold(x[tr], y[tr], groups[tr], x[te], fold, lam)
            oofs[str(lam)][te] = p
            r = metrics(y[te], p)
            folds[str(lam)].append({"fold": fold, **r, "n_test": int(len(te)),
                                    "train_subjects": int(len(np.unique(groups[tr]))),
                                    "test_subjects": int(len(np.unique(groups[te])))})
            print(f"[result] lambda={lam:g} fold={fold}/5 AUC={r['AUC']:.5f} "
                  f"bACC={r['bACC']:.5f} elapsed={time.perf_counter()-tick:.1f}s", flush=True)
    results = {
        "protocol": {
            "cv": "StratifiedGroupKFold(5, shuffle=True, random_state=0)",
            "input": "highdim_deep_features.npz X (858) + emg_features_v2.npz E (20)",
            "window": "strict current 6-second feature window; all available training-subject windows",
            "standardization": "per-fold training-window mean/std; constant columns removed per fold",
            "model": "MLP encoder 878->128->32 + binary state head",
            "loss": "BCE + lambda * mean same-class subject-mean squared distance to same-class training-subject grand mean",
            "lambdas": list(LAMBDAS), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-3)", "seed": SEED,
            "test_history": False, "test_subject_statistics": False,
            "alignment_uses": "training-fold labels and training-fold subjects only; no R/T mixing"
        },
        "models": {}, "paired_subject_bootstrap_vs_lambda0": {}
    }
    for lam in LAMBDAS:
        key = str(lam); p = oofs[key]
        r = metrics(y, p)
        r["fold_mean_AUC"] = float(np.mean([v["AUC"] for v in folds[key]]))
        r["fold_mean_bACC"] = float(np.mean([v["bACC"] for v in folds[key]]))
        r["folds"] = folds[key]
        results["models"][key] = r
        print(f"[final] lambda={lam:g} OOF AUC={r['AUC']:.5f} bACC={r['bACC']:.5f} "
              f"fivefold AUC={r['fold_mean_AUC']:.5f}", flush=True)
    for lam in LAMBDAS[1:]:
        results["paired_subject_bootstrap_vs_lambda0"][str(lam)] = paired_bootstrap(
            y, groups, oofs[str(lam)], oofs["0.0"])
    results["runtime_seconds"] = float(time.perf_counter() - tick)
    np.savez_compressed(OUT / "class_conditional_alignment_window6_oof.npz", y=y, groups=groups, **oofs)
    (OUT / "class_conditional_alignment_window6_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] elapsed={results['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
