"""Training-fold subject-window consistency experiment for single-window inference.

Each outer fold fits preprocessing and models using only training subjects. The
consistency model sees two independently masked/noisy views of each training
window and optimizes label BCE plus representation/logit consistency. Test
inference receives one current 6-second, 856-feature window only.
"""
import copy
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)


class WindowNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 192), nn.GELU(), nn.Dropout(0.10),
                                     nn.Linear(192, 48), nn.GELU())
        self.head = nn.Linear(48, 1)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(1), z


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        tx = torch.from_numpy(X.astype("float32"))
        for start in range(0, len(X), 512):
            logits, _ = model(tx[start:start + 512].to(DEVICE))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def metrics(y, p):
    return {"AUC": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, p >= 0.5))}


def train_one(X, y, groups, fold, consistency):
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + fold)
    it, iv = next(inner.split(X, y, groups))
    model = WindowNet(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    posw = torch.tensor(float((y[it] == 0).sum() / max((y[it] == 1).sum(), 1)), device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=posw)
    tx = torch.from_numpy(X[it]).to(DEVICE)
    ty = torch.from_numpy(y[it].astype("float32")).to(DEVICE)
    rng = np.random.default_rng(SEED + (1000 if consistency else 0) + fold)
    best, state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, 61):
        model.train(); order = rng.permutation(len(it)); losses = []
        for start in range(0, len(order), 256):
            ix = order[start:start + 256]
            clean = tx[ix]
            opt.zero_grad()
            if consistency:
                v1 = clean * (torch.rand_like(clean) > 0.08).float() + 0.03 * torch.randn_like(clean)
                v2 = clean * (torch.rand_like(clean) > 0.08).float() + 0.03 * torch.randn_like(clean)
                l1, z1 = model(v1); l2, z2 = model(v2)
                loss = 0.5 * (bce(l1, ty[ix]) + bce(l2, ty[ix]))
                loss = loss + 0.20 * nn.functional.mse_loss(nn.functional.normalize(z1, dim=1), nn.functional.normalize(z2, dim=1))
                loss = loss + 0.10 * nn.functional.mse_loss(torch.sigmoid(l1), torch.sigmoid(l2))
            else:
                logits, _ = model(clean)
                loss = bce(logits, ty[ix])
            loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        p = np.clip(predict(model, X[iv]), 1e-6, 1 - 1e-6)
        val_loss = float(-np.mean(y[iv] * np.log(p) + (1 - y[iv]) * np.log(1 - p)))
        if val_loss < best - 1e-5:
            best, state, stale, best_epoch = val_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            tag = "consistency" if consistency else "supervised"
            print(f"[fold {fold}/5] {tag} epoch {epoch}/60 val_loss={val_loss:.5f}", flush=True)
        if stale >= 8:
            break
    model.load_state_dict(state)
    return model, {"best_epoch": int(best_epoch), "inner_val_windows": int(len(iv))}


def main():
    tick = time.perf_counter(); np.random.seed(SEED); torch.manual_seed(SEED)
    data = np.load(OUT / "highdim_deep_features.npz")
    X0 = data["X"].astype("float32"); y = data["y"].astype(int); groups = data["groups"]
    if X0.shape[1] != 858:
        raise ValueError(f"expected 858 source features, got {X0.shape[1]}")
    oof = {"supervised": np.full(len(y), np.nan, dtype="float32"),
           "consistency": np.full(len(y), np.nan, dtype="float32")}
    folds = []
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    print(f"[data] X={X0.shape}; current_input=856; subjects={len(np.unique(groups))}; device={DEVICE}", flush=True)
    for fold, (tr, te) in enumerate(cv.split(X0, y, groups), 1):
        assert not (set(groups[tr]) & set(groups[te]))
        keep = X0[tr].std(axis=0) > 1e-8
        if int(keep.sum()) != 856:
            raise ValueError(f"fold {fold}: expected 856 nonconstant features, got {int(keep.sum())}")
        mu = X0[tr][:, keep].mean(axis=0)
        sd = np.maximum(X0[tr][:, keep].std(axis=0), 1e-6)
        X = ((X0[:, keep] - mu) / sd).astype("float32")
        row = {"fold": fold, "n_train": int(len(tr)), "n_test": int(len(te)),
               "train_subjects": int(len(np.unique(groups[tr]))),
               "test_subjects": int(len(np.unique(groups[te])))}
        for mode in ("supervised", "consistency"):
            model, info = train_one(X[tr], y[tr], groups[tr], fold, mode == "consistency")
            p = predict(model, X[te]); oof[mode][te] = p
            r = metrics(y[te], p); row[mode] = {**r, **info}
            print(f"[result] {mode} fold={fold}/5 AUC={r['AUC']:.5f} bACC={r['bACC']:.5f}", flush=True)
        folds.append(row)
    final = {mode: metrics(y, oof[mode]) for mode in oof}
    fold_mean = {mode: {m: float(np.mean([r[mode][m] for r in folds])) for m in ("AUC", "bACC")} for mode in oof}
    for mode in oof:
        print(f"[final] {mode} OOF AUC={final[mode]['AUC']:.5f} bACC={final[mode]['bACC']:.5f}; fold_mean AUC={fold_mean[mode]['AUC']:.5f} bACC={fold_mean[mode]['bACC']:.5f}", flush=True)
    results = {"protocol": {"cv": "StratifiedGroupKFold(5, shuffle=True, random_state=0)",
                             "source": "test/highdim_deep_features.npz", "source_features": 858,
                             "current_window_features": 856, "training": "all windows from outer training subjects only; fold-fitted standardization",
                             "consistency_training": "two feature-masked/noisy views per training window; BCE + representation consistency + logit consistency",
                             "inference": "one current 6-second window only", "test_subject_isolation": True,
                             "test_history": False, "window_position_schedule_r_t": False, "test_subject_calibration": False},
               "models": {"supervised": {"oof": final["supervised"], "fold_mean": fold_mean["supervised"]},
                          "consistency": {"oof": final["consistency"], "fold_mean": fold_mean["consistency"]}},
               "folds": folds, "runtime_seconds": float(time.perf_counter() - tick)}
    np.savez_compressed(OUT / "subject_consistency_window6_oof.npz", y=y, groups=groups,
                        supervised_oof=oof["supervised"], consistency_oof=oof["consistency"])
    (OUT / "subject_consistency_window6_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] elapsed={results['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
