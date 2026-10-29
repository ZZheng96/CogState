"""Strict subject-disjoint 5-fold evaluation of frozen SingLEM window tokens."""
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

OUT = Path(__file__).resolve().parent
SEED = 0
EPOCHS = 40
PATIENCE = 7
BATCH = 256


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data():
    emb = OUT / "singlem_embed.npz"
    if not emb.exists():
        print("[data] singlem_embed.npz 不存在，运行现有 singlem_embed.py", flush=True)
        subprocess.run([sys.executable, "-u", str(OUT / "singlem_embed.py")], check=True)
    e = np.load(emb)
    d = np.load(OUT / "highdim_deep_features.npz")
    R = e["R"].astype(np.float32)
    y = d["y"].astype(np.int64)
    groups = d["groups"]
    if R.shape != (len(y), 6, 16):
        raise ValueError(f"unexpected R shape: {R.shape}")
    return R, y, groups


def trend_features(R):
    t = np.arange(R.shape[1], dtype=np.float32)
    tc = t - t.mean()
    slope = np.sum((R - R.mean(1, keepdims=True)) * tc[None, :, None], axis=1) / np.sum(tc ** 2)
    return np.concatenate([R.mean(1), R.std(1), slope], axis=1)


def split_train_val(train_idx, y, groups, seed):
    subs = np.unique(groups[train_idx])
    sy = np.array([int(np.mean(y[train_idx][groups[train_idx] == s]) >= .5) for s in subs])
    if len(np.unique(sy)) == 2 and len(subs) >= 6:
        sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        a, b = next(sg.split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]))
        return train_idx[a], train_idx[b]
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    val_subs = set(subs[:max(1, len(subs) // 5)])
    return train_idx[~np.isin(groups[train_idx], list(val_subs))], train_idx[np.isin(groups[train_idx], list(val_subs))]


def subject_metrics(y, p, groups):
    aucs, baccs = [], []
    for s in np.unique(groups):
        m = groups == s
        if len(np.unique(y[m])) == 2:
            aucs.append(roc_auc_score(y[m], p[m]))
        baccs.append(balanced_accuracy_score(y[m], p[m] >= .5))
    return {"auc": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, p >= .5)),
            "subject_macro_auc": float(np.mean(aucs)),
            "subject_macro_bACC": float(np.mean(baccs))}


class AttnClassifier(nn.Module):
    def __init__(self, use_gated=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 32), nn.GELU())
        self.use_gated = use_gated
        self.score = nn.Linear(32, 1)
        if use_gated:
            self.gate = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
            self.gate2 = nn.Sequential(nn.Linear(32, 32), nn.Sigmoid())
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))

    def forward(self, x, return_z=False):
        h = self.proj(x)
        if self.use_gated:
            a = self.score(h) + (self.gate(h) * self.gate2(h)).sum(-1, keepdim=True) / 32**.5
        else:
            a = self.score(h)
        w = torch.softmax(a, dim=1)
        z = (w * h).sum(1)
        logit = self.head(z).squeeze(-1)
        return (logit, z) if return_z else logit


def supcon_loss(z, y, temp=.15):
    z = nn.functional.normalize(z, dim=1)
    sim = z @ z.T / temp
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    pos = (y[:, None] == y[None, :]) & ~eye
    logits = sim.masked_fill(eye, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    count = pos.sum(1)
    valid = count > 0
    if not valid.any():
        return z.sum() * 0
    return -(logp.masked_fill(~pos, 0).sum(1)[valid] / count[valid]).mean()


def fit_torch(R, y, groups, tr, te, method, fold, device):
    fit, val = split_train_val(tr, y, groups, SEED + fold)
    # Standardization is estimated from training subjects only.
    mu, sd = R[fit].mean((0, 1), keepdims=True), R[fit].std((0, 1), keepdims=True) + 1e-5
    X = ((R - mu) / sd).astype(np.float32)
    model = AttnClassifier(use_gated=method == "attention_gated").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(X[fit]), torch.from_numpy(y[fit]).float())
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=torch.Generator().manual_seed(SEED + fold))
    best, best_state, wait = -np.inf, None, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logit, z = model(xb, return_z=True)
            loss = loss_fn(logit, yb)
            if method == "attention_gated_ssl":
                loss = loss + .08 * supcon_loss(z, yb.long())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(torch.from_numpy(X[val]).to(device))).cpu().numpy()
        score = roc_auc_score(y[val], vp)
        if score > best + 1e-5:
            best, best_state, wait = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(X[te]).to(device))).cpu().numpy()
    return p, epoch + 1, best


def main():
    tick = time.perf_counter(); seed_all()
    R, y, groups = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[start] R={R.shape}, subjects={len(np.unique(groups))}, device={device}", flush=True)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(R, y, groups))
    methods = ["pooling_logistic", "attention_gated", "attention_gated_ssl"]
    oof = {m: np.full(len(y), np.nan, dtype=np.float32) for m in methods}
    fold_info = []
    for fold, (tr, te) in enumerate(splits, 1):
        print(f"[fold {fold}/5] train={len(tr)} test={len(te)} subjects={len(np.unique(groups[te]))}", flush=True)
        F = trend_features(R)
        scaler_lr = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=2000, random_state=SEED))
        scaler_lr.fit(F[tr], y[tr]); oof["pooling_logistic"][te] = scaler_lr.predict_proba(F[te])[:, 1]
        print(f"[fold {fold}/5] pooling_logistic done", flush=True)
        info = {"fold": fold, "test_subjects": np.unique(groups[te]).tolist()}
        for method in methods[1:]:
            p, epochs, val_auc = fit_torch(R, y, groups, tr, te, method, fold, device)
            oof[method][te] = p
            info[method] = {"epochs": epochs, "val_auc": float(val_auc)}
            print(f"[fold {fold}/5] {method} done epochs={epochs} val_auc={val_auc:.4f}", flush=True)
        fold_info.append(info)
    results = {m: subject_metrics(y, p, groups) for m, p in oof.items()}
    params = {"pooling_logistic": int(sum(np.prod(v.shape) for v in scaler_lr[-1].coef_)) + 1,
              "attention_gated": sum(p.numel() for p in AttnClassifier(True).parameters()),
              "attention_gated_ssl": sum(p.numel() for p in AttnClassifier(True).parameters())}
    result = {"seed": SEED, "folds": 5, "subjects": int(len(np.unique(groups))), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "elapsed_sec": time.perf_counter() - tick, "parameter_counts": params,
              "methods": results, "fold_info": fold_info}
    np.savez_compressed(OUT / "singlem_token_ssl_window6_oof.npz", y=y, groups=groups, **oof)
    (OUT / "singlem_token_ssl_window6_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for m, r in results.items():
        print(f"[result] {m:22s} AUC={r['auc']:.4f} bACC={r['bACC']:.4f} subjectAUC={r['subject_macro_auc']:.4f} subjectbACC={r['subject_macro_bACC']:.4f}", flush=True)
    print(f"[done] elapsed={result['elapsed_sec']:.1f}s outputs written", flush=True)


if __name__ == "__main__":
    main()
