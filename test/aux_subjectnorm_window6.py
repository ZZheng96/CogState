"""多任务辅助目标 + 训练期受试者内标准化（6 秒窗，严格受试者隔离）。

主任务：relax/task BCE。辅助任务：从当前 6 秒 raw 波形直接计算 9 个生理量
（5 个频带 log 功率、谱熵、Hjorth activity/mobility/complexity），
并在训练折内部按训练受试者做 z-score，使目标描述"该受试者自身的相对状态"。
推理只用当前窗特征，不接触任何受试者统计量。
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
SEED = 0
FS = 250.0
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
LAMBDAS = (0.1, 0.3, 1.0)
MAX_EPOCHS = 40
PATIENCE = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)


# ---------------- 辅助目标 ----------------
def aux_targets(raw):
    """每个窗口的生理量，仅用当前 6 秒波形，不含跨窗口信息。"""
    x = raw.astype("float64")
    n = x.shape[1]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft((x - x.mean(1, keepdims=True)) * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    feats, names = [], []
    for lo, hi in BANDS:
        sel = (freqs >= lo) & (freqs < hi)
        feats.append(np.log(spec[:, sel].sum(1) + 1e-30))
        names.append(f"log_power_{lo}_{hi}")
    band = (freqs >= 1) & (freqs <= 45)
    p = spec[:, band] / np.maximum(spec[:, band].sum(1, keepdims=True), 1e-30)
    feats.append(-(p * np.log(p + 1e-30)).sum(1) / np.log(p.shape[1]))
    names.append("spectral_entropy_1_45")

    x0 = x - x.mean(1, keepdims=True)
    d1 = np.diff(x0, axis=1)
    v0 = np.var(x0, axis=1)
    v1 = np.var(d1, axis=1)
    mob = np.sqrt(v1 / np.maximum(v0, 1e-30))
    mob1 = np.sqrt(np.var(np.diff(d1, axis=1), axis=1) / np.maximum(v1, 1e-30))
    feats.append(np.log(v0 + 1e-30))
    names.append("hjorth_activity_logvar")
    feats.append(mob)
    names.append("hjorth_mobility")
    feats.append(mob1 / np.maximum(mob, 1e-6))
    names.append("hjorth_complexity")
    return np.stack(feats, axis=1).astype("float32"), names


def subject_zscore(T, groups, ref):
    """按受试者 z-score；统计量只取 ref 指定的行（内层训练行）。"""
    Z = np.zeros_like(T)
    for s in np.unique(groups):
        m = groups == s
        r = m & ref
        if r.sum() < 2:
            continue
        mu = T[r].mean(0)
        sd = T[r].std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        Z[m] = (T[m] - mu) / sd
    return np.clip(Z, -5.0, 5.0).astype("float32")


# ---------------- 模型与训练 ----------------
class Net(nn.Module):
    def __init__(self, dim, n_aux):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, 96), nn.GELU(), nn.Dropout(0.15),
                                   nn.Linear(96, 32), nn.GELU())
        self.cls = nn.Linear(32, 1)
        self.aux = nn.Linear(32, n_aux)

    def forward(self, x):
        z = self.trunk(x)
        return self.cls(z).squeeze(1), self.aux(z)


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            logit, _ = model(torch.from_numpy(X[i:i + 512]).to(DEVICE))
            out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def val_bce(model, Xv, yv):
    p = np.clip(predict(model, Xv), 1e-6, 1 - 1e-6)
    return float(-np.mean(yv * np.log(p) + (1 - yv) * np.log(1 - p)))


def train_model(Xtr, ytr, Atr, lam, epochs, seed, Xval=None, yval=None):
    """lam=0 即纯 BCE 对照；结构完全相同。Xval 为 None 时按固定轮数训练（refit）。"""
    torch.manual_seed(seed)
    model = Net(Xtr.shape[1], Atr.shape[1] if Atr is not None else 1).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    posw = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)), device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=posw)
    rng = np.random.default_rng(seed)
    xt = torch.from_numpy(Xtr).to(DEVICE)
    yt = torch.from_numpy(ytr.astype("float32")).to(DEVICE)
    at = torch.from_numpy(Atr).to(DEVICE) if Atr is not None else None
    use_aux = lam > 0 and at is not None
    best, state, stale, best_ep = float("inf"), None, 0, epochs
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(xt))
        for start in range(0, len(order), 256):
            ix = order[start:start + 256]
            opt.zero_grad()
            logit, aux_pred = model(xt[ix])
            loss = bce(logit, yt[ix])
            if use_aux:
                loss = loss + lam * nn.functional.smooth_l1_loss(aux_pred, at[ix])
            loss.backward()
            opt.step()
        if Xval is None:
            continue
        v = val_bce(model, Xval, yval)
        if v < best - 1e-5:
            best, state, stale, best_ep = v, copy.deepcopy(model.state_dict()), 0, ep
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if state is not None:
        model.load_state_dict(state)
    return model, best_ep, best


def metrics(y, p):
    return {"AUC": float(roc_auc_score(y, p)), "bACC": float(balanced_accuracy_score(y, p >= 0.5))}


def main():
    tick = time.perf_counter()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    data = np.load(OUT / "highdim_deep_features.npz")
    X0 = data["X"].astype("float32")
    raw = data["raw"]
    y = data["y"].astype(int)
    groups = data["groups"]

    keep = X0.std(axis=0) > 1e-8
    A_all, aux_names = aux_targets(raw)
    print(f"[data] X={X0.shape} kept={int(keep.sum())} aux_dim={A_all.shape[1]} "
          f"subjects={len(np.unique(groups))} device={DEVICE}", flush=True)

    n = len(y)
    oof_base = np.full(n, np.nan, dtype="float32")
    oof_aux = np.full(n, np.nan, dtype="float32")
    oof_cand = {lam: np.full(n, np.nan, dtype="float32") for lam in LAMBDAS}
    folds = []

    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(cv.split(X0, y, groups), 1):
        assert not (set(groups[tr]) & set(groups[te]))
        print(f"[fold {fold}/5] start train_subj={len(np.unique(groups[tr]))} "
              f"test_subj={len(np.unique(groups[te]))}", flush=True)

        mu = X0[tr][:, keep].mean(axis=0)
        sd = np.maximum(X0[tr][:, keep].std(axis=0), 1e-6)
        X = ((X0[:, keep] - mu) / sd).astype("float32")
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        gtr = groups[tr]

        inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + fold)
        it, iv = next(inner.split(Xtr, ytr, gtr))
        it_mask = np.zeros(len(tr), dtype=bool)
        it_mask[it] = True
        A_it = subject_zscore(A_all[tr], gtr, it_mask)
        A_tr = subject_zscore(A_all[tr], gtr, np.ones(len(tr), dtype=bool))

        # 对照：纯 BCE，同样轮数预算
        base_model, base_ep, base_val = train_model(
            Xtr[it], ytr[it], None, 0.0, MAX_EPOCHS, SEED + 10 * fold, Xtr[iv], ytr[iv])
        p_base_cand = predict(base_model, Xte)
        r_base_cand = metrics(yte, p_base_cand)
        print(f"[fold {fold}/5] baseline inner_bce={base_val:.5f} best_epoch={base_ep} "
              f"test_AUC={r_base_cand['AUC']:.4f} bACC={r_base_cand['bACC']:.4f}", flush=True)

        cand = {}
        for lam in LAMBDAS:
            model, ep, v = train_model(Xtr[it], ytr[it], A_it[it], lam, MAX_EPOCHS,
                                       SEED + 10 * fold + int(lam * 100), Xtr[iv], ytr[iv])
            p = predict(model, Xte)
            oof_cand[lam][te] = p
            r = metrics(yte, p)
            cand[str(lam)] = {"inner_val_bce": float(v), "best_epoch": int(ep), **r}
            print(f"[fold {fold}/5] lambda={lam} inner_val_bce={v:.5f} best_epoch={ep} "
                  f"test_AUC={r['AUC']:.4f} bACC={r['bACC']:.4f}", flush=True)

        sel = min(LAMBDAS, key=lambda s: cand[str(s)]["inner_val_bce"])
        print(f"[fold {fold}/5] selected lambda={sel} (inner val BCE)", flush=True)

        # refit：全部外层训练受试者，轮数取内层早停轮数
        aux_model, _, _ = train_model(Xtr, ytr, A_tr, sel, cand[str(sel)]["best_epoch"],
                                      SEED + 100 + fold)
        p_aux = predict(aux_model, Xte)
        oof_aux[te] = p_aux
        r_aux = metrics(yte, p_aux)
        base_model2, _, _ = train_model(Xtr, ytr, None, 0.0, base_ep, SEED + 200 + fold)
        p_base = predict(base_model2, Xte)
        oof_base[te] = p_base
        r_base = metrics(yte, p_base)
        print(f"[fold {fold}/5] refit baseline AUC={r_base['AUC']:.4f} bACC={r_base['bACC']:.4f} | "
              f"aux AUC={r_aux['AUC']:.4f} bACC={r_aux['bACC']:.4f}", flush=True)
        print(f"[fold {fold}/5] done", flush=True)

        folds.append({"fold": fold, "selected_lambda": sel,
                      "inner_val_bce": {"baseline": float(base_val),
                                        **{k: v["inner_val_bce"] for k, v in cand.items()}},
                      "candidates": {"baseline": {"best_epoch": int(base_ep), **r_base_cand},
                                     **cand},
                      "refit": {"baseline": {**r_base, "epochs": int(base_ep)},
                                "aux": {**r_aux, "epochs": int(cand[str(sel)]["best_epoch"])}}})

    base_oof = metrics(y, oof_base)
    aux_oof = metrics(y, oof_aux)
    results = {
        "protocol": {
            "cv": "StratifiedGroupKFold(5, shuffle=True, random_state=0)",
            "source": "test/highdim_deep_features.npz",
            "input": "current single 6-second window features only (858 -> 856 non-constant)",
            "aux_targets_source": "raw 6-second waveform in the same npz (current window only)",
            "aux_target_dim": int(A_all.shape[1]),
            "aux_subject_zscore": "per training subject, stats from that outer training fold "
                                  "(candidate: inner-train rows only; refit: all training rows)",
            "lambda_selection": "inner-val BCE (StratifiedGroupKFold on training subjects); "
                                "test fold never used",
            "test_subject_isolation": True,
            "test_history": False,
            "window_position_schedule_r_t": False,
            "test_subject_calibration": False,
            "causal": True,
        },
        "aux_targets": {"names": aux_names, "dim": int(A_all.shape[1]),
                        "selection": "physiological quantities computed directly from the current "
                                     "6-s raw waveform: 5 log band powers, spectral entropy, "
                                     "3 Hjorth parameters"},
        "lambda_grid": list(LAMBDAS),
        "baseline": {"OOF": base_oof,
                     "fold_mean": {k: float(np.mean([f["refit"]["baseline"][k] for f in folds]))
                                   for k in ("AUC", "bACC")}},
        "aux": {"OOF": aux_oof,
                "fold_mean": {k: float(np.mean([f["refit"]["aux"][k] for f in folds]))
                              for k in ("AUC", "bACC")},
                "selected_lambdas": [f["selected_lambda"] for f in folds]},
        "per_lambda_candidate": {
            str(lam): {"OOF": metrics(y, oof_cand[lam]),
                       "fold_mean": {k: float(np.mean([f["candidates"][str(lam)][k] for f in folds]))
                                     for k in ("AUC", "bACC")}}
            for lam in LAMBDAS},
        "folds": folds,
        "runtime_seconds": float(time.perf_counter() - tick),
    }
    np.savez_compressed(OUT / "aux_subjectnorm_window6_oof.npz", y=y, groups=groups,
                        baseline_oof=oof_base, aux_oof=oof_aux,
                        **{f"cand_l{lam}_oof": oof_cand[lam] for lam in LAMBDAS})
    (OUT / "aux_subjectnorm_window6_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[final] baseline OOF AUC={base_oof['AUC']:.5f} bACC={base_oof['bACC']:.5f} | "
          f"aux OOF AUC={aux_oof['AUC']:.5f} bACC={aux_oof['bACC']:.5f}", flush=True)
    print(f"[done] elapsed={results['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
