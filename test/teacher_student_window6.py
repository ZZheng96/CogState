"""严格受试者隔离的 teacher-student 6 秒窗实验。
Teacher 训练时可使用训练受试者的长时段聚合上下文；Student 训练和推理只接收当前窗的 856 维特征。
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
        return torch.cat([torch.sigmoid(model(z[i:i + 512].to(DEVICE))).cpu() for i in range(0, len(X), 512)]).numpy()


def metrics(y, p):
    return {"AUC": float(roc_auc_score(y, p)),
            "bACC": float(balanced_accuracy_score(y, p >= 0.5))}


def subject_context(X, groups):
    """Only called with the current outer training fold."""
    result = np.empty_like(X, dtype="float32")
    for subject in np.unique(groups):
        mask = groups == subject
        result[mask] = X[mask].mean(axis=0, keepdims=True)
    return result


def fit_teacher(X, context, y, inner_train, inner_val, fold):
    model = MLP(X.shape[1] + context.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    posw = torch.tensor(float((y[inner_train] == 0).sum() / max((y[inner_train] == 1).sum(), 1)), device=DEVICE)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=posw)
    rng = np.random.default_rng(SEED + fold)
    best, state, stale, best_epoch = float("inf"), None, 0, 0
    train_x = np.hstack([X[inner_train], context[inner_train]])
    val_x = np.hstack([X[inner_val], context[inner_val]])
    xt = torch.from_numpy(train_x).to(DEVICE); yt = torch.from_numpy(y[inner_train].astype("float32")).to(DEVICE)
    for epoch in range(1, 41):
        model.train()
        order = rng.permutation(len(xt)); losses = []
        for start in range(0, len(order), 256):
            ix = order[start:start + 256]
            opt.zero_grad(); loss = lossfn(model(xt[ix]), yt[ix]); loss.backward(); opt.step(); losses.append(float(loss))
        p = np.clip(predict(model, val_x), 1e-6, 1 - 1e-6)
        val_loss = float(-np.mean(y[inner_val] * np.log(p) + (1 - y[inner_val]) * np.log(1 - p)))
        if val_loss < best - 1e-5:
            best, state, stale, best_epoch = val_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[fold {fold}/5] teacher epoch {epoch}/40 val_loss={val_loss:.5f}", flush=True)
        if stale >= 7:
            break
    model.load_state_dict(state)
    return model, best_epoch


def fit_student(X, y, soft, inner_train, inner_val, fold):
    model = MLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    posw = torch.tensor(float((y[inner_train] == 0).sum() / max((y[inner_train] == 1).sum(), 1)), device=DEVICE)
    bce = nn.BCEWithLogitsLoss(pos_weight=posw)
    rng = np.random.default_rng(SEED + 100 + fold)
    best, state, stale, best_epoch = float("inf"), None, 0, 0
    xt = torch.from_numpy(X[inner_train]).to(DEVICE); yt = torch.from_numpy(y[inner_train].astype("float32")).to(DEVICE)
    st = torch.from_numpy(soft[inner_train].astype("float32")).to(DEVICE)
    for epoch in range(1, 61):
        model.train(); order = rng.permutation(len(xt)); losses = []
        for start in range(0, len(order), 256):
            ix = order[start:start + 256]; opt.zero_grad()
            logits = model(xt[ix]); loss = 0.65 * bce(logits, yt[ix]) + 0.35 * nn.functional.mse_loss(torch.sigmoid(logits), st[ix])
            loss.backward(); opt.step(); losses.append(float(loss))
        p = np.clip(predict(model, X[inner_val]), 1e-6, 1 - 1e-6)
        val_loss = float(-np.mean(y[inner_val] * np.log(p) + (1 - y[inner_val]) * np.log(1 - p)))
        if val_loss < best - 1e-5:
            best, state, stale, best_epoch = val_loss, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"[fold {fold}/5] student epoch {epoch}/60 val_loss={val_loss:.5f}", flush=True)
        if stale >= 8:
            break
    model.load_state_dict(state)
    return model, best_epoch


def main():
    tick = time.perf_counter(); np.random.seed(SEED); torch.manual_seed(SEED)
    data = np.load(OUT / "highdim_deep_features.npz")
    X0 = data["X"].astype("float32"); y = data["y"].astype(int); groups = data["groups"]
    if X0.shape[1] != 858:
        raise ValueError(f"expected 858 source features, got {X0.shape[1]}")
    oof = np.full(len(y), np.nan, dtype="float32"); folds = []
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    print(f"[data] X={X0.shape}; student_input=856; subjects={len(np.unique(groups))}; device={DEVICE}", flush=True)
    for fold, (tr, te) in enumerate(cv.split(X0, y, groups), 1):
        assert not (set(groups[tr]) & set(groups[te]))
        keep = X0[tr].std(axis=0) > 1e-8
        if int(keep.sum()) != 856:
            raise ValueError(f"fold {fold}: expected 856 nonconstant features, got {int(keep.sum())}")
        mu = X0[tr][:, keep].mean(axis=0); sd = np.maximum(X0[tr][:, keep].std(axis=0), 1e-6)
        X = ((X0[:, keep] - mu) / sd).astype("float32")
        inner_cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED + fold)
        it, iv = next(inner_cv.split(X[tr], y[tr], groups[tr]))
        context = subject_context(X[tr], groups[tr])
        teacher, teacher_epoch = fit_teacher(X[tr], context, y[tr], it, iv, fold)
        teacher.eval()
        teacher_input = np.hstack([X[tr], context]).astype("float32")
        teacher_soft = predict(teacher, teacher_input)
        student, student_epoch = fit_student(X[tr], y[tr], teacher_soft, it, iv, fold)
        p = predict(student, X[te]); oof[te] = p
        r = metrics(y[te], p); row = {"fold": fold, **r, "n_test": int(len(te)), "train_subjects": int(len(np.unique(groups[tr]))), "test_subjects": int(len(np.unique(groups[te]))), "teacher_best_epoch": int(teacher_epoch), "student_best_epoch": int(student_epoch)}
        folds.append(row); print(f"[result] fold={fold}/5 AUC={r['AUC']:.5f} bACC={r['bACC']:.5f}", flush=True)
    final = metrics(y, oof)
    results = {"protocol": {"cv": "StratifiedGroupKFold(5, shuffle=True, random_state=0)", "source": "highdim_deep_features.npz", "source_features": 858, "student_input_features": 856, "teacher_training": "training-fold labels plus per-training-subject long-window mean context", "student_training": "BCE labels + 0.35 sigmoid MSE distillation target", "student_inference": "current single 6-second 856-d feature window only", "test_subject_isolation": True, "test_history": False, "window_position_schedule_r_t": False, "test_subject_calibration": False}, "oof": final, "fold_mean": {"AUC": float(np.mean([r['AUC'] for r in folds])), "bACC": float(np.mean([r['bACC'] for r in folds]))}, "folds": folds, "runtime_seconds": float(time.perf_counter() - tick)}
    np.savez_compressed(OUT / "teacher_student_window6_oof.npz", y=y, groups=groups, student_oof=oof)
    (OUT / "teacher_student_window6_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[final] OOF AUC={final['AUC']:.5f} bACC={final['bACC']:.5f}; fold_mean AUC={results['fold_mean']['AUC']:.5f} bACC={results['fold_mean']['bACC']:.5f}", flush=True)
    print(f"[done] elapsed={results['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
