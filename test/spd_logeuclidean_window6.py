"""当前 6 秒窗的固定滤波器组与 Log-Euclidean SPD 子窗网络。"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

OUT = Path(__file__).resolve().parent
FS, N_TOKENS, TOKEN_SAMPLES = 250, 6, 250
BATCH_SIZE, EVAL_BATCH_SIZE = 512, 1024
MAX_EPOCHS, PATIENCE = 30, 6
SEEDS = (0, 1)
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
START = time.perf_counter()

class Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, path.open("w", encoding="utf-8")
    def write(self, text): self.stream.write(text); self.file.write(text); self.file.flush()
    def flush(self): self.stream.flush(); self.file.flush()
    def close(self): self.file.close()

def log(message): print(f"[elapsed {time.perf_counter()-START:8.2f}s] {message}", flush=True)

def subject_macro_bacc(y, groups, pred):
    values = [balanced_accuracy_score(y[groups == s], pred[groups == s])
              for s in np.unique(groups) if np.unique(y[groups == s]).size == 2]
    return float(np.mean(values))

def metrics(y, groups, score, pred):
    return {"AUC": float(roc_auc_score(y, score)), "subject_macro_bACC": subject_macro_bacc(y, groups, pred),
            "window_micro_bACC": float(balanced_accuracy_score(y, pred)), "n_eval": int(len(y)),
            "n_subjects": int(np.unique(groups).size)}

def select_threshold(y, groups, score):
    candidates = np.unique(np.r_[0.5, np.quantile(score, np.linspace(.02, .98, 97))])
    best, threshold = -np.inf, .5
    for t in candidates:
        value = subject_macro_bacc(y, groups, score >= t)
        if value > best + 1e-12 or (abs(value-best) <= 1e-12 and abs(t-.5) < abs(threshold-.5)):
            best, threshold = value, float(t)
    return threshold, best

def balanced_weights(labels, groups, indices):
    weights = np.zeros(len(labels), dtype=np.float32)
    for s in np.unique(groups[indices]):
        for y in (0, 1):
            ids = indices[(groups[indices] == s) & (labels[indices] == y)]
            if ids.size: weights[ids] = 1.0 / ids.size
    weights[indices] /= weights[indices].mean()
    return weights

class SPDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(21), nn.Linear(21, 64), nn.GELU(), nn.Dropout(.15), nn.Linear(64, 48), nn.GELU())
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 48), nn.GELU(), nn.Dropout(.2), nn.Linear(48, 1))
    def forward(self, features):
        z = self.encoder(features)
        pooled = torch.cat((z.mean(1), z.std(1, unbiased=False)), dim=1)
        return self.head(pooled).squeeze(1)

@torch.no_grad()
def predict(model, features, indices):
    model.eval(); out = []
    for start in range(0, len(indices), EVAL_BATCH_SIZE):
        ids = indices[start:start+EVAL_BATCH_SIZE]
        with torch.autocast(device_type="cuda", dtype=torch.float16): out.append(torch.sigmoid(model(features[ids])).float().cpu().numpy())
    return np.concatenate(out)

def build_spd_features(raw):
    """固定六带通掩码；协方差/eigh/log 全程 FP32，输入只为当前 1500 点。"""
    freq = torch.fft.rfftfreq(1536, 1 / FS).to(raw.device)
    masks = torch.stack([((freq >= low) & (freq < high)).float() for low, high in BANDS])
    upper = torch.triu_indices(6, 6, device=raw.device)
    eye = torch.eye(6, device=raw.device, dtype=torch.float32)
    chunks = []
    for start in range(0, raw.shape[0], 2048):
        x = raw[start:start + 2048].float(); x = x - x.mean(1, keepdim=True)
        x = x / x.var(1, keepdim=True, unbiased=False).add(1e-6).sqrt(); x = x.clamp(-10, 10)
        fft = torch.fft.rfft(x, n=1536)
        filtered = torch.fft.irfft(fft[:, None, :] * masks[None, :, :], n=1536)[..., :1500]
        filtered = filtered.reshape(x.shape[0], 6, N_TOKENS, TOKEN_SAMPLES).permute(0, 2, 1, 3)
        filtered = filtered - filtered.mean(-1, keepdim=True)
        cov = torch.matmul(filtered, filtered.transpose(-1, -2)) / (TOKEN_SAMPLES - 1) + 1e-4 * eye
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        logcov = torch.matmul(eigenvectors * torch.log(eigenvalues.clamp_min(1e-6)).unsqueeze(-2), eigenvectors.transpose(-1, -2))
        chunks.append(logcov[:, :, upper[0], upper[1]].contiguous())
        del fft, filtered, cov, eigenvalues, eigenvectors, logcov
    return torch.cat(chunks, dim=0)

def build_folds(dense_groups, dense_labels, canonical_y, canonical_groups, canon_idx):
    folds, fold_id = [], np.full(len(canonical_y), -1, dtype=np.int64)
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for fold, (train_outer, test_can) in enumerate(outer.split(canonical_y, canonical_y, canonical_groups), 1):
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=fold)
        inner_train, inner_val = next(inner.split(train_outer, canonical_y[train_outer], canonical_groups[train_outer]))
        train_can, val_can = train_outer[inner_train], train_outer[inner_val]
        train_subjects, val_subjects, test_subjects = (np.unique(canonical_groups[v]) for v in (train_can, val_can, test_can))
        train_dense = np.flatnonzero(np.isin(dense_groups, train_subjects)); val_dense, test_dense = canon_idx[val_can], canon_idx[test_can]
        assert not set(train_subjects)&set(val_subjects) and not set(train_subjects)&set(test_subjects) and not set(val_subjects)&set(test_subjects)
        fold_id[test_can] = fold - 1
        folds.append(dict(fold=fold, train=train_dense, val=val_dense, test=test_dense, val_can=val_can, test_can=test_can,
                          train_subjects=train_subjects, val_subjects=val_subjects, test_subjects=test_subjects))
    assert np.all(fold_id >= 0); return folds, fold_id

def train_fold(features, labels, weights, row, canonical_y, canonical_groups, seed, max_epochs=MAX_EPOCHS):
    torch.manual_seed(1000 * seed + row["fold"]); torch.cuda.manual_seed_all(1000 * seed + row["fold"])
    model, opt = SPDNet().cuda(), torch.optim.AdamW(SPDNet().parameters(), lr=1e-3) if False else None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3); scaler = torch.amp.GradScaler("cuda")
    train_ids, val_ids = torch.as_tensor(row["train"], device="cuda"), torch.as_tensor(row["val"], device="cuda")
    best_auc, best_state, stale, history = -np.inf, None, 0, []; tick = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train(); order = train_ids[torch.randperm(train_ids.numel(), device="cuda")]; total = 0.
        for start in range(0, order.numel(), BATCH_SIZE):
            ids = order[start:start+BATCH_SIZE]; opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = (nn.functional.binary_cross_entropy_with_logits(model(features[ids]), labels[ids], reduction="none") * weights[ids]).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.); scaler.step(opt); scaler.update(); total += float(loss.detach()) * ids.numel()
        val_score = predict(model, features, val_ids); auc = float(roc_auc_score(canonical_y, val_score)); macro = subject_macro_bacc(canonical_y, canonical_groups, val_score >= .5)
        rowh = {"epoch": epoch, "train_loss": total/len(row["train"]), "val_AUC": auc, "val_macro_bACC_at_0.5": macro, "gpu_max_memory_GiB": torch.cuda.max_memory_allocated()/1024**3}; history.append(rowh)
        log(f"fold {row['fold']}/5 seed {seed} epoch {epoch:02d}/{max_epochs} loss={rowh['train_loss']:.5f} valAUC={auc:.5f} valMacro={macro:.5f} GPUmax={rowh['gpu_max_memory_GiB']:.2f}GiB")
        if auc > best_auc + 1e-4: best_auc, best_state, stale, best_epoch = auc, copy.deepcopy(model.state_dict()), 0, epoch
        else: stale += 1
        if stale >= PATIENCE: break
    model.load_state_dict(best_state)
    return dict(val_score=predict(model, features, val_ids), test_score=predict(model, features, torch.as_tensor(row["test"], device="cuda")), best_val_AUC=float(best_auc), best_epoch=best_epoch, epochs_run=epoch, history=history, seconds=time.perf_counter()-tick, parameters=sum(p.numel() for p in model.parameters()))

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args = parser.parse_args()
    result_path, oof_path = OUT/"spd_logeuclidean_window6_results.json", OUT/"spd_logeuclidean_window6_oof.npz"
    if not args.smoke and (result_path.exists() or oof_path.exists()): raise FileExistsError("拒绝覆盖已有 SPD 正式结果")
    if not torch.cuda.is_available(): raise RuntimeError("本实验要求 CUDA GPU")
    tee = Tee(sys.stdout, OUT/"spd_logeuclidean_window6.log"); sys.stdout = tee
    try:
        torch.set_float32_matmul_precision("high"); cache = np.load(OUT/"dense_window_protocol_cache.npz"); dense = np.load(OUT/"dense_window_protocol_raw6.npy", mmap_mode="r"); high = np.load(OUT/"highdim_deep_features.npz")
        y, groups, canon_idx = high["y"].astype(np.int64), high["groups"], np.flatnonzero(cache["canon"]); dense_y, dense_groups = cache["label"].astype(np.int64), cache["subject"]
        assert np.array_equal(dense[canon_idx], high["raw"]) and np.array_equal(dense_y[canon_idx], y) and np.array_equal(dense_groups[canon_idx], groups)
        raw = torch.from_numpy(np.array(dense, copy=True)).cuda(); labels = torch.from_numpy(dense_y.astype(np.float32)).cuda(); log(f"GPU={torch.cuda.get_device_name(0)} raw={tuple(raw.shape)} batch={BATCH_SIZE} AMP=True")
        log("precomputing fixed six-band SPD log features (FP32 eig/log)"); features = build_spd_features(raw); assert torch.isfinite(features).all(); log(f"features={tuple(features.shape)} finite=True GPUmem={torch.cuda.max_memory_allocated()/1024**3:.2f}GiB")
        folds, fold_id = build_folds(dense_groups, dense_y, y, groups, canon_idx)
        if args.smoke:
            row = folds[0]; w = torch.from_numpy(balanced_weights(dense_y, dense_groups, row["train"])).cuda(); run = train_fold(features, labels, w, row, y[row["val_can"]], groups[row["val_can"]], 0, 1); log(f"smoke passed parameters={run['parameters']} valAUC={run['best_val_AUC']:.6f}"); return
        runs = {}
        for row in folds:
            w = torch.from_numpy(balanced_weights(dense_y, dense_groups, row["train"])).cuda()
            for seed in SEEDS:
                log(f"start fold {row['fold']}/5 seed {seed}: train={len(row['train'])} val={len(row['val'])} test={len(row['test'])}"); runs[(row['fold'], seed)] = train_fold(features, labels, w, row, y[row["val_can"]], groups[row["val_can"]], seed)
        seed_scores, seed_metrics, val_scores = {}, {}, {}
        for seed in SEEDS:
            score = np.full(len(y), np.nan); pred = np.zeros(len(y), np.int8)
            for row in folds:
                run = runs[(row["fold"], seed)]; score[row["test_can"]] = run["test_score"]; val_scores[(row["fold"], seed)] = run["val_score"]; t, _ = select_threshold(y[row["val_can"]], groups[row["val_can"]], run["val_score"]); pred[row["test_can"]] = run["test_score"] >= t
            seed_scores[seed], seed_metrics[str(seed)] = score, metrics(y, groups, score, pred)
        mean_score = np.mean([seed_scores[s] for s in SEEDS], axis=0); mean_pred = np.zeros(len(y), np.int8); thresholds=[]; folds_out=[]
        for row in folds:
            val = np.mean([val_scores[(row["fold"], s)] for s in SEEDS], axis=0); t, vm = select_threshold(y[row["val_can"]], groups[row["val_can"]], val); mean_pred[row["test_can"]] = mean_score[row["test_can"]] >= t; thresholds.append(t)
            folds_out.append({"fold":row["fold"], "threshold":t, "val_subject_macro_bACC":vm, "test":metrics(y[row["test_can"]], groups[row["test_can"]], mean_score[row["test_can"]], mean_pred[row["test_can"]]), "train_subjects":[str(v) for v in row["train_subjects"]], "val_subjects":[str(v) for v in row["val_subjects"]], "test_subjects":[str(v) for v in row["test_subjects"]]})
        overall = metrics(y, groups, mean_score, mean_pred); log(f"seed-mean OOF AUC={overall['AUC']:.6f} macro={overall['subject_macro_bACC']:.6f} micro={overall['window_micro_bACC']:.6f}")
        np.savez_compressed(oof_path, y=y, groups=groups, fold_id=fold_id, score_mean=mean_score, pred_mean=mean_pred, thresholds=np.asarray(thresholds), **{f"score_seed{s}":seed_scores[s] for s in SEEDS})
        result = {"protocol":{"input":"current 1500 points / 6 s only; six 1 s subwindows; no cross-window history","outer_cv":"StratifiedGroupKFold(5,shuffle=True,random_state=0) canonical","inner_cv":"first StratifiedGroupKFold(4,shuffle=True,random_state=outer_fold)","train_windows":"dense 1 s-step windows from inner-training subjects only","val_test_windows":"canonical windows only","early_stopping":"inner-validation AUC; test labels unused","threshold":"inner-validation subject-macro bACC","loss":"equal total weight per subject-class cell","seeds":list(SEEDS),"test_labels_used_for_training_or_selection":False},"model":{"architecture":"six fixed FFT band-pass pseudo-channels; regularized 6x6 covariance; FP32 Log-Euclidean SPD upper triangle; shared MLP; mean+std pooling","parameters":runs[(1,0)]["parameters"],"batch_size":BATCH_SIZE,"mixed_precision":"FP16 AMP; eig/log FP32","feature_shape":list(features.shape[1:])},"data":{"dense_shape":list(dense.shape),"canonical_shape":list(high["raw"].shape),"canonical_raw_label_group_exact_match":True,"features_finite":bool(torch.isfinite(features).all())},"seed_metrics":seed_metrics,"seed_mean_metrics":overall,"folds":folds_out,"training_runs":{f"fold{f}_seed{s}":{k:v for k,v in r.items() if k not in ("val_score","test_score")} for (f,s),r in runs.items()},"gpu":{"name":torch.cuda.get_device_name(0),"torch":torch.__version__,"peak_memory_GiB":torch.cuda.max_memory_allocated()/1024**3},"seconds":time.perf_counter()-START}
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); log(f"done seconds={result['seconds']:.2f}")
    finally: sys.stdout = tee.stream; tee.close()
if __name__ == "__main__": main()
