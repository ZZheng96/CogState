"""当前 6 秒单通道 EEG 的多频带瞬时相位-幅值耦合实验。"""
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
FS = 250
N = 1500
SUB = 250
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
SEEDS = (0, 1)
BATCH = 2048
EVAL_BATCH = 8192
MAX_EPOCHS = 30
PATIENCE = 6
START = time.perf_counter()


class Tee:
    def __init__(self, stream, path):
        self.stream, self.file = stream, path.open("w", encoding="utf-8")
    def write(self, text):
        self.stream.write(text); self.file.write(text); self.file.flush()
    def flush(self):
        self.stream.flush(); self.file.flush()
    def close(self): self.file.close()


def log(msg): print(f"[elapsed {time.perf_counter()-START:8.2f}s] {msg}", flush=True)


def extract_features(raw):
    """GPU-only batched FFT analytic signals; returns 112 fixed statistics."""
    x = raw.float() - raw.float().mean(1, keepdim=True)
    spec = torch.fft.fft(x, n=N, dim=1)
    freq = torch.fft.fftfreq(N, d=1.0 / FS, device=x.device)
    pos = freq >= 0
    feats = []
    analytic = []
    for low, high in BANDS:
        mask = ((freq >= low) & (freq < high)) | ((freq <= -low) & (freq > -high))
        posmask = mask & pos
        a_spec = torch.where(posmask, spec * 2.0, torch.zeros_like(spec))
        a_spec[:, 0] = 0
        analytic.append(torch.fft.ifft(a_spec, dim=1))
    z = torch.stack(analytic, dim=1)
    amp = z.abs().clamp_min(1e-8)
    unit = z / amp
    metrics = []
    # Six per-band metrics: log mean amplitude, amplitude CV, phase resultant.
    for b in range(len(BANDS)):
        a, u = amp[:, b], unit[:, b]
        aa, uu = a.reshape(-1, 6, SUB), u.reshape(-1, 6, SUB)
        mean_amp = aa.mean(-1)
        metrics.extend((torch.log(mean_amp.clamp_min(1e-8)),
                        aa.std(-1, unbiased=False) / mean_amp.clamp_min(1e-8),
                        uu.mean(-1).abs()))
    # Adjacent-band phase-difference resultant and envelope correlation.
    for b in range(len(BANDS) - 1):
        aa, bb = amp[:, b].reshape(-1, 6, SUB), amp[:, b + 1].reshape(-1, 6, SUB)
        uu, vv = unit[:, b].reshape(-1, 6, SUB), unit[:, b + 1].reshape(-1, 6, SUB)
        metrics.append((uu * vv.conj()).mean(-1).abs())
        ac, bc = aa - aa.mean(-1, keepdim=True), bb - bb.mean(-1, keepdim=True)
        metrics.append((ac * bc).mean(-1) / (ac.square().mean(-1).sqrt() * bc.square().mean(-1).sqrt()).clamp_min(1e-8))
    per_sec = torch.stack(metrics, dim=2)
    t = torch.arange(6, device=raw.device, dtype=per_sec.dtype) - 2.5
    slope = (per_sec * t.view(1, 6, 1)).mean(1) / (t.square().mean() + 1e-8)
    summary = torch.cat((per_sec.mean(1), per_sec.std(1, unbiased=False), slope), dim=1)
    full = torch.stack([m.reshape(m.shape[0], -1).mean(1) for m in metrics], dim=1)
    out = torch.cat((summary, full), dim=1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def make_features(raw):
    chunks = []
    for start in range(0, len(raw), BATCH):
        chunks.append(extract_features(raw[start:start+BATCH]))
    return torch.cat(chunks, 0)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                                 nn.Dropout(0.25), nn.Linear(256, 128), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(self, x): return self.net(x).squeeze(1)


def macro_bacc(y, groups, pred):
    vals = [balanced_accuracy_score(y[groups == s], pred[groups == s])
            for s in np.unique(groups) if np.unique(y[groups == s]).size == 2]
    return float(np.mean(vals))


def select_threshold(y, groups, score):
    best, threshold = -np.inf, 0.5
    for q in np.unique(np.r_[0.5, np.quantile(score, np.linspace(.02, .98, 97))]):
        v = macro_bacc(y, groups, score >= q)
        if v > best + 1e-12 or (abs(v-best) <= 1e-12 and abs(q-.5) < abs(threshold-.5)):
            best, threshold = v, float(q)
    return threshold, best


def weights(y, groups, idx):
    w = np.zeros(len(y), np.float32)
    for s in np.unique(groups[idx]):
        for c in (0, 1):
            z = idx[(groups[idx] == s) & (y[idx] == c)]
            if len(z): w[z] = 1.0 / len(z)
    w[idx] /= w[idx].mean()
    return w


@torch.no_grad()
def predict(model, x, idx):
    model.eval(); out = []
    for i in range(0, len(idx), EVAL_BATCH):
        with torch.autocast("cuda", dtype=torch.float16): out.append(torch.sigmoid(model(x[idx[i:i+EVAL_BATCH]])).float().cpu())
    return torch.cat(out).numpy()


def train_fold(x, y, wt, train, val, test, val_y, val_groups, seed, fold, epochs=MAX_EPOCHS):
    torch.manual_seed(1000 * seed + fold); torch.cuda.manual_seed_all(1000 * seed + fold)
    model = MLP(x.shape[1]).cuda(); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scaler = torch.amp.GradScaler("cuda"); order_base = torch.as_tensor(train, device="cuda")
    best_auc, best_state, stale, history = -np.inf, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train(); order = order_base[torch.randperm(len(order_base), device="cuda")]; total = 0.0
        for i in range(0, len(order), BATCH):
            ids = order[i:i+BATCH]; opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss_each = nn.functional.binary_cross_entropy_with_logits(model(x[ids]), y[ids], reduction="none")
                loss = (loss_each * wt[ids]).mean()
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); total += float(loss.detach()) * len(ids)
        score = predict(model, x, val); auc = float(roc_auc_score(val_y, score)); macro = macro_bacc(val_y, val_groups, score >= .5)
        row = {"epoch": epoch, "train_loss": total/len(train), "val_AUC": auc, "val_macro_bACC_at_0.5": macro,
               "gpu_max_memory_GiB": torch.cuda.max_memory_allocated()/1024**3}; history.append(row)
        log(f"fold {fold}/5 seed {seed} epoch {epoch:02d}/{epochs} loss={row['train_loss']:.5f} valAUC={auc:.5f} valMacro={macro:.5f} GPUmax={row['gpu_max_memory_GiB']:.2f}GiB")
        if auc > best_auc + 1e-4: best_auc, best_state, stale = auc, copy.deepcopy(model.state_dict()), 0
        else: stale += 1
        if stale >= PATIENCE: break
    model.load_state_dict(best_state)
    return {"val_score": predict(model, x, val), "test_score": predict(model, x, test), "best_val_AUC": best_auc,
            "best_epoch": epoch, "history": history, "parameters": sum(p.numel() for p in model.parameters())}


def build_folds(dense_groups, dense_labels, cy, cg, canon_idx):
    folds=[]; outer=StratifiedGroupKFold(5, shuffle=True, random_state=0); fold_id=np.full(len(cy), -1)
    for fold, (tr, te) in enumerate(outer.split(cy, cy, cg), 1):
        inner=StratifiedGroupKFold(4, shuffle=True, random_state=fold); a,b=next(inner.split(tr, cy[tr], cg[tr]))
        train_can, val_can, train_sub = tr[a], tr[b], np.unique(cg[tr[a]])
        val_sub, test_sub = np.unique(cg[tr[b]]), np.unique(cg[te]); assert not (set(train_sub)&set(val_sub)|set(train_sub)&set(test_sub)|set(val_sub)&set(test_sub))
        train=np.flatnonzero(np.isin(dense_groups, train_sub)); val=canon_idx[val_can]; test=canon_idx[te]
        assert not set(dense_groups[train]) & set(dense_groups[test]); fold_id[te]=fold-1
        folds.append(dict(fold=fold, train=train, val=val, test=test, val_can=val_can, test_can=te, train_subjects=train_sub, val_subjects=val_sub, test_subjects=test_sub))
    return folds, fold_id


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); args=ap.parse_args()
    result_path=OUT/"phase_amplitude_window6_results.json"; oof_path=OUT/"phase_amplitude_window6_oof.npz"
    if not torch.cuda.is_available(): raise RuntimeError("本实验要求 CUDA GPU")
    if not args.smoke and (result_path.exists() or oof_path.exists()): raise FileExistsError("拒绝覆盖已有正式结果")
    original_stdout=sys.stdout; tee=Tee(original_stdout, OUT/"phase_amplitude_window6.log"); sys.stdout=tee
    try:
        cache=np.load(OUT/"dense_window_protocol_cache.npz", allow_pickle=False); dense=np.load(OUT/"dense_window_protocol_raw6.npy", mmap_mode="r"); hi=np.load(OUT/"highdim_deep_features.npz", allow_pickle=False)
        cy=hi["y"].astype(np.int64); cg=hi["groups"]; canon_idx=np.flatnonzero(cache["canon"]); dy=cache["label"].astype(np.int64); dg=cache["subject"]
        assert np.array_equal(dense[canon_idx], hi["raw"]); assert np.array_equal(dy[canon_idx], cy); assert np.array_equal(dg[canon_idx], cg)
        raw=torch.from_numpy(np.array(dense, copy=True)).cuda(); log(f"GPU={torch.cuda.get_device_name(0)} raw={tuple(raw.shape)}")
        x=make_features(raw); assert x.shape[1] <= 200 and torch.isfinite(x).all(); log(f"features shape={tuple(x.shape)} finite=True GPU={torch.cuda.get_device_name(0)}")
        folds, fold_id=build_folds(dg,dy,cy,cg,canon_idx)
        if args.smoke:
            row=folds[0]; wt=torch.from_numpy(weights(dy,dg,row["train"])).cuda(); run=train_fold(x,torch.from_numpy(dy.astype(np.float32)).cuda(),wt,row["train"],row["val"],row["test"],cy[row["val_can"]],cg[row["val_can"]],0,1,1); log(f"smoke passed parameters={run['parameters']} valAUC={run['best_val_AUC']:.6f}"); return
        y=torch.from_numpy(dy.astype(np.float32)).cuda(); runs={}
        for row in folds:
            wt=torch.from_numpy(weights(dy,dg,row["train"])).cuda()
            for seed in SEEDS:
                log(f"start fold {row['fold']}/5 seed {seed} train={len(row['train'])} val={len(row['val'])} test={len(row['test'])}")
                run=train_fold(x,y,wt,row["train"],row["val"],row["test"],cy[row["val_can"]],cg[row["val_can"]],seed,row["fold"])
                runs[(row["fold"],seed)]=run
        scores={}; preds=np.zeros(len(cy), np.int8); fold_summaries=[]
        for seed in SEEDS:
            score=np.full(len(cy), np.nan)
            for row in folds: score[row["test_can"]]=runs[(row["fold"],seed)]["test_score"]
            scores[seed]=score
        mean_score=np.mean([scores[s] for s in SEEDS], axis=0)
        for row in folds:
            val_score=np.mean([runs[(row["fold"],s)]["val_score"] for s in SEEDS], axis=0)
            thr, val_macro=select_threshold(cy[row["val_can"]],cg[row["val_can"]],val_score)
            preds[row["test_can"]]=(mean_score[row["test_can"]]>=thr)
            fold_summaries.append({"fold":row["fold"],"threshold":thr,"val_subject_macro_bACC":val_macro})
        overall={"AUC":float(roc_auc_score(cy,mean_score)),"subject_macro_bACC":macro_bacc(cy,cg,preds),"window_micro_bACC":float(balanced_accuracy_score(cy,preds)),"n_eval":len(cy),"n_subjects":len(np.unique(cg))}
        np.savez_compressed(oof_path,y=cy,groups=cg,fold_id=fold_id,score_mean=mean_score,pred_mean=preds,score_seed0=scores[0],score_seed1=scores[1])
        result={"protocol":{"input":"current 6 s / 1500 points only","feature_dim":int(x.shape[1]),"bands":BANDS,"outer":"StratifiedGroupKFold(5, shuffle=True, random_state=0)","inner":"first StratifiedGroupKFold(4, shuffle=True, random_state=fold)","seeds":SEEDS,"amp":True,"test_labels_used":False},"overall":overall,"folds":fold_summaries,"parameters":runs[(1,0)]["parameters"]}
        result_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        log(f"seed-mean OOF macro={overall['subject_macro_bACC']:.6f} micro={overall['window_micro_bACC']:.6f} AUC={overall['AUC']:.6f}")
    finally:
        sys.stdout=original_stdout; tee.close()


if __name__ == "__main__": main()
