"""当前 6 秒内 raw+谱 token 的单向 GRU，严格被试外推与 GPU 训练。"""
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
N_TOKENS = 6
TOKEN_SAMPLES = 250
BATCH_SIZE = 1024
EVAL_BATCH_SIZE = 2048
MAX_EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
SEEDS = (0, 1)
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (52, 100))
START = time.perf_counter()


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = path.open("w", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def log(message):
    print(f"[elapsed {time.perf_counter() - START:8.2f}s] {message}", flush=True)


class TokenGRU(nn.Module):
    """每个 token 仅含 1 秒；单向 GRU 的最后状态总感受野恰为 6 秒。"""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, 15, stride=3, padding=7, bias=False),
            nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv1d(32, 64, 11, stride=3, padding=5, bias=False),
            nn.GroupNorm(4, 64), nn.GELU(),
            nn.Conv1d(64, 96, 9, stride=2, padding=4, bias=False),
            nn.GroupNorm(4, 96), nn.GELU(),
        )
        self.project = nn.Linear(192, 96)
        self.token_norm = nn.LayerNorm(114)
        self.gru = nn.GRU(114, 128, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Dropout(0.3), nn.Linear(128, 1))

    def spectral_features(self, tokens):
        x = tokens.reshape(-1, TOKEN_SAMPLES).float()
        power = torch.fft.rfft(x, n=256).abs().square() / 256.0
        freq = torch.linspace(0.0, FS / 2.0, power.shape[1], device=x.device)
        positive = power[:, 1:]
        total = positive.sum(dim=1, keepdim=True).clamp_min(1e-12)
        features = []
        for low, high in BANDS:
            band = power[:, (freq >= low) & (freq < high)].sum(dim=1, keepdim=True)
            features.extend((torch.log(band + 1e-12), band / total))
        probability = positive / total
        entropy = -(probability * torch.log(probability + 1e-12)).sum(
            dim=1, keepdim=True) / np.log(positive.shape[1])
        flatness = torch.exp(torch.log(positive + 1e-12).mean(
            dim=1, keepdim=True)) / positive.mean(dim=1, keepdim=True).clamp_min(1e-12)
        centroid = (positive * freq[1:]).sum(dim=1, keepdim=True) / total / (FS / 2.0)
        features.extend((torch.log(total), entropy, flatness, centroid))
        return torch.cat(features, dim=1)

    def forward(self, raw):
        batch = raw.shape[0]
        x = raw.float()
        x = x - x.mean(dim=1, keepdim=True)
        x = x / x.var(dim=1, keepdim=True, unbiased=False).add(1e-6).sqrt()
        x = x.clamp(-10.0, 10.0)
        if self.training:
            scale = torch.exp(torch.empty(batch, 1, device=x.device).uniform_(
                np.log(0.8), np.log(1.25)))
            x = x * scale + 0.05 * torch.randn_like(x)
        tokens = x.reshape(batch * N_TOKENS, 1, TOKEN_SAMPLES)
        encoded = self.encoder(tokens)
        pooled = torch.cat((encoded.mean(dim=2), encoded.amax(dim=2)), dim=1)
        embedding = self.project(pooled)
        spectral = self.spectral_features(tokens).to(embedding.dtype)
        sequence = self.token_norm(torch.cat((embedding, spectral), dim=1)).reshape(
            batch, N_TOKENS, -1)
        output, _ = self.gru(sequence)
        return self.head(output[:, -1]).squeeze(1)


def subject_macro_bacc(y, groups, pred):
    values = []
    for subject in np.unique(groups):
        selected = groups == subject
        if np.unique(y[selected]).size == 2:
            values.append(balanced_accuracy_score(y[selected], pred[selected]))
    return float(np.mean(values))


def metrics(y, groups, score, pred):
    return {
        "AUC": float(roc_auc_score(y, score)),
        "subject_macro_bACC": subject_macro_bacc(y, groups, pred),
        "window_micro_bACC": float(balanced_accuracy_score(y, pred)),
        "n_eval": int(len(y)),
        "n_subjects": int(np.unique(groups).size),
    }


def select_threshold(y, groups, score):
    """仅用内层验证被试选择 macro-bACC 阈值。"""
    candidates = np.unique(np.r_[0.5, np.quantile(score, np.linspace(0.02, 0.98, 97))])
    best_value, best_threshold = -np.inf, np.nan
    for threshold in candidates:
        value = subject_macro_bacc(y, groups, score >= threshold)
        if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_value, best_threshold = value, float(threshold)
    return best_threshold, best_value


def paired_bootstrap(y, groups, baseline_pred, variant_pred, n_boot=2000):
    subjects = np.unique(groups)
    baseline, variant = [], []
    for subject in subjects:
        selected = groups == subject
        if np.unique(y[selected]).size == 2:
            baseline.append(balanced_accuracy_score(y[selected], baseline_pred[selected]))
            variant.append(balanced_accuracy_score(y[selected], variant_pred[selected]))
    delta = np.asarray(variant) - np.asarray(baseline)
    rng = np.random.default_rng(0)
    sampled = delta[rng.integers(0, delta.size, size=(n_boot, delta.size))].mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(v) for v in np.percentile(sampled, (2.5, 97.5))],
        "n_subjects": int(delta.size),
        "n_bootstrap": n_boot,
    }


@torch.no_grad()
def predict(model, raw, indices):
    model.eval()
    output = []
    for start in range(0, len(indices), EVAL_BATCH_SIZE):
        ids = indices[start:start + EVAL_BATCH_SIZE]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(raw[ids])
        output.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(output)


def balanced_training_weights(labels, groups, indices):
    weights = np.zeros(len(labels), dtype=np.float32)
    for subject in np.unique(groups[indices]):
        for label in (0, 1):
            selected = indices[(groups[indices] == subject) & (labels[indices] == label)]
            if selected.size:
                weights[selected] = 1.0 / selected.size
    weights[indices] /= weights[indices].mean()
    return weights


def train_fold(raw, labels, sample_weights, train_idx, val_idx, test_idx, val_y, val_groups,
               seed, fold, max_epochs=MAX_EPOCHS):
    torch.manual_seed(1000 * seed + fold)
    torch.cuda.manual_seed_all(1000 * seed + fold)
    model = TokenGRU().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")
    train_gpu = torch.as_tensor(train_idx, device="cuda")
    val_gpu = torch.as_tensor(val_idx, device="cuda")
    test_gpu = torch.as_tensor(test_idx, device="cuda")
    best_auc, best_state, best_epoch, stale = -np.inf, None, 0, 0
    history = []
    tick = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = train_gpu[torch.randperm(train_gpu.numel(), device="cuda")]
        total_loss = 0.0
        for start in range(0, order.numel(), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(raw[ids])
                losses = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[ids], reduction="none")
                loss = (losses * sample_weights[ids]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * ids.numel()
        val_score = predict(model, raw, val_gpu)
        val_auc = float(roc_auc_score(val_y, val_score))
        val_macro = subject_macro_bacc(val_y, val_groups, val_score >= 0.5)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / train_idx.size,
            "val_AUC": val_auc,
            "val_macro_bACC_at_0.5": val_macro,
            "gpu_max_memory_GiB": torch.cuda.max_memory_allocated() / 1024 ** 3,
        }
        history.append(row)
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        log(f"fold {fold}/5 seed {seed} epoch {epoch:02d}/{max_epochs} "
            f"loss={row['train_loss']:.5f} valAUC={val_auc:.5f} "
            f"valMacro@0.5={val_macro:.5f} best={best_auc:.5f}@{best_epoch} "
            f"GPUmax={row['gpu_max_memory_GiB']:.2f}GiB")
        if stale >= PATIENCE:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    val_score = predict(model, raw, val_gpu)
    test_score = predict(model, raw, test_gpu)
    return {
        "val_score": val_score,
        "test_score": test_score,
        "best_val_AUC": float(best_auc),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "history": history,
        "seconds": time.perf_counter() - tick,
        "parameters": int(sum(p.numel() for p in model.parameters())),
    }


def build_folds(dense_groups, dense_labels, canonical_y, canonical_groups, canon_idx):
    folds = []
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    fold_id = np.full(len(canonical_y), -1, dtype=np.int64)
    for fold, (train_outer, test_can) in enumerate(outer.split(
            canonical_y, canonical_y, canonical_groups), 1):
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=fold)
        inner_train, inner_val = next(inner.split(
            train_outer, canonical_y[train_outer], canonical_groups[train_outer]))
        train_can = train_outer[inner_train]
        val_can = train_outer[inner_val]
        train_subjects = np.unique(canonical_groups[train_can])
        val_subjects = np.unique(canonical_groups[val_can])
        test_subjects = np.unique(canonical_groups[test_can])
        assert not set(train_subjects) & set(val_subjects)
        assert not set(train_subjects) & set(test_subjects)
        assert not set(val_subjects) & set(test_subjects)
        train_dense = np.flatnonzero(np.isin(dense_groups, train_subjects))
        val_dense = canon_idx[val_can]
        test_dense = canon_idx[test_can]
        assert not set(dense_groups[train_dense]) & set(dense_groups[test_dense])
        fold_id[test_can] = fold - 1
        folds.append({
            "fold": fold,
            "train": train_dense,
            "val": val_dense,
            "test": test_dense,
            "train_can": train_can,
            "val_can": val_can,
            "test_can": test_can,
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
        })
    assert np.all(fold_id >= 0)
    return folds, fold_id


def evaluate_oof(runs, folds, canonical_y, canonical_groups):
    seed_scores = {}
    seed_metrics = {}
    val_by_fold = {}
    for seed in SEEDS:
        oof = np.full(len(canonical_y), np.nan, dtype=np.float64)
        pred = np.zeros(len(canonical_y), dtype=np.int8)
        for row in folds:
            fold, seed_key = row["fold"], (row["fold"], seed)
            run = runs[seed_key]
            val_can, test_can = row["val_can"], row["test_can"]
            oof[test_can] = run["test_score"]
            val_by_fold[(fold, seed)] = run["val_score"]
            threshold, _ = select_threshold(
                canonical_y[val_can], canonical_groups[val_can], run["val_score"])
            pred[test_can] = (run["test_score"] >= threshold).astype(np.int8)
        assert np.all(np.isfinite(oof))
        seed_scores[seed] = oof
        seed_metrics[str(seed)] = metrics(canonical_y, canonical_groups, oof, pred)

    mean_score = np.mean([seed_scores[seed] for seed in SEEDS], axis=0)
    mean_pred = np.zeros(len(canonical_y), dtype=np.int8)
    fold_summaries = []
    thresholds = []
    for row in folds:
        fold = row["fold"]
        val_can, test_can = row["val_can"], row["test_can"]
        val_score = np.mean([val_by_fold[(fold, seed)] for seed in SEEDS], axis=0)
        threshold, val_macro = select_threshold(
            canonical_y[val_can], canonical_groups[val_can], val_score)
        mean_pred[test_can] = (mean_score[test_can] >= threshold).astype(np.int8)
        thresholds.append(threshold)
        fold_summaries.append({
            "fold": fold,
            "threshold": threshold,
            "val_subject_macro_bACC": val_macro,
            "test": metrics(canonical_y[test_can], canonical_groups[test_can],
                            mean_score[test_can], mean_pred[test_can]),
            "train_subjects": [str(v) for v in row["train_subjects"]],
            "val_subjects": [str(v) for v in row["val_subjects"]],
            "test_subjects": [str(v) for v in row["test_subjects"]],
        })
    mean_metrics = metrics(canonical_y, canonical_groups, mean_score, mean_pred)
    return seed_scores, seed_metrics, mean_score, mean_pred, mean_metrics, fold_summaries, thresholds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="单折单轮，不写正式结果")
    args = parser.parse_args()
    result_path = OUT / "token_gru_window6_results.json"
    oof_path = OUT / "token_gru_window6_oof.npz"
    if not args.smoke and (result_path.exists() or oof_path.exists()):
        raise FileExistsError("拒绝覆盖已有 token_gru_window6 正式结果")
    if not torch.cuda.is_available():
        raise RuntimeError("本实验要求 CUDA GPU")

    tee = Tee(sys.stdout, OUT / "token_gru_window6.log")
    sys.stdout = tee
    try:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()
        cache = np.load(OUT / "dense_window_protocol_cache.npz", allow_pickle=False)
        dense = np.load(OUT / "dense_window_protocol_raw6.npy", mmap_mode="r")
        highdim = np.load(OUT / "highdim_deep_features.npz", allow_pickle=False)
        canonical_y = highdim["y"].astype(np.int64)
        canonical_groups = highdim["groups"]
        canon_idx = np.flatnonzero(cache["canon"])
        dense_labels = cache["label"].astype(np.int64)
        dense_groups = cache["subject"]
        assert np.array_equal(dense[canon_idx], highdim["raw"])
        assert np.array_equal(dense_labels[canon_idx], canonical_y)
        assert np.array_equal(dense_groups[canon_idx], canonical_groups)

        log(f"loading full dense raw to GPU: shape={dense.shape} dtype={dense.dtype}")
        raw = torch.from_numpy(np.array(dense, copy=True)).to("cuda")
        labels = torch.from_numpy(dense_labels.astype(np.float32)).to("cuda")
        log(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} "
            f"raw_memory={raw.element_size() * raw.nelement() / 1024 ** 3:.3f}GiB "
            f"batch={BATCH_SIZE} AMP=True")
        folds, fold_id = build_folds(
            dense_groups, dense_labels, canonical_y, canonical_groups, canon_idx)
        if args.smoke:
            row = folds[0]
            weights = balanced_training_weights(dense_labels, dense_groups, row["train"])
            sample_weights = torch.from_numpy(weights).to("cuda")
            run = train_fold(raw, labels, sample_weights, row["train"], row["val"], row["test"],
                             canonical_y[row["val_can"]], canonical_groups[row["val_can"]],
                             0, 1, max_epochs=1)
            log(f"smoke passed parameters={run['parameters']} valAUC={run['best_val_AUC']:.6f}")
            return

        runs = {}
        for row in folds:
            fold = row["fold"]
            weights = balanced_training_weights(dense_labels, dense_groups, row["train"])
            sample_weights = torch.from_numpy(weights).to("cuda")
            for seed in SEEDS:
                log(f"start fold {fold}/5 seed {seed}: train={row['train'].size} "
                    f"val={row['val'].size} test={row['test'].size} "
                    f"subjects={len(row['train_subjects'])}/{len(row['val_subjects'])}/{len(row['test_subjects'])}")
                runs[(fold, seed)] = train_fold(
                    raw, labels, sample_weights, row["train"], row["val"], row["test"],
                    canonical_y[row["val_can"]], canonical_groups[row["val_can"]], seed, fold)
                log(f"finished fold {fold}/5 seed {seed}: "
                    f"best_epoch={runs[(fold, seed)]['best_epoch']} "
                    f"best_val_AUC={runs[(fold, seed)]['best_val_AUC']:.6f}")

        # 所有训练结束后才计算测试 OOF 指标。
        seed_scores, seed_metrics, mean_score, mean_pred, mean_metrics, fold_summaries, thresholds = \
            evaluate_oof(runs, folds, canonical_y, canonical_groups)
        log(f"seed-mean OOF macro={mean_metrics['subject_macro_bACC']:.6f} "
            f"micro={mean_metrics['window_micro_bACC']:.6f} AUC={mean_metrics['AUC']:.6f}")
        # 主结果先于可选基线落盘，防止旧基线的排序或版本问题丢失训练产物。
        saved = {
            "y": canonical_y,
            "groups": canonical_groups,
            "fold_id": fold_id,
            "score_mean": mean_score,
            "pred_mean": mean_pred,
            "thresholds": np.asarray(thresholds),
        }
        for seed in SEEDS:
            saved[f"score_seed{seed}"] = seed_scores[seed]
        np.savez_compressed(oof_path, **saved)

        baselines = {}
        name, filename, key = ("handcrafted_seed_mean", "seed_noise_band_oof.npz", "oof_seed_mean")
        with np.load(OUT / filename, allow_pickle=False) as data:
            assert np.array_equal(data["y"], canonical_y)
            assert np.array_equal(data["groups"], canonical_groups)
            score = data[key].astype(np.float64)
        baseline_pred = (score >= 0.5).astype(np.int8)
        baselines[name] = {
            "fixed_0.5": metrics(canonical_y, canonical_groups, score, baseline_pred),
            "paired_bootstrap_new_minus_baseline": paired_bootstrap(
                canonical_y, canonical_groups, baseline_pred, mean_pred),
        }
        timefreq_published = json.loads((OUT / "timefreq_ssl_window6_results.json").read_text(
            encoding="utf-8"))
        baselines["timefreq_InfoNCE_published"] = {
            "overall_fixed_0.5": timefreq_published["overall_fixed_0.5"],
            "subject_fixed_0.5": timefreq_published["subject_fixed_0.5"],
            "row_aligned_with_current_cache": False,
            "paired_bootstrap": None,
            "reason": "legacy database ordering differs from highdim canonical ordering; only published aggregate metrics are comparable",
        }

        result = {
            "protocol": {
                "input": "current 6 s only; six chronological 1 s tokens; no cross-window state",
                "effective_receptive_field_seconds": 6,
                "outer_cv": "StratifiedGroupKFold(5,shuffle=True,random_state=0) on canonical windows",
                "inner_cv": "first StratifiedGroupKFold(4,shuffle=True,random_state=outer_fold) split on outer-training subjects",
                "train_windows": "all dense 1 s-step windows from inner-training subjects only",
                "val_test_windows": "canonical non-overlapping 6 s windows only",
                "early_stopping": "inner-validation AUC; patience 6; no test labels",
                "threshold": "maximum inner-validation subject-macro bACC; 97 quantiles plus 0.5",
                "loss": "BCE with equal total weight for each inner-training subject-class cell",
                "seeds": list(SEEDS),
                "final_prediction": "predetermined mean of two seed probabilities",
                "test_labels_used_for_training_or_selection": False,
            },
            "model": {
                "architecture": "shared raw Conv1d token encoder + 18 spectral features + unidirectional GRU + final-token classifier",
                "parameters": runs[(1, 0)]["parameters"],
                "batch_size": BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "mixed_precision": "FP16 autocast with FP32 FFT and master weights",
                "full_dense_raw_resident_on_gpu": True,
            },
            "data": {
                "dense_shape": list(dense.shape),
                "canonical_shape": list(highdim["raw"].shape),
                "n_subjects": int(np.unique(canonical_groups).size),
                "canonical_raw_label_group_exact_match": True,
            },
            "seed_metrics": seed_metrics,
            "seed_mean_metrics": mean_metrics,
            "folds": fold_summaries,
            "baselines": baselines,
            "training_runs": {f"fold{fold}_seed{seed}": {
                key: value for key, value in run.items() if key not in ("val_score", "test_score")
            } for (fold, seed), run in runs.items()},
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "peak_memory_GiB": torch.cuda.max_memory_allocated() / 1024 ** 3,
            },
            "seconds": time.perf_counter() - START,
        }
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"done seconds={result['seconds']:.2f} peakGPU={result['gpu']['peak_memory_GiB']:.2f}GiB")
    finally:
        sys.stdout = tee.stream
        tee.close()


if __name__ == "__main__":
    main()
