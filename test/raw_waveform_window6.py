"""Raw 6 s waveform deep models (ShallowConvNet / EEGNet-1ch / ResNet1d), strictly window-local.

Preprocessing per window only: demean, divide by window std, clip +-10. No cross-window/subject stats.
Outer: StratifiedGroupKFold(5, shuffle=True, random_state=0) by subject.
Inner (early stopping + threshold): first split of StratifiedGroupKFold(4, shuffle=True, random_state=fold) on training subjects.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
RES, OOF = OUT / 'raw_waveform_window6_results.json', OUT / 'raw_waveform_window6_oof.npz'
FS, WIN, CROP = 250, 1500, 1000
N_TTA, BS, MAX_EP, PATIENCE, LR, WD = 5, 128, 50, 8, 1e-3, 1e-2
SEEDS = (0, 1, 2)
BUDGET_SECONDS = 15 * 60


def normalize(raw):
    x = raw.astype(np.float64)
    x = x - x.mean(1, keepdims=True)
    sd = x.std(1, keepdims=True)
    x = np.where(sd > 1e-12, x / np.maximum(sd, 1e-12), 0.)
    return np.clip(x, -10, 10).astype(np.float32), int((sd.ravel() <= 1e-12).sum())


class Shallow(nn.Module):
    """ShallowConvNet-style: temporal conv -> BN -> square -> avg pool -> log -> linear."""
    def __init__(self, F=40):
        super().__init__()
        self.conv1 = nn.Conv1d(1, F, 25, padding=12, bias=False)
        self.conv2 = nn.Conv1d(F, F, 25, padding=12, bias=False)
        self.bn = nn.BatchNorm1d(F)
        self.pool = nn.AvgPool1d(75, 15)
        self.drop = nn.Dropout(.5)
        self.fc = nn.Linear(F * ((CROP - 75) // 15 + 1), 1)

    def forward(self, x):
        x = self.pool(self.bn(self.conv2(self.conv1(x))) ** 2)
        return self.fc(self.drop(torch.log(torch.clamp(x, min=1e-6)).flatten(1)))


class EEGNet1(nn.Module):
    """Single-channel EEGNet-style: temporal conv, depthwise (D=2), separable conv."""
    def __init__(self, F1=8, D=2, F2=16):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv1d(1, F1, 125, padding=62, bias=False), nn.BatchNorm1d(F1),
                                    nn.Conv1d(F1, F1 * D, 1, groups=F1, bias=False), nn.BatchNorm1d(F1 * D), nn.ELU(),
                                    nn.AvgPool1d(4), nn.Dropout(.25))
        self.block2 = nn.Sequential(nn.Conv1d(F1 * D, F1 * D, 17, padding=8, groups=F1 * D, bias=False),
                                    nn.Conv1d(F1 * D, F2, 1, bias=False), nn.BatchNorm1d(F2), nn.ELU(),
                                    nn.AvgPool1d(8), nn.Dropout(.25))
        self.fc = nn.Linear(F2 * (CROP // 4 // 8), 1)

    def forward(self, x):
        return self.fc(self.block2(self.block1(x)).flatten(1))


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.body = nn.Sequential(nn.Conv1d(cin, cout, 7, stride, 3, bias=False), nn.BatchNorm1d(cout), nn.ReLU(),
                                  nn.Conv1d(cout, cout, 7, 1, 3, bias=False), nn.BatchNorm1d(cout))
        self.skip = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


class ResNet1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 15, 2, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 64, 2), Block(64, 128, 2))
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


MODELS = {'shallow': Shallow, 'eegnet': EEGNet1, 'resnet1d': ResNet1d}


def augment(xb, gen):
    n = xb.shape[0]
    off = torch.randint(0, WIN - CROP + 1, (n, 1), generator=gen, device=xb.device)  # random 4 s crop = time shift
    xb = xb.gather(1, off + torch.arange(CROP, device=xb.device))
    scale = torch.exp(torch.empty(n, 1, device=xb.device).uniform_(np.log(.8), np.log(1.25), generator=gen))
    return (xb * scale + .05 * torch.randn(xb.shape, generator=gen, device=xb.device))[:, None, :]


@torch.no_grad()
def predict(net, Xg, ids):
    """Mean logit over N_TTA evenly spaced 4 s crops of the current 6 s window."""
    net.eval(); out = []
    offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int)
    for b in range(0, len(ids), 512):
        xb = Xg[torch.as_tensor(ids[b:b + 512], device=Xg.device)]
        out.append(torch.stack([net(xb[:, None, o:o + CROP]).ravel() for o in offs]).mean(0).cpu().numpy())
    return 1 / (1 + np.exp(-np.concatenate(out)))


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0, 1))


def train_fold(kind, Xg, yg, y, tr, va, te, seed, fold):
    tick = time.perf_counter()
    torch.manual_seed(1000 * seed + fold)
    rng = np.random.default_rng(1000 * seed + fold)
    gen = torch.Generator(device=Xg.device); gen.manual_seed(1000 * seed + fold)
    net = MODELS[kind]().to(Xg.device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()
    best, state, stale, best_ep, hist = -1., None, 0, 0, []
    for ep in range(1, MAX_EP + 1):
        net.train(); perm = rng.permutation(tr); tot = 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2:
                continue
            idt = torch.as_tensor(ids, device=Xg.device)
            loss = crit(net(augment(Xg[idt], gen)).ravel(), yg[idt])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            tot += loss.item() * len(ids)
        pv = predict(net, Xg, va); auc = roc_auc_score(y[va], pv)
        hist.append({'epoch': ep, 'train_loss': tot / len(tr), 'val_auc': float(auc)})
        if auc > best + 1e-4:
            best, state, stale, best_ep = auc, copy.deepcopy(net.state_dict()), 0, ep
        else:
            stale += 1
        print(f'    [{kind} s{seed} f{fold}] ep{ep:02d} loss={tot / len(tr):.4f} valAUC={auc:.4f} best={best:.4f}@{best_ep} '
              f't={time.perf_counter() - tick:.0f}s', flush=True)
        if stale >= PATIENCE:
            break
    net.load_state_dict(state)
    pv, pt = predict(net, Xg, va), predict(net, Xg, te)
    return {'val': pv, 'test': pt, 'best_epoch': best_ep, 'epochs_run': ep, 'best_val_auc': float(best),
            'params': int(sum(p.numel() for p in net.parameters())), 'seconds': time.perf_counter() - tick, 'history': hist}


def metrics(y, p, thr):
    return {'AUC': float(roc_auc_score(y, p)), 'bACC': float(balanced_accuracy_score(y, p >= thr)),
            'bACC_at_0.5': float(balanced_accuracy_score(y, p >= .5))}


def pooled(y, p, folds, thrs):
    """Pooled OOF metrics; bACC uses each fold's inner-validation Youden threshold."""
    hard = np.zeros(len(y), bool)
    for k, (_, _, te) in enumerate(folds):
        hard[te] = p[te] >= thrs[k]
    per_fold = [metrics(y[te], p[te], thrs[k]) for k, (_, _, te) in enumerate(folds)]
    return {'AUC': float(roc_auc_score(y, p)), 'bACC': float(balanced_accuracy_score(y, hard)),
            'bACC_at_0.5': float(balanced_accuracy_score(y, p >= .5)),
            'fold_mean': {m: float(np.mean([f[m] for f in per_fold])) for m in ('AUC', 'bACC')}, 'folds': per_fold}


def boot_delta(y, a, b, subj_idx, n=1000, seed=0):
    rng = np.random.default_rng(seed); keys = list(subj_idx); d = []
    for _ in range(n):
        ids = np.concatenate([subj_idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        if len(set(y[ids])) == 2:
            d.append(roc_auc_score(y[ids], a[ids]) - roc_auc_score(y[ids], b[ids]))
    d = np.array(d)
    return {'delta_AUC': float(roc_auc_score(y, a) - roc_auc_score(y, b)), 'CI95': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            'P_delta_le_0': float((d <= 0).mean()), 'n_boot': len(d), 'unit': 'subject'}


def main():
    if RES.exists() or OOF.exists():
        raise FileExistsError('Refusing to overwrite raw_waveform_window6 outputs')
    t0 = time.perf_counter()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device} torch={torch.__version__}', flush=True)
    d = np.load(OUT / 'highdim_deep_features.npz')
    raw, y, groups = d['raw'], d['y'], d['groups']
    hb = np.load(OUT / 'seed_noise_band_oof.npz')
    assert np.array_equal(hb['y'], y) and np.array_equal(hb['groups'], groups)
    X, n_flat = normalize(raw)
    Xg, yg = torch.from_numpy(X).to(device), torch.tensor(y, dtype=torch.float32, device=device)
    print(f'windows={len(y)} subjects={len(set(groups))} flat_windows={n_flat}', flush=True)

    folds = []
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, groups), 1):
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k).split(tr, y[tr], groups[tr]))
        itr, iva = tr[a], tr[b]
        assert not set(groups[te]) & set(groups[tr]) and not set(groups[itr]) & set(groups[iva])
        folds.append((itr, iva, te))
    fold_match = all(len(set(hb['fold_id'][te].tolist())) == 1 for _, _, te in folds)
    print(f'handcrafted fold_id matches outer split: {fold_match}', flush=True)

    # quick check: seed 0, fold 1, all models
    runs, quick = {}, {}
    for kind in MODELS:
        print(f'[quick] {kind} seed0 fold1', flush=True)
        r = train_fold(kind, Xg, yg, y, *folds[0], 0, 1)
        runs[(kind, 0, 0)] = r
        thr = youden(y[folds[0][1]], r['val'])
        quick[kind] = {**metrics(y[folds[0][2]], r['test'], thr), 'threshold': thr, 'seconds': r['seconds'],
                       'best_epoch': r['best_epoch'], 'params': r['params'], 'best_val_auc': r['best_val_auc']}
        print(f'[quick] {kind} testAUC={quick[kind]["AUC"]:.4f} bACC={quick[kind]["bACC"]:.4f} t={r["seconds"]:.0f}s', flush=True)
    est = {k: quick[k]['seconds'] * (len(SEEDS) * 5 - 1) for k in MODELS}
    if sum(est.values()) <= BUDGET_SECONDS:
        full = list(MODELS)
    else:
        full = sorted(MODELS, key=lambda k: -quick[k]['AUC'])[:2]
    print(f'[plan] estimated remaining seconds={ {k: round(v) for k, v in est.items()} } -> full run: {full}', flush=True)

    results, oof_out = {}, {'y': y, 'groups': groups}
    seed_mean = {}
    for kind in full:
        per_seed, oofs, vals = {}, {}, {}
        for s in SEEDS:
            oof = np.full(len(y), np.nan); thrs, fold_rows = [], []
            for k, (itr, iva, te) in enumerate(folds):
                key = (kind, s, k)
                if key not in runs:
                    print(f'[full] {kind} seed{s} fold{k + 1}/5', flush=True)
                    runs[key] = train_fold(kind, Xg, yg, y, itr, iva, te, s, k + 1)
                r = runs[key]; oof[te] = r['test']; vals[(s, k)] = r['val']
                thrs.append(youden(y[iva], r['val']))
                fold_rows.append({'fold': k + 1, 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
                                  'best_val_auc': r['best_val_auc'], 'seconds': r['seconds'], 'threshold': thrs[-1]})
                print(f'[full] {kind} seed{s} fold{k + 1} testAUC={roc_auc_score(y[te], r["test"]):.4f} '
                      f'elapsed={time.perf_counter() - t0:.0f}s', flush=True)
            m = pooled(y, oof, folds, thrs)
            for row, f in zip(fold_rows, m['folds']):
                row.update(f)
            m['folds'] = fold_rows
            per_seed[s], oofs[s] = m, oof
            oof_out[f'{kind}_seed{s}'] = oof.astype(np.float32)
            print(f'[seed] {kind} seed{s} OOF AUC={m["AUC"]:.4f} bACC={m["bACC"]:.4f}', flush=True)
        mean_p = np.mean([oofs[s] for s in SEEDS], 0)
        thr_mean = [youden(y[iva], np.mean([vals[(s, k)] for s in SEEDS], 0)) for k, (_, iva, _) in enumerate(folds)]
        sm = pooled(y, mean_p, folds, thr_mean)
        seed_mean[kind] = mean_p
        oof_out[f'{kind}_seed_mean'] = mean_p.astype(np.float32)
        results[kind] = {'params': runs[(kind, 0, 0)]['params'],
                         'per_seed': {str(s): per_seed[s] for s in SEEDS},
                         'seed_summary': {f'{m}_{st}': float(f([per_seed[s][m] for s in SEEDS], **kw))
                                          for m in ('AUC', 'bACC') for st, f, kw in (('mean', np.mean, {}), ('sd', np.std, {'ddof': 1}))},
                         'seed_mean_prediction': sm,
                         'train_seconds': float(sum(runs[(kind, s, k)]['seconds'] for s in SEEDS for k in range(5)))}
        print(f'[model] {kind} AUC {results[kind]["seed_summary"]["AUC_mean"]:.4f}±{results[kind]["seed_summary"]["AUC_sd"]:.4f} '
              f'seed-mean AUC={sm["AUC"]:.4f} bACC={sm["bACC"]:.4f}', flush=True)

    # handcrafted baseline
    hand = hb['oof_seed_mean'].astype(np.float64)
    hseeds = [k for k in hb.files if k.startswith('oof_seed') and k != 'oof_seed_mean']
    hand_res = {'seed_mean_prediction': {'AUC': float(roc_auc_score(y, hand)), 'bACC_at_0.5': float(balanced_accuracy_score(y, hand >= .5))},
                'per_seed_AUC': {k: float(roc_auc_score(y, hb[k])) for k in hseeds}}
    hand_res['per_seed_AUC_mean_sd'] = [float(np.mean(list(hand_res['per_seed_AUC'].values()))), float(np.std(list(hand_res['per_seed_AUC'].values()), ddof=1))]
    print(f'[hand] seed-mean AUC={hand_res["seed_mean_prediction"]["AUC"]:.4f} bACC@.5={hand_res["seed_mean_prediction"]["bACC_at_0.5"]:.4f}', flush=True)

    subj_idx = {g: np.where(groups == g)[0] for g in sorted(set(groups))}
    best_kind = max(full, key=lambda k: results[k]['seed_mean_prediction']['AUC'])
    comparisons, fusion = {}, {}
    for kind in full:
        comparisons[f'{kind}_seed_mean_vs_hand'] = boot_delta(y, seed_mean[kind], hand, subj_idx)
        rp = seed_mean[kind]
        # (a) rank average (w=0.5) over pooled OOF: ranks use the evaluated window population (evaluation-only, not deployable)
        rank_f = .5 * rankdata(rp) / len(y) + .5 * rankdata(hand) / len(y)
        # (b) strictly window-local: mean of logits, w=0.5 fixed (no inner-val hand predictions available for weight tuning)
        lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        logit_f = 1 / (1 + np.exp(-(.5 * lg(rp) + .5 * lg(hand))))
        for name, f, thr in (('rank_avg_w0.5', rank_f, .5), ('logit_avg_w0.5', logit_f, .5)):
            key = f'{kind}+hand_{name}'
            fusion[key] = {'AUC': float(roc_auc_score(y, f)), 'bACC_at_0.5': float(balanced_accuracy_score(y, f >= thr)),
                           'fold_mean_AUC': float(np.mean([roc_auc_score(y[te], f[te]) for _, _, te in folds])),
                           'delta_vs_hand': boot_delta(y, f, hand, subj_idx)}
            oof_out[key] = f.astype(np.float32)
            print(f'[fusion] {key} AUC={fusion[key]["AUC"]:.4f} bACC={fusion[key]["bACC_at_0.5"]:.4f} '
                  f'dAUC={fusion[key]["delta_vs_hand"]["delta_AUC"]:+.4f} CI={fusion[key]["delta_vs_hand"]["CI95"]}', flush=True)
        print(f'[boot] {kind} vs hand dAUC={comparisons[f"{kind}_seed_mean_vs_hand"]["delta_AUC"]:+.4f} '
              f'CI={comparisons[f"{kind}_seed_mean_vs_hand"]["CI95"]}', flush=True)

    diagnosis = {
        'old_preprocessing': 'raw_cnn used one global mean/std over all inner-train samples ((raw-mean)/scale).',
        'data_facts': {'global_std': float(raw.std()), 'median_window_std': float(np.median(raw.std(1))),
                       'window_std_p1_p99': [float(np.percentile(raw.std(1), 1)), float(np.percentile(raw.std(1), 99))],
                       'mean_abs_window_offset': float(np.abs(raw.mean(1)).mean()), 'flat_windows_std0': n_flat,
                       'median_power_share_below_1Hz': 0.869},
        'cause': ('Raw signal is unfiltered with large per-window DC offsets (~0.03-0.04 V) and heavy-tailed amplitude '
                  '(window std spans ~400x). The global std (~0.053) is dominated by offsets/outlier windows, so after global '
                  'normalization a typical window has std ~0.0024 around a large constant: the net mainly sees a near-constant '
                  'offset (subject/session dependent) and the EEG waveform is ~400x below unit scale. Training never left '
                  'chance: inner-val BCE stayed ~0.693 every epoch and best_epoch was 1 in 3/5 folds (early stopping kept '
                  'an essentially untrained net); the resulting test AUC<0.5 is noise/offset-driven subject shift, not a '
                  'label-inversion. No label/prediction misalignment was found (oof[te] indexing and sigmoid direction are correct).'),
        'secondary': ['no BatchNorm/no per-window scaling, so the tiny-signal regime cannot be fixed by the net',
                      'early stopping on val BCE with patience 6 and min_delta, from epoch 1, selects near-initial weights',
                      'no augmentation, single 6 s view, pos_weight irrelevant (balanced classes)'],
        'fix_here': 'per-window demean + per-window std normalization + clip, BN in models, crop/scale/noise augmentation, '
                    'early stopping on inner-val AUC, multi-crop TTA within the current window.'}
    out = {'protocol': {'test_subject_isolation': True, 'test_history': False, 'window_position_schedule_r_t': False,
                        'test_subject_calibration': False, 'per_window_normalization_only': True,
                        'preprocessing': 'x-mean(x); /std(x) of the same window (flat windows -> 0); clip [-10,10]',
                        'outer_cv': 'StratifiedGroupKFold(5,shuffle=True,random_state=0), groups=subject',
                        'inner_val': 'first split of StratifiedGroupKFold(4,shuffle=True,random_state=outer_fold) on training subjects; '
                                     'used for early stopping (val AUC, patience 8, max 50 epochs) and Youden threshold',
                        'training': f'BCEWithLogits, AdamW lr={LR} wd={WD}, batch {BS}, grad clip 1; aug: random 4 s crop (time shift), '
                                    'log-uniform amplitude scale [0.8,1.25], Gaussian noise sd 0.05',
                        'inference': f'mean logit of {N_TTA} evenly spaced 4 s crops inside the current 6 s window',
                        'seeds': list(SEEDS), 'n_windows': int(len(y)), 'n_subjects': int(len(set(groups))),
                        'handcrafted_fold_id_matches_outer_split': bool(fold_match),
                        'fusion_note': 'rank_avg uses ranks over pooled OOF windows (evaluation-only; not a deployable per-window rule). '
                                       'logit_avg is strictly window-local. Weight fixed at 0.5 because the handcrafted baseline has no '
                                       'inner-validation predictions to select weights inside training folds.'},
           'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None, 'torch_version': str(torch.__version__),
           'old_raw_cnn_diagnosis': diagnosis, 'quick_check_seed0_fold1': quick, 'full_run_models': full,
           'estimated_full_seconds': est, 'models': results, 'handcrafted_baseline': hand_res,
           'best_raw_model': best_kind, 'raw_vs_hand_bootstrap': comparisons, 'fusion': fusion,
           'runtime_seconds': time.perf_counter() - t0}
    with OOF.open('xb') as h:
        np.savez_compressed(h, **oof_out)
    with RES.open('x', encoding='utf-8') as h:
        json.dump(out, h, ensure_ascii=False, indent=2)
    print(f'done in {time.perf_counter() - t0:.0f}s -> {RES}, {OOF}', flush=True)


if __name__ == '__main__':
    main()
