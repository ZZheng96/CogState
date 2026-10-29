"""Question: does training the 300-dim MLP for more epochs help?

Reuses the cached 858-dim feature pool from highdim_deep_features.npz.
Every fold keeps the same subject-disjoint split and the same seed as the
previous high-dimensional experiment, so only the epoch budget / early
stopping policy changes.
"""
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test'))
from highdim_deep_ablation import FeatureTransform, network, predict

OUT = ROOT / 'test'
FEATURES = OUT / 'highdim_deep_features.npz'
RESULTS = OUT / 'epoch_ablation_results.json'
FIGURE = OUT / 'epoch_ablation_results.png'


def score(y, p):
    return {'AUC': float(roc_auc_score(y, p)),
            'bACC': float(balanced_accuracy_score(y, p >= .5))}


def fit(X, y, V, vy, device, seed, max_epochs, patience, lr, weight_decay):
    """Train one fold; patience=None means run the full fixed budget."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = network('large', X.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float((y == 0).sum() / max((y == 1).sum(), 1)), device=device))
    rng = np.random.default_rng(seed)
    best, state, stale, best_epoch = float('inf'), None, 0, 0
    history, train_loss = [], []
    for epoch in range(1, max_epochs + 1):
        net.train()
        order = rng.permutation(len(y))
        batch_loss = []
        for start in range(0, len(y), 128):
            ids = order[start:start + 128]
            xx = torch.from_numpy(X[ids]).to(device)
            yy = torch.tensor(y[ids], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = criterion(net(xx).ravel(), yy)
            loss.backward()
            optimizer.step()
            batch_loss.append(float(loss.item()))
        train_loss.append(float(np.mean(batch_loss)))
        p = np.clip(predict(net, V, device), 1e-7, 1 - 1e-7)
        val = float(-np.mean(vy * np.log(p) + (1 - vy) * np.log(1 - p)))
        history.append(val)
        if val < best - 1e-5:
            best, state, stale, best_epoch = val, copy.deepcopy(net.state_dict()), 0, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f'      epoch {epoch}/{max_epochs} train={train_loss[-1]:.4f} '
                  f'val={val:.5f} best_epoch={best_epoch}', flush=True)
        if patience is not None and stale >= patience:
            break

    if patience is None:
        # deliberately keep the final epoch to expose pure over-training
        net.eval()
        return net, {'epochs_run': max_epochs, 'best_epoch': None, 'selected_epoch': max_epochs,
                     'best_val': best, 'final_val': history[-1],
                     'final_train': train_loss[-1], 'history': history}
    net.load_state_dict(state)
    net.eval()
    return net, {'epochs_run': len(history), 'best_epoch': best_epoch, 'selected_epoch': best_epoch,
                 'best_val': best, 'final_val': history[-1],
                 'final_train': train_loss[-1], 'history': history}


def evaluate(config, X, y, groups, device):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    folds, tick = [], time.perf_counter()
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        assert not set(groups[tr]) & set(groups[te])
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold)
                    .split(X[tr], y[tr], groups[tr]))
        inner_train, inner_val = tr[a], tr[b]
        assert not set(groups[inner_train]) & set(groups[inner_val])
        transform = FeatureTransform(300, 'kbest').fit(X[inner_train], y[inner_train])
        A, V, T = [transform.transform(X[ids]) for ids in (inner_train, inner_val, te)]
        print(f'  [{config["name"]}] fold {fold}/5 '
              f'train={len(inner_train)} val={len(inner_val)} test={len(te)}', flush=True)
        model, info = fit(A, y[inner_train], V, y[inner_val], device, fold,
                          config['max_epochs'], config['patience'],
                          config['lr'], config['weight_decay'])
        oof[te] = predict(model, T, device)
        row = {'fold': fold, 'n_test': len(te),
               'parameters': int(sum(p.numel() for p in model.parameters())),
               **info, **score(y[te], oof[te])}
        folds.append(row)
        print(f'  [{config["name"]}] fold {fold} AUC={row["AUC"]:.5f} '
              f'bACC={row["bACC"]:.5f} epochs={row["epochs_run"]} '
              f'best={row["selected_epoch"]}', flush=True)
    subject = {s: score(y[groups == s], oof[groups == s]) for s in sorted(set(groups))}
    summary = {
        'oof': score(y, oof),
        'subject_macro': {m: float(np.mean([v[m] for v in subject.values()])) for m in ('AUC', 'bACC')},
        'fold_mean': {m: float(np.mean([f[m] for f in folds])) for m in ('AUC', 'bACC')},
        'epochs_run': [f['epochs_run'] for f in folds],
        'selected_epochs': [f['selected_epoch'] for f in folds],
        'folds': folds,
        'seconds': time.perf_counter() - tick,
    }
    return summary, oof


def main():
    if RESULTS.exists() or FIGURE.exists():
        raise FileExistsError('Refusing to overwrite epoch ablation artifacts')
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cached = np.load(FEATURES)
    X, y, groups = cached['X'], cached['y'], cached['groups']
    print(f'device={device} windows={len(y)} pool={X.shape[1]} '
          f'subjects={len(set(groups))}', flush=True)

    configs = [
        {'name': 'base_e40_p6', 'max_epochs': 40, 'patience': 6, 'lr': 1e-3, 'weight_decay': .01},
        {'name': 'e120_p6', 'max_epochs': 120, 'patience': 6, 'lr': 1e-3, 'weight_decay': .01},
        {'name': 'e120_p20', 'max_epochs': 120, 'patience': 20, 'lr': 1e-3, 'weight_decay': .01},
        {'name': 'e400_p30', 'max_epochs': 400, 'patience': 30, 'lr': 1e-3, 'weight_decay': .01},
        {'name': 'fixed_e80', 'max_epochs': 80, 'patience': None, 'lr': 1e-3, 'weight_decay': .01},
        {'name': 'fixed_e200', 'max_epochs': 200, 'patience': None, 'lr': 1e-3, 'weight_decay': .01},
        {'name': 'e120_p20_lr3e4', 'max_epochs': 120, 'patience': 20, 'lr': 3e-4, 'weight_decay': .01},
    ]

    result = {
        'protocol': {
            'n_windows': int(len(y)),
            'n_subjects': int(len(set(groups))),
            'feature_pool': int(X.shape[1]),
            'selected_features': 300,
            'network': '300->128->64->1, ReLU, dropout 0.2',
            'outer_cv': 'StratifiedGroupKFold(5,shuffle=True,random_state=0)',
            'inner_cv': 'first split of StratifiedGroupKFold(4,shuffle=True,random_state=fold)',
            'batch': 128,
            'seed': 'torch/numpy seeded with outer fold id (identical across configs)',
            'selection': 'ANOVA SelectKBest(300) + StandardScaler, fitted on inner training subjects only',
            'device': device,
            'torch_version': str(torch.__version__),
            'note': 'early-stopped configs select the epoch by inner subject-disjoint validation loss; '
                    'fixed_e* runs the whole budget and keeps the final epoch on purpose',
        },
        'variants': {},
    }
    predictions = {}
    tick = time.perf_counter()
    for config in configs:
        print(f'[config] {config}', flush=True)
        result['variants'][config['name']], predictions[config['name']] = evaluate(
            config, X, y, groups, device)
    result['runtime_seconds'] = time.perf_counter() - tick
    np.savez_compressed(OUT / 'epoch_ablation_oof.npz', y=y, groups=groups, **predictions)
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    names = list(result['variants'])
    pos = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for delta, metric in ((-.18, 'AUC'), (.18, 'bACC')):
        axes[0].bar(pos + delta,
                    [result['variants'][n]['subject_macro'][metric] for n in names],
                    .36, label='Subject macro ' + metric)
    axes[0].set(xticks=pos, xticklabels=names, ylim=(.45, .75), ylabel='score',
                title='Epoch budget vs. subject-macro metrics')
    axes[0].legend(); axes[0].tick_params(axis='x', rotation=25)
    for name in names:
        hist = [f['history'] for f in result['variants'][name]['folds']]
        longest = max(map(len, hist))
        padded = np.full((len(hist), longest), np.nan)
        for i, curve in enumerate(hist):
            padded[i, :len(curve)] = curve
        axes[1].plot(np.arange(1, longest + 1), np.nanmean(padded, axis=0), label=name)
    axes[1].set(xlabel='epoch', ylabel='inner validation loss', yscale='log',
                title='Mean inner-validation loss per epoch')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    plt.close(fig)
    print('COMPLETE', json.dumps({n: {'AUC': v['oof']['AUC'], 'bACC': v['oof']['bACC'],
                                      'macro_AUC': v['subject_macro']['AUC'],
                                      'macro_bACC': v['subject_macro']['bACC'],
                                      'epochs': v['epochs_run'],
                                      'selected': v['selected_epochs']}
                                 for n, v in result['variants'].items()}, indent=1), flush=True)


if __name__ == '__main__':
    main()
