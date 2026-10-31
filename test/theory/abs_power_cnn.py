"""D4: does absolute power (lost by per-window z-scoring) carry the cross-subject signal for a Saini-style 1D CNN?

Input: raw 6 s windows (1500 samples, database/windows/{relax,task}/<sub>.npz 'x', canonical order == audit load_all).
  absmean   x - mean(x) per window (window-local, causal), then divided by ONE global std fitted on outer-train rows
  zscore    (x - mean) / std per window (window-local)
Net (no BN): Conv1d(1,16,5)-ReLU-MaxPool2 - Conv1d(16,32,5)-ReLU-MaxPool2 - Dropout.25 - Dense64-ReLU - Dense1.
Adam 1e-3, batch 128, BCE, fixed EPOCHS (no early stopping, no val peeking). Outer SGKF(5) folds == npz; seeds 0,1.
Report: AUC, subject-macro bACC, per-group macro bACC / mean per-subject AUC, paired subject bootstrap absmean - zscore,
and vs strict LR889 baseline (audit npz oof 'p_base' if present).
Run: python test/theory/abs_power_cnn.py [--smoke]
"""
import argparse, time

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from _common import GROUPS, TEST, boot, el, impute_scale, load, metrics, open_log, save, bacc

EPOCHS, BS, LR = 30, 128, 1e-3
SEEDS = (0, 1)


def net():
    return nn.Sequential(nn.Conv1d(1, 16, 5), nn.ReLU(), nn.MaxPool1d(2), nn.Conv1d(16, 32, 5), nn.ReLU(),
                         nn.MaxPool1d(2), nn.Dropout(.25), nn.Flatten(), nn.Linear(32 * 372, 64), nn.ReLU(),
                         nn.Linear(64, 1))


def inputs(X, tr, mode):
    Xc = X - X.mean(1, keepdims=True)
    if mode == 'zscore':
        return Xc / np.maximum(Xc.std(1, keepdims=True), 1e-12)
    if mode == 'absmean_globalstd':
        return Xc / Xc[tr].std()
    return Xc / np.median(Xc[tr].std(1))  # absmean: one robust global scale (median per-window std, train rows)


def train_predict(Xin, y, tr, te, seed, dev, epochs, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    m = net().to(dev); opt = torch.optim.Adam(m.parameters(), LR); lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xin[tr, None], dtype=torch.float32, device=dev)
    yt = torch.tensor(y[tr], dtype=torch.float32, device=dev)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(tr), generator=g).to(dev); tot = 0.
        for i in range(0, len(tr), BS):
            b = perm[i:i + BS]; opt.zero_grad()
            loss = lossf(m(Xt[b]).squeeze(1), yt[b]); loss.backward(); opt.step(); tot += loss.item() * len(b)
        if ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == epochs:
            print(f'    {tag} epoch {ep + 1}/{epochs} train_loss={tot / len(tr):.4f} {el()}', flush=True)
    m.eval(); out = []
    Xe = torch.tensor(Xin[te, None], dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(te), 1024):
            out.append(torch.sigmoid(m(Xe[i:i + 1024].to(dev)).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


def gstats(y, g, p, grp_of):
    out = {}
    for G in ('all',) + tuple(GROUPS):
        ss = [s for s in np.unique(g[np.isfinite(p)]) if G == 'all' or grp_of[s] == G]
        b = [bacc(y[g == s], p[g == s]) for s in ss]; a = [roc_auc_score(y[g == s], p[g == s]) for s in ss]
        out[G] = {'n': len(ss), 'macro_bACC': float(np.mean(b)) if ss else None,
                  'mean_subject_AUC': float(np.mean(a)) if ss else None}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'abs_power_cnn' + ('_smoke' if args.smoke else ''); tee = open_log(stem)
    D = load(); y, g, subs, grp_of, fold = D['y'], D['g'], D['subs'], D['grp_of'], D['fold']
    DB = TEST.parent / 'database' / 'windows'
    X, yy = [], []
    for s in subs:
        parts = [np.load(DB / k / f'{s}.npz') for k in ('relax', 'task')]
        X.append(np.vstack([p['x'] for p in parts]).astype(np.float64))
        yy.append(np.concatenate([p['label'] for p in parts]).astype(int))
    X = np.vstack(X); assert np.array_equal(np.concatenate(yy), y) and X.shape == (len(y), 1500)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    folds = [0] if args.smoke else range(5); seeds = SEEDS[:1] if args.smoke else SEEDS
    epochs = 3 if args.smoke else EPOCHS
    print(f'[raw] X={X.shape} device={dev} folds={list(folds)} seeds={seeds} epochs={epochs} {el()}', flush=True)
    modes = ('absmean', 'zscore', 'absmean_globalstd')
    P = {(m, sd): np.full(len(y), np.nan) for m in modes for sd in seeds}
    res = {'meta': {'net': 'Saini-style Conv1D(16,32,k5)+Pool2+Dropout.25+Dense64, no BN', 'epochs': epochs,
                    'batch': BS, 'lr': LR, 'seeds': list(seeds), 'device': dev, 'smoke': args.smoke,
                    'absmean': 'per-window demean, one global scale = median per-window std over outer-train rows',
                    'absmean_globalstd': 'per-window demean, one global std over all outer-train samples (as spec; '
                                         'artifact windows inflate it ~55x vs median -> typical input ~0.02, stalls)',
                    'zscore': 'per-window (x-mean)/std'}, 'checks': {}, 'methods': {}}
    for k in folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        assert not set(g[tr]) & set(g[te])
        for m in modes:
            Xin = inputs(X, tr, m)
            if m.startswith('absmean'):  # scale must not depend on test rows
                Xalt = X.copy(); Xalt[te] *= 7.; assert np.allclose(inputs(Xalt, tr, m)[tr], Xin[tr])
                res['checks'][f'fold{k}_{m}_scale_train_only'] = 'ok'
            for sd in seeds:
                P[(m, sd)][te] = train_predict(Xin, y, tr, te, sd, dev, epochs, f'fold{k} {m} seed{sd}')
                mk = fold == k
                print(f'[fold {k}] {m} seed{sd} AUC={roc_auc_score(y[mk], P[(m, sd)][mk]):.4f} {el()}', flush=True)
    mask = np.isin(fold, list(folds))
    base = np.full(len(y), np.nan)  # strict LR(C=.01) on 889, impute+scale fitted on outer-train rows
    for k in folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(D['Fc'], tr), impute_scale(D['Fn'], tr)
        Z = np.hstack([tfo(D['Fc']), tfn(D['Fn'])])
        base[te] = LogisticRegression(C=.01, max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    avg = {}
    for m in modes:
        avg[m] = np.mean([P[(m, sd)] for sd in seeds], 0)
        for sd in seeds:
            res['methods'][f'{m}_seed{sd}'] = {**metrics(y, g, P[(m, sd)], mask), 'groups': gstats(y, g, P[(m, sd)], grp_of)}
        r = {**metrics(y, g, avg[m], mask), 'groups': gstats(y, g, avg[m], grp_of)}
        r['vs_LR889'] = boot(avg[m], base, y, g, mask, n=200 if args.smoke else 1000)
        res['methods'][f'{m}_seedavg'] = r
    res['absmean_minus_zscore'] = boot(avg['absmean'], avg['zscore'], y, g, mask, n=200 if args.smoke else 1000)
    res['methods']['LR889'] = {**metrics(y, g, base, mask), 'groups': gstats(y, g, base, grp_of)}
    for name, r in res['methods'].items():
        print(f'[result] {name:18s} AUC={r["AUC"]:.4f} mac={r["subject_macro_bACC"]:.4f} ' +
              ' '.join(f'{G}={r["groups"][G]["macro_bACC"]:.3f}/{r["groups"][G]["mean_subject_AUC"]:.3f}'
                       for G in GROUPS if r['groups'].get(G, {}).get('n')), flush=True)
    d = res['absmean_minus_zscore']
    print(f'[boot] absmean-zscore dAUC={d["dAUC"]["mean"]:+.4f}[{d["dAUC"]["p2.5"]:+.4f},{d["dAUC"]["p97.5"]:+.4f}] '
          f'dmac={d["dmacro_bACC"]["mean"]:+.4f}[{d["dmacro_bACC"]["p2.5"]:+.4f},{d["dmacro_bACC"]["p97.5"]:+.4f}]',
          flush=True)
    save(stem, res, tee)


if __name__ == '__main__':
    main()
