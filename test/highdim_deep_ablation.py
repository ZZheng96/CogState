"""Window-local high-dimensional ablation; independent subjects in every split."""
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np
import pywt
import torch
from torch import nn
from scipy.signal import stft, butter, sosfilt, hilbert
from scipy.spatial import cKDTree
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cogstate import BANDS, load_trials, bad_intervals, valid_window
from preprocessing import causal_filter, PreprocessingConfig
from nonstationary_ablation import feature_blocks, check_window, score, WAVE_NAMES, TF_NAMES, QUALITY_NAMES

OUT = ROOT / 'test'
STAT_NAMES = ('mean', 'std', 'skew', 'kurtosis', 'trend', 'q10', 'q25', 'q50', 'q75', 'q90', 'min', 'max', 'entropy')


def statistics(x):
    x = np.asarray(x, dtype=float)
    sd = x.std()
    z = (x - x.mean()) / sd if sd > 1e-15 else np.zeros_like(x)
    t = np.linspace(-.5, .5, len(x))
    p = np.abs(x) / max(np.abs(x).sum(), 1e-30)
    return [x.mean(), sd, np.mean(z**3), np.mean(z**4)-3 if sd > 1e-15 else 0,
            np.dot(t, x) / max(np.dot(t, t), 1e-30), *np.quantile(x, [.1, .25, .5, .75, .9]),
            x.min(), x.max(), -np.sum(p*np.log(np.maximum(p, 1e-30)))/max(np.log(len(x)), 1)]


def nonlinear(x):
    x = (x-x.mean()) / max(x.std(), 1e-15)
    d, dd = np.diff(x), np.diff(x, 2)
    mobility = np.sqrt(np.var(d) / max(np.var(x), 1e-15))
    complexity = np.sqrt(np.var(dd)/max(np.var(d), 1e-15))/max(mobility, 1e-15)
    # Fixed five-sample decimation within the window bounds pair-counting cost.
    v = x[::5]
    counts = []
    for m in (2, 3):
        emb = np.lib.stride_tricks.sliding_window_view(v, m)[:len(v)-2]
        counts.append((cKDTree(emb).count_neighbors(cKDTree(emb), .2, p=np.inf)-len(emb))/2)
    sampen = -np.log((counts[1]+1)/(counts[0]+1))
    patterns = np.argsort(np.lib.stride_tricks.sliding_window_view(v, 3), axis=1, kind='stable')
    _, count = np.unique(patterns, axis=0, return_counts=True)
    p = count/count.sum()
    ac = [np.dot(x[:-lag], x[lag:])/max(np.dot(x, x), 1e-15) for lag in (1, 5, 25, 50, 125)]
    return [np.var(x), mobility, complexity, sampen, -np.sum(p*np.log(p))/np.log(6), np.mean(x[:-1]*x[1:] < 0), *ac]


def highdim_features(x, fs=250., return_names=False):
    x = check_window(x, fs)
    b = feature_blocks(x, fs)
    values = list(np.r_[b['wavelet'], b['timefreq'], b['quality']])
    names = list(WAVE_NAMES)+list(TF_NAMES)+list(QUALITY_NAMES)
    filtered = causal_filter(x, PreprocessingConfig(sfreq=fs))
    def add(prefix, row):
        values.extend(statistics(row)); names.extend(f'{prefix}_{s}' for s in STAT_NAMES)
    for seconds in (.5, 1., 2.):
        n = round(seconds*fs)
        f, _, z = stft(filtered, fs=fs, nperseg=n, noverlap=n//2, boundary=None, padded=False)
        bp = np.array([np.sum(abs(z[(f >= lo) & (f < hi)])**2, axis=0) for lo, hi in BANDS])
        for kind, matrix in (('log', np.log(np.maximum(bp, 1e-30))), ('relative', bp/np.maximum(bp.sum(axis=0), 1e-30))):
            for band, row in enumerate(matrix):
                add(f'stft_{seconds}_{kind}_{band}', row)
    for wave in ('db4', 'sym5', 'coif2'):
        for level, c in enumerate(pywt.wavedec(filtered, wave, mode='zero', level=5)):
            add(f'dwt_{wave}_{level}', c)
    for band, (lo, hi) in enumerate(BANDS):
        band_signal = sosfilt(butter(3, [lo, hi], fs=fs, btype='bandpass', output='sos'), filtered)
        analytic = hilbert(band_signal)
        add(f'hilbert_envelope_{band}', np.abs(analytic))
        # Endpoint-only FFT Hilbert; no samples outside this six-second window.
        frequency = np.diff(np.unwrap(np.angle(analytic)))*fs/(2*np.pi)
        add(f'hilbert_frequency_{band}', np.clip(frequency, -fs/2, fs/2))
    nl_names = ('activity_normalized', 'mobility', 'complexity', 'sample_entropy', 'permutation_entropy', 'zero_crossing', 'ac1', 'ac5', 'ac25', 'ac50', 'ac125')
    for label, signal in (('filtered', filtered), ('raw', x)):
        values.extend(nonlinear(signal)); names.extend(f'{label}_{n}' for n in nl_names)
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite feature')
    return (result, names) if return_names else result


class WindowFeatures:
    def __init__(self, fs=250.):
        self.fs, self.buffer = fs, np.empty(0)

    def update(self, samples):
        self.buffer = np.r_[self.buffer, samples][-round(6*self.fs):]
        return highdim_features(self.buffer, self.fs) if len(self.buffer) == round(6*self.fs) else None


class FeatureTransform:
    def __init__(self, k, mode='kbest'):
        self.k, self.mode = k, mode

    def fit(self, X, y):
        candidates = np.arange(min(self.k, X.shape[1])) if self.mode == 'prefix' else np.arange(X.shape[1])
        if self.mode != 'prefix':
            candidates = candidates[np.var(X[:, candidates], axis=0) > 1e-20]
        if self.mode == 'corr':
            corr = np.abs(np.corrcoef(X[:, candidates], rowvar=False))
            keep = []
            for j in range(len(candidates)):
                if not keep or np.all(corr[j, keep] < .98):
                    keep.append(j)
            candidates = candidates[keep]
        if self.mode in ('kbest', 'corr'):
            selector = SelectKBest(f_classif, k=min(self.k, len(candidates))).fit(X[:, candidates], y)
            candidates = candidates[selector.get_support()]
        elif self.mode == 'variance':
            candidates = candidates[:self.k]
        self.indices = candidates
        self.scaler = StandardScaler().fit(X[:, candidates])
        return self

    def transform(self, X):
        return self.scaler.transform(X[:, self.indices]).astype(np.float32)


def network(kind, dim):
    if kind == 'cnn':
        return nn.Sequential(nn.Conv1d(1, 32, 15, stride=3, padding=7), nn.ReLU(),
                             nn.Conv1d(32, 64, 9, stride=3, padding=4), nn.ReLU(),
                             nn.Conv1d(64, 64, 7, stride=2, padding=3), nn.ReLU(),
                             nn.AdaptiveAvgPool1d(8), nn.Flatten(), nn.Linear(512, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1))
    widths = (128, 64) if kind == 'large' else (32, 16)
    return nn.Sequential(nn.Linear(dim, widths[0]), nn.ReLU(), nn.Dropout(.2),
                         nn.Linear(widths[0], widths[1]), nn.ReLU(), nn.Linear(widths[1], 1))


def predict(net, X, device):
    net.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(net(torch.from_numpy(X[i:i+256]).to(device))).cpu().numpy().ravel() for i in range(0, len(X), 256)])


def fit_network(kind, X, y, V, vy, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    net = network(kind, X.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum()/max((y == 1).sum(), 1)), device=device))
    best, state, stale, history = float('inf'), None, 0, []
    rng = np.random.default_rng(seed)
    for epoch in range(1, 41):
        net.train()
        order = rng.permutation(len(y))
        for start in range(0, len(y), 128):
            ids = order[start:start+128]
            xx = torch.from_numpy(X[ids]).to(device)
            yy = torch.tensor(y[ids], dtype=torch.float32, device=device)
            optimizer.zero_grad(); loss = criterion(net(xx).ravel(), yy); loss.backward(); optimizer.step()
        p = np.clip(predict(net, V, device), 1e-7, 1-1e-7)
        val = float(-np.mean(vy*np.log(p)+(1-vy)*np.log(1-p)))
        history.append(val)
        if val < best-1e-5:
            best, state, stale, best_epoch = val, copy.deepcopy(net.state_dict()), 0, epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f'    epoch {epoch}/40 validation_loss={val:.5f}', flush=True)
        if stale >= 6:
            break
    net.load_state_dict(state); net.eval()
    return net, {'epochs_run': epoch, 'best_epoch': best_epoch, 'validation_loss': history}


def extract():
    features, signals, labels, groups, meta = [], [], [], [], []
    paths = sorted((ROOT/'EEG_dataset').glob('*/*_events.csv'))
    tick = time.perf_counter()
    for i, csv in enumerate(paths, 1):
        bdf = next(csv.parent.glob('*.bdf'))
        raw = mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
        fs = float(raw.info['sfreq'])
        if not np.isclose(fs, 250.):
            raise ValueError('raw CNN requires 250Hz')
        bad = bad_intervals(raw)
        for trial in load_trials(str(csv), str(bdf)):
            end = trial.start+6
            while end <= trial.end+1e-6:
                start = end-6
                if valid_window(start, end, bad, raw.n_times/fs):
                    x = raw.get_data(start=round(start*fs), stop=round(end*fs))[0]
                    if len(x) == 1500 and np.isfinite(x).all():
                        features.append(highdim_features(x, fs)); signals.append(x.astype(np.float32))
                        labels.append(trial.label); groups.append(trial.subject)
                        meta.append({'subject': trial.subject, 'trial_id': trial.trial_id, 'end': end})
                end += 6
        raw.close()
        print(f'[extract {i}/{len(paths)}] {len(labels)} windows elapsed={time.perf_counter()-tick:.1f}s', flush=True)
    return np.array(features), np.array(signals), np.array(labels), np.array(groups), meta


def evaluate(name, kind, k, mode, X, raw, y, groups, device):
    oof = np.full(len(y), np.nan); folds = []; tick = time.perf_counter()
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for fold, (tr, te) in enumerate(cv.split(X, y, groups), 1):
        start = time.perf_counter()
        assert not set(groups[tr]) & set(groups[te])
        info = {}
        if kind == 'logistic':
            transform = FeatureTransform(k, mode).fit(X[tr], y[tr])
            model = LogisticRegression(C=1, max_iter=2000, class_weight='balanced', random_state=0)
            model.fit(transform.transform(X[tr]), y[tr])
            oof[te] = model.predict_proba(transform.transform(X[te]))[:, 1]
            params = model.coef_.size + model.intercept_.size
            inner_train, inner_val = tr, np.array([], dtype=int)
        else:
            a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=fold).split(X[tr], y[tr], groups[tr]))
            inner_train, inner_val = tr[a], tr[b]
            assert not set(groups[inner_train]) & set(groups[inner_val])
            if kind == 'cnn':
                mean, scale = float(raw[inner_train].mean()), max(float(raw[inner_train].std()), 1e-15)
                A, V, T = [((raw[ids]-mean)/scale).astype(np.float32)[:, None, :] for ids in (inner_train, inner_val, te)]
                transform = {'mean': mean, 'scale': scale}
            else:
                transform = FeatureTransform(k, mode).fit(X[inner_train], y[inner_train])
                A, V, T = [transform.transform(X[ids]) for ids in (inner_train, inner_val, te)]
            model, info = fit_network(kind, A, y[inner_train], V, y[inner_val], device, fold)
            oof[te] = predict(model, T, device)
            params = sum(p.numel() for p in model.parameters())
        checkpoint = OUT/f'highdim_{name}_fold{fold}.pt'
        with checkpoint.open('xb') as handle:
            torch.save({'kind': kind, 'model': model if kind == 'logistic' else None,
                        'state_dict': None if kind == 'logistic' else {n: p.cpu() for n, p in model.state_dict().items()},
                        'input_dim': 1500 if kind == 'cnn' else len(transform.indices),
                        'transform': transform if kind == 'cnn' else {'indices': transform.indices, 'scaler': transform.scaler}}, handle)
        row = {'fold': fold, 'test_subjects': sorted(set(groups[te])), 'train_subjects': sorted(set(groups[inner_train])),
               'validation_subjects': sorted(set(groups[inner_val])), 'n_test': len(te), 'parameters': int(params),
               'input_dim': 1500 if kind == 'cnn' else len(transform.indices),
               'selected_indices': None if kind == 'cnn' else transform.indices.tolist(),
               'seconds': time.perf_counter()-start, **info, **score(y[te], oof[te])}
        folds.append(row)
        print(f'[{name} fold {fold}/5] AUC={row["AUC"]:.5f} bACC={row["bACC"]:.5f} params={params}', flush=True)
    subject = {s: score(y[groups == s], oof[groups == s]) for s in sorted(set(groups))}
    return {'oof': score(y, oof), 'subject_macro': {m: float(np.mean([s[m] for s in subject.values()])) for m in ('AUC', 'bACC')},
            'fold_mean': {m: float(np.mean([f[m] for f in folds])) for m in ('AUC', 'bACC')}, 'subjects': subject,
            'folds': folds, 'seconds': time.perf_counter()-tick, 'kind': kind, 'requested_dim': k, 'selection': mode}, oof


def main():
    outputs = [OUT/n for n in ('highdim_deep_results.json', 'highdim_deep_oof.npz', 'highdim_deep_comparison.png', 'highdim_deep_features.npz')]
    if any(p.exists() for p in outputs) or list(OUT.glob('highdim_*_fold*.pt')):
        raise FileExistsError('Refusing to overwrite highdim experiment artifacts')
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)
    tick = time.perf_counter()
    X, raw, y, groups, meta = extract()
    extraction = time.perf_counter()-tick
    _, names = highdim_features(raw[0], return_names=True)
    with outputs[3].open('xb') as handle:
        np.savez_compressed(handle, X=X, raw=raw, y=y, groups=groups)
    candidates = [('logistic82', 'logistic', 82, 'prefix'), ('logistic150', 'logistic', 150, 'kbest'),
                  ('logistic300', 'logistic', 300, 'kbest'), ('logistic600', 'logistic', 600, 'kbest'),
                  ('logistic300_variance', 'logistic', 300, 'variance'), ('logistic300_corr', 'logistic', 300, 'corr'),
                  ('mlp82', 'mlp', 82, 'prefix'), ('mlp300', 'mlp', 300, 'kbest'),
                  ('large_mlp300', 'large', 300, 'kbest'), ('raw_cnn', 'cnn', 1500, 'raw')]
    result = {'protocol': {'n_windows': len(y), 'n_subjects': len(set(groups)), 'feature_pool': X.shape[1],
              'window_seconds': 6, 'stride_seconds': 6, 'outer_cv': 'StratifiedGroupKFold(5,shuffle=True,random_state=0)',
              'inner_cv': 'first split of StratifiedGroupKFold(4,shuffle=True,random_state=outer_fold)',
              'deep_training': 'inner-training only; no outer refit; max40 epochs, batch128, patience6, AdamW lr=.001 weight_decay=.01, dropout=.2',
              'selection': 'variance >1e-20; optional greedy |r|<.98; ANOVA SelectKBest(f_classif); scaler fit only training partition',
              'features': '82 baseline + multiresolution STFT .5/1/2s (floor half overlap) + db4/sym5/coif2 level5 zero-extension DWT + Hilbert band envelope/frequency + nonlinear',
              'entropy': 'sample entropy m2 r.2 on standardized fivefold-decimated signal, pseudocount1; permutation order3 stable ties',
              'endpoint': 'Hilbert and STFT use whole current six-second window only; not sample-by-sample causal within window',
              'polarization': 'not applicable to single-channel EEG; no polarization features invented',
              'device': device, 'torch_version': str(torch.__version__), 'limitations': 'exploratory repeated CV, no independent test; subject/session confounded; fixed seeds not seed robustness study'},
              'feature_names': names, 'metadata': meta, 'extraction_seconds': extraction, 'variants': {}}
    predictions = {}
    with threadpool_limits(limits=1):
        for name, kind, k, mode in candidates:
            print(f'[model] {name}', flush=True)
            result['variants'][name], predictions[name] = evaluate(name, kind, k, mode, X, raw, y, groups, device)
    with outputs[1].open('xb') as handle:
        np.savez_compressed(handle, y=y, groups=groups, **predictions)
    labels = list(predictions); pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6))
    for delta, metric in ((-.18, 'AUC'), (.18, 'bACC')):
        ax.bar(pos+delta, [result['variants'][n]['subject_macro'][metric] for n in labels], .36, label='Subject macro '+metric)
    ax.set(xticks=pos, xticklabels=labels, ylim=(.4, .8)); ax.legend()
    plt.xticks(rotation=25, ha='right'); fig.tight_layout(); fig.savefig(outputs[2], dpi=160); plt.close(fig)
    result['runtime_seconds'] = time.perf_counter()-tick
    with outputs[0].open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print('COMPLETE', result['runtime_seconds'], flush=True)


if __name__ == '__main__':
    main()
