"""Signal ceiling: leave-one-trial-out per subject with a low-dimensional model.

The earlier within-subject number used a random window split, which puts
adjacent 6 s windows of the same block on both sides of the split. Here every
held-out fold is a whole trial, and the model only sees 10 band-power features
with strong regularisation, which is the honest per-subject estimate.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / 'test'
RESULTS = OUT / 'bottleneck_signal_ceiling.json'


def loto(X, y, groups, trials, cols, C):
    aucs, skipped = [], 0
    for subject in sorted(set(groups)):
        ids = np.flatnonzero(groups == subject)
        yy, tt = y[ids], trials[ids]
        if np.bincount(yy, minlength=2).min() < 10:
            continue
        oof = np.full(len(ids), np.nan)
        for tr, te in LeaveOneGroupOut().split(ids, yy, tt):
            if len(np.unique(yy[tr])) < 2:
                continue
            model = Pipeline([('scale', StandardScaler()),
                              ('clf', LogisticRegression(C=C, max_iter=2000,
                                                         class_weight='balanced'))])
            model.fit(X[ids[tr]][:, cols], yy[tr])
            oof[te] = model.predict_proba(X[ids[te]][:, cols])[:, 1]
        mask = np.isfinite(oof)
        if yy[mask].min() == yy[mask].max():
            skipped += 1
            continue
        aucs.append(float(roc_auc_score(yy[mask], oof[mask])))
    aucs = np.array(aucs)
    return {'n_subjects': int(len(aucs)), 'skipped': skipped,
            'mean': float(aucs.mean()), 'median': float(np.median(aucs)),
            'q25': float(np.quantile(aucs, .25)), 'q75': float(np.quantile(aucs, .75)),
            'min': float(aucs.min()), 'max': float(aucs.max()),
            'fraction_above_0.65': float((aucs > .65).mean()),
            'fraction_below_0.55': float((aucs < .55).mean())}


def main():
    if RESULTS.exists():
        raise FileExistsError('Refusing to overwrite signal-ceiling artifact')
    cached = np.load(OUT / 'highdim_deep_features.npz')
    X, y, groups = cached['X'].astype(np.float64), cached['y'], cached['groups']
    meta = json.load(open(OUT / 'highdim_deep_results.json', encoding='utf-8'))['metadata']
    names = json.load(open(OUT / 'highdim_deep_results.json', encoding='utf-8'))['feature_names']
    trials = np.array([f"{m['subject']}_{m['trial_id']}" for m in meta])
    index = {n: i for i, n in enumerate(names)}
    cols = [index[n] for n in names if n.startswith('stft_log_power_') and '_mean' in n]
    cols += [index[n] for n in names if n.startswith('stft_relative_') and '_mean' in n]

    result = {'protocol': {
        'cv': 'LeaveOneGroupOut on trials, inside each subject',
        'features': [names[j] for j in cols],
        'note': '10 band-power features, Strong regularisation avoids memorising the six blocks per subject'},
        'regularisation': {}}
    for C in (0.003, 0.01, 0.03, 0.1, 1.0):
        result['regularisation'][str(C)] = loto(X, y, groups, trials, cols, C)
        print(C, json.dumps({k: round(v, 4) if isinstance(v, float) else v
                             for k, v in result['regularisation'][str(C)].items()}), flush=True)

    singles = ['stft_log_power_4_8_mean', 'stft_log_power_8_13_mean',
               'stft_log_power_13_30_mean', 'stft_relative_8_13_mean']
    result['single_feature'] = {}
    for name in singles:
        j = index[name]
        per = [roc_auc_score(y[groups == s], X[groups == s, j])
               for s in sorted(set(groups)) if y[groups == s].min() != y[groups == s].max()]
        result['single_feature'][name] = {
            'pooled_auc': float(roc_auc_score(y, X[:, j])),
            'subject_mean_auc': float(np.mean(per)),
            'subject_median_auc': float(np.median(per)),
            'fraction_direction_positive': float(np.mean(np.array(per) > .5))}
        print(name, json.dumps({k: round(v, 3) for k, v in result['single_feature'][name].items()}), flush=True)

    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
