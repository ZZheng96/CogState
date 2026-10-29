"""Diagnose where the cross-subject ceiling comes from.

Three questions, all on the cached 858-dim window features:

1. Within-subject separability: if a model only ever sees one subject's windows,
   how well can it separate task/relax? This is the signal/label ceiling.
2. Cross-subject gap: re-use the existing out-of-fold predictions of the best
   model to see how much is lost when the subject is unseen.
3. Amplitude/domain shift: how much of the feature variance is between-subject,
   and what happens with an oracle per-subject standardisation (diagnostic only,
   it uses unlabeled test windows and is therefore NOT rule compliant).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
CACHE = OUT / 'highdim_deep_features.npz'
OOF = OUT / 'highdim_deep_oof.npz'
RESULTS = OUT / 'bottleneck_diagnosis.json'
FIGURE = OUT / 'bottleneck_diagnosis.png'


def score(y, p):
    return {'AUC': float(roc_auc_score(y, p)), 'bACC': float(balanced_accuracy_score(y, p >= .5))}


def pipeline(seed=0):
    return [('scale', StandardScaler()), ('select', SelectKBest(f_classif, k=300)),
            ('clf', LogisticRegression(C=1, max_iter=2000, class_weight='balanced',
                                       random_state=seed))]


def within_subject(X, y, groups):
    rows = {}
    for subject in sorted(set(groups)):
        ids = np.flatnonzero(groups == subject)
        yy = y[ids]
        counts = np.bincount(yy, minlength=2)
        if counts.min() < 10:
            rows[subject] = {'skipped': True, 'counts': counts.tolist()}
            continue
        oof = np.full(len(ids), np.nan)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        for tr, te in cv.split(ids, yy):
            from sklearn.pipeline import Pipeline
            model = Pipeline(pipeline()).fit(X[ids[tr]], yy[tr])
            oof[te] = model.predict_proba(X[ids[te]])[:, 1]
        rows[subject] = {**score(yy, oof), 'n': int(len(ids)), 'task_fraction': float(yy.mean())}
    valid = {s: r for s, r in rows.items() if 'AUC' in r}
    aucs = np.array([r['AUC'] for r in valid.values()])
    return rows, {'n_subjects': len(valid), 'mean': float(aucs.mean()),
                  'median': float(np.median(aucs)),
                  'q25': float(np.quantile(aucs, .25)), 'q75': float(np.quantile(aucs, .75)),
                  'min': float(aucs.min()), 'max': float(aucs.max()),
                  'fraction_above_0.65': float((aucs > .65).mean()),
                  'fraction_below_0.55': float((aucs < .55).mean())}


def cross_subject_oracle(X, y, groups):
    """Per-subject z-score using that subject's unlabeled windows (diagnostic only)."""
    from sklearn.pipeline import Pipeline
    Z = np.empty_like(X)
    for subject in sorted(set(groups)):
        ids = groups == subject
        Z[ids] = (X[ids] - X[ids].mean(0)) / np.maximum(X[ids].std(0), 1e-12)
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    for tr, te in cv.split(Z, y, groups):
        model = Pipeline(pipeline()).fit(Z[tr], y[tr])
        oof[te] = model.predict_proba(Z[te])[:, 1]
    return oof, score(y, oof)


def variance_share(X, groups):
    """Between-subject share of total variance per feature (eta squared)."""
    subjects = sorted(set(groups))
    overall = X.var(0) + 1e-12
    between = np.zeros(X.shape[1])
    for subject in subjects:
        ids = groups == subject
        between += ids.sum() * (X[ids].mean(0) - X.mean(0)) ** 2
    eta = between / (len(X) * overall)
    return {'median_eta_squared': float(np.median(eta)),
            'mean_eta_squared': float(eta.mean()),
            'fraction_eta_above_0.5': float((eta > .5).mean()),
            'fraction_eta_above_0.8': float((eta > .8).mean())}


def main():
    if RESULTS.exists() or FIGURE.exists():
        raise FileExistsError('Refusing to overwrite bottleneck diagnosis artifacts')
    cached = np.load(CACHE)
    X, y, groups = cached['X'].astype(np.float64), cached['y'], cached['groups']
    print(f'pool={X.shape} windows={len(y)} subjects={len(set(groups))}', flush=True)

    print('[1/3] within-subject 5-fold', flush=True)
    rows, summary = within_subject(X, y, groups)
    print(json.dumps(summary, indent=1), flush=True)

    print('[2/3] cross-subject OOF from the current best model', flush=True)
    oof_best = np.load(OOF)['large_mlp300']
    per_subject_cross = {s: score(y[groups == s], oof_best[groups == s]) for s in sorted(set(groups))}
    cross_auc = np.array([v['AUC'] for v in per_subject_cross.values()])
    cross_summary = {'mean': float(cross_auc.mean()), 'median': float(np.median(cross_auc)),
                     'q25': float(np.quantile(cross_auc, .25)), 'q75': float(np.quantile(cross_auc, .75)),
                     'fraction_above_0.65': float((cross_auc > .65).mean()),
                     'fraction_below_0.55': float((cross_auc < .55).mean())}
    print(json.dumps(cross_summary, indent=1), flush=True)

    print('[3/3] variance share + oracle per-subject standardisation', flush=True)
    share = variance_share(X, groups)
    oof_oracle, oracle_score = cross_subject_oracle(X, y, groups)
    print(json.dumps({'variance': share, 'oracle_subject_norm': oracle_score}, indent=1), flush=True)

    paired = np.array([[rows[s]['AUC'], per_subject_cross[s]['AUC']] for s in rows if 'AUC' in rows[s]])
    result = {
        'protocol': {'feature_pool': int(X.shape[1]), 'selected': 300,
                     'within_subject_cv': 'StratifiedKFold(5,shuffle=True,random_state=0) per subject',
                     'cross_subject_cv': 'StratifiedGroupKFold(5) from highdim_deep_results large_mlp300',
                     'oracle_note': 'per-subject z-score over that subject\'s full record uses unlabeled test data; '
                                    'diagnostic only, not compliant with the "no whole-record normalisation" rule'},
        'within_subject': {'summary': summary, 'subjects': rows},
        'cross_subject': {'summary': cross_summary, 'subjects': per_subject_cross},
        'paired_gap_mean': float((paired[:, 0] - paired[:, 1]).mean()),
        'paired_gap_median': float(np.median(paired[:, 0] - paired[:, 1])),
        'variance_share': share,
        'oracle_subject_norm': oracle_score,
    }
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(paired[:, 0], bins=15, alpha=.7, label='within-subject')
    axes[0].hist(paired[:, 1], bins=15, alpha=.7, label='cross-subject')
    axes[0].axvline(.5, color='k', ls='--', lw=1)
    axes[0].set(xlabel='per-subject AUC', ylabel='subjects', title='Same signal, seen vs unseen subject')
    axes[0].legend()
    axes[1].scatter(paired[:, 1], paired[:, 0], s=22)
    axes[1].plot([.4, .9], [.4, .9], 'k--', lw=1)
    axes[1].set(xlabel='cross-subject AUC', ylabel='within-subject AUC', title='Domain gap per subject')
    fig.tight_layout(); fig.savefig(FIGURE, dpi=160); plt.close(fig)
    print('COMPLETE within=%.4f cross=%.4f gap=%.4f oracle=%.4f' %
          (summary['mean'], cross_summary['mean'], result['paired_gap_mean'],
           oracle_score['AUC']), flush=True)


if __name__ == '__main__':
    main()
