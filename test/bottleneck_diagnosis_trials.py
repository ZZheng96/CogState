"""Second bottleneck check: is the within-subject advantage real, or window leakage?

Windows inside one trial are adjacent 6 s blocks of the same state, so a random
within-subject split puts near-duplicate neighbours on both sides. Here the
within-subject CV is grouped by trial, and we also measure whether the
label->feature direction is consistent across subjects.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'
RESULTS = OUT / 'bottleneck_diagnosis_trials.json'
FIGURE = OUT / 'bottleneck_diagnosis_trials.png'


def main():
    if RESULTS.exists() or FIGURE.exists():
        raise FileExistsError('Refusing to overwrite trial-blocked diagnosis artifacts')
    cached = np.load(OUT / 'highdim_deep_features.npz')
    X, y, groups = cached['X'].astype(np.float64), cached['y'], cached['groups']
    meta = json.load(open(OUT / 'highdim_deep_results.json', encoding='utf-8'))['metadata']
    names = json.load(open(OUT / 'highdim_deep_results.json', encoding='utf-8'))['feature_names']
    trials = np.array([f"{m['subject']}_{m['trial_id']}" for m in meta])
    assert len(trials) == len(y)

    print('[1/2] within-subject CV grouped by trial', flush=True)
    rows = {}
    for subject in sorted(set(groups)):
        ids = np.flatnonzero(groups == subject)
        yy, tt = y[ids], trials[ids]
        if np.bincount(yy, minlength=2).min() < 10 or len(set(tt)) < 4:
            rows[subject] = {'skipped': True}
            continue
        oof = np.full(len(ids), np.nan)
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=0)
        for tr, te in cv.split(ids, yy, tt):
            model = Pipeline([('scale', StandardScaler()),
                              ('select', SelectKBest(f_classif, k=300)),
                              ('clf', LogisticRegression(C=1, max_iter=2000,
                                                         class_weight='balanced', random_state=0))])
            model.fit(X[ids[tr]], yy[tr])
            oof[te] = model.predict_proba(X[ids[te]])[:, 1]
        rows[subject] = {'AUC': float(roc_auc_score(yy, oof)), 'n': int(len(ids)),
                         'n_trials': int(len(set(tt))), 'task_fraction': float(yy.mean())}
        print(f'  subject {subject} AUC={rows[subject]["AUC"]:.4f} '
              f'windows={len(ids)} trials={rows[subject]["n_trials"]}', flush=True)
    aucs = np.array([r['AUC'] for r in rows.values() if 'AUC' in r])
    trial_summary = {'n_subjects': len(aucs), 'mean': float(aucs.mean()),
                     'median': float(np.median(aucs)),
                     'q25': float(np.quantile(aucs, .25)), 'q75': float(np.quantile(aucs, .75)),
                     'min': float(aucs.min()), 'max': float(aucs.max()),
                     'fraction_above_0.65': float((aucs > .65).mean())}
    print(json.dumps(trial_summary, indent=1), flush=True)

    print('[2/2] sign consistency of feature-label correlation across subjects', flush=True)
    fscore = f_classif(X, y)[0]
    fscore = np.nan_to_num(fscore, nan=0.0)
    top = np.argsort(fscore)[-100:]
    per_subject = {}
    for subject in sorted(set(groups)):
        ids = np.flatnonzero(groups == subject)
        yy = y[ids]
        if yy.min() == yy.max():
            continue
        rho = np.array([spearmanr(X[ids, j], yy).statistic for j in top])
        per_subject[subject] = np.nan_to_num(rho, nan=0.0)
    matrix = np.array(list(per_subject.values()))
    signs = np.sign(matrix)
    agreement = np.maximum((signs > 0).mean(0), (signs < 0).mean(0))
    strength = np.abs(matrix).mean(0)
    order = np.argsort(-strength)
    feature_rows = [{'feature': names[top[j]], 'mean_abs_rho': float(strength[j]),
                     'sign_agreement': float(agreement[j])} for j in order[:25]]
    sign_summary = {'median_sign_agreement': float(np.median(agreement)),
                    'mean_sign_agreement': float(agreement.mean()),
                    'fraction_agreement_above_0.8': float((agreement > .8).mean()),
                    'median_mean_abs_rho': float(np.median(strength))}
    print(json.dumps(sign_summary, indent=1), flush=True)
    print(json.dumps(feature_rows[:10], ensure_ascii=False, indent=1), flush=True)

    result = {'protocol': {
        'within_subject_cv': 'StratifiedGroupKFold(5) grouped by trial inside each subject',
        'features': '858-dim pool, SelectKBest(300) refitted inside every training fold',
        'sign_check': 'Spearman(label, feature) per subject on the 100 highest global F features'},
        'within_subject_trial_blocked': {'summary': trial_summary, 'subjects': rows},
        'sign_consistency': {'summary': sign_summary, 'top_features': feature_rows}}
    RESULTS.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].hist(aucs, bins=14)
    axes[0].axvline(.5, color='k', ls='--', lw=1)
    axes[0].set(xlabel='per-subject AUC (trial-blocked)', ylabel='subjects',
                title='Within-subject separability without adjacent-window leakage')
    axes[1].hist(agreement, bins=14)
    axes[1].set(xlabel='sign agreement across subjects', ylabel='features',
                title='Is the task direction the same for everyone?')
    fig.tight_layout(); fig.savefig(FIGURE, dpi=160); plt.close(fig)
    print('COMPLETE trial_blocked=%.4f sign_agreement=%.3f' %
          (trial_summary['mean'], sign_summary['median_sign_agreement']), flush=True)


if __name__ == '__main__':
    main()
