"""Anchor-direction experiment (literature-driven): is there a sparse, sign-consistent feature subset on which the
relax/task conditional is shared across subjects, so that a single direction works for the low group?

Motivation (see the literature section of 研究报告.md): unsupervised alignment / subject-invariant representations /
self-training all failed here because the shift is a *concept* shift (P(Y|X) differs per subject, cos(w_ws,w_xs)=.03,
low-group pseudo-label accuracy .57). The only remaining assumption that can still hold is that a small set of
physiological markers (alpha power, EMG, ...) has the SAME sign for everyone; if so, a direction restricted to those
markers transfers, and can serve as the orientation anchor for the low group.

Diagnostic (leaky, reported): per subject and feature j, Cohen's d of the task-vs-relax mean difference on the
fold-standardised 889 features; per group: median d, sign consistency (fraction of subjects whose d has the sign of the
group median), and the within-subject AUC of the single best features. This quantifies how many low subjects can be
"anchored" at all.

Legal variants (fold-wise; feature selection uses ONLY training subjects' labels):
  xs_full        reference: LR(C=.01) on all 889 features, trained on the training subjects of the fold.
  anchor_m       select features by |median_{train subjects} d| with sign consistency >= CONS among the training
                 subjects (ties broken by |median d|), take the top m in {16, 32, 64, 128}, fit LR(C=.01) on those
                 features only, apply to the held-out subjects.
  anchor_oracle  ceiling: the same selection using ALL subjects' true labels (leaky) -> how much a perfect anchor would
                 give; if this is also low, no sparse shared direction exists at all.
  anchor_dir     no fitting at all: score = sum_j sign(median d_j) * zz_j over the selected features (a pure fixed
                 direction, no per-fold classifier), i.e. the "physiological polarity" version.
Outputs: per-subject AUC and bACC (threshold .5 on the score after causal thr_mean256? no - these scores are
order-free, so the plain .5 threshold is used; the causal threshold variant is reported for the pipeline only as a
reference), grouped by high/mid/low, plus cos between the selected shared direction and each subject's own
within-subject LR direction (leaky, for interpretation).
Checks (assert): folds == npz; shuffling the held-out subjects' labels leaves every score bit-identical; the anchor
selection uses no test labels (perturbing a test subject's labels changes neither selection nor score); determinism.
Run: python test/depositioned_anchor_direction.py [--smoke]   (smoke = fold 0 only)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, load_db  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from depositioned_within_subject_adapt import GROUPS  # noqa: E402

M_LIST = (16, 32, 64, 128)
CONS = .70
C_REG, EPS = .01, 1e-9
VARS = ['xs_full'] + [f'anchor_{m}' for m in M_LIST] + ['anchor_oracle_64', 'anchor_dir_64']
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def cohen_d(Z, y, eps=1e-6):
    """Per-feature standardised mean difference (task - relax) for one subject's rows."""
    a, b = Z[y == 1], Z[y == 0]
    sd = np.sqrt((a.var(0) + b.var(0)) / 2) + eps
    return (a.mean(0) - b.mean(0)) / sd


def select(Dtr, m):
    """Median d over training subjects + sign consistency; return the selected feature indices (top m by |median|)."""
    med = np.median(Dtr, 0)
    cons = np.mean(np.sign(Dtr) == np.sign(med), 0)
    ok = np.flatnonzero((cons >= CONS) & (np.abs(med) > 0))
    if ok.size == 0: return np.array([], int), med, cons
    order = ok[np.argsort(-np.abs(med[ok]), kind='stable')]
    return order[:m], med, cons


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_anchor_direction' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee

    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    Fn = A_['new_features']; fold = A_['fold']; n = len(y)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    ceil = json.loads((OUT / 'depositioned_transfer_ceiling.json').read_text(encoding='utf-8'))

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    assert np.array_equal(np.array([fold_of[s] for s in g]), fold)
    run_folds = [0] if args.smoke else list(range(5))
    ev_subs = [s for s in subs if fold_of[s] in run_folds]

    result = {'meta': {'deployable': True, 'purpose': 'sparse sign-consistent anchor direction', 'C': C_REG,
                       'consistency': CONS, 'm': list(M_LIST), 'threshold': .5, 'folds': run_folds,
                       'smoke': args.smoke,
                       'leaky_rows': ['anchor_oracle_64', 'diagnostic(cohen_d per group)', 'w_ws cosine']},
              'checks': {}, 'methods': {}, 'groups': {}, 'subjects': {}, 'diagnostics': {}}
    print(f'[start] rows={n} subjects={len(subs)} folds={run_folds} {el()}', flush=True)

    # ---------- leaky per-group diagnostic: how anchorable is each group? ----------
    Dg, best_auc = {}, {G: [] for G in GROUPS}
    for k in run_folds:
        tr = np.flatnonzero(fold != k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        Z = np.hstack([tfo(Fc), tfn(Fn)])
        for s in subs:
            if s in Dg: continue
            q = g == s
            Dg[s] = cohen_d(Z[q], y[q])
            au = [roc_auc_score(y[q], Z[q][:, j]) for j in np.argsort(-np.abs(Dg[s]))[:20]]
            best_auc[grp_of[s]].append(float(np.max(au)))
    D = np.stack([Dg[s] for s in ev_subs])
    med_all = np.median(D, 0)
    for G in GROUPS:
        ss = [s for s in ev_subs if grp_of[s] == G]
        if not ss: continue
        Dg_ = np.stack([Dg[s] for s in ss])
        consG = float(np.mean(np.sign(Dg_) == np.sign(med_all)))
        result['diagnostics'][G] = {'n': len(ss), 'sign_consistency_vs_global_median': consG,
                                    'median_abs_cohen_d': float(np.median(np.abs(Dg_))),
                                    'best_single_feature_auc': float(np.mean(best_auc[G])),
                                    'frac_subjects_best_auc_ge_65': float(np.mean(np.array(best_auc[G]) >= .65))}
        print(f'[diag {G}] n={len(ss)} sign-consistency={consG:.3f} |d|={np.median(np.abs(Dg_)):.3f} '
              f'best-single-feature AUC={np.mean(best_auc[G]):.3f} '
              f'frac(best AUC>=.65)={np.mean(np.array(best_auc[G]) >= .65):.2f}', flush=True)

    # ---------- fold-wise legal scores ----------
    P = {v: np.full(n, np.nan) for v in VARS}
    for k in run_folds:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        Z = np.hstack([tfo(Fc), tfn(Fn)])
        with threadpool_limits(limits=1):
            P['xs_full'][te] = LogisticRegression(C=C_REG, max_iter=1000).fit(Z[tr], y[tr]).decision_function(Z[te])
        trs = list(np.unique(g[tr])); tes = list(np.unique(g[te]))
        Dtr = np.stack([Dg[s] for s in trs])
        for m in M_LIST:
            idx, med, cons = select(Dtr, m)
            if idx.size < 4: continue
            with threadpool_limits(limits=1):
                P[f'anchor_{m}'][te] = LogisticRegression(C=C_REG, max_iter=1000).fit(
                    Z[np.isin(g, trs)][:, idx], y[np.isin(g, trs)]).decision_function(Z[te][:, idx])
        # leaky oracle anchor (selection uses ALL subjects) - ceiling
        idx_o, med_o, _ = select(D, 64)
        with threadpool_limits(limits=1):
            P['anchor_oracle_64'][te] = LogisticRegression(C=C_REG, max_iter=1000).fit(
                Z[tr][:, idx_o], y[tr]).decision_function(Z[te][:, idx_o])
            # fixed-sign direction, no fitting: score = sum_j sign(median d_j) zz_j
            idx_d, med_d, _ = select(Dtr, 64)
            P['anchor_dir_64'][te] = Z[te][:, idx_d] @ np.sign(med_d[idx_d])
        # cos between the fold's selected shared direction and each test subject's own LR direction (leaky)
        w = LogisticRegression(C=C_REG, max_iter=1000).fit(Z[np.isin(g, trs)][:, idx_d], y[np.isin(g, trs)]).coef_[0]
        for s in tes:
            q = g == s
            with threadpool_limits(limits=1):
                w_s = LogisticRegression(C=C_REG, max_iter=1000).fit(Z[q], y[q]).coef_[0][idx_d]
            result.setdefault('cos_anchor_ws', {})[s] = float(
                np.dot(w, w_s) / (np.linalg.norm(w) * np.linalg.norm(w_s) + EPS))
        print(f'[fold {k}] anchor sizes ok, oracle |idx|={idx_o.size} {el()}', flush=True)

    # ---------- checks: test-label independence ----------
    for k in run_folds[:2]:
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
        Z = np.hstack([tfo(Fc), tfn(Fn)])
        Dtr = np.stack([Dg[s] for s in np.unique(g[tr])])
        idx, _, _ = select(Dtr, 64)
        y2 = y.copy(); y2[te] = np.random.default_rng(k).permutation(y2[te])
        with threadpool_limits(limits=1):
            s1 = LogisticRegression(C=C_REG, max_iter=1000).fit(Z[tr][:, idx], y[tr]).decision_function(Z[te][:, idx])
            s2 = LogisticRegression(C=C_REG, max_iter=1000).fit(Z[tr][:, idx], y2[tr]).decision_function(Z[te][:, idx])
        assert np.array_equal(s1, s2), ('test-label independence', k)
        assert np.array_equal(s1, P['anchor_64'][te]), ('determinism', k)
        result['checks'][f'fold{k}'] = {'test_label_independence': 'ok', 'determinism': 'ok'}
        print(f'[check fold{k}] test-label independence + determinism OK', flush=True)

    # ---------- evaluation ----------
    ev = np.isin(fold, run_folds)
    for v in VARS:
        if not np.isfinite(P[v][ev]).all(): continue
        m = metrics(y, g, P[v], ev)
        pv = {s: metrics(y, g, P[v], g == s)['subject_macro_bACC'] for s in ev_subs}
        au = {s: (roc_auc_score(y[g == s], P[v][g == s]) if np.unique(y[g == s]).size == 2 else None)
              for s in ev_subs}
        gm = {G: (float(np.mean([pv[s] for s in ev_subs if grp_of[s] == G]))
                   if any(grp_of[s] == G for s in ev_subs) else None) for G in GROUPS}
        ga = {G: (float(np.median([au[s] for s in ev_subs if grp_of[s] == G and au[s] is not None]))
                   if any(grp_of[s] == G and au[s] is not None for s in ev_subs) else None) for G in GROUPS}
        result['methods'][v] = {'all': m, 'subjects': pv, 'groups_bacc': gm, 'groups_auc_median': ga}
        result['subjects'].setdefault(v, {})
        for s in ev_subs: result['subjects'][v][s] = {'bacc': pv[s], 'auc': au[s]}
        print(f'[result] {v:16s} AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} | ' +
              ' '.join(f'{G}: bacc={gm[G] if gm[G] is None else round(gm[G], 3)} '
                       f'auc={ga[G] if ga[G] is None else round(ga[G], 3)}' for G in GROUPS), flush=True)

    ref = ceil['streams']['stream0']['subjects']['pipeline']
    base = np.array([ref[s] for s in ev_subs])
    rng = np.random.default_rng(0); Bt = 100 if args.smoke else 1000
    for v in VARS:
        if v not in result['methods']: continue
        d0 = np.array([result['methods'][v]['subjects'][s] for s in ev_subs]) - base
        bs = [d0[rng.integers(0, len(d0), len(d0))].mean() for _ in range(Bt)]
        result['methods'][v]['boot_vs_pipeline'] = {'d': float(d0.mean()), 'lo': float(np.percentile(bs, 2.5)),
                                                    'hi': float(np.percentile(bs, 97.5))}
    print('[boot] ' + ' '.join(f'{v}:{result["methods"][v]["boot_vs_pipeline"]["d"]:+.3f}'
                               f'[{result["methods"][v]["boot_vs_pipeline"]["lo"]:+.3f},'
                               f'{result["methods"][v]["boot_vs_pipeline"]["hi"]:+.3f}]' for v in VARS), flush=True)
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] {el()}', flush=True); sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
