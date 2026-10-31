"""Step-4: LEGAL local-direction estimators - can pseudo labels supply the ~64-128 recent windows that
depositioned_local_direction.py showed are sufficient (last64 recovers 90% of the low-group oracle gap, cos(last64,
exp) = .75, cos(last64, xs) = .01)?

Everything here uses only the past (causal): the test subject's unlabelled windows ZZ[:t0] and its own past fused
score L[:t0] = mean logit(all_prank, resnet_whiten_logit_delta, resnet_fbank_logit_delta). Test labels are used ONLY
in the (explicitly leaky) pseudo-label-accuracy diagnostic and in the final evaluation.

Protocol identical to the parents: 889 features imputed/scaled with the fold-train rows; per-subject stream order =
concat(relax, task) permuted by default_rng([stream_seed, subject_index]); folds = StratifiedGroupKFold(5, shuffle,
rs0) (asserted == npz); refit every R=16 from t0=48; pseudo labels = top / bottom 30% of a score; LR C=.01 on the
past-scaled 889 features; z = (s_new - mean(s_past)) / std(s_past) clipped +-3; score = L + z; causal thr_mean256;
threshold .5. Rows t0<48 fall back to L.
Variants (fixed before running; no test-set selection):
  ref        v2 pseudo_lr_full exactly (pseudo labels from L[:t0], uniform weights) -> reference, must reproduce
             v2 fuse_pseudo_lr_full_thr per subject (1e-9)
  rec64      same pseudo labels, sample_weight = exp(-(t0-1-i)/64): the legal analogue of tau64 (.96 of the gap)
  last128    pseudo labels from the last 128 past L values only (the legal analogue of last128, .95 of the gap)
  last64     pseudo labels from the last 64 past L values only (legal analogue of last64, .90 of the gap)
  md64/md128 mean-difference direction instead of a full 889-d LR: d = mean(x|pseudo-task) - mean(x|pseudo-relax)
             over the same pseudo-labelled rows as last64/last128, score = d.x (identity-covariance LDA; the true-label
             version reached .78 of the gap with far fewer parameters)
  pca_rec64  rec64's labels and weights but the LR lives on the past PCA-16 subspace (lower variance direction)
  em2        two rounds of soft labels: 1-D 2-component GaussianMixture on L[:t0] (means initialised at the 30/70
             percentiles) -> responsibilities r -> weighted LR (y = r>=.5, w = |2r-1|); recompute the score on the
             past, refit the mixture on L2 = L + z, refit the LR. Replaces hard top/bottom-30% labels by calibrated
             soft ones, which is the only place where a legal mechanism can differ from `ref` in *label quality*.
Leaky diagnostic (reported, never used by a score): pseudo-label accuracy of `ref`/`last128` labels vs the true labels
of the same rows, per group and per recency band (last 64 / 128 / all past), at every refit point.
Checks (assert): folds == npz; ref reproduces v2 fuse_pseudo_lr_full_thr per subject (1e-9); shuffling a subject's
test labels leaves every variant bit-identical; prefix invariance (perturb future ZZ and L -> earlier scores
unchanged, later change); subject isolation (single subject == pooled).
Evaluation: all / mature AUC, subject-macro bACC, micro bACC; groups (high/mid/low, reporting); paired subject
bootstrap (1000; 100 in smoke) of every variant vs fuse_pseudo_lr_full_thr, and the low-group recovery fraction of
the oracle gap (.5393 -> .6955) with the pre-registered rule reused from step 3: a variant "closes the locality gap"
if it recovers >= 50% of it with a CI>0.
Parallel over subjects (joblib threading, 8 jobs, 1 BLAS thread each). Deterministic.
Run: python test/depositioned_local_legal.py [--smoke]   (smoke: stream0, fold0, bootstrap 100)
"""
import argparse, json, sys, time, warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from depositioned_history_delta import arrange, load_db  # noqa: E402
from depositioned_resnet_history_delta import stream_order, logit, sig  # noqa: E402
from depositioned_adaptive_threshold import shift_subject  # noqa: E402
from depositioned_symmetric_contrast import impute_scale  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402
from depositioned_within_subject_adapt import GROUPS, MATURE, STREAM_SEEDS, THR_K, R_STEP, MIN_PAST, MIN_CLS, NPC, \
    PSEUDO_Q, pseudo_labels  # noqa: E402

VARS = ('ref', 'rec64', 'last128', 'last64', 'md64', 'md128', 'pca_rec64', 'em2')
LEAKY_DIAG = ('ref', 'last128')
CLIP, EPS, N_JOBS, C_REG, TAU_REC, NPC_ = 3., 1e-6, 8, .01, 64., NPC
GAP = (.5392642517348247, .6954721330329693)  # stored pipeline / oracle_past low-group means (rule denominators)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def zc(sp, sn):
    return np.clip((sn - sp.mean()) / max(sp.std(), EPS), -CLIP, CLIP)


def lr():
    return LogisticRegression(C=C_REG, max_iter=1000)


def em_resp(lp):
    """1-D 2-component Gaussian mixture responsibilities for 'task' (component with the larger mean)."""
    q = np.quantile(lp, [.3, .7])
    if q[1] - q[0] < 1e-9: return np.full(len(lp), .5)
    g = GaussianMixture(2, means_init=q[:, None], random_state=0, n_init=1, reg_covar=1e-6).fit(lp[:, None])
    hi = int(np.argmax(g.means_.ravel()))
    return g.predict_proba(lp[:, None])[:, hi]


def adapt(ZZ, L, y):
    """One subject in stream order. y is used only for the leaky pseudo-label-accuracy diagnostic."""
    n = len(L); out = {v: np.full(n, np.nan) for v in VARS}
    diag = {k: [] for k in ('ref', 'ref_64', 'ref_128', 'last64', 'last128')}
    for t0 in range(MIN_PAST, n, R_STEP):
        t1 = min(n, t0 + R_STEP)
        with threadpool_limits(limits=1):
            sc = StandardScaler().fit(ZZ[:t0]); xp, xn = sc.transform(ZZ[:t0]), sc.transform(ZZ[t0:t1])
            idx, yl = pseudo_labels(L[:t0])
            m = lr().fit(xp[idx], yl)
            sp, sn = m.decision_function(xp), m.decision_function(xn)
            out['ref'][t0:t1] = zc(sp, sn)
            sw = np.exp(-(t0 - 1 - idx) / TAU_REC)
            m = lr().fit(xp[idx], yl, sample_weight=sw)
            out['rec64'][t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
            for K, v in ((128, 'last128'), (64, 'last64')):
                lo = max(0, t0 - K)
                i2, y2 = pseudo_labels(L[lo:t0])
                m = lr().fit(xp[lo:][i2], y2)
                out[v][t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
                d = xp[lo:][i2][y2 == 1].mean(0) - xp[lo:][i2][y2 == 0].mean(0)
                out['md' + str(K)][t0:t1] = zc(xp @ d, xn @ d)
                # leaky accuracy of these labels on the rows they label
                diag[v].append(float((y[lo:t0][i2] == y2).mean()))
            # pca_rec64: recency-weighted LR on the past PCA-16 subspace
            p = PCA(NPC_, svd_solver='full').fit(xp); pp, pn = p.transform(xp), p.transform(xn)
            m = LogisticRegression(C=.1, max_iter=1000).fit(pp[idx], yl, sample_weight=sw)
            out['pca_rec64'][t0:t1] = zc(m.decision_function(pp), m.decision_function(pn))
            # em2: soft labels from L, then from L + z
            r = em_resp(L[:t0])
            yw = (r >= .5).astype(int)
            if len(np.unique(yw)) == 2:
                m = lr().fit(xp, yw, sample_weight=np.abs(2 * r - 1) + 1e-3)
                zp = m.decision_function(xp)
                r2 = em_resp(L[:t0] + zc(zp, zp))
                yw2 = (r2 >= .5).astype(int)
                if len(np.unique(yw2)) == 2:
                    m = lr().fit(xp, yw2, sample_weight=np.abs(2 * r2 - 1) + 1e-3)
                    out['em2'][t0:t1] = zc(m.decision_function(xp), m.decision_function(xn))
            # leaky pseudo-label accuracy per recency band
            diag['ref'].append(float((y[:t0][idx] == yl).mean()))
            for b in (64, 128):
                lo = max(0, t0 - b)
                i3, y3 = pseudo_labels(L[lo:t0])
                diag[f'ref_{b}'].append(float((y[lo:t0][i3] == y3).mean()))
    sc_ = {v: L + np.where(np.isfinite(out[v]), out[v], 0.) for v in VARS}
    dg = {k: float(np.mean(v)) if v else None for k, v in diag.items()}
    return sc_, dg


def per_subject_bacc(y, g, p, subs):
    return {s: metrics(y, g, p, g == s)['subject_macro_bACC'] for s in subs}


def main():
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_local_legal' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS

    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(OUT / 'depositioned_aperiodic_emg_oof.npz'); B_ = np.load(OUT / 'depositioned_resnet_timefreq_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    for q in ('y', 'groups', 'fold'): assert np.array_equal(A_[q], B_[q]), q
    Fn = A_['new_features']; assert Fc.shape[1] == 858 and Fn.shape == (len(y), 31)
    fold = A_['fold']; n = len(y)
    diag = json.loads((OUT / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    v2 = json.loads((OUT / 'depositioned_within_subject_adapt_v2.json').read_text(encoding='utf-8'))
    ceil = json.loads((OUT / 'depositioned_transfer_ceiling.json').read_text(encoding='utf-8'))

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    assert np.array_equal(np.array([fold_of[s] for s in g]), fold)
    run_folds = [0] if args.smoke else list(range(5))
    ev_subs = [s for s in subs if fold_of[s] in run_folds]
    sidx = {s: j for j, s in enumerate(subs)}

    result = {'meta': {'deployable': True, 'purpose': 'legal local-direction estimators (recency / mean-diff / EM)',
                       'reference': 'fuse_pseudo_lr_full_thr (v2)', 'refit_step': R_STEP, 'min_past': MIN_PAST,
                       'pseudo_q': PSEUDO_Q, 'C': C_REG, 'tau': TAU_REC, 'pca': NPC_, 'clip': CLIP,
                       'thr': {'k': THR_K, 'min_past': 16, 'mode': 'mean'}, 'threshold': .5,
                       'mature_index': MATURE, 'stream_seeds': list(sseeds), 'folds': run_folds, 'smoke': args.smoke,
                       'rule': 'closes locality gap if (low bACC - .5393)/(.6955-.5393) >= .50 with CI>0'},
              'checks': {}, 'streams': {}, 'pseudo_acc': {}, 'rule': {}}
    print(f'[start] rows={n} subjects={len(subs)} folds={run_folds} sseeds={sseeds} {el()}', flush=True)

    for sd in sseeds:
        order, t_c = stream_order(g, subs, sd)
        Bl = np.c_[logit(A_[f'stream{sd}__all_prank']), logit(B_[f'stream{sd}__resnet_whiten_logit_delta']),
                   logit(B_[f'stream{sd}__resnet_fbank_logit_delta'])]
        assert np.isfinite(Bl).all()
        L = Bl.mean(1)
        ZZt = np.full((n, 889), np.nan)
        for k in run_folds:
            tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
            tfo, tfn = impute_scale(Fc, tr), impute_scale(Fn, tr)
            ZZt[te] = np.hstack([tfo(Fc[te]), tfn(Fn[te])])
        ev = np.isin(fold, run_folds)
        P = {v: np.full(n, np.nan) for v in VARS}; DG = {}
        ts = time.perf_counter()
        gen = Parallel(n_jobs=N_JOBS, backend='threading', return_as='generator')(
            delayed(adapt)(ZZt[order[sidx[s]]], Bl[order[sidx[s]]].mean(1), y[order[sidx[s]]]) for s in ev_subs)
        for i, (s, (sc_, dg)) in enumerate(zip(ev_subs, gen)):
            loc = order[sidx[s]]
            for v in VARS: P[v][loc] = sig(shift_subject(sc_[v], THR_K, 'mean'))
            DG[s] = dg
            if (i + 1) % 10 == 0 or i + 1 == len(ev_subs):
                print(f'[s{sd}] adapted {i + 1}/{len(ev_subs)} {time.perf_counter() - ts:.1f}s {el()}', flush=True)
        for v in VARS: assert np.isfinite(P[v][ev]).all(), v

        # ---- checks
        for s in ev_subs[:2]:
            loc = order[sidx[s]]; ns = len(loc)
            sc0, _ = adapt(ZZt[loc], Bl[loc].mean(1), y[loc])
            for v in VARS:
                d = float(np.nanmax(np.abs(sig(shift_subject(sc0[v], THR_K, 'mean')) - P[v][loc])))
                assert d < 1e-9, ('isolation', s, v, d)
            s1, _ = adapt(ZZt[loc], Bl[loc].mean(1), np.random.default_rng(1).permutation(y[loc]))
            for v in VARS:
                assert np.array_equal(sig(shift_subject(sc0[v], THR_K, 'mean')),
                                      sig(shift_subject(s1[v], THR_K, 'mean')), equal_nan=True), ('label shuffle', s, v)
            cut = MIN_PAST + R_STEP * ((ns - MIN_PAST) // (2 * R_STEP)) + 5
            rng = np.random.default_rng(int(sidx[s])); Z1, B1 = ZZt[loc].copy(), Bl[loc].copy()
            Z1[cut:] += rng.normal(0, 2, Z1[cut:].shape); B1[cut:] += rng.normal(0, 2, B1[cut:].shape)
            sc2, _ = adapt(Z1, B1.mean(1), y[loc])
            for v in VARS:
                a, b = sc0[v], sc2[v]
                assert np.array_equal(a[:cut], b[:cut], equal_nan=True), ('prefix', s, v)
                d = np.abs(a[cut:] - b[cut:])
                assert np.isfinite(d).any() and d[np.isfinite(d)].max() > 0, ('prefix-changed', s, v)
            print(f'[check s{sd}] {s}: isolation / label-shuffle / prefix invariance OK (cut={cut})', flush=True)

        # ---- per-subject / per-group
        R = result['streams'][f'stream{sd}'] = {'methods': {}, 'groups': {}, 'subjects': {}}
        pv = {v: per_subject_bacc(y, g, P[v], ev_subs) for v in VARS}
        for v in VARS:
            R['methods'][v] = {sg: metrics(y, g, P[v], ev & (True if sg == 'all' else t_c >= MATURE))
                               for sg in ('all', 'mature')}
            gm = {G: float(np.mean([pv[v][s] for s in ev_subs if grp_of[s] == G and pv[v][s] is not None]))
                  for G in GROUPS}
            R['groups'][v] = {G: {'n': int(sum(1 for s in ev_subs if grp_of[s] == G)), 'mean': gm[G]} for G in GROUPS}
            R['subjects'][v] = {s: pv[v][s] for s in ev_subs}
            m = R['methods'][v]['all']
            print(f'[result s{sd}] {v:10s} all: AUC={m["AUC"]:.4f} mac={m["subject_macro_bACC"]:.4f} | ' +
                  ' '.join(f'{G}:{gm[G]:.3f}' for G in GROUPS), flush=True)
        # ---- reference reproduction (v2 fuse_pseudo_lr_full_thr per subject, stored in the ceiling table)
        if not args.smoke:
            ref = ceil['streams'][f'stream{sd}']['subjects']['pipeline']
            for s in ev_subs:
                assert abs(pv['ref'][s] - ref[s]) < 1e-9, ('v2 reproduction', sd, s, pv['ref'][s], ref[s])
            print(f'[check s{sd}] ref reproduces v2 fuse_pseudo_lr_full_thr per subject', flush=True)
        # ---- paired bootstrap vs stored legal pipeline per-subject bACC
        rng = np.random.default_rng(0); Bt = 100 if args.smoke else 1000
        base = np.array([ceil['streams'][f'stream{sd}']['subjects']['pipeline'][s] for s in ev_subs])
        bs = {v: [] for v in VARS}
        for _ in range(Bt):
            ii = rng.integers(0, len(ev_subs), len(ev_subs))
            b0 = base[ii].mean()
            for v in VARS:
                bs[v].append(np.array([pv[v][ev_subs[j]] for j in ii]).mean() - b0)
        R['boot'] = {v: {'d': float(np.mean([pv[v][s] for s in ev_subs]) - base.mean()),
                         'lo': float(np.percentile(bs[v], 2.5)), 'hi': float(np.percentile(bs[v], 97.5))}
                     for v in VARS}
        print(f'[boot s{sd}] ' + ' '.join(f'{v}:{R["boot"][v]["d"]:+.3f}[{R["boot"][v]["lo"]:+.3f},'
                                          f'{R["boot"][v]["hi"]:+.3f}]' for v in VARS), flush=True)
        # ---- leaky pseudo-label accuracy per group
        PA = result['pseudo_acc'][f'stream{sd}'] = {}
        keys = sorted({k for s in ev_subs for k in DG[s]})
        for k in keys:
            PA[k] = {G: float(np.mean([DG[s][k] for s in ev_subs if grp_of[s] == G and DG[s].get(k) is not None]))
                     for G in GROUPS}
        print(f'[pseudo-acc s{sd}] ' + ' | '.join(
            f'{k}: ' + ' '.join(f'{G}={PA[k][G]:.3f}' for G in GROUPS) for k in keys), flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # ---- pre-registered rule on the low group (stream 0 unless smoke)
    sd0 = f'stream{STREAM_SEEDS[0]}'; G0 = result['streams'][sd0]['groups']
    pipe, orc = GAP; gap = orc - pipe
    rec = {v: (G0[v]['low']['mean'] - pipe) / gap for v in VARS}
    closes = [v for v in VARS if rec[v] >= .5 and result['streams'][sd0]['boot'][v]['lo'] > 0]
    result['rule'] = {'recovery_low': rec, 'closes_gap': closes}
    print('[rule] low-group recovery fraction: ' + ' '.join(f'{v}:{rec[v]:.2f}' for v in VARS), flush=True)
    print(f'[rule] closes >=50% of the locality gap with CI>0: {closes}', flush=True)
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] {el()}', flush=True); sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
