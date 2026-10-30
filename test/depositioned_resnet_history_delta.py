"""De-positioned causal unlabeled history delta for the ResNet / merge_best-style model (database/, no timestamps).

Stream order per subject = concat(relax, task) permuted by default_rng([stream_seed, subject_index]) (as
depositioned_history_delta.arrange(mode='independent')). Stream seed 0 main, 1/2 robustness.
Outer CV: StratifiedGroupKFold(5, shuffle, rs=0) on seed-0 arrangement; the subject->fold map is reused for seeds 1/2.

ResNet (raw_waveform_window6: ResNet1d, per-window demean/std/clip, 4 s crop aug, 5-crop TTA, AdamW 1e-3/1e-2, BS 128,
max 50 ep, patience 8 on val AUC). Per outer fold: train on 3/4 of train subjects (A), early-stop on the other 1/4 (B) =
first split of SGKF(4, shuffle, rs=fold) (same as raw_waveform_window6). ResNet seeds 0,1,2 (seed*1000+fold).
The ResNet itself sees only the current window -> its scores do not depend on stream order, so it is trained once and
reused for all stream seeds; history features are recomputed in each stream order.
Heads on the 128-d penultimate embedding (TTA-mean) are fit on B (subjects the ResNet did not train on), one head per
ResNet seed, probability averaged over seeds. Embedding standardized with B statistics.
Variants (fixed in advance):
  position_only__{lr,hgb}            [t]                         control, must be AUC ~.5
  hgb_current / hgb_delta32          HGB on 858 f [z] / [z, z - mean(past 32 own z)] on full train (prior script)
  resnet_current                     sigmoid(mean logit over ResNet seeds)
  resnet_logit_delta                 LR on B: [l, l - mean(past 32 l)]  (l = ResNet logit)
  resnet_emb_current__{lr,hgb}       head on B: [e]
  resnet_emb_delta__{lr,hgb}         head on B: [e, e - mean(past 32 e)]
  fusion_current / fusion_delta      sigmoid(mean(logit hgb_current, logit resnet_emb_current__lr)) / delta counterparts
  fusionR_current / fusionR_delta    sigmoid(mean(logit hgb_*, logit resnet_current / resnet_logit_delta))
Raw-channel delta (x - mean past 32 raw windows) is not trained: windows are not phase-aligned, so the past mean of raw
waveforms ~ 0 and the extra channel ~ duplicate of x; the script logs the measured rms ratio / correlation as evidence.
History: past 32 own windows, no prior, 0 when no past. Threshold .5. No test labels used anywhere.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
DB = OUT.parent / 'database'
sys.path.insert(0, str(OUT))
from raw_waveform_window6 import ResNet1d, augment, normalize, WIN, CROP, N_TTA, BS, MAX_EP, PATIENCE, LR, WD  # noqa: E402
from depositioned_history_delta import arrange, make_clf, hist_subject, boot, load_db  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

K, MATURE = 32, 120
RESNET_SEEDS = (0, 1, 2)
STREAM_SEEDS = (0, 1, 2)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def load_raw(subs):
    """Canonical order: per subject concat(relax, task) -- identical to load_db / arrange before permutation."""
    X, F, y, g = [], [], [], []
    for s in subs:
        parts = [np.load(DB / 'windows' / k / f'{s}.npz') for k in ('relax', 'task')]
        X.append(np.vstack([p['x'] for p in parts])); F.append(np.vstack([p['f'] for p in parts]).astype(float))
        y.append(np.concatenate([p['label'] for p in parts]).astype(int)); g.append(np.full(len(y[-1]), s))
    return np.vstack(X), np.vstack(F), np.concatenate(y), np.concatenate(g)


def stream_order(g, subs, seed):
    """Return canonical row indices in stream order (subject-major) and per-row position t (canonical index)."""
    order, t = [], np.zeros(len(g), int)
    for j, s in enumerate(subs):
        loc = np.flatnonzero(g == s)
        perm = np.random.default_rng([seed, j]).permutation(len(loc))
        order.append(loc[perm]); t[loc[perm]] = np.arange(len(loc))
    return order, t


def hist_delta(Z, order):
    """Z canonical rows; delta[i] = Z[i] - mean of Z over previous <=K stream rows of same subject (0 if none)."""
    D = np.zeros_like(Z)
    for loc in order:
        zs = Z[loc]; h32 = hist_subject(zs)[1]
        D[loc] = zs - h32
    return D


def train_resnet(Xg, yg, y, tr, va, seed, tag, max_ep):
    tick = time.perf_counter(); dev = Xg.device
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    net = ResNet1d().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); crit = nn.BCEWithLogitsLoss()
    best, state, stale, best_ep = -1., None, 0, 0
    for ep in range(1, max_ep + 1):
        net.train(); perm = rng.permutation(tr); tot = 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2: continue
            idt = torch.as_tensor(ids, device=dev)
            loss = crit(net(augment(Xg[idt], gen)).ravel(), yg[idt])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            tot += loss.item() * len(ids)
        auc = roc_auc_score(y[va], embed(net, Xg, va)[1])
        if auc > best + 1e-4:
            best, state, stale, best_ep = auc, {k: v.detach().clone() for k, v in net.state_dict().items()}, 0, ep
        else:
            stale += 1
        print(f'    [{tag}] ep{ep:02d}/{max_ep} loss={tot / len(tr):.4f} valAUC={auc:.4f} best={best:.4f}@{best_ep} '
              f't={time.perf_counter() - tick:.0f}s {el()}', flush=True)
        if stale >= PATIENCE: break
    net.load_state_dict(state); net.eval()
    return net, {'best_epoch': int(best_ep), 'epochs_run': int(ep), 'best_val_auc': float(best),
                 'seconds': float(time.perf_counter() - tick)}


@torch.no_grad()
def embed(net, Xg, ids, X_override=None):
    """TTA-mean (5 equispaced 4 s crops) penultimate 128-d embedding and logit (= head(mean emb), linear in eval)."""
    net.eval(); E = []
    offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int)
    src = Xg if X_override is None else X_override
    for b in range(0, len(ids), 512):
        xb = src[torch.as_tensor(ids[b:b + 512], device=src.device)] if X_override is None else src[b:b + 512]
        E.append(torch.stack([net.blocks(net.stem(xb[:, None, o:o + CROP])).mean(-1) for o in offs]).mean(0))
    E = torch.cat(E)
    lg = net.head[3](E).ravel()  # head = [AvgPool, Flatten, Dropout, Linear]
    return E.cpu().numpy().astype(np.float64), lg.cpu().numpy().astype(np.float64)


def raw_delta_evidence(Xn, g, subs):
    order, _ = stream_order(g, subs, 0); r, c = [], []
    for loc in order[:10]:
        xs = Xn[loc].astype(np.float64); h = hist_subject(xs)[1]
        d = xs - h
        r.append(np.sqrt((h[1:] ** 2).mean()) / np.sqrt((xs ** 2).mean()))
        c.append(np.nanmean([np.corrcoef(xs[i], d[i])[0, 1] for i in range(1, len(loc)) if xs[i].std() > 0]))
    return {'rms_ratio_pastmean_over_x': float(np.mean(r)), 'mean_corr_x_vs_delta': float(np.mean(c))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_resnet_history_delta' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rseeds = RESNET_SEEDS[:1] if args.smoke else RESNET_SEEDS
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    max_ep = 2 if args.smoke else MAX_EP
    nb = 100 if args.smoke else 1000

    man, subs, data = load_db()
    Xraw, Fc, y, g = load_raw(subs)
    assert np.array_equal(Fc, np.vstack([data[s][0] for s in subs])) and np.array_equal(y, np.concatenate([data[s][1] for s in subs]))
    print(f'[start] rows={len(y)} subjects={len(subs)} device={dev} smoke={args.smoke} rseeds={rseeds} sseeds={sseeds}', flush=True)
    # consistency with prior arrange(): stream order reproduces its feature/label sequence exactly
    for sd in sseeds:
        Xa, ya, ga, ta = arrange(subs, data, sd, 'independent')
        order, t = stream_order(g, subs, sd); oc = np.concatenate(order)
        assert np.array_equal(Fc[oc], Xa) and np.array_equal(y[oc], ya) and np.array_equal(t[oc], ta)
    print('[check] stream_order == depositioned_history_delta.arrange(independent) for all stream seeds', flush=True)

    # folds (seed-0 arrangement, like prior script) -> subject->fold map
    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent')
    fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g])
    folds = []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k + 1).split(tr[:, None], y[tr], g[tr]))
        A, B = tr[a], tr[b]
        assert not (set(g[A]) & set(g[B])) and not (set(g[tr]) & set(g[te]))
        folds.append((tr, te, A, B))

    Xn, n_flat = normalize(Xraw)
    ev_raw = raw_delta_evidence(Xn, g, subs)
    print(f'[raw-delta evidence] {ev_raw} flat_windows={n_flat}', flush=True)
    Xg = torch.from_numpy(Xn).to(dev); yg = torch.tensor(y, dtype=torch.float32, device=dev)

    # ---------------- ResNet training (once; order-independent) ----------------
    EMB = {}; LOG = {}; runs = []; nets = {}
    all_ids = np.arange(len(y))
    for k, (tr, te, A, B) in enumerate(folds):
        if args.smoke and k > 0: break
        for rs in rseeds:
            tag = f'resnet rs{rs} fold{k + 1}/5'
            net, info = train_resnet(Xg, yg, y, A, B, 1000 * rs + k + 1, tag, max_ep)
            E, L = embed(net, Xg, all_ids)
            EMB[(k, rs)], LOG[(k, rs)] = E, L
            if rs == rseeds[0]: nets[k] = net
            runs.append({'fold': k, 'resnet_seed': rs, **info,
                         'test_auc': float(roc_auc_score(y[te], L[te])), 'B_auc': float(roc_auc_score(y[B], L[B]))})
            print(f'[resnet] {tag} best_ep={info["best_epoch"]}/{info["epochs_run"]} valAUC={info["best_val_auc"]:.4f} '
                  f'testAUC={runs[-1]["test_auc"]:.4f} {info["seconds"]:.0f}s {el()}', flush=True)
    done_folds = sorted({k for k, _ in EMB})

    # ---------------- causality checks on the ResNet embedding (order-free, per window) ----------------
    checks = {}
    for k in done_folds:
        net = nets[k]; te = folds[k][1]; s0 = np.unique(g[te])[0]
        loc = np.flatnonzero(g == s0)
        E_alone, L_alone = embed(net, Xg, loc)
        d_iso = float(np.abs(E_alone - EMB[(k, rseeds[0])][loc]).max())
        Xalt = Xg[torch.as_tensor(loc, device=dev)].clone(); cut = len(loc) // 2
        Xalt[cut:] = Xalt[cut:] * -3. + 1.
        E_alt, _ = embed(net, Xg, loc, X_override=Xalt)
        d_pre = float(np.abs(E_alt[:cut] - E_alone[:cut]).max()); d_fut = float(np.abs(E_alt[cut:] - E_alone[cut:]).max())
        assert d_iso < 1e-3 and d_pre < 1e-3 and d_fut > 1e-2, (d_iso, d_pre, d_fut)
        checks[f'emb_fold{k}'] = {'subject': str(s0), 'max_abs_alone_vs_pooled': d_iso, 'max_abs_past_after_future_change': d_pre,
                                  'max_abs_future_changed': d_fut}
    print(f'[check] embedding subject isolation + per-window locality ok {checks} {el()}', flush=True)

    # ---------------- heads per stream seed ----------------
    result = {'meta': {'K': K, 'mature_index': MATURE, 'threshold': .5, 'rows': int(len(y)), 'subjects': len(subs),
                       'resnet_seeds': list(rseeds), 'stream_seeds': list(sseeds), 'device': dev,
                       'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None, 'smoke': args.smoke,
                       'raw_delta_evidence': ev_raw},
              'resnet_runs': runs, 'checks': checks}
    pairs = [('resnet_logit_delta', 'resnet_current'), ('resnet_emb_delta__lr', 'resnet_emb_current__lr'),
             ('resnet_emb_delta__hgb', 'resnet_emb_current__hgb'), ('hgb_delta32', 'hgb_current'),
             ('fusion_delta', 'fusion_current'), ('fusionR_delta', 'fusionR_current'),
             ('fusion_current', 'hgb_current'), ('fusion_delta', 'hgb_delta32'), ('resnet_emb_current__lr', 'resnet_current')]
    oof = {'y': y, 'groups': g, 'fold': fold}
    for sd in sseeds:
        order, t = stream_order(g, subs, sd)
        S = {}
        names = ['position_only__lr', 'position_only__hgb', 'hgb_current', 'hgb_delta32', 'resnet_current', 'resnet_logit_delta',
                 'resnet_emb_current__lr', 'resnet_emb_delta__lr', 'resnet_emb_current__hgb', 'resnet_emb_delta__hgb']
        for n in names: S[n] = np.full(len(y), np.nan)
        with threadpool_limits(limits=8):
            for k in done_folds:
                tr, te, A, B = folds[k]
                # position control
                T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                # HGB on 858 features (prior protocol)
                med = np.nanmedian(np.where(np.isfinite(Fc[tr]), Fc[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
                Ff = np.where(np.isfinite(Fc), Fc, med[None, :])
                Z = StandardScaler().fit(Ff[tr]).transform(Ff); DZ = hist_delta(Z, order)
                t1 = time.perf_counter()
                S['hgb_current'][te] = make_clf('hgb').fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
                ZD = np.hstack([Z, DZ])
                hgb_d = make_clf('hgb').fit(ZD[tr], y[tr])
                S['hgb_delta32'][te] = hgb_d.predict_proba(ZD[te])[:, 1]
                print(f'[stream {sd} fold {k + 1}/5] hgb current/delta32 fit_s={time.perf_counter() - t1:.1f} {el()}', flush=True)
                # ResNet-based heads, per ResNet seed, prob-averaged
                acc = {n: [] for n in names if n.startswith('resnet_') and n != 'resnet_current'}
                lg = []
                head_store = {}
                for rs in rseeds:
                    E, L = EMB[(k, rs)], LOG[(k, rs)]
                    lg.append(L[te])
                    Es = StandardScaler().fit(E[B]).transform(E); DE = hist_delta(Es, order)
                    ls = (L - L[B].mean()) / max(L[B].std(), 1e-8); DL = hist_delta(ls[:, None], order)
                    feats = {'resnet_logit_delta': (np.c_[ls, DL], 'lr'),
                             'resnet_emb_current__lr': (Es, 'lr'), 'resnet_emb_delta__lr': (np.hstack([Es, DE]), 'lr'),
                             'resnet_emb_current__hgb': (Es, 'hgb'), 'resnet_emb_delta__hgb': (np.hstack([Es, DE]), 'hgb')}
                    for n, (Fm, c) in feats.items():
                        m = make_clf(c).fit(Fm[B], y[B]); acc[n].append(m.predict_proba(Fm[te])[:, 1])
                        if rs == rseeds[0] and n == 'resnet_emb_delta__lr': head_store = {'m': m, 'sc': StandardScaler().fit(E[B])}
                S['resnet_current'][te] = sig(np.mean(lg, 0))
                for n, v in acc.items(): S[n][te] = np.mean(v, 0)
                # strict causality on the full pipeline for 2 test subjects: prefix invariance + subject isolation
                for s0 in np.unique(g[te])[:2]:
                    loc = order[list(subs).index(s0)]; cut = len(loc) // 2  # stream-ordered rows of this subject
                    # (a) 858-feature HGB delta: alone and with altered future
                    zs = Z[loc]; alone = np.hstack([zs, zs - hist_subject(zs)[1]])
                    assert np.array_equal(alone, ZD[loc])
                    alt = zs.copy(); alt[cut:] += 100.; fa = np.hstack([alt, alt - hist_subject(alt)[1]])
                    assert np.array_equal(hgb_d.predict_proba(fa)[:cut, 1], S['hgb_delta32'][loc[:cut]])
                    # (b) ResNet embedding delta head: recompute subject alone from raw, then alter future raw windows
                    net = nets[k]; m, sc = head_store['m'], head_store['sc']
                    def pipe(Xsub):
                        e, _ = embed(net, Xg, loc, X_override=Xsub); es = sc.transform(e)
                        return m.predict_proba(np.hstack([es, es - hist_subject(es)[1]]))[:, 1]
                    Xs = Xg[torch.as_tensor(loc, device=dev)].clone(); p0 = pipe(Xs)
                    Es0 = sc.transform(EMB[(k, rseeds[0])]); DE0 = hist_delta(Es0, order)
                    p_pool = m.predict_proba(np.hstack([Es0, DE0])[loc])[:, 1]
                    Xs2 = Xs.clone(); Xs2[cut:] = Xs2[cut:] * -3. + 1.; p1 = pipe(Xs2)
                    d_iso, d_pre, d_fut = (float(np.abs(p0 - p_pool).max()), float(np.abs(p1[:cut] - p0[:cut]).max()),
                                           float(np.abs(p1[cut:] - p0[cut:]).max()))
                    assert d_iso < 1e-4 and d_pre < 1e-4 and d_fut > 1e-4, (d_iso, d_pre, d_fut)
                    checks[f'pipeline_stream{sd}_fold{k}_{s0}'] = {'alone_vs_pooled': d_iso, 'past_after_future_change': d_pre,
                                                                    'future_changed': d_fut}
                print(f'[stream {sd} fold {k + 1}/5] resnet heads + prefix/isolation checks ok {el()}', flush=True)
        S['fusion_current'] = sig((logit(S['hgb_current']) + logit(S['resnet_emb_current__lr'])) / 2)
        S['fusion_delta'] = sig((logit(S['hgb_delta32']) + logit(S['resnet_emb_delta__lr'])) / 2)
        S['fusionR_current'] = sig((logit(S['hgb_current']) + logit(S['resnet_current'])) / 2)
        S['fusionR_delta'] = sig((logit(S['hgb_delta32']) + logit(S['resnet_logit_delta'])) / 2)

        ev = np.isfinite(S['hgb_current']); M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'methods': {}, 'bootstrap': {}, 'subjects': {}}
        for n, p in S.items():
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            ba = np.array([v for s in np.unique(g[ev]) if (v := metrics(y, g, p, ev & (g == s))['subject_macro_bACC']) is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:26s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
            oof[f'stream{sd}__{n}'] = p
        oof[f't_stream{sd}'] = t
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
        for a_, b_ in pairs:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:24s} vs {b_:24s} {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] {el()}',
                      flush=True)
    result['checks'] = checks
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '_results.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    for sd in sseeds:
        for c in ('lr', 'hgb'):
            assert result[f'stream{sd}']['sanity'][f'position_only__{c}']['ok'], f'position control failed stream {sd} {c}'
    print(f'[done] {el()} -> {stem}_results.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
