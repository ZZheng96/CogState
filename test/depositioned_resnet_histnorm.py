"""Direction 1: causal per-subject normalization of the raw signal by past unlabeled window statistics, ResNet retrained.

Protocol identical to depositioned_resnet_history_delta.py (stream order = concat(relax,task) permuted by
default_rng([stream_seed, subject_index]); outer SGKF(5, rs0) on seed-0 arrangement; per fold A/B = first split of
SGKF(4, rs=k+1); ResNet1d training / early stop on B val AUC / 5-crop TTA unchanged; ResNet seeds 0,1,2 -> 1000*rs+k+1).
Input variants (fixed before running), both 2-channel [c0, c1] with c0 = normalize(x) (per-window demean/std/clip 10):
  resnet_histscale   c1 = clip((x - mean(x)) / rms_past32, +-10); rms_past32 = RMS of demeaned signal over this
                     subject's previous <=32 stream windows (pooled samples); no history -> median per-window RMS of
                     the fold's training rows. Keeps amplitude relative to the subject's own past.
  resnet_whiten      c1 = normalize(irfft(rFFT(x - mean) / sqrt(P_past32 + eps))); P_past32 = mean |rFFT|^2 over this
                     subject's previous <=32 stream windows; no history -> mean power spectrum of fold training rows;
                     eps = 1e-6 * mean(P_fallback). 2-channel variant chosen (c0 kept so no information is removed).
History statistics depend on stream order, so inputs (train subjects included) and the ResNets are rebuilt per stream
seed: stream 0 uses ResNet seeds 0,1,2; streams 1,2 use ResNet seed 0 only.
Augmentation: the same 4 s crop offset and amplitude scale for both channels, independent noise per channel.
Heads per variant X: X = sigmoid(mean logit over seeds); X_logit_delta = LR on B [l, l - mean(past 32 l)] (prob mean
over seeds); fusionH = sigmoid(mean(logit hgb_delta32, logit X_logit_delta)); fusionH_cur = same with X.
Baselines: hgb_delta32 + position controls recomputed here and asserted equal to the prior script's OOF; resnet_current /
fusionR_delta loaded from depositioned_resnet_history_delta_oof.npz (deterministic prior run; same folds/rows).
No test labels used anywhere; threshold .5.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from raw_waveform_window6 import ResNet1d, normalize, WIN, CROP, N_TTA, BS, MAX_EP, PATIENCE, LR, WD  # noqa: E402
from depositioned_history_delta import arrange, make_clf, hist_subject, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, hist_delta, logit, sig  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

K, MATURE, CLIP = 32, 120, 10.
VARIANTS = ('resnet_histscale', 'resnet_whiten')
RESNET_SEEDS = (0, 1, 2)
STREAM_SEEDS = (0, 1, 2)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


class ResNet1dC(ResNet1d):
    """ResNet1d with in_ch input channels (only the first conv changes)."""
    def __init__(self, in_ch=2):
        super().__init__()
        self.stem[0] = nn.Conv1d(in_ch, 32, 15, 2, 7, bias=False)


def augment_mc(xb, gen):
    """xb (n, C, WIN): same random 4 s crop + amplitude scale for all channels, Gaussian noise sd .05."""
    n, c, _ = xb.shape
    off = torch.randint(0, WIN - CROP + 1, (n, 1, 1), generator=gen, device=xb.device)
    xb = xb.gather(2, (off + torch.arange(CROP, device=xb.device)).expand(n, c, CROP))
    scale = torch.exp(torch.empty(n, 1, 1, device=xb.device).uniform_(np.log(.8), np.log(1.25), generator=gen))
    return xb * scale + .05 * torch.randn(xb.shape, generator=gen, device=xb.device)


def past_mean(v, k=K):
    """Row t: mean of v[max(0,t-k):t] along axis 0 (nan row when no past)."""
    cs = np.concatenate([np.zeros((1,) + v.shape[1:]), np.cumsum(v, 0)])
    n = len(v); t = np.arange(n); lo = np.maximum(0, t - k)
    cnt = (t - lo).reshape((-1,) + (1,) * (v.ndim - 1)).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return (cs[t] - cs[lo]) / cnt


def build_subject(xs, variant, fb):
    """xs: raw windows of ONE subject in stream order -> (n, 2, WIN) float32. fb: fold-train fallback statistic."""
    c0, _ = normalize(xs)
    xd = xs.astype(np.float64) - xs.astype(np.float64).mean(1, keepdims=True)
    if variant == 'resnet_histscale':
        rms = np.sqrt(past_mean((xd ** 2).mean(1)))
        rms = np.where(np.isfinite(rms) & (rms > 1e-12), rms, fb)
        c1 = np.clip(xd / rms[:, None], -CLIP, CLIP).astype(np.float32)
    else:
        Fx = np.fft.rfft(xd, axis=1); P = past_mean(np.abs(Fx) ** 2)
        P = np.where(np.isfinite(P).all(1, keepdims=True), P, fb[None, :])
        c1, _ = normalize(np.fft.irfft(Fx / np.sqrt(P + 1e-6 * fb.mean()), n=WIN, axis=1))
    return np.stack([c0, c1], 1)


def fallback(Xraw, rows, variant):
    xd = Xraw[rows].astype(np.float64); xd -= xd.mean(1, keepdims=True)
    if variant == 'resnet_histscale':
        return float(np.median(np.sqrt((xd ** 2).mean(1))))
    P = np.zeros(WIN // 2 + 1)
    for b in range(0, len(xd), 2048): P += (np.abs(np.fft.rfft(xd[b:b + 2048], axis=1)) ** 2).sum(0)
    return P / len(xd)


def build_all(Xraw, order, variant, fb):
    out = np.empty((len(Xraw), 2, WIN), np.float32)
    for loc in order: out[loc] = build_subject(Xraw[loc], variant, fb)
    return out


def train_resnet(Xg, yg, y, tr, va, seed, tag, max_ep):
    tick = time.perf_counter(); dev = Xg.device
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    net = ResNet1dC(Xg.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); crit = nn.BCEWithLogitsLoss()
    best, state, stale, best_ep = -1., None, 0, 0
    for ep in range(1, max_ep + 1):
        net.train(); perm = rng.permutation(tr); tot = 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2: continue
            idt = torch.as_tensor(ids, device=dev)
            loss = crit(net(augment_mc(Xg[idt], gen)).ravel(), yg[idt])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            tot += loss.item() * len(ids)
        auc = roc_auc_score(y[va], predict(net, Xg[torch.as_tensor(va, device=dev)]))
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
def predict(net, Xs):
    """Xs (n, C, WIN) tensor -> TTA-mean logit (5 equispaced 4 s crops)."""
    net.eval(); offs = np.linspace(0, WIN - CROP, N_TTA).round().astype(int); out = []
    for b in range(0, len(Xs), 512):
        xb = Xs[b:b + 512]
        out.append(torch.stack([net(xb[:, :, o:o + CROP]).ravel() for o in offs]).mean(0))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_resnet_histnorm' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    max_ep = 2 if args.smoke else MAX_EP
    nb = 100 if args.smoke else 1000

    man, subs, data = load_db()
    Xraw, Fc, y, g = load_raw(subs)
    prior = np.load(OUT / 'depositioned_resnet_history_delta_oof.npz')
    assert np.array_equal(prior['y'], y) and np.array_equal(prior['groups'], g)
    print(f'[start] rows={len(y)} subjects={len(subs)} device={dev} smoke={args.smoke} sseeds={sseeds} variants={VARIANTS}', flush=True)

    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent')
    fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g]); assert np.array_equal(fold, prior['fold'])
    folds = []
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k + 1).split(tr[:, None], y[tr], g[tr]))
        A, B = tr[a], tr[b]
        assert not (set(g[A]) & set(g[B])) and not (set(g[tr]) & set(g[te]))
        folds.append((tr, te, A, B))
    run_folds = [0] if args.smoke else list(range(5))
    yg = torch.tensor(y, dtype=torch.float32, device=dev)
    c0_ref, _ = normalize(Xraw)

    result = {'meta': {'K': K, 'mature_index': MATURE, 'threshold': .5, 'rows': int(len(y)), 'subjects': len(subs),
                       'variants': VARIANTS, 'resnet_seeds_stream0': list(RESNET_SEEDS), 'resnet_seeds_other_streams': [0],
                       'stream_seeds': list(sseeds), 'device': dev, 'smoke': args.smoke,
                       'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None,
                       'baseline_source': 'hgb_delta32/position recomputed (asserted == prior OOF); resnet_current/'
                                          'fusionR_delta from depositioned_resnet_history_delta_oof.npz'},
              'runs': [], 'checks': {}}
    checks = result['checks']
    for sd in sseeds:
        rseeds = (RESNET_SEEDS[:1] if args.smoke else RESNET_SEEDS) if sd == 0 else RESNET_SEEDS[:1]
        order, t = stream_order(g, subs, sd)
        S = {n: np.full(len(y), np.nan) for n in ('position_only__lr', 'position_only__hgb', 'hgb_delta32')}
        for v in VARIANTS:
            for n in (v, v + '_logit_delta'): S[n] = np.full(len(y), np.nan)
        # ---------- baselines ----------
        with threadpool_limits(limits=8):
            for k in run_folds:
                tr, te, A, B = folds[k]; T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
                med = np.nanmedian(np.where(np.isfinite(Fc[tr]), Fc[tr], np.nan), 0); med = np.where(np.isfinite(med), med, 0.)
                Ff = np.where(np.isfinite(Fc), Fc, med[None, :])
                Z = StandardScaler().fit(Ff[tr]).transform(Ff); ZD = np.hstack([Z, hist_delta(Z, order)])
                S['hgb_delta32'][te] = make_clf('hgb').fit(ZD[tr], y[tr]).predict_proba(ZD[te])[:, 1]
        ev = np.isfinite(S['hgb_delta32'])
        for n in ('position_only__lr', 'position_only__hgb', 'hgb_delta32'):
            dmax = float(np.abs(S[n][ev] - prior[f'stream{sd}__{n}'][ev]).max())
            assert dmax < 1e-9, (n, dmax)
            checks[f'baseline_reproduced_stream{sd}_{n}'] = dmax
        for n in ('resnet_current', 'fusionR_delta'):
            S[n] = np.where(ev, prior[f'stream{sd}__{n}'], np.nan)
        print(f'[stream {sd}] baselines: hgb_delta32/position recomputed == prior OOF; resnet_current/fusionR_delta loaded {el()}', flush=True)
        # ---------- histnorm ResNets ----------
        for v in VARIANTS:
            for k in run_folds:
                tr, te, A, B = folds[k]
                t1 = time.perf_counter(); fb = fallback(Xraw, tr, v)
                Xin = build_all(Xraw, order, v, fb)
                assert np.array_equal(Xin[:, 0], c0_ref)
                Xg = torch.from_numpy(Xin).to(dev)
                print(f'[stream {sd} {v} fold {k + 1}/5] inputs built ({time.perf_counter() - t1:.1f}s) '
                      f'c1 std={Xin[:, 1].std():.3f} clip_frac={(np.abs(Xin[:, 1]) >= CLIP).mean():.4f} {el()}', flush=True)
                lgs, pdl = [], []
                for rs in rseeds:
                    tag = f's{sd} {v} rs{rs} fold{k + 1}/5'
                    net, info = train_resnet(Xg, yg, y, A, B, 1000 * rs + k + 1, tag, max_ep)
                    L = predict(net, Xg)
                    result['runs'].append({'stream': sd, 'variant': v, 'fold': k, 'resnet_seed': rs, **info,
                                           'test_auc': float(roc_auc_score(y[te], L[te])), 'B_auc': float(roc_auc_score(y[B], L[B]))})
                    print(f'[resnet] {tag} best_ep={info["best_epoch"]}/{info["epochs_run"]} valAUC={info["best_val_auc"]:.4f} '
                          f'testAUC={result["runs"][-1]["test_auc"]:.4f} {info["seconds"]:.0f}s {el()}', flush=True)
                    mu, sdv = L[B].mean(), max(L[B].std(), 1e-8)
                    ls = (L - mu) / sdv; F2 = np.c_[ls, hist_delta(ls[:, None], order)]
                    head = make_clf('lr').fit(F2[B], y[B])
                    lgs.append(L[te]); pdl.append(head.predict_proba(F2[te])[:, 1])
                    if rs == rseeds[0]:
                        # causality: subject alone == pooled; future raw change leaves prefix inputs & scores unchanged
                        for s0 in np.unique(g[te])[:2]:
                            loc = order[list(subs).index(s0)]; cut = len(loc) // 2
                            xa = build_subject(Xraw[loc], v, fb)
                            assert np.array_equal(xa, Xin[loc])
                            raw2 = Xraw[loc].astype(np.float64).copy(); raw2[cut:] = raw2[cut:] * 50. + 1.
                            xb2 = build_subject(raw2, v, fb)
                            in_pre = float(np.abs(xb2[:cut] - xa[:cut]).max()); in_fut = float(np.abs(xb2[cut:] - xa[cut:]).max())

                            def score(xs):
                                l = (predict(net, torch.from_numpy(xs).to(dev)) - mu) / sdv
                                return l, head.predict_proba(np.c_[l, l - hist_subject(l[:, None])[1][:, 0]])[:, 1]
                            l0, p0 = score(xa); l1, p1 = score(xb2)
                            d_iso = float(np.abs(p0 - pdl[-1][np.searchsorted(te, loc)]).max())
                            d_pre = float(max(np.abs(l1[:cut] - l0[:cut]).max(), np.abs(p1[:cut] - p0[:cut]).max()))
                            d_fut = float(np.abs(p1[cut:] - p0[cut:]).max())
                            assert in_pre == 0. and in_fut > 1e-3 and d_iso < 1e-4 and d_pre < 1e-4 and d_fut > 1e-4, \
                                (in_pre, in_fut, d_iso, d_pre, d_fut)
                            checks[f'causal_stream{sd}_{v}_fold{k}_{s0}'] = {
                                'input_prefix_maxabs': in_pre, 'input_future_maxabs': in_fut, 'score_alone_vs_pooled': d_iso,
                                'score_prefix_after_future_change': d_pre, 'score_future_changed': d_fut}
                        print(f'[check] s{sd} {v} fold{k + 1}: isolation + prefix invariance ok {el()}', flush=True)
                    del net
                S[v][te] = sig(np.mean(lgs, 0)); S[v + '_logit_delta'][te] = np.mean(pdl, 0)
                del Xg, Xin
                if dev == 'cuda': torch.cuda.empty_cache()
        for v in VARIANTS:
            S[v.replace('resnet_', 'fusionH_')] = sig((logit(S['hgb_delta32']) + logit(S[v + '_logit_delta'])) / 2)
            S[v.replace('resnet_', 'fusionH_cur_')] = sig((logit(S['hgb_delta32']) + logit(S[v])) / 2)

        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'resnet_seeds': list(rseeds), 'methods': {}, 'bootstrap': {}, 'subjects': {}}
        for n, p in S.items():
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            ba = np.array([q for s in np.unique(g[ev]) if (q := metrics(y, g, p, ev & (g == s))['subject_macro_bACC']) is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:32s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
            if not args.smoke: assert abs(a - .5) < .03
        new = [n for v in VARIANTS for n in (v, v + '_logit_delta', v.replace('resnet_', 'fusionH_'), v.replace('resnet_', 'fusionH_cur_'))]
        for a_ in new:
            for b_ in ('resnet_current', 'fusionR_delta'):
                for seg in M:
                    bb = R['bootstrap'][f'{a_}__vs__{b_}__{seg}'] = boot(S[a_], S[b_], y, g, M[seg], nb)
                    print(f'[boot stream {sd}] {a_:30s} vs {b_:15s} {seg:6s} '
                          f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                          f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                          f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                          flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
