"""Direction C: alternative input representations (filter bank / STFT + causal history spectrum), de-positioned streams.

Protocol identical to depositioned_resnet_histnorm.py (stream order = concat(relax,task) permuted by
default_rng([stream_seed, subject_index]); outer SGKF(5, rs0) on seed-0 arrangement (asserted == stored fold); per fold
A/B = first split of SGKF(4, rs=k+1); train on A, early stop on B val AUC (MAX_EP 50, PATIENCE 8, AdamW LR 1e-3 WD 1e-2,
BS 128, grad clip 1, BCE); 5 equispaced 4 s crop TTA; network seeds 0,1,2 -> torch seed 1000*rs+k+1; stream 0 uses
3 seeds, streams 1/2 seed 0 only). History = previous <=32 stream windows of the SAME subject (train subjects too), no
labels. Variants (fixed before running):
  resnet_fbank   ResNet1dC(in_ch=7): [normalize(x), w = whiten channel of depositioned_resnet_histnorm (rFFT / sqrt(mean
                 past-32 power spectrum), fallback = fold-train mean spectrum), normalize(bandpass_b(w)) for b in
                 delta 1-4 / theta 4-8 / alpha 8-13 / beta 13-30 / gamma 30-45 Hz]; bandpass = 4th-order Butterworth SOS,
                 zero-phase sosfiltfilt WITHIN the current 6 s window (no cross-window state; causal across windows).
                 Augmentation: same 4 s crop + amplitude scale for all channels, independent noise sd .05 (augment_mc).
  cnn_stft_hist  2D CNN on 2 x 24 freq x frames: S = log(|STFT|^2 + eps) of the demeaned window (Hann periodic,
                 nperseg 128, hop 16, no padding, bins 0..23 = 0-44.9 Hz, 86 frames per 6 s); eps = 1e-6 * mean fold-train
                 power. c0 = (S - mu_f) / sd_f, c1 = (S - Hpast_f) / sd_f where Hpast = mean over the subject's previous
                 <=32 windows of the time-averaged log spectrum (per frequency; no past -> fold-train mean log spectrum);
                 mu_f / sd_f = per-frequency mean / std of S over fold-train rows; both clipped +-10.
                 Net: 4 blocks conv3x3-BN-ReLU (16, 32, 64, 64), MaxPool2d(2) after blocks 1-3, GAP, Dropout .3, Linear(64,1).
                 Augmentation: random 55-frame crop (= 4 s, 1000 samples) same for both channels, amplitude scale
                 U(log .8, log 1.25) applied as + 2 log(s) / sd_f to c0 and c1, Gaussian noise sd .05. TTA: 5 equispaced
                 55-frame crops.
Heads per variant X: X = sigmoid(mean logit over seeds); X_logit_delta = LR(C .05) on B [l, l - mean(past 32 l)]
(prob mean over seeds). Fusions: fusionT_X = sigmoid(mean(logit hgb_prank, logit X_logit_delta));
fusion3_X = sigmoid(mean(logit hgb_prank, logit resnet_whiten_logit_delta, logit X_logit_delta)).
hgb_prank / resnet_whiten_logit_delta / fusionM_prank / position-only loaded from depositioned_feature_multiscale_oof.npz
(y/groups/fold asserted; fusionM_prank asserted == recomputed from its parts; position-only recomputed and asserted).
Paired subject bootstrap 1000 vs fusionM_prank. Threshold .5. No test labels used anywhere. CPU threads limited to 4.
Checks: position-only AUC ~.5; subject alone == pooled inputs/scores; future raw change leaves prefix inputs/scores
unchanged and changes future scores.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '4'
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from raw_waveform_window6 import normalize, FS, WIN, CROP, N_TTA, BS, MAX_EP, PATIENCE, LR, WD  # noqa: E402
from depositioned_history_delta import arrange, make_clf, hist_subject, boot, load_db  # noqa: E402
from depositioned_resnet_history_delta import load_raw, stream_order, hist_delta, logit, sig  # noqa: E402
from depositioned_resnet_histnorm import (ResNet1dC, augment_mc, past_mean, build_subject as hn_build,  # noqa: E402
                                          fallback as hn_fallback, train_resnet, predict as predict_1d)
from causal_kernel_state_mapping import metrics  # noqa: E402
from switch_dual_memory import Tee  # noqa: E402

MATURE = 120
VARIANTS = ('resnet_fbank', 'cnn_stft_hist')
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
SOS = [butter(4, b, btype='bandpass', fs=FS, output='sos') for b in BANDS]
NPERSEG, HOP, NF = 128, 16, 24
NFR = (WIN - NPERSEG) // HOP + 1           # 86 frames per 6 s window
CFR = (CROP - NPERSEG) // HOP + 1          # 55 frames per 4 s crop
HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NPERSEG) / NPERSEG))
CLIP = 10.
NET_SEEDS = (0, 1, 2)
STREAM_SEEDS = (0, 1, 2)
T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


# ---------------------------------------------------------------- resnet_fbank inputs
def fbank_subject(xs, fb):
    """xs: raw windows of ONE subject in stream order -> (n, 7, WIN) float32; fb = whiten fallback spectrum."""
    base = hn_build(xs, 'resnet_whiten', fb)               # [normalize(x), normalize(whitened)]
    xd = xs.astype(np.float64) - xs.astype(np.float64).mean(1, keepdims=True)
    Fx = np.fft.rfft(xd, axis=1); P = past_mean(np.abs(Fx) ** 2)
    P = np.where(np.isfinite(P).all(1, keepdims=True), P, fb[None, :])
    w = np.fft.irfft(Fx / np.sqrt(P + 1e-6 * fb.mean()), n=WIN, axis=1)
    bands = [normalize(sosfiltfilt(s, w, axis=1))[0] for s in SOS]
    return np.concatenate([base, np.stack(bands, 1)], 1)


# ---------------------------------------------------------------- cnn_stft_hist inputs
def logstft(xs, eps):
    xd = xs.astype(np.float64) - xs.astype(np.float64).mean(1, keepdims=True)
    fr = np.lib.stride_tricks.sliding_window_view(xd, NPERSEG, axis=1)[:, ::HOP]   # (n, 86, 128)
    S = np.abs(np.fft.rfft(fr * HANN, axis=2)[:, :, :NF]) ** 2
    return np.log(S + eps).transpose(0, 2, 1)                                      # (n, 24, 86)


def stft_stats(Xraw, rows):
    xd = Xraw[rows].astype(np.float64); xd -= xd.mean(1, keepdims=True)
    eps = 1e-6 * float((xd ** 2).mean()) * (HANN ** 2).sum()
    s1, s2, n = np.zeros(NF), np.zeros(NF), 0
    for b in range(0, len(rows), 1024):
        S = logstft(Xraw[rows[b:b + 1024]], eps)
        s1 += S.sum((0, 2)); s2 += (S ** 2).sum((0, 2)); n += S.shape[0] * S.shape[2]
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 1e-12))
    return {'eps': eps, 'mu': mu, 'sd': sd}


def stft_subject(xs, st):
    S = logstft(xs, st['eps']); m = S.mean(2)
    H = past_mean(m); H = np.where(np.isfinite(H).all(1, keepdims=True), H, st['mu'][None, :])
    c0 = (S - st['mu'][None, :, None]) / st['sd'][None, :, None]
    c1 = (S - H[:, :, None]) / st['sd'][None, :, None]
    return np.clip(np.stack([c0, c1], 1), -CLIP, CLIP).astype(np.float32)


class CNN2d(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        L, c = [], in_ch
        for i, co in enumerate((16, 32, 64, 64)):
            L += [nn.Conv2d(c, co, 3, 1, 1, bias=False), nn.BatchNorm2d(co), nn.ReLU()]
            if i < 3: L.append(nn.MaxPool2d(2))
            c = co
        self.body = nn.Sequential(*L)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.3), nn.Linear(64, 1))

    def forward(self, x):
        return self.head(self.body(x))


def augment_2d(xb, gen, ssd):
    """xb (n, 2, NF, NFR): same random 55-frame crop for both channels; amplitude scale as log-power shift; noise .05."""
    n = xb.shape[0]
    off = torch.randint(0, NFR - CFR + 1, (n, 1, 1, 1), generator=gen, device=xb.device)
    xb = xb.gather(3, (off + torch.arange(CFR, device=xb.device)).expand(n, 2, NF, CFR))
    ls = torch.empty(n, 1, 1, 1, device=xb.device).uniform_(np.log(.8), np.log(1.25), generator=gen)
    xb = xb + 2 * ls / ssd                                  # ssd (1, 1, NF, 1)
    return xb + .05 * torch.randn(xb.shape, generator=gen, device=xb.device)


@torch.no_grad()
def predict_2d(net, Xs):
    net.eval(); offs = np.linspace(0, NFR - CFR, N_TTA).round().astype(int); out = []
    for b in range(0, len(Xs), 512):
        xb = Xs[b:b + 512]
        out.append(torch.stack([net(xb[..., o:o + CFR]).ravel() for o in offs]).mean(0))
    return torch.cat(out).cpu().numpy().astype(np.float64)


def train_cnn(Xg, yg, y, tr, va, seed, tag, max_ep, ssd):
    tick = time.perf_counter(); dev = Xg.device
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    net = CNN2d(Xg.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD); crit = nn.BCEWithLogitsLoss()
    best, state, stale, best_ep = -1., None, 0, 0
    for ep in range(1, max_ep + 1):
        net.train(); perm = rng.permutation(tr); tot = 0.
        for b in range(0, len(perm), BS):
            ids = perm[b:b + BS]
            if len(ids) < 2: continue
            idt = torch.as_tensor(ids, device=dev)
            loss = crit(net(augment_2d(Xg[idt], gen, ssd)).ravel(), yg[idt])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
            tot += loss.item() * len(ids)
        auc = roc_auc_score(y[va], predict_2d(net, Xg[torch.as_tensor(va, device=dev)]))
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


# ---------------------------------------------------------------- dispatch
def fold_stat(Xraw, tr, v):
    return hn_fallback(Xraw, tr, 'resnet_whiten') if v == 'resnet_fbank' else stft_stats(Xraw, tr)


def build_subject(xs, v, st):
    return fbank_subject(xs, st) if v == 'resnet_fbank' else stft_subject(xs, st)


def build_all(Xraw, order, v, st):
    out = None
    for loc in order:
        F = build_subject(Xraw[loc], v, st)
        if out is None: out = np.empty((len(Xraw),) + F.shape[1:], np.float32)
        out[loc] = F
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--smoke', action='store_true'); args = ap.parse_args()
    stem = 'depositioned_resnet_timefreq' + ('_smoke' if args.smoke else '')
    tee = Tee(sys.stdout, OUT / (stem + '.log')); sys.stdout = tee
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    sseeds = STREAM_SEEDS[:1] if args.smoke else STREAM_SEEDS
    max_ep = 2 if args.smoke else MAX_EP
    nb = 100 if args.smoke else 1000

    man, subs, data = load_db()
    Xraw, Fc, y, g = load_raw(subs)
    prior = np.load(OUT / 'depositioned_feature_multiscale_oof.npz')
    assert np.array_equal(prior['y'], y) and np.array_equal(prior['groups'], g)
    print(f'[start] rows={len(y)} subjects={len(subs)} device={dev} smoke={args.smoke} sseeds={sseeds} '
          f'variants={VARIANTS} stft frames={NFR} crop_frames={CFR} bins={NF}', flush=True)

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
    n_params = {'resnet_fbank': sum(p.numel() for p in ResNet1dC(7).parameters()),
                'cnn_stft_hist': sum(p.numel() for p in CNN2d(2).parameters())}
    print(f'[params] {n_params}', flush=True)

    result = {'meta': {'mature_index': MATURE, 'threshold': .5, 'rows': int(len(y)), 'subjects': len(subs),
                       'variants': VARIANTS, 'bands_hz': BANDS, 'stft': {'nperseg': NPERSEG, 'hop': HOP, 'bins': NF,
                       'frames': NFR, 'crop_frames': CFR}, 'n_params': n_params,
                       'net_seeds_stream0': list(NET_SEEDS), 'net_seeds_other_streams': [0],
                       'stream_seeds': list(sseeds), 'device': dev, 'smoke': args.smoke,
                       'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else None,
                       'baseline_source': 'hgb_prank / resnet_whiten_logit_delta / fusionM_prank from '
                                          'depositioned_feature_multiscale_oof.npz; position-only recomputed (asserted)'},
              'runs': [], 'checks': {}}
    checks = result['checks']; oof = {'y': y, 'groups': g, 'fold': fold}
    for sd in sseeds:
        ts = time.perf_counter()
        rseeds = (NET_SEEDS[:1] if args.smoke else NET_SEEDS) if sd == 0 else NET_SEEDS[:1]
        order, t = stream_order(g, subs, sd)
        S = {n: np.full(len(y), np.nan) for n in ('position_only__lr', 'position_only__hgb')}
        for v in VARIANTS:
            for n in (v, v + '_logit_delta'): S[n] = np.full(len(y), np.nan)
        with threadpool_limits(limits=4):
            for k in run_folds:
                tr, te, A, B = folds[k]; T = t[:, None].astype(float)
                for c in ('lr', 'hgb'):
                    S[f'position_only__{c}'][te] = make_clf(c).fit(T[tr], y[tr]).predict_proba(T[te])[:, 1]
        ev = np.isfinite(S['position_only__lr'])
        for c in ('lr', 'hgb'):
            n = f'position_only__{c}'; dmax = float(np.abs(S[n][ev] - prior[f'stream{sd}__{n}'][ev]).max())
            assert dmax < 1e-9, (n, dmax); checks[f'position_reproduced_s{sd}_{c}'] = dmax
        for n in ('hgb_prank', 'resnet_whiten_logit_delta', 'fusionM_prank'):
            S[n] = np.where(ev, prior[f'stream{sd}__{n}'], np.nan)
        fm = sig((logit(S['hgb_prank']) + logit(S['resnet_whiten_logit_delta'])) / 2)
        dmax = float(np.abs(fm[ev] - S['fusionM_prank'][ev]).max()); assert dmax < 1e-9, dmax
        checks[f'fusionM_prank_recomputed_s{sd}'] = dmax
        print(f'[stream {sd}] baselines loaded & aligned (position recomputed, fusionM_prank recomputed) {el()}', flush=True)

        for v in VARIANTS:
            for k in run_folds:
                tr, te, A, B = folds[k]
                t1 = time.perf_counter()
                with threadpool_limits(limits=4):
                    st = fold_stat(Xraw, tr, v); Xin = build_all(Xraw, order, v, st)
                if v == 'resnet_fbank':
                    assert np.array_equal(Xin[:, 0], c0_ref)
                Xg = torch.from_numpy(Xin).to(dev)
                ssd = torch.tensor(st['sd'], dtype=torch.float32, device=dev).view(1, 1, NF, 1) if v == 'cnn_stft_hist' else None
                print(f'[stream {sd} {v} fold {k + 1}/5] inputs built {tuple(Xin.shape)} ({time.perf_counter() - t1:.1f}s) '
                      f'ch std={np.round(Xin.std(axis=tuple(i for i in range(Xin.ndim) if i != 1)), 3).tolist()} '
                      f'clip_frac={(np.abs(Xin) >= CLIP).mean():.4f} {el()}', flush=True)
                pred = (lambda net, X: predict_1d(net, X)) if v == 'resnet_fbank' else predict_2d
                lgs, pdl = [], []
                for rs in rseeds:
                    tag = f's{sd} {v} rs{rs} fold{k + 1}/5'
                    if v == 'resnet_fbank':
                        net, info = train_resnet(Xg, yg, y, A, B, 1000 * rs + k + 1, tag, max_ep)
                    else:
                        net, info = train_cnn(Xg, yg, y, A, B, 1000 * rs + k + 1, tag, max_ep, ssd)
                    L = pred(net, Xg)
                    result['runs'].append({'stream': sd, 'variant': v, 'fold': k, 'net_seed': rs, **info,
                                           'test_auc': float(roc_auc_score(y[te], L[te])), 'B_auc': float(roc_auc_score(y[B], L[B]))})
                    print(f'[net] {tag} best_ep={info["best_epoch"]}/{info["epochs_run"]} valAUC={info["best_val_auc"]:.4f} '
                          f'testAUC={result["runs"][-1]["test_auc"]:.4f} {info["seconds"]:.0f}s {el()}', flush=True)
                    mu, sdv = L[B].mean(), max(L[B].std(), 1e-8)
                    ls = (L - mu) / sdv; F2 = np.c_[ls, hist_delta(ls[:, None], order)]
                    with threadpool_limits(limits=4):
                        head = make_clf('lr').fit(F2[B], y[B])
                    lgs.append(L[te]); pdl.append(head.predict_proba(F2[te])[:, 1])
                    if rs == rseeds[0]:
                        for s0 in np.unique(g[te])[:2]:
                            loc = order[list(subs).index(s0)]; cut = len(loc) // 2
                            xa = build_subject(Xraw[loc], v, st)
                            assert np.array_equal(xa, Xin[loc])
                            raw2 = Xraw[loc].astype(np.float64).copy(); raw2[cut:] = raw2[cut:] * 50. + 1.
                            xb2 = build_subject(raw2, v, st)
                            in_pre = float(np.abs(xb2[:cut] - xa[:cut]).max()); in_fut = float(np.abs(xb2[cut:] - xa[cut:]).max())

                            def score(xs):
                                l = (pred(net, torch.from_numpy(xs).to(dev)) - mu) / sdv
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
                print(f'[progress] stream {sd} {v} fold {k + 1}/5 done; stream time {time.perf_counter() - ts:.0f}s {el()}', flush=True)
        for v in VARIANTS:
            lx = logit(S[v + '_logit_delta'])
            S[f'fusionT_{v}'] = sig((logit(S['hgb_prank']) + lx) / 2)
            S[f'fusion3_{v}'] = sig((logit(S['hgb_prank']) + logit(S['resnet_whiten_logit_delta']) + lx) / 3)

        M = {'all': ev, 'mature': ev & (t >= MATURE)}
        R = result[f'stream{sd}'] = {'net_seeds': list(rseeds), 'methods': {}, 'bootstrap': {}, 'subjects': {}}
        for n, p in S.items():
            oof[f'stream{sd}__{n}'] = p
            r = R['methods'][n] = {s: metrics(y, g, p, M[s]) for s in M}
            ba = np.array([q for s in np.unique(g[ev]) if (q := metrics(y, g, p, ev & (g == s))['subject_macro_bACC']) is not None])
            R['subjects'][n] = {'n_ge_0.9': int((ba >= .9).sum()), 'max': float(ba.max()), 'n': int(len(ba))}
            print(f'[result stream {sd}] {n:30s} ' + ' | '.join(
                f'{s}: AUC={r[s]["AUC"]:.4f} mac={r[s]["subject_macro_bACC"]:.4f} mic={r[s]["micro_bACC"]:.4f}' for s in M) +
                f' | subj>=.9={R["subjects"][n]["n_ge_0.9"]} max={R["subjects"][n]["max"]:.3f}', flush=True)
        for c in ('lr', 'hgb'):
            a = R['methods'][f'position_only__{c}']['all']['AUC']
            R.setdefault('sanity', {})[f'position_only__{c}'] = {'AUC': a, 'ok': abs(a - .5) < .03}
            print(f'[sanity stream {sd}] position_only__{c} AUC={a:.4f} -> {"OK" if abs(a - .5) < .03 else "FLAG"}', flush=True)
            if not args.smoke: assert abs(a - .5) < .03
        new = [n for v in VARIANTS for n in (v, v + '_logit_delta', f'fusionT_{v}', f'fusion3_{v}')]
        for a_ in new:
            for seg in M:
                bb = R['bootstrap'][f'{a_}__vs__fusionM_prank__{seg}'] = boot(S[a_], S['fusionM_prank'], y, g, M[seg], nb)
                print(f'[boot stream {sd}] {a_:32s} vs fusionM_prank {seg:6s} '
                      f'dAUC={bb["dAUC"]["mean"]:+.4f}[{bb["dAUC"]["p2.5"]:+.4f},{bb["dAUC"]["p97.5"]:+.4f}] '
                      f'dmac={bb["dmacro_bACC"]["mean"]:+.4f}[{bb["dmacro_bACC"]["p2.5"]:+.4f},{bb["dmacro_bACC"]["p97.5"]:+.4f}] '
                      f'dmic={bb["dmicro_bACC"]["mean"]:+.4f}[{bb["dmicro_bACC"]["p2.5"]:+.4f},{bb["dmicro_bACC"]["p97.5"]:+.4f}]',
                      flush=True)
        print(f'[progress] stream {sd} finished in {time.perf_counter() - ts:.0f}s {el()}', flush=True)
        (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    result['seconds'] = time.perf_counter() - T0
    (OUT / (stem + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(OUT / (stem + '_oof.npz'), **oof)
    print(f'[done] {el()} -> {stem}.json', flush=True)
    sys.stdout = tee.stream; tee.f.close()


if __name__ == '__main__':
    main()
