"""实验 B（可部署候选）：跨被试 R/T 模板对比特征。

每窗向量 v（只用当前 1500 点窗）：窗内 z-score 后 Welch(1 Hz 分辨) 1–45 Hz 对数谱（减去自身均值）+ 52–100 Hz 相对对数功率，
外加 6 个 1 秒子窗的 7 频带相对对数功率跨子窗 std 与子窗对数总功率 std（共 54 维）。
外层训练折内：每个训练被试用自身无标签均值做被试内中心化（仅用于构造模板），按被试平均得到 R、T 模板与差分方向 D=T-R；
训练窗特征用留一被试模板（避免本人参与本人模板）。测试窗：
  window 变体  —— 只用当前窗 v（严格单窗），模板放回训练被试总体均值 G 的绝对坐标；
  causal 变体  —— v 减去本人过去窗的无标签均值（先验 32 窗 = G，预测后更新），再加 G。
特征（24 维）：到 R/T 欧氏距离及差、谱相关(R/T)、与 D 的相关、沿 D 的投影（总/谱/子窗）、7 个谱频带（含 52–100 Hz）和 8 个子窗统计对 D 的分量。
分类：LR(StandardScaler,C=1,balanced) 与 HGB(150,3,0.06,balanced) 固定超参；阈值在内层 SGKF(4,rs=fold) 第一划分验证上按 subject-macro bACC 选。
与 merge_best_window6 combo__C4_bb_resnet_eq 做 logit 等权融合（阈值固定 0，与基线 bACC@0 同口径）。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45))
SUB_BANDS = ((1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (30, 45), (52, 100))
PRIOR = 32.
BOOT_SEED, N_BOOT = 20261029, 1000
T0 = time.perf_counter()
LOGF = None


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")} +{time.perf_counter() - T0:7.1f}s] {msg}'
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + '\n'); LOGF.flush()


def window_vectors(raw):
    """每窗 54 维，仅依赖该窗 1500 点。"""
    assert raw.ndim == 2 and raw.shape[1] == 1500
    x = raw.astype(np.float64)
    x = (x - x.mean(1, keepdims=True)) / np.maximum(x.std(1, keepdims=True), 1e-12)
    f, P = welch(x, fs=250, nperseg=250, noverlap=125, axis=1)
    lo = np.log(P[:, (f >= 1) & (f <= 45)] + 1e-12)            # 45 bins
    hi = np.log(P[:, (f >= 52) & (f < 100)].mean(1) + 1e-12)
    ref = lo.mean(1, keepdims=True)
    S = np.column_stack([lo - ref, hi[:, None] - ref])          # 46
    sub = x.reshape(len(x), 6, 250)
    sp = np.abs(np.fft.rfft(sub - sub.mean(-1, keepdims=True), axis=-1)) ** 2
    fr = np.fft.rfftfreq(250, 1 / 250)
    tot = sp[..., (fr >= 1) & (fr < 100)].sum(-1) + 1e-12
    rel = np.stack([np.log(sp[..., (fr >= a) & (fr < b)].sum(-1) / tot + 1e-12) for a, b in SUB_BANDS], -1)
    Q = np.column_stack([rel.std(1), np.log(tot).std(1)])       # 8
    return np.column_stack([S, Q])


SPEC = slice(0, 46)
SUBQ = slice(46, 54)
FREQS = np.arange(1, 46)


def band_masks():
    m = [np.r_[(FREQS >= a) & (FREQS < b), False] for a, b in BANDS]
    m[-1] = np.r_[(FREQS >= 30) & (FREQS <= 45), False]
    m.append(np.r_[np.zeros(45, bool), True])  # 52-100 Hz
    return m


BM = band_masks()


def subject_stats(V, y, g):
    subs = np.unique(g)
    mu = np.stack([V[g == s].mean(0) for s in subs])
    r = np.stack([(V[(g == s) & (y == 0)] - mu[i]).mean(0) for i, s in enumerate(subs)])
    t = np.stack([(V[(g == s) & (y == 1)] - mu[i]).mean(0) for i, s in enumerate(subs)])
    return subs, mu, r, t


def feats(v, G, R, T, sig):
    """v: [n,54] 绝对坐标；模板 R/T 为被试内中心化后的类均值；G 为总体均值；sig 为被试内 std。"""
    z = (v - G) / sig
    Rz, Tz = R / sig, T / sig
    if Rz.ndim == 1:
        Rz, Tz = Rz[None], Tz[None]
    D = Tz - Rz
    Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
    c = z - (Rz + Tz) / 2
    dR, dT = np.linalg.norm(z - Rz, axis=1), np.linalg.norm(z - Tz, axis=1)

    def corr(a, b):
        a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-12)

    vs = v[:, SPEC]
    GR, GT = (G + R)[..., SPEC], (G + T)[..., SPEC]
    GR, GT = np.broadcast_to(GR, vs.shape), np.broadcast_to(GT, vs.shape)
    comp = c * Dn
    out = [dR, dT, dT - dR, corr(vs, GR), corr(vs, GT), corr(z, np.broadcast_to(D, z.shape)),
           comp.sum(1), comp[:, SPEC].sum(1), comp[:, SUBQ].sum(1)]
    out += [comp[:, SPEC][:, m].sum(1) for m in BM]
    out += [comp[:, SUBQ][:, k] for k in range(8)]
    F = np.column_stack(out)
    assert F.shape[1] == 24
    return F


def causal_center(V, g, G):
    """本人过去窗无标签均值（先验 PRIOR 窗于 G），当前窗先输出后更新。"""
    out = np.empty_like(V)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s)
        tot, n = PRIOR * G.copy(), PRIOR
        for i in ii:
            out[i] = V[i] - tot / n + G
            tot += V[i]; n += 1
    return out


def build(V, y, g, tr, te, variant):
    """在 tr 上拟合模板，返回 (Ftr, Fte)。tr 特征用留一被试模板。"""
    subs, mu, r, t = subject_stats(V[tr], y[tr], g[tr])
    sig = np.sqrt(np.mean([(V[tr][g[tr] == s] - mu[i]).var(0) for i, s in enumerate(subs)], 0)) + 1e-9
    G, R, T = mu.mean(0), r.mean(0), t.mean(0)
    n = len(subs)
    Vtr = V[tr] if variant == 'window' else causal_center(V[tr], g[tr], G)
    pos = {s: i for i, s in enumerate(subs)}
    k = np.array([pos[s] for s in g[tr]])
    Gl = (mu.sum(0) - mu[k]) / (n - 1)
    Rl = (r.sum(0) - r[k]) / (n - 1)
    Tl = (t.sum(0) - t[k]) / (n - 1)
    if variant == 'causal':
        Vtr = Vtr - G + Gl
    Ftr = feats(Vtr, Gl, Rl, Tl, sig)
    Vte = V[te] if variant == 'window' else causal_center(V[te], g[te], G)
    Fte = feats(Vte, G, R, T, sig)
    return Ftr, Fte


def clf(kind):
    if kind == 'lr':
        return make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced', max_iter=3000))
    return HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=.06,
                                          class_weight='balanced', random_state=0)


def macro_bacc(y, pred, g):
    v = [balanced_accuracy_score(y[g == s], pred[g == s]) for s in np.unique(g) if len(np.unique(y[g == s])) == 2]
    return float(np.mean(v))


def pick_thr(y, p, g):
    cand = np.unique(np.quantile(p, np.linspace(.02, .98, 97)))
    sc = [macro_bacc(y, (p >= c).astype(int), g) for c in cand]
    return float(cand[int(np.argmax(sc))])


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def summarize(y, g, s, thr):
    pred = (s >= thr).astype(int)
    return {'AUC': float(roc_auc_score(y, s)), 'macro_bACC': macro_bacc(y, pred, g),
            'micro_bACC': float(balanced_accuracy_score(y, pred))}


def bootstrap(y, g, scores, thrs, ref):
    subs = np.unique(g)
    idx = {s: np.flatnonzero(g == s) for s in subs}
    sb = {k: {s: balanced_accuracy_score(y[idx[s]], s_[idx[s]] >= thrs[k]) for s in subs}
          for k, s_ in scores.items()}
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(subs, len(subs)) for _ in range(N_BOOT)]
    res = {k: {m: [] for m in ('AUC', 'macro_bACC', 'micro_bACC')} for k in scores}
    for i, d in enumerate(draws):
        ii = np.concatenate([idx[s] for s in d])
        for k, s_ in scores.items():
            res[k]['AUC'].append(roc_auc_score(y[ii], s_[ii]))
            res[k]['macro_bACC'].append(np.mean([sb[k][s] for s in d]))
            res[k]['micro_bACC'].append(balanced_accuracy_score(y[ii], s_[ii] >= thrs[k]))
        if (i + 1) % 250 == 0:
            log(f'[boot] {i + 1}/{N_BOOT}')
    out = {}
    for k in scores:
        if k == ref:
            continue
        out[k] = {}
        for m in ('AUC', 'macro_bACC', 'micro_bACC'):
            dl = np.array(res[k][m]) - np.array(res[ref][m])
            point = summarize(y, g, scores[k], thrs[k])[m] - summarize(y, g, scores[ref], thrs[ref])[m]
            out[k][m] = {'delta': float(point), 'ci95': [float(np.percentile(dl, 2.5)), float(np.percentile(dl, 97.5))],
                         'p_gt0': float(np.mean(dl > 0))}
    return out


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    stem = 'template_contrast_features' + ('_smoke' if args.smoke else '')
    paths = [OUT / f'{stem}.log', OUT / f'{stem}_results.json', OUT / f'{stem}_oof.npz']
    if not args.smoke and any(p.exists() for p in paths):
        raise FileExistsError('refusing to overwrite template_contrast_features artifacts')
    LOGF = paths[0].open('w', encoding='utf-8')

    d = np.load(OUT / 'highdim_deep_features.npz')
    raw, y, g = d['raw'], d['y'].astype(int), d['groups'].astype(str)
    mb = np.load(OUT / 'merge_best_window6_oof.npz')
    assert np.array_equal(mb['y'].astype(int), y) and np.array_equal(mb['groups'].astype(str), g)
    sn = np.load(OUT / 'seed_noise_band_oof.npz')
    assert np.array_equal(sn['y'].astype(int), y) and np.array_equal(sn['groups'].astype(str), g)
    base = mb['combo__C4_bb_resnet_eq'].astype(np.float64)
    if args.smoke:
        keep = np.isin(g, np.unique(g)[:15])
        raw, y, g, base = raw[keep], y[keep], g[keep], base[keep]
    V = window_vectors(raw)
    assert np.isfinite(V).all()
    log(f'[data] windows={len(y)} subjects={len(np.unique(g))} vec_dim={V.shape[1]} raw_len={raw.shape[1]} smoke={args.smoke}')

    outer = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(V, y, g))
    fold_id = np.zeros(len(y), int)
    for k, (_, te) in enumerate(outer, 1):
        fold_id[te] = k
    if not args.smoke:
        assert np.array_equal(fold_id, sn['fold_id'].astype(int)), 'outer folds differ from seed_noise_band fold_id'

    variants = ('window', 'causal')
    kinds = ('lr', 'hgb')
    S = {f'{v}_{c}': np.full(len(y), np.nan) for v in variants for c in kinds}
    thr = {k: [] for k in S}
    Fdump = {}
    checks = []
    for k, (tr, te) in enumerate(outer, 1):
        assert not set(g[tr]) & set(g[te])
        a, b = next(StratifiedGroupKFold(4, shuffle=True, random_state=k).split(tr[:, None], y[tr], g[tr]))
        itr, iva = tr[a], tr[b]
        assert not set(g[itr]) & set(g[iva])
        for v in variants:
            Fi_tr, Fi_va = build(V, y, g, itr, iva, v)
            Ftr, Fte = build(V, y, g, tr, te, v)
            if v == 'causal':
                # 因果核验：篡改每个测试被试后半窗，前半特征逐位不变
                s0 = np.unique(g[te])[0]
                ii = np.flatnonzero(g == s0); cut = len(ii) // 2
                V2 = V.copy(); V2[ii[cut:]] += 5.
                _, Fte2 = build(V2, y, g, tr, te, v)
                pos = np.searchsorted(te, ii[:cut])
                assert np.array_equal(Fte[pos], Fte2[pos])
                checks.append({'fold': k, 'subject': str(s0), 'prefix_invariant_under_future_perturbation': True})
            if v == 'window':
                # 单窗核验：测试窗特征与同折其它测试窗无关
                _, Fsingle = build(V, y, g, tr, te[:3], v)
                assert np.allclose(Fsingle, Fte[:3])
                checks.append({'fold': k, 'single_window_independence': True})
            Fdump[(k, v)] = Fte
            for c in kinds:
                pv = clf(c).fit(Fi_tr, y[itr]).predict_proba(Fi_va)[:, 1]
                t_ = pick_thr(y[iva], pv, g[iva])
                pt = clf(c).fit(Ftr, y[tr]).predict_proba(Fte)[:, 1]
                key = f'{v}_{c}'
                S[key][te] = pt; thr[key].append(t_)
                log(f'[fold {k}/5] {key:11s} innerAUC={roc_auc_score(y[iva], pv):.4f} thr={t_:.3f} '
                    f'testAUC={roc_auc_score(y[te], pt):.4f}')
    thr_vec = {key: np.array(thr[key])[fold_id - 1] for key in S}

    results = {}
    for key in S:
        assert np.isfinite(S[key]).all()
        r = summarize(y, g, S[key], 0.)  # AUC only from here
        r['macro_bACC_inner_thr'] = macro_bacc(y, (S[key] >= thr_vec[key]).astype(int), g)
        r['micro_bACC_inner_thr'] = float(balanced_accuracy_score(y, S[key] >= thr_vec[key]))
        r['macro_bACC_at_0.5'] = macro_bacc(y, (S[key] >= .5).astype(int), g)
        r['micro_bACC_at_0.5'] = float(balanced_accuracy_score(y, S[key] >= .5))
        r['thresholds'] = thr[key]
        r.pop('macro_bACC'); r.pop('micro_bACC')
        results[key] = r
        log(f'[result] {key:11s} AUC={r["AUC"]:.4f} macro@thr={r["macro_bACC_inner_thr"]:.4f} micro@thr={r["micro_bACC_inner_thr"]:.4f} '
            f'macro@0.5={r["macro_bACC_at_0.5"]:.4f}')

    # 融合：logit 等权，阈值 0（与 merge_best bACC@0 同口径）
    scores = {'merge_best': base}
    for key in S:
        scores[f'fuse_{key}'] = .5 * to_logit(S[key]) + .5 * base
        scores[f'B_{key}'] = to_logit(S[key])
    thrs = {k: 0. for k in scores}
    fusion = {k: summarize(y, g, s, 0.) for k, s in scores.items()}
    for k, r in fusion.items():
        log(f'[fusion@0] {k:20s} AUC={r["AUC"]:.4f} macro={r["macro_bACC"]:.4f} micro={r["micro_bACC"]:.4f}')
    boot = bootstrap(y, g, scores, thrs, 'merge_best')
    for k, r in boot.items():
        log(f'[delta vs merge_best] {k:20s} ' + ' '.join(
            f'{m}={r[m]["delta"]:+.4f}[{r[m]["ci95"][0]:+.4f},{r[m]["ci95"][1]:+.4f}]' for m in r))

    res = {'protocol': {
        'deployable': True, 'window_points': 1500, 'test_subject_isolation': True,
        'window_variant': 'strict single window: features depend only on current 1500-sample window and train-fold templates',
        'causal_variant': f'subtract own past unlabeled mean (prior {PRIOR:.0f} windows at train grand mean G), predict-then-update',
        'templates': 'outer/inner train only: per-subject unlabeled mean-centering, class means per subject averaged over subjects; '
                     'train-window features use leave-one-subject-out templates; sigma = pooled within-subject std (unlabeled)',
        'features': 24, 'vector': '54 = 45 log-PSD bins 1-45 Hz (self-referenced) + 52-100 Hz rel + 8 sub-window stats',
        'classifiers': {'lr': 'StandardScaler+LR(C=1,balanced)', 'hgb': 'HistGB(150,depth3,lr0.06,balanced)'},
        'threshold': 'inner SGKF(4,shuffle,rs=fold) first split, maximize subject-macro bACC over 97 quantile candidates; refit on outer train',
        'outer_cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=0) == seed_noise_band fold_id',
        'fusion': '0.5*logit(B) + 0.5*merge_best combo__C4_bb_resnet_eq; fixed threshold 0 (same as baseline bACC@0); no weight search',
        'bootstrap': f'paired subject bootstrap {N_BOOT}, seed {BOOT_SEED}; macro = mean of per-subject bACC over drawn subjects',
        'hyperparameter_search': False},
        'methods': results, 'fusion_at0': fusion, 'bootstrap_vs_merge_best': boot, 'checks': checks,
        'smoke': args.smoke, 'seconds': time.perf_counter() - T0}
    paths[1].write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(paths[2], y=y, groups=g, fold_id=fold_id, merge_best=base,
                        **{f'p_{k}': v for k, v in S.items()},
                        **{f'thr_{k}': v for k, v in thr_vec.items()},
                        **{f'fuse_{k}': scores[f'fuse_{k}'] for k in S})
    log(f'[done] -> {paths[1].name}, {paths[2].name}')


if __name__ == '__main__':
    main()
