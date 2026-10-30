"""诊断（不可部署，用全部被试标签）：被试 R/T 效应量谱与“模式”聚类。

特征（54 维）：Welch(nperseg=250, 1 Hz 分辨率) 1..45 Hz 每 1 Hz bin 对数 PSD（45）
+ 52-100 Hz 5 个对数带功率（52-60,60-70,70-80,80-90,90-100）
+ 6 个 1 秒子窗：log 方差均值、log 方差标准差、谱熵均值、谱熵标准差。
效应量 d = (mean_T - mean_R)/pooled_sd（y=1 为 task）。
聚类：被试 d 向量（54 维）做 KMeans（k=2..6）与余弦层次聚类；同时报告“去广带偏移”（每被试 d 减去 1-45Hz 均值）版本；
silhouette 与高斯零模型（同协方差多元正态，200 次）比较判断模式是否显著。
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUT = Path(__file__).resolve().parent
T0 = time.perf_counter()
LOGF = None
HF = ((52, 60), (60, 70), (70, 80), (80, 90), (90, 100))
TOP = '051,054,016,025,036,059,043,003,031,018,048,030,007,017'.split(',')
BOT = '024,057,026,040,009,015,053,019,002,005,028,041,004,029'.split(',')


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")} +{time.perf_counter() - T0:7.1f}s] {msg}'
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + '\n'); LOGF.flush()


def fine_features(raw, chunk=2000):
    assert raw.shape[1] == 1500
    out = []
    for i in range(0, len(raw), chunk):
        r = raw[i:i + chunk].astype(np.float64)
        f, p = welch(r, fs=250, nperseg=250, noverlap=125, axis=1)
        lp = np.log(p[:, (f >= 1) & (f <= 45)] + 1e-20)
        hf = np.stack([np.log(p[:, (f >= a) & (f < b)].sum(1) + 1e-20) for a, b in HF], 1)
        sub = r.reshape(len(r), 6, 250)
        sub = sub - sub.mean(-1, keepdims=True)
        lv = np.log(sub.var(-1) + 1e-20)
        ps = np.abs(np.fft.rfft(sub * np.hanning(250), axis=-1)) ** 2
        fr = np.fft.rfftfreq(250, 1 / 250)
        q = ps[..., (fr >= 1) & (fr < 100)] + 1e-20
        q = q / q.sum(-1, keepdims=True)
        se = -(q * np.log(q)).sum(-1)
        out.append(np.column_stack([lp, hf, lv.mean(1), lv.std(1), se.mean(1), se.std(1)]))
    return np.concatenate(out)


FEAT_NAMES = [f'lp{h}Hz' for h in range(1, 46)] + [f'hf{a}-{b}' for a, b in HF] + \
             ['sub_logvar_mean', 'sub_logvar_std', 'sub_spent_mean', 'sub_spent_std']


def cohen_d(F, y):
    a, b = F[y == 0], F[y == 1]
    sp = np.sqrt(((len(a) - 1) * a.var(0, ddof=1) + (len(b) - 1) * b.var(0, ddof=1)) / (len(a) + len(b) - 2)) + 1e-12
    return (b.mean(0) - a.mean(0)) / sp


BANDS_DIAG = {'delta1-3': range(0, 3), 'theta4-7': range(3, 7), 'alpha8-12': range(7, 12), 'beta13-30': range(12, 30),
              'gamma31-45': range(30, 45), 'hf52-100': range(45, 50), 'subwin': range(50, 54)}


def band_summary(d):
    return {k: float(np.mean(d[list(v)])) for k, v in BANDS_DIAG.items()}


def main_band(d):
    best = max(BANDS_DIAG, key=lambda k: np.max(np.abs(d[list(BANDS_DIAG[k])])))
    return best


def cluster_eval(D, ks, rng_seed=0, n_null=200):
    res = {}
    cov = np.cov(D.T) + 1e-9 * np.eye(D.shape[1])
    rng = np.random.default_rng(rng_seed)
    nulls = rng.multivariate_normal(D.mean(0), cov, size=(n_null, len(D)))
    for k in ks:
        km = KMeans(k, n_init=20, random_state=0).fit(D)
        sil = silhouette_score(D, km.labels_)
        ns = np.array([silhouette_score(Z, KMeans(k, n_init=5, random_state=0).fit_predict(Z)) for Z in nulls])
        res[k] = {'silhouette': float(sil), 'null_mean': float(ns.mean()), 'null_q95': float(np.quantile(ns, .95)),
                  'p_null': float((np.sum(ns >= sil) + 1) / (n_null + 1)), 'labels': km.labels_.tolist()}
        log(f'  k={k} silhouette={sil:.3f} null mean={ns.mean():.3f} q95={np.quantile(ns, .95):.3f} p={res[k]["p_null"]:.3f}')
    return res


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    stem = 'subject_mode_analysis' + ('_smoke' if args.smoke else '')
    paths = [OUT / f'{stem}.log', OUT / f'{stem}_results.json', OUT / f'{stem}_features.npz']
    if not args.smoke and any(p.exists() for p in paths):
        raise FileExistsError('refusing to overwrite subject_mode_analysis artifacts')
    LOGF = paths[0].open('w', encoding='utf-8')
    d = np.load(OUT / 'highdim_deep_features.npz')
    raw, y, g = d['raw'], d['y'].astype(int), d['groups'].astype(str)
    if args.smoke:
        keep = np.isin(g, np.unique(g)[:12]); raw, y, g = raw[keep], y[keep], g[keep]
    F = fine_features(raw)
    log(f'[feat] {F.shape}')
    ceil = json.loads((OUT / 'per_subject_ceiling_results.json').read_text(encoding='utf-8'))
    crow = {r['subject']: r for r in ceil['per_subject']}
    subs = np.unique(g)
    Dm = np.stack([cohen_d(F[g == s], y[g == s]) for s in subs])
    info = []
    for s, dv in zip(subs, Dm):
        r = crow.get(s, {})
        info.append({'subject': s, 'ceiling_lr_bacc': r.get('lr_bacc'), 'ceiling_mean3_bacc': r.get('mean3_bacc'),
                     'merge_best_auc': r.get('merge_best_auc'), 'flip': r.get('direction_flip'),
                     'group': 'top' if s in TOP else 'bottom' if s in BOT else 'mid',
                     'max_abs_d': float(np.max(np.abs(dv))), 'argmax_feat': FEAT_NAMES[int(np.argmax(np.abs(dv)))],
                     'main_band': main_band(dv), 'band_mean_d': band_summary(dv),
                     'alpha_d': float(dv[7:12].mean()), 'theta_d': float(dv[3:7].mean()),
                     'broadband_offset_1_45': float(dv[:45].mean())})
        log(f'  {s} grp={info[-1]["group"]:6s} ceil={info[-1]["ceiling_lr_bacc"]} mbAUC={info[-1]["merge_best_auc"]} '
            f'max|d|={info[-1]["max_abs_d"]:.2f} ({info[-1]["argmax_feat"]}) band={info[-1]["main_band"]} '
            f'off={info[-1]["broadband_offset_1_45"]:+.2f} alpha={info[-1]["alpha_d"]:+.2f} theta={info[-1]["theta_d"]:+.2f}')

    def grp_stats(mask):
        D = Dm[mask]
        return {'n': int(mask.sum()), 'mean_d_spectrum': D.mean(0).tolist(), 'median_abs_d_spectrum': np.median(np.abs(D), 0).tolist(),
                'frac_positive_d': (D > 0).mean(0).tolist(), 'band_mean_d': band_summary(D.mean(0)),
                'band_frac_positive': {k: float((D[:, list(v)].mean(1) > 0).mean()) for k, v in BANDS_DIAG.items()},
                'main_band_counts': {b: int(sum(info[i]['main_band'] == b for i in np.flatnonzero(mask))) for b in BANDS_DIAG},
                'median_max_abs_d': float(np.median(np.abs(D).max(1)))}
    gm = np.array([i['group'] for i in info])
    flip = np.array([i['flip'] == 1 for i in info])
    groups = {'top': grp_stats(gm == 'top'), 'bottom': grp_stats(gm == 'bottom'), 'flip': grp_stats(flip),
              'all': grp_stats(np.ones(len(subs), bool))}
    for k, v in groups.items():
        if v['n']:
            log(f'[group {k}] n={v["n"]} band_mean_d={json.dumps({a: round(b, 2) for a, b in v["band_mean_d"].items()})} '
                f'frac+={json.dumps({a: round(b, 2) for a, b in v["band_frac_positive"].items()})} '
                f'main_band={v["main_band_counts"]} med max|d|={v["median_max_abs_d"]:.2f}')
    # 被试间 d 向量相关：方向一致性
    C = np.corrcoef(Dm)
    iu = np.triu_indices(len(subs), 1)
    log(f'[consistency] median pairwise corr of d-vectors all={np.median(C[iu]):.3f} frac<0={np.mean(C[iu] < 0):.3f}')
    for nm, msk in (('top', gm == 'top'), ('bottom', gm == 'bottom')):
        ii = np.flatnonzero(msk); cc = C[np.ix_(ii, ii)][np.triu_indices(len(ii), 1)]
        log(f'[consistency] {nm}: median pairwise corr={np.median(cc):.3f}' if len(cc) else '')

    ks = range(2, 7)
    n_null = 30 if args.smoke else 200
    variants = {}
    unit = Dm / (np.linalg.norm(Dm, axis=1, keepdims=True) + 1e-12)
    shape = Dm.copy(); shape[:, :45] -= Dm[:, :45].mean(1, keepdims=True)
    shape_unit = shape / (np.linalg.norm(shape, axis=1, keepdims=True) + 1e-12)
    for nm, D in (('raw_d', Dm), ('unit_d', unit), ('shape_unit_d', shape_unit)):
        log(f'[cluster] variant={nm}')
        variants[nm] = cluster_eval(D, ks, n_null=n_null)
    # 余弦层次聚类（average）k=2..6 的簇大小
    Z = linkage(unit, 'average', metric='cosine')
    hier = {k: fcluster(Z, k, 'maxclust').tolist() for k in ks}
    log(f'[hier cosine] sizes ' + ' '.join(f'k={k}:{np.bincount(hier[k])[1:].tolist()}' for k in ks))

    def describe(labels, D_used):
        labels = np.asarray(labels); out = []
        for c in np.unique(labels):
            m = labels == c
            cv = [info[i]['ceiling_lr_bacc'] for i in np.flatnonzero(m) if info[i]['ceiling_lr_bacc'] is not None]
            au = [info[i]['merge_best_auc'] for i in np.flatnonzero(m) if info[i]['merge_best_auc'] is not None]
            md = Dm[m].mean(0)
            top3 = [FEAT_NAMES[j] for j in np.argsort(-np.abs(md))[:5]]
            out.append({'cluster': int(c), 'n': int(m.sum()), 'subjects': subs[m].tolist(),
                        'ceiling_median': float(np.median(cv)) if cv else None,
                        'merge_best_auc_median': float(np.median(au)) if au else None,
                        'n_top': int(np.sum(gm[m] == 'top')), 'n_bottom': int(np.sum(gm[m] == 'bottom')), 'n_flip': int(flip[m].sum()),
                        'band_mean_d': band_summary(md), 'top_features': top3,
                        'within_cos_median': float(np.median((unit[m] @ unit[m].T)[np.triu_indices(m.sum(), 1)])) if m.sum() > 1 else None})
            o = out[-1]
            log(f'  c{c} n={o["n"]} ceil_med={o["ceiling_median"]} mbAUC_med={o["merge_best_auc_median"]} top/bot/flip='
                f'{o["n_top"]}/{o["n_bottom"]}/{o["n_flip"]} bands={json.dumps({a: round(b, 2) for a, b in o["band_mean_d"].items()})} '
                f'feat={o["top_features"]} wcos={o["within_cos_median"]}')
        return out

    descr = {}
    for nm in variants:
        best_k = max(ks, key=lambda k: variants[nm][k]['silhouette'] - variants[nm][k]['null_mean'])
        log(f'[describe] {nm} k={best_k} (max silhouette excess over null)')
        descr[nm] = {'k': best_k, 'clusters': describe(variants[nm][best_k]['labels'], None)}
    for k in (3, 4):
        log(f'[describe] unit_d fixed k={k}')
        descr[f'unit_d_k{k}'] = {'k': k, 'clusters': describe(variants['unit_d'][k]['labels'], None)}
    # 符号无关：|cos| 聚类（模式与其翻转视为同一模式）
    A = np.abs(unit @ unit.T)
    Zs = linkage(1 - A[np.triu_indices(len(subs), 1)], 'average')
    sign_free = {k: fcluster(Zs, k, 'maxclust').tolist() for k in ks}
    log('[sign-free |cos| hier] sizes ' + ' '.join(f'k={k}:{np.bincount(sign_free[k])[1:].tolist()}' for k in ks))

    result = {'protocol': {'deployable': False, 'uses_all_labels': True, 'features': FEAT_NAMES,
                           'cohen_d': '(mean_T - mean_R)/pooled_sd, y=1 task', 'ceiling': 'per_subject_ceiling lr_bacc',
                           'null_model': f'multivariate normal with same mean/cov as d-vectors, {n_null} draws'},
              'per_subject': info, 'groups': groups, 'cluster_variants': {k: {str(kk): vv for kk, vv in v.items()} for k, v in variants.items()},
              'hier_cosine': {str(k): v for k, v in hier.items()}, 'sign_free_hier': {str(k): v for k, v in sign_free.items()},
              'described': descr, 'subjects': subs.tolist(), 'smoke': args.smoke, 'seconds': time.perf_counter() - T0}
    paths[1].write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(paths[2], F=F.astype(np.float32), y=y, groups=g, D=Dm, subjects=subs)
    log(f'[done] -> {paths[1].name}')


if __name__ == '__main__':
    main()
