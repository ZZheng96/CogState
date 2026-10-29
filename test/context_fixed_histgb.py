"""紧凑复现实验：q、200窗均值去趋势、200窗均值/标准差归一化，以及排除当前窗的对应对照。

固定 HistGB；5折 StratifiedGroupKFold(seed=0)。每名受试者前120窗既不训练也不评估。
Z 为仅用前120窗校准的分位数特征；所有历史变换均为受试者内因果计算。
注意：200窗历史及 smooth5 的有效感受野均超过6秒，本实验不声称严格6秒合规。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
K, L, SEED = 120, 200, 0
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile


def causal_mean_std(z, g, k):
    mean = np.empty_like(z); std = np.empty_like(z)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; v = z[ii]
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
        t = np.arange(len(ii)); lo = np.maximum(t-k+1, 0); n = t+1-lo
        m = (cs[t+1]-cs[lo])/n[:, None]
        cs2 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v*v, axis=0)])
        q = (cs2[t+1]-cs2[lo])/n[:, None]
        mean[ii], std[ii] = m, np.sqrt(np.maximum(q-m*m, 0))
    return mean, std


def past_mean_std(z, g, k):
    mean = np.empty_like(z); std = np.empty_like(z)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; v = z[ii]
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)])
        t = np.arange(len(ii)); lo = np.maximum(t-k, 0); n = t-lo
        m = np.divide(cs[t]-cs[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
        cs2 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v*v, axis=0)])
        q = np.divide(cs2[t]-cs2[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
        std[ii] = np.sqrt(np.maximum(q-m*m, 0)); mean[ii] = m
    return mean, std


def smooth(v, g, k=5):
    out = np.full_like(v, np.nan)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; x = v[ii]
        good = np.isfinite(x); x0 = np.where(good, x, 0.)
        cs = np.cumsum(np.r_[0., x0]); t = np.arange(len(ii)); lo = np.maximum(t-k+1, 0)
        n = np.cumsum(np.r_[0., good.astype(int)]); nn = n[t+1]-n[lo]
        out[ii] = np.divide(cs[t+1]-cs[lo], nn, out=np.full(len(ii), np.nan), where=nn > 0)
    return out


def direction_auc(v, y, g, mask):
    out = {'R->T': [], 'T->R': []}
    for s in np.unique(g):
        ii = np.where(g == s)[0]; cuts = np.flatnonzero(np.r_[True, np.diff(y[ii]) != 0, True])
        blocks = [ii[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
        for a, b in zip(blocks[:-1], blocks[1:]):
            sel = np.r_[a, b]; sel = sel[mask[sel]]
            if len(sel) and len(np.unique(y[sel])) == 2:
                key = 'R->T' if y[a[0]] == 0 else 'T->R'
                out[key].append(float(roc_auc_score(y[sel], v[sel])))
    return {k: (float(np.mean(x)) if x else None) for k, x in out.items()}


def metrics(v, y, g, mask):
    m = mask & np.isfinite(v)
    r = {'n': int(m.sum()), 'AUC': float(roc_auc_score(y[m], v[m])),
         'bACC_fixed05': float(balanced_accuracy_score(y[m], v[m] >= .5)),
         'bACC_median_diagnostic': float(balanced_accuracy_score(y[m], v[m] >= np.median(v[m])))}
    r.update(direction_auc(v, y, g, m))
    return r


def boot(a, b, y, g, mask, n=500):
    subs = np.unique(g); pools = [np.where((g == s) & mask)[0] for s in subs]
    rng = np.random.default_rng(0); d = []
    for _ in range(n):
        ix = np.concatenate([pools[j] for j in rng.integers(0, len(subs), len(subs))])
        if len(np.unique(y[ix])) == 2: d.append(roc_auc_score(y[ix], a[ix])-roc_auc_score(y[ix], b[ix]))
    d = np.asarray(d)
    return {'mean': float(d.mean()), 'p2.5': float(np.percentile(d, 2.5)),
            'p97.5': float(np.percentile(d, 97.5)), 'win': float(np.mean(d > 0))}


def hist_rank(z, g, k):
    """当前窗在本人过去 k 窗（不含当前窗）中的经验分位：(小于数 + 0.5*相等数)/n；无历史时为 0.5。"""
    out = np.empty_like(z)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; v = z[ii]
        less = np.zeros(v.shape, np.float32); eq = np.zeros(v.shape, np.float32)
        for lag in range(1, min(k, len(ii)-1)+1):
            less[lag:] += v[:-lag] < v[lag:]; eq[lag:] += v[:-lag] == v[lag:]
        n = np.minimum(np.arange(len(ii)), k)[:, None]
        out[ii] = np.divide(less+.5*eq, n, out=np.full(v.shape, .5), where=n > 0)
    return out


def positions(g):
    pos = np.empty(len(g), int)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; pos[ii] = np.arange(len(ii))
    return pos


def make_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=0)


def fit_oof(F, y, g, mask, tag, tick):
    oof = np.full(len(y), np.nan)
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, y, g), 1):
        tr = tr[mask[tr]]; te_eval = te[mask[te]]
        oof[te_eval] = make_model().fit(F[tr], y[tr]).predict_proba(F[te_eval])[:, 1]
        print(f'[progress] {tag} fold {fold}/5 train={len(tr)} test={len(te_eval)} '
              f'elapsed={time.perf_counter()-tick:.1f}s', flush=True)
    return oof


def summarize(oof, y, g, mask, tag):
    r = {'window': metrics(oof, y, g, mask), 'smooth5': metrics(smooth(oof, g, 5), y, g, mask)}
    print(f'[result] {tag} window AUC={r["window"]["AUC"]:.4f} bACC05={r["window"]["bACC_fixed05"]:.4f}; '
          f'smooth5 AUC={r["smooth5"]["AUC"]:.4f} bACC05={r["smooth5"]["bACC_fixed05"]:.4f} '
          f'R->T={r["smooth5"]["R->T"]:.4f} T->R={r["smooth5"]["T->R"]:.4f}', flush=True)
    return r


def check_features(Z, g):
    mu, sd = causal_mean_std(Z, g, L); pmu, psd = past_mean_std(Z, g, L)
    return {'q': Z, 'd200_mean': Z-mu, 'd200_mean_std': (Z-mu)/np.maximum(sd, .1),
            'past200_mean_std': (Z-pmu)/np.maximum(psd, .1)}


def block_perm(y, g, rng, samelabel):
    perm = []
    for s in np.unique(g):
        i = np.where(g == s)[0]
        blocks = np.split(i, np.flatnonzero(np.r_[1, np.diff(y[i]) != 0])[1:])
        if samelabel:
            lab = [y[b[0]] for b in blocks]
            pools = {c: list(rng.permutation([j for j in range(len(blocks)) if lab[j] == c])) for c in (0, 1)}
            perm += [blocks[pools[lab[j]].pop()] for j in range(len(blocks))]
        else:
            perm += [blocks[j] for j in rng.permutation(len(blocks))]
    return np.concatenate(perm)


def save_key(key, value):
    path = OUT/'context_fixed_histgb_results.json'
    res = json.loads(path.read_text(encoding='utf-8'))
    res[key] = value
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')


def run_check(Z, y, g, tick):
    out = {'meta': {
        'orders': {'orig': '原始顺序', 'samelabel0-2': '同标签块互换（R块间互换、T块间互换，保留交替），rng=default_rng(rep)',
                   'blockshuf0': '整块随机重排，rng=default_rng(0)'},
        'approximation': 'Z 的分位数校准沿用原始顺序前120窗（与 detrend_context.py 相同，对 Z 整体重排而非重做校准）；'
                         '重排后按新序列重新计算历史均值/标准差；训练与评估剔除的是重排后序列的前120窗，'
                         '因此原始校准窗可能进入训练/评估，而新的前120窗不一定是校准窗。',
        'split': 'StratifiedGroupKFold(5, shuffle=True, random_state=0) 在重排后数组上重新划分（受试者组成不变）',
        'delta': 'boot(variant_smooth5, q_smooth5)，受试者 bootstrap 500 次，seed 0'}}
    orders = {'orig': np.arange(len(y))}
    for rep in range(3):
        orders[f'samelabel{rep}'] = block_perm(y, g, np.random.default_rng(rep), True)
    orders['blockshuf0'] = block_perm(y, g, np.random.default_rng(0), False)
    with threadpool_limits(limits=4):
        for oname, perm in orders.items():
            ys, gs, Zs = y[perm], g[perm], Z[perm]
            ms = positions(gs) >= K
            res, sm = {}, {}
            for name, F in check_features(Zs, gs).items():
                oof = fit_oof(F, ys, gs, ms, f'{oname}/{name}', tick)
                res[name] = summarize(oof, ys, gs, ms, f'{oname}/{name}')
                sm[name] = smooth(oof, gs, 5)
            for name in sm:
                if name == 'q': continue
                b = res[name]['vs_q_smooth5'] = boot(sm[name], sm['q'], ys, gs, ms)
                print(f'[boot] {oname} {name}-q Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
            out[oname] = res
            out['seconds'] = time.perf_counter()-tick
            save_key('check', out)
    print(f'[done] check seconds={out["seconds"]:.1f}', flush=True)


def run_ext(Z, y, g, mask, tick):
    from scipy.stats import rankdata
    out = {'meta': {
        'hist_rank200': '每个 Z 特征在本人过去200窗（不含当前窗）的经验分位 (less+0.5*equal)/n；t0 无历史取0.5；'
                        'Z 已离散化（121级并截断），故用半计数处理并列',
        'ms_ens': 'd100/d200/d400_mean_std（含当前窗因果均值/标准差，std floor 0.1）三个 HistGB 的逐窗 OOF，'
                  '在评估窗上 rankdata/n 后平均，再 smooth5',
        'ms_ens_threshold_note': 'rank 尺度上固定0.5阈值约等于池化中位数阈值，且 rank 需要全体测试窗，不能在线部署；'
                                 '另附 ms_ens_probmean（概率平均）仅用于固定0.5 bACC 参考',
        'delta': 'boot(ext_smooth5, d200_mean_std_smooth5)，受试者 bootstrap 500 次，seed 0'}}
    feats = {}
    t0 = time.perf_counter()
    feats['hist_rank200'] = hist_rank(Z, g, L)
    out['meta']['hist_rank200_compute_seconds'] = time.perf_counter()-t0
    print(f'[info] hist_rank200 computed in {time.perf_counter()-t0:.1f}s', flush=True)
    for k in (100, 200, 400):
        mu, sd = causal_mean_std(Z, g, k); feats[f'd{k}_mean_std'] = (Z-mu)/np.maximum(sd, .1)
    oofs = {}
    with threadpool_limits(limits=4):
        for name, F in feats.items():
            oofs[name] = fit_oof(F, y, g, mask, name, tick)
            out[name] = summarize(oofs[name], y, g, mask, name)
    m = mask
    rk = np.full(len(y), np.nan)
    rk[m] = np.mean([rankdata(oofs[f'd{k}_mean_std'][m])/m.sum() for k in (100, 200, 400)], 0)
    oofs['ms_ens'] = rk
    out['ms_ens'] = summarize(rk, y, g, mask, 'ms_ens')
    oofs['ms_ens_probmean'] = np.mean([oofs[f'd{k}_mean_std'] for k in (100, 200, 400)], 0)
    out['ms_ens_probmean'] = summarize(oofs['ms_ens_probmean'], y, g, mask, 'ms_ens_probmean')
    ref = smooth(oofs['d200_mean_std'], g, 5)
    for name in ('hist_rank200', 'd100_mean_std', 'd400_mean_std', 'ms_ens', 'ms_ens_probmean'):
        b = out[name]['vs_d200_mean_std_smooth5'] = boot(smooth(oofs[name], g, 5), ref, y, g, mask)
        print(f'[boot] {name}-d200_mean_std Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
    out['seconds'] = time.perf_counter()-tick
    save_key('ext', out)
    print(f'[done] ext seconds={out["seconds"]:.1f}', flush=True)


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X, y, g = d['X'].astype(np.float64), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    base = np.hstack([X[:, X.std(0) > 1e-8], E])
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; pos[ii] = np.arange(len(ii))
    eval_mask = pos >= K
    Z = subject_quantile(base, g, K)
    if '--check' in sys.argv:
        return run_check(Z, y, g, tick)
    if '--ext' in sys.argv:
        return run_ext(Z, y, g, eval_mask, tick)
    mu, sd = causal_mean_std(Z, g, L)
    past_mu, past_sd = past_mean_std(Z, g, L)
    features = {'q': Z, 'd200_mean': Z-mu, 'd200_mean_std': (Z-mu)/np.maximum(sd, .1),
                'past200_mean': Z-past_mu, 'past200_mean_std': (Z-past_mu)/np.maximum(past_sd, .1)}
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Z, y, g))
    make = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                    l2_regularization=1., random_state=0)
    result = {'meta': {'folds': 5, 'split_seed': SEED, 'warmup_windows': K,
                       'context_windows': L, 'std_floor': .1,
                       'train_excludes_warmup': True,
                       'current_window': {'causal_mean_std': 'include_current', 'past200_mean': 'exclude_current',
                                          'past200_mean_std': 'exclude_current'},
                       'past_context_initialization': 't0 uses 0 mean and 0 std before std floor; warmup t0-t119 is not evaluated',
                       'note': '200窗历史与smooth5感受野均超过6秒；不声称严格6秒合规。'}, 'variants': {}}
    oofs = {}
    with threadpool_limits(limits=4):
        for name, F in features.items():
            oof = np.full(len(y), np.nan)
            for fold, (tr, te) in enumerate(splits, 1):
                tr = tr[eval_mask[tr]]
                te_eval = te[eval_mask[te]]
                model = make().fit(F[tr], y[tr])
                oof[te_eval] = model.predict_proba(F[te_eval])[:, 1]
                print(f'[progress] {name} fold {fold}/5 train={len(tr)} test={len(te_eval)} elapsed={time.perf_counter()-tick:.1f}s', flush=True)
            oofs[name] = oof
            result['variants'][name] = {'window': metrics(oof, y, g, eval_mask),
                                        'smooth5': metrics(smooth(oof, g, 5), y, g, eval_mask)}
            print(f'[result] {name} window AUC={result["variants"][name]["window"]["AUC"]:.4f} '
                  f'bACC05={result["variants"][name]["window"]["bACC_fixed05"]:.4f}; '
                  f'smooth5 AUC={result["variants"][name]["smooth5"]["AUC"]:.4f} '
                  f'bACC05={result["variants"][name]["smooth5"]["bACC_fixed05"]:.4f}', flush=True)
    ref = 'd200_mean'
    for name in features:
        if name == ref: continue
        result['variants'][name]['vs_d200_window'] = boot(oofs[name], oofs[ref], y, g, eval_mask)
        result['variants'][name]['vs_d200_smooth5'] = boot(smooth(oofs[name], g, 5), smooth(oofs[ref], g, 5), y, g, eval_mask)
    result['seconds'] = time.perf_counter()-tick
    path = OUT/'context_fixed_histgb_results.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        result.update({k: old[k] for k in ('check', 'ext') if k in old})
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] {path} seconds={result["seconds"]:.1f}', flush=True)


if __name__ == '__main__':
    main()
