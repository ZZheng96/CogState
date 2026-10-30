"""可部署主方法（预注册）：因果无标签模式识别 + 模式专属专家（混合专家）。

预先固定：54 维细谱特征（subject_mode_analysis.fine_features）；被试内因果标准化（先验 PRIOR=32 窗的训练全局均值/方差，
统计用 ≤t 的窗，不含未来）；模式 = 训练折被试 R/T 效应量单位向量 KMeans(k=3, n_init=20, rs=0)（k=3 由诊断预先固定，
不在测试结果上选）；专家 = LR(C=0.1, balanced) 每模式一个 + 全局 LR；
识别器 = 多项 LR(C=0.1) 输入被试过去无标签窗描述子：原始特征历史均值、标准差、历史协方差 PC1 |载荷|，
以及对每个训练模式方向 mu_m 的投影 skew/kurtosis/|cos(PC1,mu_m)|（符号线索只来自训练折学到的映射）。
识别器训练样本 = 训练被试在前缀长度 {20,40,80,160,全长} 的描述子；
在线：t<20 用全局模型；t≥20 时 p = Σ_m w_m(t) p_m，w(t) 由 ≤t 的无标签窗计算。阈值 0.5。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(OUT)]
from subject_mode_analysis import fine_features, cohen_d, TOP, BOT  # noqa: E402
from causal_kernel_state_mapping import metrics  # noqa: E402

PRIOR, K, C, WARM = 32., 3, 0.1, 20
PREFIXES = (20, 40, 80, 160, None)
T0 = time.perf_counter()
LOGF = None


def log(msg):
    s = f'[{time.strftime("%H:%M:%S")} +{time.perf_counter() - T0:7.1f}s] {msg}'
    print(s, flush=True)
    if LOGF:
        LOGF.write(s + '\n'); LOGF.flush()


def causal_z(x, mu0, var0):
    """被试内因果标准化：第 t 窗用窗 0..t 与先验混合的均值/方差。"""
    n = np.arange(1, len(x) + 1)[:, None] + PRIOR
    m = (PRIOR * mu0 + np.cumsum(x, 0)) / n
    s2 = (PRIOR * (var0 + mu0 ** 2) + np.cumsum(x ** 2, 0)) / n - m ** 2
    return (x - m) / np.sqrt(np.maximum(s2, 1e-6))


def descriptor(xs, zs, mus):
    """xs: 全局缩放后的原始特征前缀; zs: 因果 z 前缀; mus: [K, D] 训练模式方向。只用无标签前缀。"""
    zc = zs - zs.mean(0)
    w, v = np.linalg.eigh(zc.T @ zc / len(zc))
    pc1 = v[:, -1]
    extra = []
    for mu in mus:
        pr = zs @ mu
        extra += [skew(pr), kurtosis(pr), abs(pc1 @ mu)]
    return np.r_[xs.mean(0), xs.std(0), np.abs(pc1), extra]


def run_fold(F, y, g, tr, te, fold, check):
    sc = StandardScaler().fit(F[tr])
    Xs = sc.transform(F)
    mu0, var0 = Xs[tr].mean(0), Xs[tr].var(0)
    Z = np.empty_like(Xs)
    for s in np.unique(g[np.r_[tr, te]]):
        ii = np.flatnonzero(g == s)
        Z[ii] = causal_z(Xs[ii], mu0, var0)
    trs = np.unique(g[tr]); tes = np.unique(g[te])
    assert not set(trs) & set(tes)
    # 训练被试模式（训练标签）
    Dv = np.stack([cohen_d(Xs[g == s], y[g == s]) for s in trs])
    U = Dv / np.linalg.norm(Dv, axis=1, keepdims=True)
    km = KMeans(K, n_init=20, random_state=0).fit(U)
    mode = dict(zip(trs, km.labels_))
    mus = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    lr = lambda: LogisticRegression(C=C, class_weight='balanced', max_iter=3000)
    glob = lr().fit(Z[tr], y[tr])
    experts = []
    for m in range(K):
        mm = tr[np.isin(g[tr], [s for s in trs if mode[s] == m])]
        experts.append(lr().fit(Z[mm], y[mm]))
    log(f'  fold {fold}: mode sizes={np.bincount(km.labels_, minlength=K).tolist()} '
        f'mode band means(1-45Hz mean d)={[round(float(c[:45].mean()), 3) for c in km.cluster_centers_]}')
    # 识别器
    Dx, Dy = [], []
    for s in trs:
        ii = np.flatnonzero(g == s)
        for L in PREFIXES:
            L = len(ii) if L is None else min(L, len(ii))
            Dx.append(descriptor(Xs[ii[:L]], Z[ii[:L]], mus)); Dy.append(mode[s])
    dsc = StandardScaler().fit(Dx)
    rec = LogisticRegression(C=C, max_iter=5000).fit(dsc.transform(Dx), Dy)
    # 测试被试在线
    out = {k: np.full(len(y), np.nan) for k in ('moe_gated', 'global_only', 'oracle_mode_nondeployable', 'uniform_mix')}
    rec_post = np.full((len(y), K), np.nan)
    true_mode = {}
    for s in tes:
        ii = np.flatnonzero(g == s)
        pe = np.stack([e.predict_proba(Z[ii])[:, 1] for e in experts], 1)
        pg = glob.predict_proba(Z[ii])[:, 1]
        W = online_weights(Xs[ii], Z[ii], mus, dsc, rec)
        rec_post[ii] = W
        pm = np.where(np.isnan(W[:, 0]), pg, np.nansum(W * pe, 1))
        out['moe_gated'][ii] = pm
        out['global_only'][ii] = pg
        out['uniform_mix'][ii] = pe.mean(1)
        dv = cohen_d(Xs[ii], y[ii]); dv /= np.linalg.norm(dv)
        tm = int(np.argmax(mus @ dv)); true_mode[s] = tm
        out['oracle_mode_nondeployable'][ii] = pe[:, tm]
        if check:
            cut = len(ii) // 2
            Xa = Xs[ii].copy(); Xa[cut:] += 50
            Za = causal_z(Xa, mu0, var0)
            assert np.allclose(Za[:cut], Z[ii][:cut], atol=0, rtol=0)
            Wa = online_weights(Xa, Za, mus, dsc, rec, upto=cut)
            assert np.allclose(np.nan_to_num(Wa[:cut], nan=-1), np.nan_to_num(W[:cut], nan=-1), atol=0, rtol=0)
            Wp = online_weights(Xs[ii][:cut], causal_z(Xs[ii][:cut], mu0, var0), mus, dsc, rec)
            assert np.allclose(np.nan_to_num(Wp, nan=-1), np.nan_to_num(W[:cut], nan=-1), atol=1e-12, rtol=0)
    return out, rec_post, true_mode


def online_weights(xs, zs, mus, dsc, rec, upto=None):
    n = len(xs) if upto is None else upto
    W = np.full((len(xs), K), np.nan)
    for t in range(WARM - 1, n):
        W[t] = rec.predict_proba(dsc.transform(descriptor(xs[:t + 1], zs[:t + 1], mus)[None]))[0]
    return W


def subj_bacc(y, g, p, subs):
    return np.array([balanced_accuracy_score(y[g == s], p[g == s] >= .5) for s in subs])


def subj_auc(y, g, p, subs):
    return np.array([roc_auc_score(y[g == s], p[g == s]) for s in subs])


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    stem = 'subject_mode_moe' + ('_smoke' if args.smoke else '')
    paths = [OUT / f'{stem}.log', OUT / f'{stem}_results.json', OUT / f'{stem}_oof.npz']
    if not args.smoke and any(p.exists() for p in paths):
        raise FileExistsError('refusing to overwrite subject_mode_moe artifacts')
    LOGF = paths[0].open('w', encoding='utf-8')
    d = np.load(OUT / 'highdim_deep_features.npz')
    raw, y, g = d['raw'], d['y'].astype(int), d['groups'].astype(str)
    F = fine_features(raw)
    log(f'[data] F={F.shape} subjects={len(np.unique(g))}')
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(F, y, g))
    P = {k: np.full(len(y), np.nan) for k in ('moe_gated', 'global_only', 'oracle_mode_nondeployable', 'uniform_mix')}
    post = np.full((len(y), K), np.nan); fold_id = np.full(len(y), -1); tmode = {}
    for fold, (tr, te) in enumerate(splits):
        if args.smoke:
            tr = tr[np.isin(g[tr], np.unique(g[tr])[:15])]; te = te[np.isin(g[te], np.unique(g[te])[:2])]
        out, rp, tm = run_fold(F, y, g, tr, te, fold, check=True)
        for k in P:
            P[k][te] = out[k][te]
        post[te] = rp[te]; fold_id[te] = fold; tmode.update(tm)
        log(f'[progress] fold {fold + 1}/5 done; prefix/future invariance + isolation asserts passed')
        if args.smoke and fold == 1:
            break
    mask = fold_id >= 0
    subs = np.unique(g[mask])
    mb = np.load(OUT / 'merge_best_window6_oof.npz')
    assert np.array_equal(mb['y'], y) and np.array_equal(mb['groups'].astype(str), g)
    P['merge_best'] = expit(mb['combo__C4_bb_resnet_eq'])
    res = {k: metrics(y, g, v, mask) for k, v in P.items()}
    for k, v in res.items():
        log(f'[result] {k:28s} {json.dumps(v)}')
    # 识别器准确率（用测试标签定义的“真模式”，仅报告）
    acc = [np.nanmean(np.argmax(post[g == s][WARM:], 1) == tmode[s]) for s in subs]
    log(f'[recognizer] mean frac windows argmax==oracle mode: {np.mean(acc):.3f}')
    rng = np.random.default_rng(0)
    ceil = {r['subject']: r for r in json.loads((OUT / 'per_subject_ceiling_results.json').read_text(encoding='utf-8'))['per_subject']}
    flips = [s for s in subs if ceil.get(s, {}).get('direction_flip') == 1]
    ba_mb, au_mb = subj_bacc(y, g, P['merge_best'], subs), subj_auc(y, g, P['merge_best'], subs)
    boot, groups = {}, {}
    for k in ('moe_gated', 'global_only', 'uniform_mix', 'oracle_mode_nondeployable'):
        ba, au = subj_bacc(y, g, P[k], subs), subj_auc(y, g, P[k], subs)
        db, da = ba - ba_mb, au - au_mb
        bs = np.array([[db[i].mean(), da[i].mean()] for i in (rng.integers(0, len(subs), len(subs)) for _ in range(1000))])
        boot[k] = {'delta_subject_bacc_mean': float(db.mean()), 'ci95_bacc': np.quantile(bs[:, 0], [.025, .975]).tolist(),
                   'delta_subject_auc_mean': float(da.mean()), 'ci95_auc': np.quantile(bs[:, 1], [.025, .975]).tolist()}
        groups[k] = {}
        for nm, ss in (('top', TOP), ('bottom', BOT), ('flip', flips)):
            m = np.isin(subs, ss)
            groups[k][nm] = {'n': int(m.sum()), 'median_delta_bacc': float(np.median(db[m])) if m.any() else None,
                             'median_delta_auc': float(np.median(da[m])) if m.any() else None,
                             'median_bacc': float(np.median(ba[m])) if m.any() else None,
                             'median_bacc_merge_best': float(np.median(ba_mb[m])) if m.any() else None}
        log(f'[delta vs merge_best] {k}: {json.dumps(boot[k])}')
        log(f'[groups] {k}: {json.dumps(groups[k])}')
    result = {'protocol': {'deployable_main': 'moe_gated', 'K': K, 'C': C, 'prior': PRIOR, 'warmup_windows': WARM,
                           'recognizer_prefixes': [str(p) for p in PREFIXES], 'threshold': .5, 'hyperparameter_search': False,
                           'outer_cv': 'StratifiedGroupKFold(5, shuffle, rs=0)',
                           'nondeployable': ['oracle_mode_nondeployable (mode from test-subject labels)'],
                           'checks': 'per test subject: causal z + gating weights unchanged under future perturbation (+50) '
                                     'and equal to prefix-only recomputation; train/test subjects disjoint'},
              'metrics': res, 'delta_vs_merge_best': boot, 'groups': groups, 'recognizer_acc_vs_oracle': float(np.mean(acc)),
              'flip_subjects': flips, 'smoke': args.smoke, 'seconds': time.perf_counter() - T0}
    paths[1].write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(paths[2], y=y, groups=g, fold_id=fold_id, gate_posterior=post, **P)
    log(f'[done] -> {paths[1].name}')


if __name__ == '__main__':
    main()
