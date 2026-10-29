"""特征层因果上下文：先把过去 k 窗的特征做因果平均，再训练分类器。

此前的时序积分都在分数层（先逐窗打分，再平均分数）。分数层平均只能压低分类器输出的方差；
如果特征本身的窗间噪声很大，分类器在单窗上学到的边界就会被噪声主导。
这里把积分前移到特征层：训练和测试都使用 Z_t' = mean(Z_{t-k+1..t})（只看过去，受试者内）。
不使用窗口序号、块序号或标签；与 30 s 分数平均同属"利用状态连续性"的合法操作。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot

K = 120


def feat_smooth(Z, g, k):
    """受试者内因果滑动平均（逐列），窗口 i 只用 i-k+1..i。"""
    out = np.empty_like(Z)
    for s in np.unique(g):
        idx = np.where(g == s)[0]; v = Z[idx]
        c = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, 0)])
        t = np.arange(len(v)); lo = np.maximum(t-k+1, 0)
        out[idx] = (c[t+1]-c[lo])/(t+1-lo)[:, None]
    return out


def metrics(v, y, mask):
    return {'AUC': float(roc_auc_score(y[mask], v[mask])),
            'bACC': float(balanced_accuracy_score(y[mask], v[mask] >= np.median(v[mask])))}


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X0, y, g = d['X'].astype(np.float64), d['y'].astype(int), d['groups']
    E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
    base = np.hstack([X0[:, X0.std(0) > 1e-8], E])
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= K
    Z = subject_quantile(base, g, K)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Z, y, g))
    print(f'[data] {Z.shape} 校准完成 {time.perf_counter()-tick:.0f}s', flush=True)

    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k: z[k] for k in z.files if k not in ('y', 'groups')})
    P3 = ['quant120_histgb', 'quant120w_histgb', 'quant120_svc', 'base_svc']
    p3r = np.mean([rankdata(pool[k])/len(y) for k in P3], 0)
    p3 = smooth(p3r, g, 5)
    res = {'base_p3': metrics(p3, y, mask), 'variants': {}}
    print(f'[base] P3 {res["base_p3"]}', flush=True)

    gb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                                l2_regularization=1., random_state=0)
    sv = lambda: make_pipeline(StandardScaler(), SVC(C=1., gamma='scale', cache_size=4096))
    oofs = {}

    def run(F, tag, make):
        oof = np.full(len(y), np.nan)
        for j, (tr, te) in enumerate(splits):
            m = make().fit(F[tr], y[tr])
            oof[te] = m.predict_proba(F[te])[:, 1] if hasattr(m, 'predict_proba') and not tag.endswith('svc') \
                else m.decision_function(F[te])
            print(f'  [{tag}] fold {j+1}/5 ({time.perf_counter()-tick:.0f}s)', flush=True)
        oofs[tag] = oof
        r = {}
        for ks in (1, 3, 5):
            r[f'post{ks}'] = metrics(smooth(oof, g, ks), y, mask)
        b = boot(smooth(oof, g, 5), p3, y, g, mask)
        r['vs_p3_post5'] = b
        res['variants'][tag] = r
        print(f'[result] {tag:18s} ' + ' '.join(f'{k}={v["AUC"]:.4f}/{v["bACC"]:.3f}' for k, v in r.items()
                                               if k.startswith('post'))
              + f' Δ5={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
        (OUT/'feat_context_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')

    # HistGB：特征层平均窗长扫描，与原始逐窗特征做对照
    run(Z, 'q_histgb', gb)
    for k in (3, 5, 8):
        run(feat_smooth(Z, g, k), f'qs{k}_histgb', gb)
    # 当前窗 + 过去平均拼接：保留瞬时信息同时给出上下文
    run(np.hstack([Z, feat_smooth(Z, g, 5)]), 'q+qs5_histgb', gb)
    best = max((k for k in res['variants'] if k.startswith('qs')),
               key=lambda k: res['variants'][k]['post5']['AUC'])
    kb = int(best[2:].split('_')[0])
    run(feat_smooth(Z, g, kb), f'qs{kb}_svc', sv)

    # 与 P3 合并（固定等权 rank，事先锁定，不搜索权重）
    for tag in (best, f'qs{kb}_svc'):
        mix = smooth(np.mean([rankdata(pool[k])/len(y) for k in P3] + [rankdata(oofs[tag])/len(y)], 0), g, 5)
        b = boot(mix, p3, y, g, mask)
        res['variants'][f'P3+{tag}'] = {**metrics(mix, y, mask), 'vs_p3': b}
        print(f'[mix] P3+{tag} {metrics(mix, y, mask)} Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)
    mix = smooth(np.mean([rankdata(oofs[t])/len(y) for t in (best, f'qs{kb}_svc')]
                         + [rankdata(pool[k])/len(y) for k in P3], 0), g, 5)
    b = boot(mix, p3, y, g, mask)
    res['variants']['P3+both'] = {**metrics(mix, y, mask), 'vs_p3': b}
    print(f'[mix] P3+both {metrics(mix, y, mask)} Δ={b["mean"]:+.4f} [{b["p2.5"]:+.4f},{b["p97.5"]:+.4f}]', flush=True)

    res['seconds'] = time.perf_counter()-tick
    np.savez_compressed(OUT/'feat_context_oof.npz', y=y, groups=g, **{k.replace('+', 'p'): v for k, v in oofs.items()})
    (OUT/'feat_context_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
