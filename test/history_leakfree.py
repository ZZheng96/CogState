"""无浮点泄漏的历史模型重跑。

旧版 causal_mean_std 用浮点 cumsum 差分，历史内恒定的特征会留下 ~1e-11 的位置相关残差，被树模型利用。
这里把分位数特征量化为整数后做 int64 累加，滑窗均值/方差精确；恒定历史的方差严格为 0。
对照：HistGB 与标准化 Logistic（线性模型无法利用微小残差）。
附加：严格单窗最佳分数（merge_best C4_bb_resnet_eq）的记忆相对化（本人过去分数 z 化）。
推理时只输入当前 6 秒窗，历史来自记忆；阈值来自 0.5 或本人过去分数中位数。
"""
import json, sys, time
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from context_history_extend import Tee
from memory_stream_window6 import subject_median_threshold

K0, SEED, SCALES, FLOOR, Q = 120, 0, (50, 100, 200, 400, 800), .1, 10**6
SMOOTHS = (1, 5, 10, 20)


def exact_mean_std(z, g, k):
    """排除当前窗的最近 k 窗均值/标准差，整数累加保证精确。"""
    zi = np.round(z*Q).astype(np.int64); mean = np.zeros_like(z); std = np.zeros_like(z)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; v = zi[ii]
        cs = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v, 0)])
        cs2 = np.vstack([np.zeros((1, v.shape[1]), np.int64), np.cumsum(v*v, 0)])
        t = np.arange(len(ii)); lo = np.maximum(t-k, 0); n = (t-lo)[:, None]
        S1, S2 = cs[t]-cs[lo], cs2[t]-cs2[lo]
        var_num = n*S2 - S1*S1  # 整数，恒定历史严格为 0
        nn = np.maximum(n, 1).astype(np.float64)
        mean[ii] = np.where(n > 0, S1/nn/Q, 0.)
        std[ii] = np.where(n > 1, np.sqrt(np.maximum(var_num, 0).astype(np.float64))/nn/Q, 0.)
    return mean, std


def causal_smooth(v, g, k):
    out = np.full_like(v, np.nan)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; x = v[ii]
        for t in range(len(ii)):
            w = x[max(0, t-k+1):t+1]; w = w[np.isfinite(w)]
            if len(w): out[ii[t]] = w.mean()
    return out


def causal_z(v, g, mask, k=400, floor=.2):
    """记忆相对化：用本人过去（eval 区开始前也可用，因分数无需标签）最多 k 个分数做 z 化。"""
    out = np.full_like(v, np.nan)
    for s in np.unique(g):
        ii = np.where(g == s)[0]; x = v[ii]
        for t in range(len(ii)):
            h = x[max(0, t-k):t]; h = h[np.isfinite(h)]
            if len(h) >= 20 and np.isfinite(x[t]): out[ii[t]] = (x[t]-h.mean())/max(h.std(), floor)
    return out


def evaluate(v, y, g, mask, center):
    r = {}
    for k in SMOOTHS:
        sm = causal_smooth(v, g, k); m = mask & np.isfinite(sm)
        r[f's{k}'] = {'AUC': float(roc_auc_score(y[m], sm[m])),
                      'bACC_fixed': float(balanced_accuracy_score(y[m], sm[m] >= center)),
                      'bACC_causal_median': float(balanced_accuracy_score(y[m], sm[m] >= subject_median_threshold(sm, g, mask)[m]))}
    return r


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3, l2_regularization=1., random_state=0)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=.01, max_iter=2000))


def line(name, r):
    return name + ' | ' + ' '.join(f'{k}:AUC={v["AUC"]:.4f}/bA={v["bACC_fixed"]:.4f}/bAmed={v["bACC_causal_median"]:.4f}' for k, v in r.items())


def main():
    start = time.perf_counter(); tee = Tee(sys.stdout, OUT/'history_leakfree.log'); sys.stdout = tee
    try:
        d = np.load(OUT/'highdim_deep_features.npz'); X = d['X'].astype(np.float64); y = d['y'].astype(int); g = d['groups']
        E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
        Z = subject_quantile(np.hstack([X[:, X.std(0) > 1e-8], E]), g, K0)
        pos = np.empty(len(y), int)
        for s in np.unique(g): pos[g == s] = np.arange(np.sum(g == s))
        mask = pos >= K0
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Z, y, g))
        res = {'meta': {'scales': list(SCALES), 'warmup': K0, 'std_floor': FLOOR, 'quant': Q, 'smooths': list(SMOOTHS),
                        'note': '整数累加精确滑窗统计；线性对照；推理输入仅当前6秒窗+记忆'}, 'single': {}, 'fusion': {}}
        P = {}
        with threadpool_limits(limits=4):
            for k in SCALES:
                mu, sd = exact_mean_std(Z, g, k); F = np.round((Z-mu)/np.maximum(sd, FLOOR), 6)
                for mname, mk in (('hgb', hgb), ('lr', lr)):
                    p = np.full(len(y), np.nan)
                    for fold, (tr, te) in enumerate(splits, 1):
                        tr, te = tr[mask[tr]], te[mask[te]]
                        p[te] = mk().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
                    key = f'd{k}_{mname}'; P[key] = p; res['single'][key] = evaluate(p, y, g, mask, .5)
                    print('[result] ' + line(key, res['single'][key]) + f' elapsed={time.perf_counter()-start:.0f}s', flush=True)
        for mname in ('hgb', 'lr'):
            key = f'D_{mname}'; P[key] = np.mean([P[f'd{k}_{mname}'] for k in SCALES], 0)
            res['fusion'][key] = evaluate(P[key], y, g, mask, .5); print('[result] ' + line(key, res['fusion'][key]), flush=True)
        P['D_hgb+lr'] = (P['D_hgb'] + P['D_lr'])/2
        res['fusion']['D_hgb+lr'] = evaluate(P['D_hgb+lr'], y, g, mask, .5); print('[result] ' + line('D_hgb+lr', res['fusion']['D_hgb+lr']), flush=True)
        # 严格单窗分数 + 记忆相对化
        mb = np.load(OUT/'merge_best_window6_oof.npz'); assert (mb['y'] == y).all() and (mb['groups'] == g).all()
        s0 = mb['combo__C4_bb_resnet_eq']; P['single6'] = s0; P['single6_memz'] = causal_z(s0, g, mask)
        res['fusion']['single6'] = evaluate(s0, y, g, mask, 0.)
        res['fusion']['single6_memz'] = evaluate(P['single6_memz'], y, g, mask, 0.)
        for key in ('single6', 'single6_memz'): print('[result] ' + line(key, res['fusion'][key]), flush=True)
        # 历史模型与单窗记忆分数的 rank 融合（单被试内因果 rank 不易做，这里用 z 化 logit 简单平均）
        lg = np.log(np.clip(P['D_hgb'], 1e-6, 1-1e-6)/(1-np.clip(P['D_hgb'], 1e-6, 1-1e-6)))
        P['D_hgb+single6_memz'] = causal_z(lg, g, mask) + P['single6_memz']
        res['fusion']['D_hgb+single6_memz'] = evaluate(P['D_hgb+single6_memz'], y, g, mask, 0.)
        print('[result] ' + line('D_hgb+single6_memz', res['fusion']['D_hgb+single6_memz']), flush=True)
        np.savez(OUT/'history_leakfree_oof.npz', y=y, groups=g, eval_mask=mask, **P)
        res['seconds'] = time.perf_counter()-start
        (OUT/'history_leakfree_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[done] seconds={res["seconds"]:.0f}', flush=True)
    finally:
        sys.stdout = tee.stream; tee.close()


if __name__ == '__main__': main()
