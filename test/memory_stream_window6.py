"""记忆流式版多尺度历史模型：每次推理只输入新来的 6 秒窗，历史以记忆状态保存。

记忆状态（每个被试一份）：
  cal   : 前 K0 个窗的原始特征（分位数校准表）
  hist  : 最近 800 个窗的校准后特征（用于多尺度均值/方差，不含当前窗）
  score : 最近 5 个窗的模型分数（因果平滑）
predict(x_6s) 只读当前窗特征 + 记忆，输出后再把当前窗写入记忆。
另比较若干提升 bACC 的变体，阈值与规则只来自训练折或因果记忆。
"""
import json, sys, time
from collections import deque
from pathlib import Path
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from b_calib_variants import subject_quantile
from context_history_extend import causal_mean_std, smooth, metrics, Tee

K0, SEED, SCALES, FLOOR = 120, 0, (50, 100, 200, 400, 800), .1
BAD = {'008', '009', '034'}
TRANS = 8  # 块开头约 48 秒视为过渡期（训练标签清洗用）


def model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3,
                                          l2_regularization=1., random_state=0)


class MemoryStream:
    """单被试在线推理器。models: {k: 已训练分类器}。"""
    def __init__(self, models, n_smooth=5):
        self.models = models; self.cal = []; self.calsort = None
        self.hist = deque(maxlen=max(SCALES)); self.scores = deque(maxlen=n_smooth)

    def _quantile(self, x):
        p = np.array([np.searchsorted(self.calsort[:, j], x[j], side='left') for j in range(len(x))])/K0
        return norm.ppf(np.clip(p, .5/K0, 1-.5/K0))

    def predict(self, x):
        """x: 当前 6 秒窗的特征向量。返回 (raw, smooth) 分数；冷启动阶段返回 None。"""
        if self.calsort is None:
            self.cal.append(x)
            if len(self.cal) < K0: return None
            self.calsort = np.sort(np.asarray(self.cal), axis=0)
            for c in self.cal: self.hist.append(self._quantile(c))
            self.cal = None
            return None
        z = self._quantile(x); H = np.asarray(self.hist); ps = []
        for k in SCALES:
            h = H[-k:]; mu = h.mean(0); sd = np.sqrt(np.maximum((h*h).mean(0)-mu*mu, 0))
            f = np.round((z-mu)/np.maximum(sd, FLOOR), 6)  # 与批量版一致，消除 1e-11 级浮点差对树阈值的影响
            ps.append(self.models[k].predict_proba(f[None])[0, 1])
        p = float(np.mean(ps)); self.scores.append(p); self.hist.append(z)
        return p, float(np.mean(self.scores))


def youden(y, p):
    t = np.unique(np.quantile(p, np.linspace(.02, .98, 97)))
    return float(t[np.argmax([balanced_accuracy_score(y, p >= v) for v in t])])


def subject_median_threshold(v, g, mask):
    """因果自适应阈值：用本人过去已输出分数的中位数作为阈值（只读记忆）。"""
    out = np.full(len(v), np.nan)
    for s in np.unique(g):
        ii = np.where((g == s) & mask & np.isfinite(v))[0]
        for t, i in enumerate(ii):
            out[i] = np.median(v[ii[:t]]) if t >= 20 else .5
    return out


def bacc_at(v, thr, y, mask):
    m = mask & np.isfinite(v)
    return float(balanced_accuracy_score(y[m], v[m] >= thr[m]))


def main():
    start = time.perf_counter(); tee = Tee(sys.stdout, OUT/'memory_stream_window6.log'); sys.stdout = tee
    try:
        d = np.load(OUT/'highdim_deep_features.npz'); X = d['X'].astype(np.float64); y = d['y'].astype(int); g = d['groups']
        E = np.load(OUT/'emg_features_v2.npz')['E'].astype(np.float64)
        base = np.hstack([X[:, X.std(0) > 1e-8], E]); Z = subject_quantile(base, g, K0)
        pos = np.empty(len(y), int); bpos = np.empty(len(y), int)
        for s in np.unique(g):
            ii = np.where(g == s)[0]; pos[ii] = np.arange(len(ii))
            cuts = np.flatnonzero(np.r_[True, np.diff(y[ii]) != 0])
            bpos[ii] = np.arange(len(ii)) - cuts[np.searchsorted(cuts, np.arange(len(ii)), side='right')-1]
        mask = pos >= K0; clean = bpos >= TRANS; good = ~np.isin(np.char.zfill(g.astype(str), 3), list(BAD))
        F = {}
        for k in SCALES:
            mu, sd = causal_mean_std(Z, g, k, True); F[k] = np.round((Z-mu)/np.maximum(sd, FLOOR), 6)
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(Z, y, g))
        variants = {'base': lambda tr: tr, 'clean_trans': lambda tr: tr[clean[tr]],
                    'drop_bad': lambda tr: tr[good[tr]], 'clean+drop': lambda tr: tr[clean[tr] & good[tr]]}
        P = {v: np.full(len(y), np.nan) for v in variants}; thr_oof = {v: np.full(len(y), .5) for v in variants}
        stream_check = {}
        with threadpool_limits(limits=4):
            for fold, (tr0, te0) in enumerate(splits, 1):
                tr0, te = tr0[mask[tr0]], te0[mask[te0]]
                for vname, sel in variants.items():
                    tr = sel(tr0); models = {}
                    for k in SCALES:
                        models[k] = model().fit(F[k][tr], y[tr])
                    P[vname][te] = np.mean([models[k].predict_proba(F[k][te])[:, 1] for k in SCALES], axis=0)
                    # 训练折内部 OOF 求 Youden 阈值（按被试分 4 折）
                    inner = np.full(len(y), np.nan)
                    for itr, ite in StratifiedGroupKFold(4, shuffle=True, random_state=SEED).split(tr, y[tr], g[tr]):
                        a, b = tr[itr], tr[ite]
                        inner[b] = np.mean([model().fit(F[k][a], y[a]).predict_proba(F[k][b])[:, 1] for k in SCALES], axis=0)
                    sm = smooth(inner, g); ok = np.isfinite(sm)
                    thr_oof[vname][te] = youden(y[ok], sm[ok])
                    print(f'[progress] fold {fold}/5 variant={vname} train={len(tr)} thr={thr_oof[vname][te[0]]:.3f} elapsed={time.perf_counter()-start:.1f}s', flush=True)
                    if vname == 'base' and fold == 1:
                        # 逐窗流式推理复现：每次只喂入当前窗特征
                        diffs = []
                        for s in np.unique(g[te0])[:3]:
                            ii = np.where(g == s)[0]; ms = MemoryStream(models); out = []
                            for i in ii:
                                r = ms.predict(base[i]); out.append(np.nan if r is None else r[0])
                            out = np.asarray(out); m = np.isfinite(out)
                            diffs.append(float(np.max(np.abs(out[m]-P['base'][ii][m]))))
                            print(f'[check] subject={s} stream_vs_batch max|diff|={diffs[-1]:.2e} n={int(m.sum())}', flush=True)
                        stream_check = {'subjects_checked': 3, 'max_abs_diff': max(diffs)}
        res = {'meta': {'scales': list(SCALES), 'warmup': K0, 'std_floor': FLOOR, 'trans_windows': TRANS,
                        'bad_subjects': sorted(BAD), 'inference_input': '仅当前 6 秒窗；历史以记忆状态保存',
                        'stream_check': stream_check}, 'variants': {}}
        for vname in variants:
            raw, sm = P[vname], smooth(P[vname], g)
            r = {'raw': metrics(raw, y, g, mask), 'smooth5': metrics(sm, y, g, mask)}
            r['bACC_youden_trainOOF'] = bacc_at(sm, thr_oof[vname], y, mask)
            r['bACC_causal_median'] = bacc_at(sm, subject_median_threshold(sm, g, mask), y, mask)
            res['variants'][vname] = r
            print(f'[result] {vname} smooth5 AUC={r["smooth5"]["AUC"]:.4f} bACC@.5={r["smooth5"]["bACC"]:.4f} '
                  f'bACC@youden={r["bACC_youden_trainOOF"]:.4f} bACC@causal_median={r["bACC_causal_median"]:.4f}', flush=True)
        np.savez(OUT/'memory_stream_window6_oof.npz', y=y, groups=g, eval_mask=mask, **{k: v for k, v in P.items()})
        res['seconds'] = time.perf_counter()-start
        (OUT/'memory_stream_window6_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[done] seconds={res["seconds"]:.1f}', flush=True)
    finally:
        sys.stdout = tee.stream; tee.close()


if __name__ == '__main__': main()
