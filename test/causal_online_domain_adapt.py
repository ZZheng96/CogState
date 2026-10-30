"""预注册的严格因果无标签在线域适配。

协议：固定 merge_best_window6 的 C4+宽频 LR+ResNet 融合分数；外层 5 折中，
训练折仅用标签估计两状态的一维分数原型与尺度。每个测试被试按原始流顺序处理，
当前窗分类只使用此前窗的在线原型；分类后才用当前分数按最大后验模式更新对应原型。
更新为固定 EMA alpha，不读取测试标签、未来窗、窗口位置或长历史特征；分类器输入仍只有当前 6 秒窗的既有融合分数。
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
SEED = 0
ALPHA = 0.02
VAR_ALPHA = 0.02
VAR_FLOOR = 0.05 ** 2


def metrics(score, pred, y, g, mask):
    m = mask & np.isfinite(score)
    per = [balanced_accuracy_score(y[g == s], pred[g == s]) for s in np.unique(g)
           if np.sum(m & (g == s)) and np.unique(y[g == s] & m[g == s]).size == 2]
    # above expression is only a guard; compute on evaluation rows explicitly
    per = []
    for s in np.unique(g[m]):
        q = m & (g == s)
        if np.unique(y[q]).size == 2:
            per.append(balanced_accuracy_score(y[q], pred[q]))
    return {'subject_macro_bACC': float(np.mean(per)),
            'window_micro_bACC': float(balanced_accuracy_score(y[m], pred[m])),
            'AUC': float(roc_auc_score(y[m], score[m])), 'n': int(m.sum())}


def fit_adapt(train_score, train_y, test_score, test_groups, alpha=ALPHA):
    """Predict-before-update, with two label-anchored modes learned on training fold."""
    mu = np.array([np.mean(train_score[train_y == c]) for c in (0, 1)], float)
    var = np.array([np.var(train_score[train_y == c]) for c in (0, 1)], float)
    pooled = np.var(train_score) + VAR_FLOOR
    var = np.maximum(var, max(VAR_FLOOR, pooled * .05))
    out = np.full(len(test_score), np.nan)
    mode = np.full(len(test_score), -1, int)
    for s in np.unique(test_groups):
        ii = np.where(test_groups == s)[0]
        m = mu.copy(); v = var.copy()
        for i in ii:
            x = test_score[i]
            ll = -0.5 * (np.log(v) + (x - m) ** 2 / v)
            k = int(np.argmax(ll))
            out[i] = expit(ll[1] - ll[0])
            mode[i] = k
            # current observation enters state only after its prediction
            m[k] = (1 - alpha) * m[k] + alpha * x
            v[k] = (1 - VAR_ALPHA) * v[k] + VAR_ALPHA * max((x - m[k]) ** 2, VAR_FLOOR)
    return out, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    tick = time.perf_counter()
    d = np.load(OUT / 'merge_best_window6_oof.npz')
    y, g, fold = d['y'].astype(int), d['groups'].astype(str), d['fold_id'].astype(int)
    raw = d['combo__C4_bb_resnet_eq'].astype(float)
    keep = np.ones(len(y), bool)
    if args.smoke:
        keep = np.zeros(len(y), bool)
        for s in np.unique(g)[:8]:
            keep |= g == s
        print(f'[smoke] subjects={np.unique(g[keep]).size} rows={keep.sum()}', flush=True)
    score = np.full(len(y), np.nan); modes = np.full(len(y), -1, int)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(raw, y, g))
    fold_metrics = {}
    for f, (tr, te) in enumerate(splits, 1):
        te = te[keep[te]]
        if not len(te): continue
        p, z = fit_adapt(raw[tr], y[tr], raw[te], g[te])
        score[te], modes[te] = p, z
        pred = p >= .5
        fold_metrics[str(f)] = metrics(p, pred, y[te], g[te], np.ones(len(te), bool))
        print(f'[progress] fold {f}/5 test={len(te)} AUC={fold_metrics[str(f)]["AUC"]:.4f} '
              f'micro={fold_metrics[str(f)]["window_micro_bACC"]:.4f} elapsed={time.perf_counter()-tick:.1f}s', flush=True)
    mask = keep & np.isfinite(score)
    # Fusion artifact is a logit-like score; its fixed 0.5 probability threshold is score >= 0.
    base_pred = raw >= 0.
    adapt_pred = score >= .5
    result = {'meta': {'protocol': 'train-fold label-anchored two-mode Gaussian prototypes; test predict-before-update EMA',
                       'alpha_mean': ALPHA, 'alpha_var': VAR_ALPHA, 'var_floor': VAR_FLOOR,
                       'features': 'current existing 6-second fusion score only', 'seed': SEED,
                       'smoke': args.smoke},
              'baseline_fixed05': metrics(raw, base_pred, y, g, mask),
              'adapt_fixed05': metrics(score, adapt_pred, y, g, mask),
              'folds': fold_metrics, 'seconds': time.perf_counter()-tick}
    # descriptive transition/steady split, using labels only for reporting
    transition = np.zeros(len(y), bool)
    for s in np.unique(g):
        ii = np.where(g == s)[0]
        transition[ii] = np.r_[True, y[ii][1:] != y[ii][:-1]]
    result['adapt_fixed05']['switching'] = metrics(score, adapt_pred, y, g, mask & transition)
    result['adapt_fixed05']['steady'] = metrics(score, adapt_pred, y, g, mask & ~transition)
    np.savez_compressed(OUT / 'causal_online_domain_adapt_oof.npz', y=y, groups=g, raw=raw,
                        score=score, pred=adapt_pred.astype(np.uint8), mode=modes, eval_mask=mask)
    (OUT / 'causal_online_domain_adapt_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[result] baseline', result['baseline_fixed05'], flush=True)
    print('[result] adapt', result['adapt_fixed05'], flush=True)
    print(f'[done] seconds={result["seconds"]:.1f}', flush=True)

if __name__ == '__main__':
    main()
