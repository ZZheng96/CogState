"""时序积分的后处理细化：K 细网格、均值 vs 中位数。直接复用已保存的 OOF。"""
# --- test layout bootstrap (generated 2026-11-02): locate test/ root, shared modules and cache ---
import sys as _sys
from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent
_TEST = _HERE
while _TEST.name != "test" and _TEST.parent != _TEST:
    _TEST = _TEST.parent
CACHE_DIR = _TEST / "cache"
if str(_TEST) not in _sys.path:
    _sys.path.append(str(_TEST))
for _d in sorted(_TEST.iterdir()):
    if _d.is_dir() and _d.name not in {"cache", "results", "pretrained_cache", "reports", "experiments", "_orphans", "__pycache__"}:
        if str(_d) not in _sys.path:
            _sys.path.append(str(_d))
# --- end bootstrap ---

import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
d = np.load(CACHE_DIR / 'temporal_integration_oof.npz')
y, g, ens = d['y'], d['groups'], d['ens']
n = len(y)


def smooth(s, k, how='mean'):
    out = np.empty_like(s)
    for sub in np.unique(g):
        idx = np.where(g == sub)[0]
        v = s[idx]
        for i in range(len(v)):
            seg = v[max(0, i-k+1):i+1]
            out[idx[i]] = np.mean(seg) if how == 'mean' else np.median(seg)
    return out


res = {}
for how in ('mean', 'median'):
    for k in range(1, 13):
        s = smooth(ens, k, how)
        res[f'{how}_K{k}'] = {'AUC': float(roc_auc_score(y, s)),
                              'bACC': float(balanced_accuracy_score(y, s >= .5)),
                              'seconds': k*6}
    best = max((v['AUC'], k) for k, v in res.items() if k.startswith(how))
    print(f'[{how}] 最佳 K={best[1]} AUC={best[0]:.4f}', flush=True)
    for k in range(1, 13):
        r = res[f'{how}_K{k}']
        print(f'   K={k:2d} ({k*6:3d}s) AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}', flush=True)

best_key = max(res, key=lambda k: res[k]['AUC'])
print(f'\n[overall best] {best_key} AUC={res[best_key]["AUC"]:.4f} bACC={res[best_key]["bACC"]:.4f}', flush=True)

splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(y[:, None], y, g))
how, k = best_key.split('_'); k = int(k[1:])
sb = smooth(ens, k, how)
res['_per_fold'] = {'raw': [float(roc_auc_score(y[te], ens[te])) for _, te in splits],
                    'smoothed': [float(roc_auc_score(y[te], sb[te])) for _, te in splits]}
print(f'[分折 raw] {np.round(res["_per_fold"]["raw"], 3).tolist()}', flush=True)
print(f'[分折 smoothed] {np.round(res["_per_fold"]["smoothed"], 3).tolist()}', flush=True)
res['_best'] = best_key
(OUT/'temporal_refine_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
print('[done]', flush=True)
