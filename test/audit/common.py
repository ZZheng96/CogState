"""Shared loading for test/audit experiments (copies the depositioned_anchor_direction.py loading / fold pattern)."""
import json, sys, time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

AUDIT = Path(__file__).resolve().parent
TEST = AUDIT.parent
sys.path.insert(0, str(TEST))
from depositioned_history_delta import arrange, boot, load_db  # noqa: E402,F401
from depositioned_symmetric_contrast import impute_scale  # noqa: E402,F401
from causal_kernel_state_mapping import metrics  # noqa: E402,F401
from switch_dual_memory import Tee  # noqa: E402,F401
from depositioned_within_subject_adapt import GROUPS, STREAM_SEEDS  # noqa: E402,F401

T0 = time.perf_counter()


def el():
    return f'elapsed={time.perf_counter() - T0:.1f}s'


def load_all():
    """-> dict(Fc [n,858], Fn [n,31], y, g, fold, subs, data, grp_of, fold_of, A_ (npz))."""
    man, subs, data = load_db()
    Fc = np.vstack([data[s][0] for s in subs]); y = np.concatenate([data[s][1] for s in subs])
    g = np.concatenate([np.full(len(data[s][1]), s) for s in subs])
    A_ = np.load(TEST / 'depositioned_aperiodic_emg_oof.npz')
    assert np.array_equal(A_['y'], y) and np.array_equal(A_['groups'], g)
    Fn = A_['new_features']; assert Fc.shape[1] == 858 and Fn.shape == (len(y), 31)
    diag = json.loads((TEST / 'depositioned_subject_diagnosis.json').read_text(encoding='utf-8'))
    grp_of = {s: G for G in GROUPS for s in diag['diagnosis']['groups'][G]['subjects']}
    assert all(s in grp_of for s in subs)
    Xa, ya, ga, _ = arrange(subs, data, 0, 'independent'); fold_of = {}
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xa, ya, ga)):
        for s in np.unique(ga[te]): fold_of[s] = k
    fold = np.array([fold_of[s] for s in g])
    assert np.array_equal(fold, A_['fold']), 'fold != npz fold'
    return dict(man=man, subs=subs, data=data, Fc=Fc, Fn=Fn, y=y, g=g, fold=fold, grp_of=grp_of,
                fold_of=fold_of, A_=A_)


def zz_fold(D, tr):
    """889-dim ZZ for ALL rows, imputation + scaling fitted on rows tr."""
    tfo, tfn = impute_scale(D['Fc'], tr), impute_scale(D['Fn'], tr)
    return np.hstack([tfo(D['Fc']), tfn(D['Fn'])])


def open_log(folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout, folder / (stem + '.log')); sys.stdout = tee
    return tee


def save(folder, stem, obj):
    (folder / (stem + '.json')).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=float),
                                           encoding='utf-8')
