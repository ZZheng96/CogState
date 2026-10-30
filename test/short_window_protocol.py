"""比较末尾 3/4/5/6 秒输入；训练使用密集 1 秒步长窗，测试使用原标准窗结束时刻。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from broadband_shape_window6 import compute_features

OUT = Path(__file__).resolve().parent
CACHE = OUT / 'dense_window_protocol_cache.npz'
RAW = OUT / 'dense_window_protocol_raw6.npy'
RES = OUT / 'short_window_protocol_results.json'
OOF = OUT / 'short_window_protocol_oof.npz'
FS = 250.; LENGTHS = (3, 4, 5, 6); SEED = 0
T0 = time.perf_counter()

def log(x): print(f'[{time.perf_counter()-T0:7.1f}s] {x}', flush=True)

def features(raw, seconds):
    L = int(seconds * FS); raw = raw[:, -L:].astype(np.float64)
    nperseg = min(500, L); noverlap = nperseg // 2
    f, P = welch(raw, fs=FS, nperseg=nperseg, noverlap=noverlap, axis=1, detrend='constant')
    fam = compute_features(raw, f, P)
    X = np.hstack([fam[k][0] for k in fam])
    bands = ((1,4),(4,8),(8,13),(13,20),(20,30),(30,45),(52,100))
    df = f[1]-f[0]
    X = np.hstack([X, np.stack([np.log(P[:, (f>=a)&(f<b)].sum(1)*df+1e-30) for a,b in bands], 1)])
    X[~np.isfinite(X)] = 0.
    return X.astype(np.float32)

def youden(y,p):
    fpr,tpr,thr = roc_curve(y,p); return float(thr[np.argmax(tpr-fpr)])

def fit_predict(kind, Xtr, ytr, gtr, Xte):
    if kind == 'lr':
        # C selected once by group OOF within the training subjects.
        best_c, best_a = .01, -1
        for C in (.001,.01,.1,1.):
            q = np.zeros(len(ytr))
            for a,b in GroupKFold(3).split(Xtr,ytr,gtr):
                sc=StandardScaler().fit(Xtr[a]); m=LogisticRegression(C=C,max_iter=1500,class_weight='balanced').fit(sc.transform(Xtr[a]),ytr[a]); q[b]=m.decision_function(sc.transform(Xtr[b]))
            a=roc_auc_score(ytr,q)
            if a>best_a: best_a,best_c=a,C
        sc=StandardScaler().fit(Xtr); m=LogisticRegression(C=best_c,max_iter=1500,class_weight='balanced').fit(sc.transform(Xtr),ytr)
        return m.decision_function(sc.transform(Xte)), 'logit', best_c
    scores=[]
    for seed in (0,1,2):
        m=HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,max_features=.5,early_stopping=False,random_state=seed).fit(Xtr,ytr)
        scores.append(m.predict_proba(Xte)[:,1])
    return np.mean(scores,0), 'prob', None

def main():
    d=np.load(CACHE); raw=np.load(RAW,mmap_mode='r'); y=d['label'].astype(int); g=d['subject']; canon=d['canon'].astype(bool)
    ref=np.load(OUT/'highdim_deep_features.npz'); cidx=np.where(canon)[0]; assert np.array_equal(y[cidx],ref['y']) and np.array_equal(g[cidx],ref['groups'])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(np.zeros(len(cidx)),y[cidx],g[cidx]))
    result={'meta':{'lengths_seconds':list(LENGTHS),'train':'all dense 1-second-step windows','test':'canonical windows, same end times','input':'tail of cached 6-second raw window','outer_cv':'StratifiedGroupKFold(5, shuffle=True, random_state=0)'},'conditions':{}}
    all_oof={}
    for sec in LENGTHS:
        log(f'extracting {sec}s features')
        X=features(raw,sec); log(f'{sec}s features shape={X.shape}')
        for kind in ('lr','hgb'):
            p=np.full(len(cidx),np.nan); hard=np.zeros(len(cidx),bool); per=[]
            for fold,(_,te0) in enumerate(folds,1):
                test_subjects=set(g[cidx[te0]]); tr=np.where(~np.isin(g,list(test_subjects)))[0]; te=te0
                pred,space,C=fit_predict(kind,X[tr],y[tr],g[tr],X[cidx[te]])
                # For LR use inner training OOF threshold; HGB use 3-fold OOF threshold.
                q=np.zeros(len(tr))
                for a,b in GroupKFold(3).split(X[tr],y[tr],g[tr]):
                    z,_,_=fit_predict(kind,X[tr][a],y[tr][a],g[tr][a],X[tr][b]); q[b]=z
                thr=youden(y[tr],q); p[te]=pred; hard[te]=pred>=thr
                per.append({'fold':fold,'auc':float(roc_auc_score(y[cidx[te]],pred)),'bacc':float(balanced_accuracy_score(y[cidx[te]],hard[te])),'threshold':thr,'n_train':len(tr),'n_test':len(te)})
                log(f'{sec}s {kind} fold {fold}/5 AUC={per[-1]["auc"]:.4f} bACC={per[-1]["bacc"]:.4f}')
            key=f'{kind}_{sec}s'; all_oof[key]=p
            result['conditions'][key]={'auc':float(roc_auc_score(y[cidx],p)),'bacc':float(balanced_accuracy_score(y[cidx],hard)),'per_fold':per}
            log(f'== {key}: AUC={result["conditions"][key]["auc"]:.4f} bACC={result["conditions"][key]["bacc"]:.4f}')
    result['seconds']=time.perf_counter()-T0
    np.savez_compressed(OOF,y=y[cidx],groups=g[cidx],**all_oof)
    RES.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); log(f'done {result["seconds"]:.0f}s')
if __name__=='__main__': main()
