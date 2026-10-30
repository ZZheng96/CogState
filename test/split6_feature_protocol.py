"""完整 6 秒窗内部细分：比较子窗特征拼接与统计融合。"""
import json, time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from broadband_shape_window6 import compute_features

OUT=Path(__file__).resolve().parent; FS=250.; T0=time.perf_counter()
CONFIGS={'base6':[(0,6)],'2x3s':[(0,3),(3,6)],'3x2s':[(0,2),(2,4),(4,6)],'6x1s':[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]}
BANDS=((1,4),(4,8),(8,13),(13,20),(20,30),(30,45),(52,100))

def log(x): print(f'[{time.perf_counter()-T0:7.1f}s] {x}',flush=True)
def one(raw):
    L=raw.shape[1]; f,P=welch(raw,fs=FS,nperseg=min(500,L),noverlap=min(250,L//2),axis=1,detrend='constant')
    fam=compute_features(raw.astype(np.float64),f,P); X=np.hstack([fam[k][0] for k in fam]); df=f[1]-f[0]
    X=np.hstack([X,np.stack([np.log(P[:,(f>=a)&(f<b)].sum(1)*df+1e-30) for a,b in BANDS],1)])
    X[~np.isfinite(X)]=0.; return X.astype(np.float32)
def extract(raw,parts):
    out=[]
    for a,b in parts:
        s=int(a*FS); e=int(b*FS); out.append(one(raw[:,s:e]))
    cat=np.hstack(out); mean=np.mean(np.stack(out,1),1); std=np.std(np.stack(out,1),1)
    return {'concat':cat,'stats':np.hstack([mean,std])}
def main():
    d=np.load(OUT/'dense_window_protocol_cache.npz'); raw=np.load(OUT/'dense_window_protocol_raw6.npy',mmap_mode='r'); y=d['label'].astype(int); g=d['subject']; c=np.where(d['canon'])[0]; yc=y[c]; gc=g[c]
    ref=np.load(OUT/'highdim_deep_features.npz'); assert np.array_equal(yc,ref['y']) and np.array_equal(gc,ref['groups'])
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(np.zeros(len(c)),yc,gc)); res={'meta':{'input':'full 6s window, split internally','configs':{k:v for k,v in CONFIGS.items()},'train':'all dense 1s-step windows','test':'canonical windows, subject-isolated'},'conditions':{}}
    for name,parts in CONFIGS.items():
        log(f'extract {name}'); feat=extract(raw,parts)
        for mode,X in feat.items():
            p=np.full(len(c),np.nan); hard=np.zeros(len(c),bool); per=[]
            for fold,(_,te) in enumerate(folds,1):
                subs=set(gc[te]); tr=np.where(~np.isin(g,list(subs)))[0]
                m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,max_features=.5,early_stopping=False,random_state=fold).fit(X[tr],y[tr])
                q=m.predict_proba(X[c[te]])[:,1]; p[te]=q
                # 固定 .5 仅作可比报告，不调测试阈值
                hard[te]=q>=.5; per.append({'fold':fold,'auc':float(roc_auc_score(yc[te],q)),'bacc':float(balanced_accuracy_score(yc[te],hard[te])),'n_train':len(tr),'n_test':len(te)})
                log(f'{name} {mode} fold {fold}/5 AUC={per[-1]["auc"]:.4f} bACC={per[-1]["bacc"]:.4f}')
            key=f'{name}_{mode}'; res['conditions'][key]={'auc':float(roc_auc_score(yc,p)),'bacc':float(balanced_accuracy_score(yc,hard)),'dim':int(X.shape[1]),'per_fold':per}; log(f'== {key} AUC={res["conditions"][key]["auc"]:.4f} bACC={res["conditions"][key]["bacc"]:.4f}')
    res['seconds']=time.perf_counter()-T0; (OUT/'split6_feature_protocol_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8'); log(f'done {res["seconds"]:.0f}s')
if __name__=='__main__': main()
