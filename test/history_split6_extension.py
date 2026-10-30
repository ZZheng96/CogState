"""将 6 秒内部 6x1s 的细粒度统计特征接入历史记忆模型。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parent; FS=250.; K0=120; Q=10**6; FLOOR=.1; SCALES=(50,100,200,400,800); T0=time.perf_counter()
sys.path.insert(0,str(OUT)); from b_calib_variants import subject_quantile; from broadband_shape_window6 import compute_features; from memory_stream_window6 import subject_median_threshold

def log(x): print(f'[{time.perf_counter()-T0:7.1f}s] {x}',flush=True)
def one(raw):
    L=raw.shape[1]; f,P=welch(raw,fs=FS,nperseg=min(500,L),noverlap=min(250,L//2),axis=1,detrend='constant'); fam=compute_features(raw.astype(np.float64),f,P); X=np.hstack([fam[k][0] for k in fam]); df=f[1]-f[0]; bands=((1,4),(4,8),(8,13),(13,20),(20,30),(30,45),(52,100)); X=np.hstack([X,np.stack([np.log(P[:,(f>=a)&(f<b)].sum(1)*df+1e-30) for a,b in bands],1)]); X[~np.isfinite(X)]=0.; return X.astype(np.float32)
def split_features(raw):
    parts=[]
    for j in range(6):
        parts.append(one(raw[:,j*250:(j+1)*250]))
        log(f'feature part {j+1}/6')
    A=np.stack(parts,1); return np.hstack([A.mean(1),A.std(1)]).astype(np.float64)
def exact(z,g,k):
    zi=np.round(z*Q).astype(np.int64); mean=np.zeros_like(z); std=np.zeros_like(z)
    for s in np.unique(g):
        ii=np.where(g==s)[0]; v=zi[ii]; cs=np.vstack([np.zeros((1,v.shape[1]),np.int64),np.cumsum(v,0)]); cs2=np.vstack([np.zeros((1,v.shape[1]),np.int64),np.cumsum(v*v,0)]); t=np.arange(len(ii)); lo=np.maximum(t-k,0); n=(t-lo)[:,None]; S1=cs[t]-cs[lo]; S2=cs2[t]-cs2[lo]; var=n*S2-S1*S1; nn=np.maximum(n,1).astype(float); mean[ii]=np.where(n>0,S1/nn/Q,0); std[ii]=np.where(n>1,np.sqrt(np.maximum(var,0).astype(float))/nn/Q,0)
    return mean,std
def smooth(v,g,k):
    o=np.full_like(v,np.nan)
    for s in np.unique(g):
        ii=np.where(g==s)[0]; x=v[ii]
        for t in range(len(ii)):
            w=x[max(0,t-k+1):t+1]; w=w[np.isfinite(w)]
            if len(w): o[ii[t]]=w.mean()
    return o
def metrics(v,y,g,mask):
    out={}
    for k in (1,5,10,20):
        p=smooth(v,g,k); m=mask&np.isfinite(p); med=subject_median_threshold(p,g,mask); out[f's{k}']={'AUC':float(roc_auc_score(y[m],p[m])),'bACC':float(balanced_accuracy_score(y[m],p[m]>=.5)),'bACC_med':float(balanced_accuracy_score(y[m],p[m]>=med[m]))}
    return out
def model(): return HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,max_depth=3,l2_regularization=1.,random_state=0)
def main():
    d=np.load(OUT/'dense_window_protocol_cache.npz'); raw=np.load(OUT/'dense_window_protocol_raw6.npy',mmap_mode='r'); y=d['label'].astype(int); g=d['subject']; c=np.where(d['canon'])[0]; yc=y[c]; gc=g[c]; ref=np.load(OUT/'highdim_deep_features.npz'); assert np.array_equal(yc,ref['y']) and np.array_equal(gc,ref['groups']); mask=np.arange(len(c))>=K0
    log('extracting dense 6x1s stats features'); F0=split_features(raw); Z=subject_quantile(F0,g,K0); Zc=Z[c]
    # Retain only canonical sequence for deployment-style historical evaluation.
    pos=np.empty(len(y),int)
    for s in np.unique(g): pos[g==s]=np.arange(np.sum(g==s))
    mask=(pos[c]>=K0); splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(Zc,yc,gc)); P={}; res={'meta':{'features':'6x1s subwindow mean+std','dim':int(Z.shape[1]),'warmup':K0,'scales':list(SCALES),'exact_integer_rolling':True},'variants':{}}
    with threadpool_limits(limits=4):
        for k in SCALES:
            mu,sd=exact(Zc,gc,k); F=np.round((Zc-mu)/np.maximum(sd,FLOOR),6); p=np.full(len(yc),np.nan)
            for fold,(tr,te) in enumerate(splits,1):
                tr=tr[mask[tr]]; te=te[mask[te]]; p[te]=model().fit(F[tr],yc[tr]).predict_proba(F[te])[:,1]; log(f'd{k} fold {fold}/5 ntr={len(tr)}')
            P[f'd{k}']=p
    P['D']=np.nanmean([P[f'd{k}'] for k in SCALES],0); res['variants']['D']=metrics(P['D'],yc,gc,mask); 
    for k,v in res['variants']['D'].items(): log(f'D {k} AUC={v["AUC"]:.4f} bACC={v["bACC"]:.4f} med={v["bACC_med"]:.4f}')
    # Compare and fuse with old history OOF and strict single-window score where aligned.
    old=np.load(OUT/'history_leakfree_oof.npz'); assert np.array_equal(old['y'],yc) and np.array_equal(old['groups'],gc)
    for key in ('D_hgb','single6_memz'):
        if key in old.files:
            P[key]=old[key]
    if 'D_hgb' in P:
        for a,b in (('newD+oldD',(P['D']+P['D_hgb'])/2),('newD+single6',(P['D']+P.get('single6_memz',np.full(len(yc),np.nan)))/2)):
            P[a]=b; res['variants'][a]=metrics(b,yc,gc,mask)
            log(f'{a} '+str(res['variants'][a]))
    res['seconds']=time.perf_counter()-T0; np.savez_compressed(OUT/'history_split6_extension_oof.npz',y=yc,groups=gc,eval_mask=mask,**P); (OUT/'history_split6_extension_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8'); log(f'done {res["seconds"]:.0f}s')
if __name__=='__main__': main()
