"""Reproducible subject-disjoint MLP and raw-waveform CNN comparisons."""
import os, sys, json, time, platform
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
from pathlib import Path
import numpy as np, torch, mne
from torch import nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cogstate import load_trials, bad_intervals, valid_window, causal_features
from kalman_deep import kalman_features, kalman_filter_window, Q_DEFAULT, R_DEFAULT, P0_DEFAULT
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); OUT=ROOT/'test'; start=time.time()

def extract():
    F, K, RAW, RAW_K, y, g, meta=[],[],[],[],[],[],[]; paths=sorted((ROOT/'EEG_dataset').glob('*/*_events.csv'))
    for ci,csv in enumerate(paths,1):
        bdf=next(csv.parent.glob('*.bdf')); sid=csv.parent.name
        raw=mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR'); fs=float(raw.info['sfreq']); dur=raw.n_times/fs; bad=bad_intervals(raw)
        for tr in load_trials(str(csv),str(bdf)):
            t=tr.start+6.0
            while t<=tr.end+1e-6:
                a,z=t-6.0,t
                if valid_window(a,z,bad,dur):
                    s,e=int(round(a*fs)),int(round(z*fs)); x=raw.get_data(start=s,stop=e)[0]
                    if len(x)==e-s and np.isfinite(x).all():
                        f=causal_features(x,fs); k=kalman_features(x,fs)
                        if f is not None and k is not None:
                            # 6-s causal block-average to 250 points; no future outside this window.
                            rw=x.astype(np.float32).reshape(250,6).mean(1)
                            kw=kalman_filter_window(x).reshape(250,6).mean(1)
                            F.append(f); K.append(k); RAW.append(rw); RAW_K.append(kw); y.append(tr.label); g.append(sid)
                            meta.append({'subject':sid,'trial_id':tr.trial_id,'time':t,'label':tr.label})
                t+=6.0
        raw.close(); print(f'[extract {ci}/{len(paths)}] subject={sid} samples={len(y)} elapsed={time.time()-start:.1f}s',flush=True)
    return *(np.asarray(v,np.float32) for v in (F,K,RAW,RAW_K)),np.asarray(y),np.asarray(g),meta

def mlp(d): return nn.Sequential(nn.Linear(d,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
class RawCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv1d(1,8,9,padding=4),nn.ReLU(),nn.AvgPool1d(4),nn.Conv1d(8,16,9,padding=4),nn.ReLU(),nn.AvgPool1d(4),nn.Flatten(),nn.Linear(16*15,16),nn.ReLU(),nn.Dropout(.15),nn.Linear(16,1))
    def forward(self,x): return self.net(x)

def fit_predict(X,y,tr,va,kind,seed):
    torch.manual_seed(seed); scaler=StandardScaler().fit(X[tr]) if kind.startswith('mlp') else None
    a=scaler.transform(X[tr]).astype('float32') if scaler else X[tr].astype('float32')
    b=scaler.transform(X[va]).astype('float32') if scaler else X[va].astype('float32')
    net=mlp(X.shape[1]) if scaler else RawCNN(); net.to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-3)
    pos=((y[tr]==0).sum()/max((y[tr]==1).sum(),1)); loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos],device=DEVICE)); hist=[]
    ta=torch.from_numpy(a).to(DEVICE); yt=torch.from_numpy(y[tr].astype('float32')).to(DEVICE)
    for ep in range(24):
        net.train(); opt.zero_grad(); inp=ta if scaler else ta[:,None,:]; l=loss(net(inp).squeeze(1),yt); l.backward(); opt.step(); hist.append(float(l.detach().cpu()))
        if (ep+1)%6==0: print(f'  {kind} epoch {ep+1}/24 loss={hist[-1]:.4f} device={DEVICE}',flush=True)
    net.eval(); tb=torch.from_numpy(b).to(DEVICE)
    with torch.no_grad(): p=torch.sigmoid(net(tb if scaler else tb[:,None,:]).squeeze(1)).cpu().numpy()
    return p,net,scaler,hist

def score(y,p): return {'bACC':float(balanced_accuracy_score(y,p>=.5)),'AUC':float(roc_auc_score(y,p))}
def subject_scores(y,p,g):
    out={}
    for sid in sorted(set(g.tolist())):
        m=g==sid; labels=y[m]
        out[str(sid)]={'n':int(m.sum()),'bACC':float(balanced_accuracy_score(labels,p[m]>=.5)),
                       'AUC':float(roc_auc_score(labels,p[m])) if len(np.unique(labels))==2 else None}
    return out
F,K,RAW,RAW_K,y,g,meta=extract(); np.savez(OUT/'kalman_deep_features.npz',X=F,K=K,raw=RAW,raw_kalman=RAW_K,y=y,groups=g)
cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED); results={'protocol':{'subjects':sorted(set(g.tolist())),'samples':len(y),'window_seconds':6,'samples_per_window':1500,'raw_embedding_dim':250,'cv':'StratifiedGroupKFold(5), subject-disjoint','seed':SEED,'device':str(DEVICE),'cuda_available':torch.cuda.is_available(),'torch':torch.__version__,'python':platform.python_version(),'kalman':{'use_only_kalman_model':True,'reset_each_window':True,'forward_only':True,'q':Q_DEFAULT,'r':R_DEFAULT,'p0':P0_DEFAULT,'units':'signal units^2/sample','no_kalman':'causal_features(raw)'}},'models':{}}
for name,Z,kind in [('no_kalman',F,'mlp_no_kalman'),('kalman',K,'mlp_kalman'),('raw_cnn',RAW,'cnn_raw'),('raw_kalman_cnn',RAW_K,'cnn_raw_kalman')]:
    oof=np.full(len(y),np.nan); fold_ids=np.full(len(y),-1); folds=[]; curves=[]
    for fold,(tr,va) in enumerate(cv.split(Z,y,g),1):
        p,net,scaler,hist=fit_predict(Z,y,tr,va,kind,SEED+fold); oof[va]=p; fold_ids[va]=fold
        folds.append({'fold':fold,'n':int(len(va)),'subjects':sorted(set(g[va].tolist())),**score(y[va],p)}); curves.append(hist)
        print(f'[{name}] fold {fold}/5 AUC={folds[-1]["AUC"]:.4f} bACC={folds[-1]["bACC"]:.4f}',flush=True)
    results['models'][name]={'input_dim':int(Z.shape[1]),'embedding_dim':16,'parameters':sum(p.numel() for p in net.parameters()),'oof':score(y,oof),'subject_metrics':subject_scores(y,oof,g),'folds':folds,'oof_file':f'kalman_deep_oof_{name}.npz'}
    np.savez(OUT/f'kalman_deep_oof_{name}.npz',prob=oof,label=y,subject=g,fold=fold_ids)
    plt.figure(figsize=(7,4)); plt.plot(np.mean(curves,axis=0)); plt.xlabel('epoch'); plt.ylabel('BCE loss'); plt.title(name); plt.tight_layout(); plt.savefig(OUT/f'kalman_deep_curve_{name}.png',dpi=150); plt.close()
# Save a separately named full-data checkpoint for each input mapping; no mismatched q/r.
for name,Z,kind,use_k in [('no_kalman',F,'mlp_no_kalman',False),('kalman',K,'mlp_kalman',True),('raw_cnn',RAW,'cnn_raw',False),('raw_kalman_cnn',RAW_K,'cnn_raw_kalman',True)]: 
    tr=np.arange(len(y)); p,net,scaler,_=fit_predict(Z,y,tr,tr,kind,SEED+100)
    ck={'state_dict':net.cpu().state_dict(),'use_kalman':use_k,'q':Q_DEFAULT,'r':R_DEFAULT,'p0':P0_DEFAULT,'epochs':24,'device_trained':str(DEVICE),'input_dim':int(Z.shape[1]),'embedding_dim':16,'model':kind,'sfreq':250.0,'window':6.0}
    if scaler is not None: ck.update(mean=scaler.mean_,scale=scaler.scale_)
    torch.save(ck,OUT/f'kalman_deep_final_{name}.pt')
results['elapsed_seconds']=time.time()-start
(OUT/'kalman_deep_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
names=list(results['models']); vals=[results['models'][n]['oof'] for n in names]; xx=np.arange(len(names)); plt.figure(figsize=(8,4)); plt.bar(xx-.18,[v['AUC'] for v in vals],.36,label='AUC'); plt.bar(xx+.18,[v['bACC'] for v in vals],.36,label='bACC'); plt.xticks(xx,names); plt.ylim(.4,1); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'kalman_deep_comparison.png',dpi=150); plt.close()
print('COMPLETE',json.dumps({'device':str(DEVICE),'results':results['models'],'elapsed':results['elapsed_seconds']},ensure_ascii=False),flush=True)
