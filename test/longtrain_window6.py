"""长训练、短推理实验：训练阶段使用训练受试者的全部6秒窗，推理仅使用当前6秒窗特征。

协议：每个外层折先在训练受试者上做带噪特征重构预训练，再用训练标签微调分类器；
测试受试者完全隔离。特征均由当前6秒EEG窗计算，不使用时间位置、历史窗或交替先验。
"""
import copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
torch.set_num_threads(2)
SEED = 0

class Encoder(nn.Module):
    def __init__(self, d):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, 384), nn.GELU(), nn.Dropout(.1),
                                                       nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 32))
    def forward(self, x): return self.net(x)

class AutoClassifier(nn.Module):
    def __init__(self, d):
        super().__init__(); self.enc = Encoder(d); self.head = nn.Linear(32, 1)
    def forward(self, x): return self.head(self.enc(x)).squeeze(1)

class AutoEncoder(nn.Module):
    def __init__(self, d):
        super().__init__(); self.enc = Encoder(d); self.dec = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, d))
    def forward(self, x): return self.dec(self.enc(x))

def predict(net, X, device):
    net.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(X), 512): out.append(torch.sigmoid(net(torch.from_numpy(X[i:i+512]).to(device))).cpu().numpy())
    return np.concatenate(out)

def score(y, p): return {'AUC': float(roc_auc_score(y,p)), 'bACC': float(balanced_accuracy_score(y,p >= .5))}

def train_one(Xtr, ytr, gtr, Xte, yte, mode, fold, device):
    subjects = np.unique(gtr)
    inner = subjects[:max(1, len(subjects)//5)]
    valmask = np.isin(gtr, inner); trainmask = ~valmask
    A, V = Xtr[trainmask], Xtr[valmask]
    yt, yv = ytr[trainmask], ytr[valmask]
    d = Xtr.shape[1]; net = AutoClassifier(d).to(device)
    if mode == 'pretrained':
        ae = AutoEncoder(d).to(device); opt = torch.optim.AdamW(ae.parameters(), lr=2e-3, weight_decay=1e-4)
        rng = np.random.default_rng(SEED + fold)
        Xt = torch.from_numpy(A).to(device)
        for ep in range(1, 21):
            ae.train(); order = rng.permutation(len(A)); losses=[]
            for st in range(0, len(A), 256):
                ix = order[st:st+256]; clean = Xt[ix]; noisy = clean + .08*torch.randn_like(clean)
                opt.zero_grad(); loss = nn.functional.smooth_l1_loss(ae(noisy), clean); loss.backward(); opt.step(); losses.append(float(loss))
            if ep == 1 or ep % 5 == 0: print(f'[fold {fold}/5] pretrain epoch {ep}/20 loss={np.mean(losses):.5f}', flush=True)
        net.enc.load_state_dict(ae.enc.state_dict()); del ae
    elif mode != 'supervised': raise ValueError(mode)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    posw = torch.tensor(float((yt == 0).sum()/max((yt == 1).sum(),1)), device=device)
    lossfn = nn.BCEWithLogitsLoss(pos_weight=posw); rng = np.random.default_rng(SEED + 100 + fold)
    Xt, Yt = torch.from_numpy(A).to(device), torch.from_numpy(yt.astype('float32')).to(device)
    best = float('inf'); state=None; stale=0; best_ep=0
    for ep in range(1, 41):
        net.train(); order=rng.permutation(len(A)); losses=[]
        for st in range(0, len(A), 256):
            ix=order[st:st+256]; opt.zero_grad(); loss=lossfn(net(Xt[ix]),Yt[ix]); loss.backward(); opt.step(); losses.append(float(loss))
        p=np.clip(predict(net,V,device),1e-6,1-1e-6); val=float(-np.mean(yv*np.log(p)+(1-yv)*np.log(1-p)))
        if val < best-1e-5: best=val; state=copy.deepcopy(net.state_dict()); stale=0; best_ep=ep
        else: stale += 1
        if ep == 1 or ep % 10 == 0: print(f'[fold {fold}/5] {mode} epoch {ep}/40 val_loss={val:.5f}', flush=True)
        if stale >= 7: break
    net.load_state_dict(state); p=predict(net,Xte,device)
    return p, {'best_epoch':best_ep,'pretrain_epochs':20 if mode=='pretrained' else 0}

def main():
    tick=time.perf_counter(); torch.manual_seed(SEED); np.random.seed(SEED)
    d=np.load(OUT/'highdim_deep_features.npz'); X=d['X'].astype('float32'); y=d['y'].astype(int); g=d['groups']
    keep=X.std(0)>1e-8; X=X[:,keep]; means=np.zeros(X.shape[1]); scales=np.ones(X.shape[1])
    device='cuda' if torch.cuda.is_available() else 'cpu'; print(f'[data] X={X.shape} subjects={len(np.unique(g))} device={device}',flush=True)
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(X,y,g)); oofs={m:np.full(len(y),np.nan) for m in ('supervised','pretrained')}; rows={}
    for fold,(tr,te) in enumerate(splits,1):
        mu=X[tr].mean(0); sd=np.maximum(X[tr].std(0),1e-6); A=((X-mu)/sd).astype('float32')
        for mode in oofs:
            p, info=train_one(A[tr],y[tr],g[tr],A[te],y[te],mode,fold,device); oofs[mode][te]=p
            r=score(y[te],p); rows[f'{mode}_fold{fold}']={**r,**info}; print(f'[result] {mode} fold={fold}/5 AUC={r["AUC"]:.5f} bACC={r["bACC"]:.5f}',flush=True)
    results={'protocol':{'input':'current 6-second EEG-derived feature vector only','training':'all windows from training subjects; supervised or 20-epoch denoising reconstruction pretraining followed by supervised fine-tuning','cv':'StratifiedGroupKFold(5), subject-disjoint','test_history':'none','feature_count':int(X.shape[1]),'pretraining':'training-fold only, no test windows'},'models':{}}
    for mode,p in oofs.items():
        results['models'][mode]={**score(y,p),'fold_mean_AUC':float(np.mean([rows[f'{mode}_fold{i}']['AUC'] for i in range(1,6)])),'folds':[rows[f'{mode}_fold{i}'] for i in range(1,6)]}
        print(f'[final] {mode} AUC={results["models"][mode]["AUC"]:.5f} bACC={results["models"][mode]["bACC"]:.5f}',flush=True)
    results['runtime_seconds']=time.perf_counter()-tick
    np.savez_compressed(OUT/'longtrain_window6_oof.npz',y=y,groups=g,**oofs)
    (OUT/'longtrain_window6_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'[done] elapsed={results["runtime_seconds"]:.1f}s',flush=True)
if __name__=='__main__': main()
