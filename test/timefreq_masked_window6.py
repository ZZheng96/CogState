"""训练折内时频掩码重建预训练 + 严格 6 秒监督微调。"""
import copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'test'
from timefreq_vision_window6 import load, make_images, DEVICE

class Encoder(nn.Module):
    def __init__(self):
        super().__init__(); self.body=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.GELU(),nn.MaxPool2d(2),
            nn.Conv2d(128,192,3,padding=1),nn.BatchNorm2d(192),nn.GELU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
    def forward(self,x): return self.body(x)

class SSLNet(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=Encoder(); self.decoder=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Linear(128,3*23*43))
    def forward(self,x): return self.decoder(self.encoder(x)).view(-1,3,23,43)

class Classifier(nn.Module):
    def __init__(self,enc): super().__init__(); self.encoder=enc; self.drop=nn.Dropout(.25); self.head=nn.Linear(192,1)
    def forward(self,x): return self.head(self.drop(self.encoder(x))).ravel()

def mask_batch(x,rng):
    z=x.copy(); mask=np.zeros(z.shape,dtype=np.float32)
    for i in range(len(z)):
        if rng.random()<.85:
            w=int(rng.integers(2,7)); s=int(rng.integers(0,z.shape[2]-w+1)); mask[i,:,s:s+w]=1
        if rng.random()<.85:
            w=int(rng.integers(2,6)); s=int(rng.integers(0,z.shape[3]-w+1)); mask[i,:,:,s:s+w]=1
    # Replace masked values with zero; loss is evaluated only on masked pixels.
    return x*(1-mask),mask

def predict(model,X):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),512): out.append(torch.sigmoid(model(torch.from_numpy(X[i:i+512]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)

def report(y,p): return {'AUC':float(roc_auc_score(y,p)),'bACC':float(balanced_accuracy_score(y,p>=.5))}

def main():
    t0=time.perf_counter(); torch.set_float32_matmul_precision('high'); x,y,g=load(); images=make_images(x); splits=StratifiedGroupKFold(5,shuffle=True,random_state=0).split(images,y,g); oof=np.full(len(y),np.nan); folds=[]
    print(f'[data] {images.shape}, device={DEVICE}',flush=True)
    for fold,(tr0,te) in enumerate(splits,1):
        tri,va=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=800+fold).split(tr0,y[tr0],g[tr0])); tr,va=tr0[tri],tr0[va]; rng=np.random.default_rng(900+fold)
        ssl=SSLNet().to(DEVICE); opt=torch.optim.AdamW(ssl.parameters(),lr=2e-3,weight_decay=2e-3); best=float('inf'); state=None; stale=0
        for ep in range(1,16):
            ssl.train(); order=rng.permutation(tr); losses=[]
            for j in range(0,len(order),256):
                ids=order[j:j+256]; masked,m=mask_batch(images[ids],rng); pred=ssl(torch.from_numpy(masked).to(DEVICE)); target=torch.from_numpy(images[ids]).to(DEVICE); mt=torch.from_numpy(m).to(DEVICE); loss=((pred-target).square()*mt).sum()/mt.sum().clamp_min(1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            ssl.eval(); vloss=0.; count=0
            with torch.no_grad():
                for j in range(0,len(va),512):
                    ids=va[j:j+512]; masked,m=mask_batch(images[ids],rng); pred=ssl(torch.from_numpy(masked).to(DEVICE)); target=torch.from_numpy(images[ids]).to(DEVICE); mt=torch.from_numpy(m).to(DEVICE); n=float(mt.sum()); vloss+=float(((pred-target).square()*mt).sum().cpu()); count+=n
            vloss/=max(count,1)
            if ep==1 or ep%5==0: print(f'[fold {fold}/5] ssl epoch {ep}/15 train={np.mean(losses):.5f} val={vloss:.5f}',flush=True)
            if vloss<best-1e-5: best,state,stale=vloss,copy.deepcopy(ssl.encoder.state_dict()),0
            else: stale+=1
            if stale>=4: break
        enc=Encoder().to(DEVICE); enc.load_state_dict(state); model=Classifier(enc).to(DEVICE); pos=float((y[tr]==0).sum()/max((y[tr]==1).sum(),1)); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device=DEVICE)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=2e-3); best=float('inf'); state2=None; stale=0
        for ep in range(1,26):
            model.train(); order=rng.permutation(tr); losses=[]
            for j in range(0,len(order),256):
                ids=order[j:j+256]; xb=torch.from_numpy(images[ids]).to(DEVICE); yb=torch.from_numpy(y[ids]).float().to(DEVICE); opt.zero_grad(set_to_none=True); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            p=predict(model,images[va]); vl=float(-np.mean(y[va]*np.log(np.clip(p,1e-7,1))+(1-y[va])*np.log(np.clip(1-p,1e-7,1))))
            if ep==1 or ep%5==0: print(f'[fold {fold}/5] ft epoch {ep}/25 train={np.mean(losses):.5f} val={vl:.5f}',flush=True)
            if vl<best-1e-4: best,state2,stale=vl,copy.deepcopy(model.state_dict()),0
            else: stale+=1
            if stale>=5: break
        model.load_state_dict(state2); p=predict(model,images[te]); oof[te]=p; r=report(y[te],p); folds.append({'fold':fold,'ssl_epochs':ep,'fixed_0.5':r}); print(f'[fold {fold}/5] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}',flush=True)
    result={'protocol':{'test_subject_isolation':True,'test_history':False,'window_position_schedule_r_t':False,'test_subject_calibration':False,'per_window_only':True,'input_seconds':6,'outer_cv':'StratifiedGroupKFold(5, shuffle=True, random_state=0)','ssl_train_subjects_only':True,'mask_loss':'masked time-frequency reconstruction'},'model':{'parameters':sum(p.numel() for p in Classifier(Encoder()).parameters()),'ssl_decoder_parameters':sum(p.numel() for p in SSLNet().decoder.parameters()),'ssl_epochs_max':15,'finetune_epochs_max':25,'batch_size':256,'ssl_lr':2e-3,'finetune_lr':1e-3,'weight_decay':2e-3},'overall_fixed_0.5':report(y,oof),'subject_fixed_0.5':{'AUC':float(np.mean([roc_auc_score(y[g==s],oof[g==s]) for s in np.unique(g)])),'bACC':float(np.mean([balanced_accuracy_score(y[g==s],oof[g==s]>=.5) for s in np.unique(g)]))},'folds':folds,'_meta':{'seconds':time.perf_counter()-t0,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}
    np.savez_compressed(OUT/'timefreq_masked_window6_oof.npz',y=y,groups=g,pred=oof); (OUT/'timefreq_masked_window6_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'[done] {result["overall_fixed_0.5"]} total={time.perf_counter()-t0:.1f}s',flush=True)
if __name__=='__main__': main()
