"""训练折内无标签自监督预训练 + 6秒时频图监督微调。
严格 subject-disjoint：每个外层折只使用训练被试做预训练和监督训练，测试窗仅用于最终评估。
"""
import copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / 'test'
from timefreq_vision_window6 import load, make_images, DEVICE

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    def forward(self, x): return self.body(x)

class SSLNet(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = Encoder(); self.project = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 64))
    def forward(self, x): return self.project(self.encoder(x))

class Classifier(nn.Module):
    def __init__(self, encoder):
        super().__init__(); self.encoder = encoder; self.drop = nn.Dropout(.25); self.head = nn.Linear(192, 1)
    def forward(self, x): return self.head(self.drop(self.encoder(x))).ravel()

def augment(x, rng):
    z = x.copy()
    z += rng.normal(0, .08, z.shape).astype(np.float32)
    if rng.random() < .7:
        w = int(rng.integers(2, 8)); s = int(rng.integers(0, z.shape[-1]-w+1)); z[:, :, s:s+w] = 0
    if rng.random() < .5:
        c = int(rng.integers(0, 3)); z[:, c] *= float(rng.uniform(.8, 1.2))
    return z

def predict(model, X):
    model.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(X),512): out.append(torch.sigmoid(model(torch.from_numpy(X[i:i+512]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)

def report(y,p): return {'AUC':float(roc_auc_score(y,p)), 'bACC':float(balanced_accuracy_score(y,p>=.5))}

def info_nce(a,b,temp=.2):
    a=nn.functional.normalize(a,dim=1); b=nn.functional.normalize(b,dim=1); logits=a@b.T/temp; labels=torch.arange(len(a),device=a.device)
    return (nn.functional.cross_entropy(logits,labels)+nn.functional.cross_entropy(logits.T,labels))/2

def main():
    t0=time.perf_counter(); torch.set_float32_matmul_precision('high'); x,y,g=load(); images=make_images(x)
    splits=StratifiedGroupKFold(5,shuffle=True,random_state=0).split(images,y,g); oof=np.full(len(y),np.nan); folds=[]
    print(f'[data] {images.shape}, device={DEVICE}',flush=True)
    for fold,(tr0,te) in enumerate(splits,1):
        tri,vi=next(GroupShuffleSplit(n_splits=1,test_size=.15,random_state=500+fold).split(tr0,y[tr0],g[tr0])); tr,va=tr0[tri],tr0[vi]
        rng=np.random.default_rng(700+fold); ssl=SSLNet().to(DEVICE); opt=torch.optim.AdamW(ssl.parameters(),lr=2e-3,weight_decay=2e-3); best=float('inf'); state=None; stale=0
        for ep in range(1,16):
            ssl.train(); order=rng.permutation(tr); losses=[]
            for j in range(0,len(order),256):
                ids=order[j:j+256]; a=torch.from_numpy(augment(images[ids],rng)).to(DEVICE); b=torch.from_numpy(augment(images[ids],rng)).to(DEVICE)
                opt.zero_grad(set_to_none=True); loss=info_nce(ssl(a),ssl(b)); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            vl=0.0; ssl.eval()
            with torch.no_grad():
                for j in range(0,len(va),512):
                    ids=va[j:j+512]; vl+=float(info_nce(ssl(torch.from_numpy(images[ids]).to(DEVICE)),ssl(torch.from_numpy(images[ids]).to(DEVICE))).cpu())*len(ids)
            vl/=len(va)
            if ep==1 or ep%5==0: print(f'[fold {fold}/5] ssl epoch {ep}/15 loss={np.mean(losses):.4f} val={vl:.4f}',flush=True)
            if vl<best-1e-4: best, state, stale=vl,copy.deepcopy(ssl.encoder.state_dict()),0
            else: stale+=1
            if stale>=4: break
        enc=Encoder().to(DEVICE); enc.load_state_dict(state); model=Classifier(enc).to(DEVICE); pos=float((y[tr]==0).sum()/max((y[tr]==1).sum(),1)); lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device=DEVICE)); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=2e-3); best=float('inf'); state2=None; stale=0
        for ep in range(1,26):
            model.train(); order=rng.permutation(tr); losses=[]
            for j in range(0,len(order),256):
                ids=order[j:j+256]; xb=torch.from_numpy(images[ids]).to(DEVICE); yb=torch.from_numpy(y[ids]).float().to(DEVICE); opt.zero_grad(set_to_none=True); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
            p=predict(model,images[va]); vl=float(-np.mean(y[va]*np.log(np.clip(p,1e-7,1))+(1-y[va])*np.log(np.clip(1-p,1e-7,1))))
            if ep==1 or ep%5==0: print(f'[fold {fold}/5] ft epoch {ep}/25 loss={np.mean(losses):.4f} val={vl:.4f}',flush=True)
            if vl<best-1e-4: best,state2,stale=vl,copy.deepcopy(model.state_dict()),0
            else: stale+=1
            if stale>=5: break
        model.load_state_dict(state2); p=predict(model,images[te]); oof[te]=p; r=report(y[te],p); folds.append({'fold':fold,'ssl_epochs':ep,'fixed_0.5':r}); print(f'[fold {fold}/5] AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f}',flush=True)
    result={'protocol':{'test_subject_isolation':True,'test_history':False,'window_position_schedule_r_t':False,'test_subject_calibration':False,'per_window_only':True,'input_seconds':6,'outer_cv':'StratifiedGroupKFold(5, shuffle=True, random_state=0)','ssl_train_subjects_only':True},'model':{'parameters':sum(p.numel() for p in Classifier(Encoder()).parameters()),'ssl_projection_parameters':sum(p.numel() for p in SSLNet().project.parameters()),'ssl_epochs_max':15,'finetune_epochs_max':25,'batch_size':256,'ssl_lr':2e-3,'finetune_lr':1e-3,'weight_decay':2e-3},'overall_fixed_0.5':report(y,oof),'subject_fixed_0.5':{'AUC':float(np.mean([roc_auc_score(y[g==s],oof[g==s]) for s in np.unique(g)])),'bACC':float(np.mean([balanced_accuracy_score(y[g==s],oof[g==s]>=.5) for s in np.unique(g)]))},'folds':folds,'_meta':{'seconds':time.perf_counter()-t0,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}
    np.savez_compressed(OUT/'timefreq_ssl_window6_oof.npz',y=y,groups=g,pred=oof); (OUT/'timefreq_ssl_window6_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(f'[done] {result["overall_fixed_0.5"]} total={time.perf_counter()-t0:.1f}s',flush=True)
if __name__=='__main__': main()
