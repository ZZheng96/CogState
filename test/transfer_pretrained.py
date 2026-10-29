"""BIOT transfer experiment on the supplied single-channel EEG recordings."""
import os, sys, json, time, platform, hashlib, urllib.request
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MKL_NUM_THREADS','1')
import numpy as np, torch, mne
from torch import nn
from scipy.signal import resample
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from cogstate import load_trials, bad_intervals, valid_window
OUT=ROOT/'test'; CACHE=OUT/'pretrained_cache'; CACHE.mkdir(exist_ok=True)
SEED=0; np.random.seed(SEED); torch.manual_seed(SEED); torch.set_num_threads(1)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); START=time.time()
URL='https://raw.githubusercontent.com/ycq091044/BIOT/main/pretrained-models/EEG-PREST-16-channels.ckpt'; CKPT=CACHE/'EEG-PREST-16-channels.ckpt'

def download():
    if not CKPT.exists():
        print('[download] BIOT EEG-PREST-16-channels.ckpt',flush=True); urllib.request.urlretrieve(URL,CKPT)
    return {'url':URL,'path':str(CKPT.relative_to(ROOT)),'bytes':CKPT.stat().st_size,'sha256':hashlib.sha256(CKPT.read_bytes()).hexdigest()}
class PatchFrequencyEmbedding(nn.Module):
    def __init__(self,emb_size=256,n_freq=101): super().__init__(); self.projection=nn.Linear(n_freq,emb_size)
    def forward(self,x): return self.projection(x.permute(0,2,1))
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=1000):
        super().__init__(); pe=torch.zeros(max_len,d_model); pos=torch.arange(max_len).unsqueeze(1).float(); div=torch.exp(torch.arange(0,d_model,2).float()*-(np.log(10000)/d_model)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer('pe',pe[None])
    def forward(self,x): return x+self.pe[:,:x.size(1)]
try:
    from linear_attention_transformer import LinearAttentionTransformer
except ImportError as e: raise RuntimeError('missing linear-attention-transformer; run python -m pip install linear-attention-transformer') from e
class BIOTEncoder(nn.Module):
    def __init__(self,emb_size=256,heads=8,depth=4,n_channels=16,n_fft=200,hop_length=100):
        super().__init__(); self.n_fft=n_fft; self.hop_length=hop_length; self.patch_embedding=PatchFrequencyEmbedding(emb_size,n_fft//2+1); self.transformer=LinearAttentionTransformer(dim=emb_size,heads=heads,depth=depth,max_seq_len=1024,attn_layer_dropout=.2,attn_dropout=.2); self.positional_encoding=PositionalEncoding(emb_size); self.channel_tokens=nn.Embedding(n_channels,256); self.index=nn.Parameter(torch.arange(n_channels),requires_grad=False)
    def forward(self,x):
        seq=[]
        for i in range(x.shape[1]):
            z=torch.abs(torch.stft(x[:,i,:],n_fft=self.n_fft,hop_length=self.hop_length,center=False,onesided=True,return_complex=True)); z=self.patch_embedding(z); b,t,_=z.shape; tok=self.channel_tokens(self.index[i]).view(1,1,-1).expand(b,t,-1); seq.append(self.positional_encoding(z+tok))
        return self.transformer(torch.cat(seq,1)).mean(1)
class Head(nn.Module):
    def __init__(self,d=256): super().__init__(); self.fc=nn.Linear(d,1)
    def forward(self,x): return self.fc(x).squeeze(1)
def extract():
    X=[]; y=[]; g=[]; meta=[]; paths=sorted((ROOT/'EEG_dataset').glob('*/*_events.csv'))
    for ci,csv in enumerate(paths,1):
        sid=csv.parent.name; bdf=next(csv.parent.glob('*.bdf')); raw=mne.io.read_raw_bdf(str(bdf),preload=True,verbose='ERROR'); fs=float(raw.info['sfreq']); bad=bad_intervals(raw); dur=raw.n_times/fs
        for tr in load_trials(str(csv),str(bdf)):
            t=tr.start+6
            while t<=tr.end+1e-6:
                a,z=t-6,t
                if valid_window(a,z,bad,dur):
                    s,e=round(a*fs),round(z*fs); q=raw.get_data(start=s,stop=e)[0]
                    if len(q)==e-s and np.isfinite(q).all(): X.append(resample(q,1200).astype('float32')); y.append(tr.label); g.append(sid); meta.append({'subject':sid,'trial_id':tr.trial_id,'time':t,'label':tr.label})
                t+=6
        raw.close(); print(f'[extract {ci}/{len(paths)}] subject={sid} windows={len(y)} elapsed={time.time()-START:.1f}s',flush=True)
    return np.asarray(X),np.asarray(y),np.asarray(g),meta
def score(y,p): return {'AUC':float(roc_auc_score(y,p)),'bACC':float(balanced_accuracy_score(y,p>=.5))}
def run_model(name,encoder,X,y,g,pretrained,freeze):
    cv=StratifiedGroupKFold(5,shuffle=True,random_state=SEED); oof=np.full(len(y),np.nan); folds=[]
    for fold,(tr,va) in enumerate(cv.split(X,y,g),1):
        torch.manual_seed(SEED+fold); enc=BIOTEncoder().to(DEVICE)
        if pretrained: enc.load_state_dict(encoder.state_dict())
        if freeze:
            for p in enc.parameters(): p.requires_grad=False
        head=Head().to(DEVICE); opt=torch.optim.AdamW([p for p in list(enc.parameters())+list(head.parameters()) if p.requires_grad],lr=2e-3,weight_decay=1e-3); loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(y[tr]==0).sum()/max((y[tr]==1).sum(),1)],device=DEVICE))
        train=torch.from_numpy(X[tr]); yt=torch.from_numpy(y[tr].astype('float32'))
        for ep in range(3):
            enc.train(); head.train(); order=np.random.default_rng(SEED+fold+ep).permutation(len(tr)); losses=[]
            for j in range(0,len(tr),64):
                ix=order[j:j+64]; xb=train[ix].to(DEVICE); yb=yt[ix].to(DEVICE); opt.zero_grad(); z=enc(xb[:,None,:].expand(-1,16,-1)); l=loss(head(z),yb); l.backward(); opt.step(); losses.append(l.item())
            print(f'[{name}] fold {fold}/5 epoch {ep+1}/3 loss={np.mean(losses):.4f}',flush=True)
        enc.eval(); head.eval(); ps=[]
        with torch.no_grad():
            for j in range(0,len(va),64): ps.append(torch.sigmoid(head(enc(torch.from_numpy(X[va[j:j+64]]).to(DEVICE)[:,None,:].expand(-1,16,-1)))).cpu().numpy())
        p=np.concatenate(ps); oof[va]=p; folds.append({'fold':fold,'subjects':sorted(set(g[va].tolist())),**score(y[va],p)}); print(f'[{name}] fold {fold} {folds[-1]}',flush=True)
    np.savez(OUT/f'pretrained_transfer_oof_{name}.npz',prob=oof,label=y,subject=g,fold=np.array([next((f['fold'] for f in folds if s in f['subjects']),-1) for s in g]))
    return {'parameters':sum(p.numel() for p in Head().parameters())+sum(p.numel() for p in encoder.parameters()),'trainable_parameters':sum(p.numel() for p in list(enc.parameters())+list(head.parameters()) if p.requires_grad),'oof':score(y,oof),'folds':folds,'oof_file':f'pretrained_transfer_oof_{name}.npz'}
meta=download(); X,y,g,records=extract(); np.savez(CACHE/'resampled_single_channel_200hz.npz',X=X,y=y,groups=g)
base=BIOTEncoder().to(DEVICE); state=torch.load(CKPT,map_location=DEVICE); base.load_state_dict(state); base.eval()
results={'protocol':{'subjects':sorted(set(g.tolist())),'n_samples':len(y),'sampling_rate_hz':200,'window_seconds':6,'input_shape':'single channel resampled 250Hz->200Hz (1200 samples), replicated 16x solely to match BIOT channel interface','cv':'StratifiedGroupKFold(5), subject-disjoint','device':str(DEVICE),'torch':torch.__version__,'mne':mne.__version__,'python':platform.python_version(),'download':meta,'license':'BIOT repository LICENSE (MIT; checkpoint provenance is the official ycq091044/BIOT repository)','causality':'BIOT uses center=False STFT but linear attention is bidirectional over the complete 6-second window; valid only as offline complete-window transfer, not strict online causal.'},'models':{}}
for name,pre,freeze in [('biot_frozen_head',True,True),('biot_finetune',True,False),('random_frozen_head',False,True),('random_finetune',False,False)]: results['models'][name]=run_model(name,base,X,y,g,pre,freeze)
(OUT/'pretrained_transfer_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
plt.figure(figsize=(8,4)); names=list(results['models']); plt.bar(np.arange(4)-.18,[results['models'][n]['oof']['AUC'] for n in names],.36,label='AUC'); plt.bar(np.arange(4)+.18,[results['models'][n]['oof']['bACC'] for n in names],.36,label='bACC'); plt.xticks(np.arange(4),names,rotation=20); plt.ylim(.4,1); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'pretrained_transfer_comparison.png',dpi=150); print('COMPLETE',json.dumps(results['models'],ensure_ascii=False),flush=True)
