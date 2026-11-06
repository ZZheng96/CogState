"""E62: subject×block equal-weight LOSO retraining.

Each training subject has equal total weight; within subject each observed block is
weighted equally; frame weight is inverse block frame count, globally normalized to
mean 1. Baseline and weighted models are trained per fold with unchanged LR/MLP
architecture, C, epochs, dropout, and seeds.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as Fnn
from sklearn.metrics import f1_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; TESTS=ROOT
sys.path.insert(0,str(TESTS/"17_compliant")); import loso as L
from multiscale import fit_mlp
E42=importlib.util.module_from_spec(s:=importlib.util.spec_from_file_location('e42',TESTS/'42_e17_loso_hmm'/'e42.py')); s.loader.exec_module(E42)
DEV='cuda' if torch.cuda.is_available() else 'cpu'; OUT=ROOT/'62_weighted_block'; FD=OUT/'corrected'/'folds'; RESULT=OUT/'corrected'/'e62_results.json'
E61=importlib.util.module_from_spec(s:=importlib.util.spec_from_file_location('e61',TESTS/'61_trainfold_initialization'/'e59_trainfold.py')); s.loader.exec_module(E61)

def weighted_fit_lr(X,y,w,C=.001,iters=200):
    mu,sd=X.mean(0),X.std(0)+1e-6; Z=(X-mu)/sd; wt=torch.tensor(w,device=DEV,dtype=torch.float32); yt=torch.tensor(y,device=DEV,dtype=torch.float32)
    vw=torch.zeros(X.shape[1],device=DEV,requires_grad=True); b=torch.zeros(1,device=DEV,requires_grad=True)
    pw=(wt*(yt==0)).sum()/(wt*(yt==1)).sum(); opt=torch.optim.LBFGS([vw,b],max_iter=iters,line_search_fn='strong_wolfe'); lam=1/(C*len(y))
    def closure():
        opt.zero_grad(); loss=(Fnn.binary_cross_entropy_with_logits(Z@vw+b,yt,pos_weight=pw,reduction='none')*wt).mean()+.5*lam*(vw**2).sum(); loss.backward(); return loss
    opt.step(closure); vw,b=vw.detach(),b.detach(); return lambda X: ((X-mu)/sd)@vw+b

def weighted_fit_mlp(X,y,w,hid=128,ep=30,wd=1e-3,drop=.3,seed=0):
    torch.manual_seed(seed); mu,sd=X.mean(0),X.std(0)+1e-6; Z=(X-mu)/sd
    net=torch.nn.Sequential(torch.nn.Dropout(.2),torch.nn.Linear(X.shape[1],hid),torch.nn.GELU(),torch.nn.Dropout(drop),torch.nn.Linear(hid,1)).to(DEV)
    opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=wd); yt=torch.tensor(y,device=DEV,dtype=torch.float32); wt=torch.tensor(w,device=DEV,dtype=torch.float32); pw=(wt*(yt==0)).sum()/(wt*(yt==1)).sum(); n=len(y)
    for _ in range(ep):
        for idx in torch.randperm(n,device=DEV).split(512):
            opt.zero_grad(); loss=Fnn.binary_cross_entropy_with_logits(net(Z[idx]).squeeze(1),yt[idx],pos_weight=pw,reduction='none'); (loss*wt[idx]).mean().backward(); opt.step()
    net.eval()
    def predict(X):
        with torch.no_grad():
            return net((X-mu)/sd).squeeze(1)
    return predict

def frame_weights(subs,D,tr):
    chunks=[]
    for q in tr:
        b=D[q][2]; blocks=np.unique(b); w=np.zeros(len(b),dtype=np.float32)
        for k in blocks: w[b==k]=1.0/(len(blocks)*int((b==k).sum()))
        chunks.append(w)
    w=np.concatenate(chunks); return w/w.mean()

def metric(y,s):
    y=np.asarray(y); s=np.asarray(s); p=s>0; pos=y==1; neg=~pos
    if not pos.any() or not neg.any(): return {k:float('nan') for k in ('bacc','auc','f1')}
    tp=(p&pos).sum(); fp=(p&neg).sum()
    return {'bacc':float(.5*(tp/pos.sum()+1-fp/neg.sum())), 'auc':float(roc_auc_score(y,s)), 'f1':float(f1_score(y,p,zero_division=0))}

def post(raw):
    a=1-0.5**(1/1200); m=float(raw[0]); z=np.empty_like(raw)
    for i,x in enumerate(raw): m+=a*(x-m); z[i]=x-m
    stay=math.log1p(-1/300); sw=-math.log(300); lp=np.array([-math.log(2.),-math.log(2.)]); o=np.empty_like(z)
    for i,x in enumerate(z):
        em=np.array([-np.logaddexp(0.,x),-np.logaddexp(0.,-x)])
        if i: lp=np.array([np.logaddexp(lp[0]+stay,lp[1]+sw),np.logaddexp(lp[0]+sw,lp[1]+stay)])
        u=lp+em; lp=u-np.logaddexp(u[0],u[1]); o[i]=lp[1]-lp[0]
    return o

def fit_scores(D,F,tr,weighted):
    X=torch.cat([F[q] for q in tr]); y=np.concatenate([D[q][1] for q in tr]); w=frame_weights(tr,D,tr) if weighted else np.ones(len(y),np.float32)
    lr=weighted_fit_lr(X,y,w) if weighted else L.fit_lr(X,y,.001); mlp=[weighted_fit_mlp(X,y,w,seed=s) for s in range(3)] if weighted else [fit_mlp(X,y,seed=s) for s in range(3)]
    with torch.no_grad():
        lp=[lr(F[q]) for q in tr]; mp=[torch.stack([f(F[q]) for f in mlp]).mean(0) for q in tr]; sl=float(torch.cat(lp).std()); sm=float(torch.cat(mp).std())
        raw={q:(lr(F[q])/sl+torch.stack([f(F[q]) for f in mlp]).mean(0)/sm).detach().cpu().numpy() for q in D}
    return raw

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--folds',type=int,default=57,choices=(1,57))
    ap.add_argument('--targets',nargs='+'); ap.add_argument('--baseline-dir',type=Path)
    a=ap.parse_args(); FD.mkdir(parents=True,exist_ok=True)
    subs,D=L.load(); targets=E61.select_targets(subs,a.targets,a.folds)
    print(f'device={DEV}; targets={targets}',flush=True)
    F=E42.build_features(subs,D); allr=[]
    for q in targets:
        i=subs.index(q)+1
        p=FD/f'fold_{i:02d}_{q}.json'
        if p.exists(): allr.append(json.loads(p.read_text(encoding='utf-8'))); continue
        t=time.time(); tr=[x for x in subs if x!=q]; fold={}
        all_raw={}
        for tag,weighted in [('baseline',False),('weighted_block',True)]:
            if not weighted and a.baseline_dir:
                source=a.baseline_dir/f'fold_{i:02d}_{q}.npz'
                with np.load(source) as z:
                    if str(z['target'])!=q or list(z['train_subjects'])!=tr: raise ValueError('Baseline split mismatch')
                    raw={k:z['raw_'+k] for k in subs}
                fold['baseline_source']=str(source.resolve())
            else:
                raw=fit_scores(D,F,tr,weighted)
                if not weighted: fold['baseline_source']='retrained same settings'
            all_raw.update({f'{tag}_{k}':v for k,v in raw.items()}); fold[tag]={}
            for variant in ('raw','first_dt1200_hmm'):
                s=raw[q] if variant=='raw' else post(raw[q]); y,b=D[q][1],D[q][2]
                fold[tag][variant]={'full':metric(y,s),'early12':None if not (b<=2).any() else metric(y[b<=2],s[b<=2]),'late45':metric(y[b>=4],s[b>=4])}
            if weighted: fold['weight_check']={'mean':float(frame_weights(subs,D,tr).mean()),'min':float(frame_weights(subs,D,tr).min()),'max':float(frame_weights(subs,D,tr).max())}
        np.savez_compressed(FD/f'fold_{i:02d}_{q}.npz', **all_raw)
        fold.update({'fold':i,'target':q,'raw_file':f'fold_{i:02d}_{q}.npz','elapsed_s':round(time.time()-t,1)}); p.write_text(json.dumps(fold,ensure_ascii=False,indent=1),encoding='utf-8'); allr.append(fold); print(f'[{i}/{len(subs)}] {q} {fold["elapsed_s"]}s',flush=True)
    agg={}
    for tag in ('baseline','weighted_block'):
        agg[tag]={}
        for v in ('raw','first_dt1200_hmm'):
            agg[tag][v]={}
            for part in ('full','early12','late45'):
                rows=[x[tag][v][part] for x in allr if x[tag][v][part] is not None]; agg[tag][v][part]={k:float(np.nanmean([r[k] for r in rows])) if rows else float('nan') for k in ('bacc','auc','f1')}
    paired=[]
    for x in allr:
        for tag in ('raw','first_dt1200_hmm'):
            row={'target':x['target'],'variant':tag}
            for part in ('full','early12','late45'):
                w,b=x['weighted_block'][tag][part],x['baseline'][tag][part]
                for k in ('bacc','auc','f1'):
                    row[f'delta_{part}_{k}']=None if w is None or b is None else w[k]-b[k]
            paired.append(row)
    RESULT.write_text(json.dumps({'experiment':'E62_weighted_subject_block','device':DEV,'n_folds':len(allr),'early12_paired_n':len([x for x in allr if x['weighted_block']['raw']['early12'] is not None]),'weight_definition':'equal subject total; equal actual block within subject; inverse block frame count; global mean=1','aggregate':agg,'paired_deltas':paired,'folds':allr},ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps(agg,ensure_ascii=False),flush=True)
if __name__=='__main__': main()
