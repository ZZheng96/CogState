from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

HERE=Path(__file__).resolve().parent; TESTS=HERE.parent
sys.path.insert(0,str(TESTS/'17_compliant')); import loso as L
sys.path.insert(0,str(TESTS/'42_e17_loso_hmm')); from multiscale import fit_mlp
spec=importlib.util.spec_from_file_location('e61',TESTS/'61_trainfold_initialization/e59_trainfold.py'); E61=importlib.util.module_from_spec(spec); spec.loader.exec_module(E61)
E53=TESTS/'53_e17_post_loso/e53_scores.npz'; E67=TESTS/'67_mlp_loso'; E63=TESTS/'63_layer_ablation'
WEIGHTS=(0.,.25,.5,.75,1.)

def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False),encoding='utf-8')
def metric(y,s):
 y=np.asarray(y,int); s=np.asarray(s,float); p=s>0; n0=int((y==0).sum()); n1=int((y==1).sum())
 return {'n':len(y),'auc':float(roc_auc_score(y,s)) if n0 and n1 else None,'f1':float(f1_score(y,p,zero_division=0)) if len(y) else None,'bacc':float(.5*(p[y==1].mean()+(~p[y==0]).mean())) if n0 and n1 else None}
def evals(y,b,s): return {k:metric(y[m],s[m]) for k,m in {'full':np.ones(len(y),bool),'early':b<=2,'late':b>=4}.items()}
def stages(raw):
 d=E61.ema_center(raw,1200,None); return {'raw':raw,'dt1200':d,'dt1200_hmm300':E61.hmm(d,300)}
def score_all(D, q, raw): return {k:evals(D[q][1],D[q][2],v) for k,v in stages(raw).items()}
def agg(rows):
 return {st:{part:{m:float(np.mean([r[st][part][m] for r in rows if r[st][part][m] is not None])) if any(r[st][part][m] is not None for r in rows) else None for m in ('auc','f1','bacc')} for part in ('full','early','late')} for st in ('raw','dt1200','dt1200_hmm300')}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
 if any(p.name not in ('e68.py','__pycache__') for p in HERE.iterdir()): raise RuntimeError('E68 directory is not empty; refusing overwrite')
 subs,D=L.load(); assert len(subs)==57
 checks={q:{'source':str(L.SLID/f'sub-{q}.npz'),'sha256':sha(L.SLID/f'sub-{q}.npz')} for q in subs}
 protocol={'experiment':'E68_strict_nested_LOSO_LR_MLP_fusion','subjects':subs,'outer_training_subjects':56,'inner_protocol':'for every outer q, each inner r in outer train is held out; fit on remaining 55 only; select by mean(0.5*full bACC+0.5*late bACC)','weights':WEIGHTS,'fusion':'2*(w*LR_scaled+(1-w)*MLP_scaled)','features':'E42/E61 fixed multiscale features','postprocessing':'ema_center(1200,None) then forward hmm(300), threshold 0','device':'cuda','baseline':'E53 scores with same postprocessing; E67 MLP fold scores reused; E63 components reused where available'}
 save(HERE/'protocol.json',{'protocol':protocol,'data_checks':checks})
 F=E61.E42.build_features(subs,D); e53=np.load(E53); e67={q:np.load(E67/f'fold_{i+1:02d}_{q}.npz')['mlp_scaled'] for i,q in enumerate(subs)}
 base={q:e53[q] for q in subs}; comps={}
 for q in subs:
  p=E63/f'fold_{q}_components.npz'
  if p.exists():
   with np.load(p) as z: comps[q]=(z['lr_scaled_'+q],z['mlp_scaled_'+q])
  else: comps[q]=(base[q]-e67[q],e67[q])
 folds=[]
 for oi,q in enumerate(subs,1):
  tr=[x for x in subs if x!=q]; inner=[]; t0=time.time()
  for ri,r in enumerate(tr,1):
   ip=HERE/f'inner_{oi:02d}_{q}_{ri:02d}_{r}.npz'
   if ip.exists():
    with np.load(ip) as z: vals={f'w{w:.2f}':z[f'w{w:.2f}'] for w in WEIGHTS}
   else:
    fit=[x for x in tr if x!=r]; X=torch.cat([F[x] for x in fit]); yy=np.concatenate([D[x][1] for x in fit]); lr=L.fit_lr(X,yy,.001); ms=[E61.fit_mlp(X,yy,seed=s) for s in range(3)] if hasattr(E61,'fit_mlp') else [__import__('multiscale').fit_mlp(X,yy,seed=s) for s in range(3)]
    with torch.no_grad():
     al=torch.cat([lr(F[x]) for x in fit]); mm=torch.cat([torch.stack([m(F[x]) for m in ms]).mean(0) for x in fit]); sl=float(al.std()); sm=float(mm.std()); a=lr(F[r]).cpu().numpy()/sl; m=torch.stack([z(F[r]) for z in ms]).mean(0).cpu().numpy()/sm
    vals={f'w{w:.2f}':2*(w*a+(1-w)*m) for w in WEIGHTS}; np.savez_compressed(ip,**vals); del X,lr,ms,al,mm; torch.cuda.empty_cache()
   inner.append({f'w{w:.2f}':score_all(D,r,vals[f'w{w:.2f}']) for w in WEIGHTS})
  means={f'w{w:.2f}':float(np.mean([.5*x[f'w{w:.2f}']['dt1200_hmm300']['full']['bacc']+.5*x[f'w{w:.2f}']['dt1200_hmm300']['late']['bacc'] for x in inner])) for w in WEIGHTS}; chosen=max(means,key=lambda k:(means[k],-abs(float(k[1:])-.5),-float(k[1:]))) ; w=float(chosen[1:]); raw=2*(w*comps[q][0]+(1-w)*comps[q][1]); chosen_metrics=score_all(D,q,raw); baseline=score_all(D,q,base[q]); row={'fold':oi,'target':q,'train_subjects':tr,'inner_loso_n':56,'inner_selection_score':means,'chosen_w':w,'metrics':chosen_metrics,'baseline_e53':baseline,'delta_vs_e53':{st:{part:{m:(chosen_metrics[st][part][m]-baseline[st][part][m]) if chosen_metrics[st][part][m] is not None and baseline[st][part][m] is not None else None for m in ('auc','f1','bacc')} for part in ('full','early','late')} for st in chosen_metrics}}
  np.savez_compressed(HERE/f'fold_{oi:02d}_{q}.npz',target=np.array(q),train_subjects=np.array(tr),chosen_w=np.array(w),raw=raw,baseline_raw=base[q],y=D[q][1],block=D[q][2],**stages(raw)); save(HERE/f'fold_{oi:02d}_{q}.json',row); folds.append(row); print(f'[{oi}/57] {q} w={w:.2f} {means[chosen]:.4f}',flush=True)
 counts={f'w{w:.2f}':sum(r['chosen_w']==w for r in folds) for w in WEIGHTS}; aggregate={st:{part:{m:float(np.mean([r['metrics'][st][part][m] for r in folds if r['metrics'][st][part][m] is not None])) for m in ('auc','f1','bacc')} for part in ('full','early','late')} for st in ('raw','dt1200','dt1200_hmm300')}; improved={st:{part:sum(r['delta_vs_e53'][st][part]['bacc']>0 for r in folds if r['delta_vs_e53'][st][part]['bacc'] is not None) for part in ('full','early','late')} for st in ('raw','dt1200','dt1200_hmm300')}
 save(HERE/'e68_results.json',{'protocol':protocol,'aggregate':aggregate,'chosen_weight_counts':counts,'improved_subjects_vs_e53_bacc':improved,'folds':folds}); print(json.dumps({'aggregate':aggregate,'counts':counts,'improved':improved},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
