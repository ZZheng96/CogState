from __future__ import annotations
import hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

HERE=Path(__file__).resolve().parent; TESTS=HERE.parent.parent
sys.path.insert(0,str(TESTS/'17_compliant')); import loso as L
spec=importlib.util.spec_from_file_location('e61',TESTS/'61_trainfold_initialization/e59_trainfold.py'); E61=importlib.util.module_from_spec(spec); spec.loader.exec_module(E61)
E63=TESTS/'63_layer_ablation'; E53=TESTS/'53_e17_post_loso/e53_scores.npz'
TARGETS=('004','009','010','018','034','050'); WEIGHTS=(0.,.25,.5,.75,1.)
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=1,allow_nan=False),encoding='utf-8')
def metric(y,s):
 y=np.asarray(y,int); s=np.asarray(s,float); p=s>0; n0=int((y==0).sum()); n1=int((y==1).sum())
 return {'n':len(y),'auc':float(roc_auc_score(y,s)) if n0 and n1 else None,'f1':float(f1_score(y,p,zero_division=0)) if len(y) else None,'bacc':float(.5*(p[y==1].mean()+(~p[y==0]).mean())) if n0 and n1 else None}
def stages(raw):
 d=E61.ema_center(raw,1200,None); return {'raw':raw,'dt1200':d,'dt1200_hmm300':E61.hmm(d,300)}
def evaluate(y,b,raw): return {st:{part:metric(y[m],s[m]) for part,m in {'full':np.ones(len(y),bool),'early':b<=2,'late':b>=4}.items()} for st,s in stages(raw).items()}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def macro(rows,st,part,m):
 v=[r[st][part][m] for r in rows if r[st][part][m] is not None]; return float(np.mean(v)) if v else None

def main():
 subs,D=L.load(); assert set(TARGETS)<=set(subs)
 checks={q:{'source':str(L.SLID/f'sub-{q}.npz'),'sha256':sha(L.SLID/f'sub-{q}.npz')} for q in subs}
 protocol={'experiment':'E68_trainfold_select','scope':'six diagnostic outer targets only','targets':TARGETS,'outer_training_subjects':56,'selection':'fixed-weight selection on all 56 outer training subjects; not inner LOSO and not strict nested','weights':WEIGHTS,'criterion':'0.5*full bACC + 0.5*late bACC after ema_center(1200,None)+hmm(300)','components':'E63 fold components; no additional model fitting','final':'selected weight applied to isolated outer target','baseline':'E53 raw scores with same postprocessing','limitation':'only six diagnostic folds; training-fold selection is optimistic for weight selection and must not be reported as strict nested LOSO'}
 save(HERE/'protocol.json',{'protocol':protocol,'data_checks':checks})
 e53=np.load(E53); folds=[]
 for q in TARGETS:
  p=E63/f'fold_{q}_components.npz'
  with np.load(p,allow_pickle=False) as z:
   assert str(z['target'])==q
   tr=list(z['train_subjects'].astype(str)); assert len(tr)==56 and q not in tr
   lr={s:z[f'lr_scaled_{s}'] for s in tr}; mlp={s:z[f'mlp_scaled_{s}'] for s in tr}; lrq=z[f'lr_scaled_{q}']; mlpq=z[f'mlp_scaled_{q}']
  candidates={}; train_scores={}
  for w in WEIGHTS:
   name=f'w{w:.2f}'; train_scores[name]=[evaluate(D[s][1],D[s][2],2*(w*lr[s]+(1-w)*mlp[s])) for s in tr]
   candidates[name]={st:{part:{m:macro(train_scores[name],st,part,m) for m in ('auc','f1','bacc')} for part in ('full','early','late')} for st in ('raw','dt1200','dt1200_hmm300')}
  selection={k:.5*candidates[k]['dt1200_hmm300']['full']['bacc']+.5*candidates[k]['dt1200_hmm300']['late']['bacc'] for k in candidates}
  chosen=max(selection,key=lambda k:(selection[k],-abs(float(k[1:])-.5),-float(k[1:])))
  w=float(chosen[1:]); final=evaluate(D[q][1],D[q][2],2*(w*lrq+(1-w)*mlpq)); baseline=evaluate(D[q][1],D[q][2],e53[q])
  delta={st:{part:{m:(final[st][part][m]-baseline[st][part][m]) if final[st][part][m] is not None and baseline[st][part][m] is not None else None for m in ('auc','f1','bacc')} for part in ('full','early','late')} for st in final}
  row={'target':q,'train_subjects':tr,'components_file':str(p),'candidate_train_metrics':candidates,'selection_score':selection,'chosen_w':w,'outer_target_metrics':final,'baseline_e53':baseline,'delta_vs_e53':delta}
  save(HERE/f'fold_{q}.json',row); np.savez_compressed(HERE/f'fold_{q}.npz',target=np.array(q),train_subjects=np.array(tr),chosen_w=np.array(w),lr_scaled=lrq,mlp_scaled=mlpq,baseline_raw=e53[q],y=D[q][1],block=D[q][2],**stages(2*(w*lrq+(1-w)*mlpq))); folds.append(row)
 counts={f'w{w:.2f}':sum(r['chosen_w']==w for r in folds) for w in WEIGHTS}
 agg={st:{part:{m:float(np.mean([r['outer_target_metrics'][st][part][m] for r in folds if r['outer_target_metrics'][st][part][m] is not None])) for m in ('auc','f1','bacc')} for part in ('full','early','late')} for st in ('raw','dt1200','dt1200_hmm300')}
 improved={st:{part:sum(r['delta_vs_e53'][st][part]['bacc']>0 for r in folds if r['delta_vs_e53'][st][part]['bacc'] is not None) for part in ('full','early','late')} for st in ('raw','dt1200','dt1200_hmm300')}
 save(HERE/'e68_trainfold_select_results.json',{'protocol':protocol,'aggregate_outer_targets':agg,'chosen_weight_counts':counts,'improved_subjects_vs_e53_bacc':improved,'folds':folds})
 print(json.dumps({'aggregate':agg,'weight_counts':counts,'improved':improved},ensure_ascii=False),flush=True)
if __name__=='__main__': main()
