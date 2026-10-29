"""组合因果 HMM 与因果个人分类器，比较是否互补。"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from sklearn.feature_selection import f_classif
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
OUT=Path(__file__).resolve().parent; sys.path.insert(0,str(OUT))
from b_calib_variants import subject_quantile
from singlem_eval import smooth, boot
from self_training import personal
from causal_hmm_tta import filt

d=np.load(OUT/'highdim_deep_features.npz'); X0,y,g=d['X'].astype(float),d['y'].astype(int),d['groups']; E=np.load(OUT/'emg_features_v2.npz')['E'].astype(float)
X0=X0[:,X0.std(0)>1e-8]; Z=subject_quantile(np.hstack([X0,E]),g,120); pos=np.empty(len(y),int)
for s in np.unique(g): i=np.where(g==s)[0]; pos[i]=np.arange(len(i))
mask=pos>=120; pool={}
for f in ('b_branch_oof.npz','b_calib_variants_oof.npz','b_calib2_oof.npz'):
 z=np.load(OUT/f); pool.update({k:z[k] for k in z.files if k not in ('y','groups')})
P3=['quant120_histgb','quant120w_histgb','quant120_svc','base_svc']; raw=np.mean([rankdata(pool[k])/len(y) for k in P3],0); gs=smooth(raw,g,5)
splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(Z,y,g)); P=np.full(len(y),.5)
for tr,te in splits:
 F,_=f_classif(Z[tr],y[tr]); cols=np.argsort(np.nan_to_num(F))[::-1][:300]
 for s in np.unique(g[te]):
  idx=np.where(g==s)[0]; P[idx]=personal(Z[np.ix_(idx,cols)],gs[idx],.3,True,.05)
Ps=smooth(P,g,5); out={}
for q,coef in ((.05,None),(.08,None)):
 h=np.full(len(y),.5)
 for tr,te in splits:
  from sklearn.linear_model import LogisticRegression
  m=LogisticRegression(max_iter=2000).fit(raw[tr,None],y[tr])
  for s in np.unique(g[te]):
   idx=np.where(g==s)[0]; h[idx]=filt(raw[idx],m.intercept_[0],m.coef_[0,0],q)
 for a in (0,.25,.5,.75,1):
  score=(1-a)*h+a*Ps
  out[f'q{q}_self{a}']={'AUC':float(roc_auc_score(y[mask],score[mask])),'bACC':float(balanced_accuracy_score(y[mask],score[mask]>=.5)),'vs_p3':boot(score,gs,y,g,mask)}
  print(f'[result] q={q} self={a}: {out[f"q{q}_self{a}"]}',flush=True)
(OUT/'blend_hmm_self_results.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
