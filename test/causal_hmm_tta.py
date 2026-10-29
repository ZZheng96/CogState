"""因果二状态 HMM：用训练折估计 score->状态的发射概率，测试流只做前向滤波。"""
import json, time
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).resolve().parent
K = 120

def filt(score, intercept, coef, q):
    # p(y=1|score) 作为发射概率；q 是每窗状态切换概率
    prior = .5; out = np.empty(len(score))
    for i, x in enumerate(score):
        em = float(expit(intercept + coef*x))
        post = prior*em / (prior*em + (1-prior)*(1-em) + 1e-12)
        out[i] = post
        prior = (1-q)*post + q*(1-post)
    return out

def main():
    tick=time.perf_counter()
    d=np.load(OUT/'highdim_deep_features.npz'); y=d['y'].astype(int); g=d['groups']
    pos=np.empty(len(y),int)
    for s in np.unique(g): i=np.where(g==s)[0]; pos[i]=np.arange(len(i))
    mask=pos>=K
    pool={}
    for f in ('b_branch_oof.npz','b_calib_variants_oof.npz','b_calib2_oof.npz'):
        z=np.load(OUT/f); pool.update({k:z[k] for k in z.files if k not in ('y','groups')})
    P3=['quant120_histgb','quant120w_histgb','quant120_svc','base_svc']
    raw=np.mean([rankdata(pool[k])/len(y) for k in P3],0)
    splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(raw,y,g))
    res={}; outs={}
    for q in (.001,.003,.005,.01,.02,.03,.05,.08,.12):
        out=np.full(len(y),.5)
        for tr,te in splits:
            # score calibration is fit only on training subjects
            m=LogisticRegression(C=1.,max_iter=2000).fit(raw[tr,None],y[tr])
            for s in np.unique(g[te]):
                idx=np.where(g==s)[0]
                out[idx]=filt(raw[idx],m.intercept_[0],m.coef_[0,0],q)
        outs[q]=out.copy()
        auc=float(roc_auc_score(y[mask],out[mask])); bacc=float(balanced_accuracy_score(y[mask],out[mask]>=.5))
        res[str(q)]={'AUC':auc,'bACC':bacc}
        print(f'[result] q={q:.3f} AUC={auc:.4f} bACC={bacc:.4f} ({time.perf_counter()-tick:.0f}s)',flush=True)
    (OUT/'causal_hmm_tta_results.json').write_text(json.dumps(res,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__': main()
