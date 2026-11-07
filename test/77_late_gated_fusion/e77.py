"""E77: causal instability protection around E76 gate, using E76/E74 cached outputs only."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

HERE=Path(__file__).resolve().parent
E76=HERE.parent/'76_conditioned_fusion'
PARTS=('full','early','late'); METRICS=('auc','f1','bacc')

def causal_stats(x,w=120):
    x=np.asarray(x,float); cs=np.cumsum(np.r_[0.,x]); cs2=np.cumsum(np.r_[0.,x*x]); n=np.arange(1,len(x)+1); lo=np.maximum(0,n-w); cnt=n-lo
    mean=(cs[n]-cs[lo])/cnt; var=np.maximum((cs2[n]-cs2[lo])/cnt-mean*mean,0.)
    return mean,np.sqrt(var)

def metric(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); p=s>0; n0=(y==0).sum(); n1=(y==1).sum()
    return {'n':int(len(y)),'auc':float(roc_auc_score(y,s)) if n0 and n1 else None,'f1':float(f1_score(y,p,zero_division=0)) if len(y) else None,'bacc':float(.5*(p[y==1].mean()+(~p[y==0]).mean())) if n0 and n1 else None}

def stage_cached(raw, cached, name):
    # Candidate is applied before the common E76 EMA1200+HMM300 pipeline.
    import importlib.util
    sys.path.insert(0,str(HERE.parent/'17_compliant'))
    spec=importlib.util.spec_from_file_location('e61',HERE.parent/'61_trainfold_initialization'/'e59_trainfold.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.hmm(m.ema_center(raw,1200,None),300)

def evaluate(d,s):
    y,b=d['y'],d['block']; masks={'full':np.ones(len(y),bool),'early':b<=2,'late':b>=4}
    return {k:metric(y[v],s[v]) for k,v in masks.items()}

def load():
    out={}
    for p in sorted(E76.glob('fold_*.npz')):
        with np.load(p) as z:
            q=str(z['target'].item()); out[q]={k:z[k].astype(float if k in ('lr_scaled','mlp_scaled','gate_score') else z[k].dtype) for k in ('t','y','block','lr_scaled','mlp_scaled','gate_score','score_w0.50','score_conditioned','score_conditioned_shrink')}
    return out

def raw_candidate(d, train, mode):
    lr,mlp,gate=d['lr_scaled'],d['mlp_scaled'],d['gate_score']; raw=.5*(lr+mlp)
    div=np.abs(lr-mlp); dm,ds=causal_stats(div); gm,gs=causal_stats(np.abs(gate-raw))
    # Thresholds are unsupervised, pooled from training subjects only.
    td=[]; tg=[]
    for q in train:
        r=.5*(q['lr_scaled']+q['mlp_scaled']); dd=np.abs(q['lr_scaled']-q['mlp_scaled']); _,ss=causal_stats(dd); _,sg=causal_stats(np.abs(q['gate_score']-r)); td.extend(ss.tolist()); tg.extend(sg.tolist())
    div_thr=float(np.quantile(td,.75)); gate_thr=float(np.quantile(tg,.75))
    unstable=(ds>div_thr)|(gs>gate_thr)
    if mode=='candidate':
        # Stable: retain E76 gate; unstable: 75% shrink toward fixed w=.5.
        out=np.where(unstable,.25*gate+.75*raw,.5*gate+.5*raw)
    elif mode=='fallback':
        # One permitted fallback: preserve E76 sign while conservatively limiting magnitude to raw.
        out=np.sign(gate)*np.minimum(np.abs(gate),np.abs(raw))
    else: raise ValueError(mode)
    return out, {'div_thr':div_thr,'gate_dev_thr':gate_thr,'unstable_rate':float(unstable.mean()),'mode':mode}

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'candidate'; data=load(); subjects=sorted(data)
    folds=[]
    for i,q in enumerate(subjects,1):
        d=data[q]; train=[data[x] for x in subjects if x!=q]
        raw,diag=raw_candidate(d,train,mode); score=stage_cached(raw,d,'candidate')
        # Baselines are the exact cached E76 outputs under identical postprocessing.
        scores={'w0.50':d['score_w0.50'],'conditioned':d['score_conditioned'],'conditioned_shrink':d['score_conditioned_shrink'], 'e77_'+mode:score}
        mets={k:evaluate(d,v) for k,v in scores.items()}
        folds.append({'fold':i,'target':q,'metrics':mets,'diagnostics':diag})
        print(f'[{i}/57] {q} late AUC={mets["e77_"+mode]["late"]["auc"]:.6f}',flush=True)
    methods=['w0.50','conditioned','conditioned_shrink','e77_'+mode]
    aggregate={k:{p:{m:(float(np.mean([r['metrics'][k][p][m] for r in folds if r['metrics'][k][p][m] is not None])) if any(r['metrics'][k][p][m] is not None for r in folds) else None) for m in METRICS} for p in PARTS} for k in methods}
    rel={}
    for k in methods[1:]:
        rel[k]={}
        for p in PARTS:
            rel[k][p]={}
            for m in METRICS:
                pairs=[(r['metrics'][k][p][m],r['metrics']['w0.50'][p][m]) for r in folds if r['metrics'][k][p][m] is not None and r['metrics']['w0.50'][p][m] is not None]
                rel[k][p][m]={'improved':sum(a>b for a,b in pairs),'regressed':sum(a<b for a,b in pairs),'equal':sum(a==b for a,b in pairs),'undefined':len(folds)-len(pairs)}
    protocol={'experiment':'E77_late_gated_fusion','status':'complete','mode':mode,'source':'E76 fold NPZ cached outputs; no src/old experiments/raw data modified','outer_split':'57-fold LOSO; each target gate and thresholds exclude target','candidate':'causal rolling disagreement/gate-deviation instability; unstable points shrink to fixed w=.5, stable points retain half gate','thresholds':'training-fold unlabeled pooled 75th percentile of causal rolling std; window=120','postprocessing':'EMA1200 + forward HMM300','parts':{'full':'all','early':'block<=2','late':'block>=4'},'fallback_policy':'If candidate fails to improve late robustness, run exactly once with sign-preserving magnitude limiter.'}
    per_subject_delta={r['target']:{p:{m:(r['metrics']['e77_'+mode][p][m]-r['metrics']['w0.50'][p][m] if r['metrics']['e77_'+mode][p][m] is not None and r['metrics']['w0.50'][p][m] is not None else None) for m in METRICS} for p in PARTS} for r in folds}
    result={'protocol':protocol,'aggregate':aggregate,'relative_to_w0.50':rel,'per_subject_delta_vs_w0.50':per_subject_delta,'folds':folds}
    (HERE/f'e77_{mode}_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[f'# E77 后段门控保护（{mode}）','', '仅使用E76逐折缓存；严格嵌套LOSO。阈值来自训练折无标签因果滚动稳定性，统一EMA1200+HMM300。','', '## 宏观指标','', '|方法|阶段|AUC|F1|bACC|','|---|---|---:|---:|---:|']
    for k in methods:
        for p in PARTS:
            a=aggregate[k][p]; fmt=lambda x:'NA' if x is None else f'{x:.6f}'; lines.append(f'|{k}|{p}|{fmt(a["auc"])}|{fmt(a["f1"])}|{fmt(a["bacc"])}|')
    lines += ['', '## 相对固定w=.5逐人差值（改善/退步/持平）','', '|方法|阶段|AUC|F1|bACC|','|---|---|---|---|---|']
    for k in methods[1:]:
        for p in PARTS:
            lines.append(f'|{k}|{p}|{rel[k][p]["auc"]["improved"]}/{rel[k][p]["auc"]["regressed"]}/{rel[k][p]["auc"]["equal"]}|{rel[k][p]["f1"]["improved"]}/{rel[k][p]["f1"]["regressed"]}/{rel[k][p]["f1"]["equal"]}|{rel[k][p]["bacc"]["improved"]}/{rel[k][p]["bacc"]["regressed"]}/{rel[k][p]["bacc"]["equal"]}|')
    lines += ['', '## 结论','', '候选是否构成实质改进须结合late与full宏观指标判断；探索候选未通过时不得称为已验证改进。逐折阈值、后段改善/退步及规则诊断见JSON。']
    (HERE/f'报告_E77_{mode}.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__': main()
