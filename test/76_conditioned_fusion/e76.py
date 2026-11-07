"""E76: nested-LOSO conditioned stacking on E74 cached component outputs."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
E74 = HERE.parent / "74_full_fusion"
PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")
SUBJECTS = [p.stem.split("_")[-1] for p in sorted(E74.glob("fold_*.npz"))]

def causal_stats(x, w=120):
    x = np.asarray(x, float); cs = np.cumsum(np.r_[0., x]); cs2 = np.cumsum(np.r_[0., x*x]); n=np.arange(1,len(x)+1)
    lo=np.maximum(0,n-w); cnt=n-lo
    mean=(cs[n]-cs[lo])/cnt; var=np.maximum((cs2[n]-cs2[lo])/cnt-mean*mean,0.)
    return mean, np.sqrt(var)

def features(lr, mlp):
    d=np.abs(lr-mlp); lm,ls=causal_stats(lr); mm,ms=causal_stats(mlp); dm,ds=causal_stats(d)
    return np.column_stack([lr,mlp,d,lm,mm,dm,ls,ms,ds]).astype(np.float32)

def metric(y,s):
    y=np.asarray(y,int); s=np.asarray(s,float); p=s>0; n0=int((y==0).sum()); n1=int((y==1).sum())
    return {"n":len(y),"n_negative":n0,"n_positive":n1,"auc":float(roc_auc_score(y,s)) if n0 and n1 else None,"f1":float(f1_score(y,p,zero_division=0)) if len(y) else None,"bacc":float(.5*(p[y==1].mean()+(~p[y==0]).mean())) if n0 and n1 else None}

def stage(raw):
    # Reuse E74 post-processing without importing src or recomputing components.
    import importlib.util, sys
    sys.path.insert(0,str(HERE.parent/"17_compliant"))
    spec=importlib.util.spec_from_file_location("e61",HERE.parent/"61_trainfold_initialization"/"e59_trainfold.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.hmm(m.ema_center(raw,1200,None),300)

def evaluate(y,b,s):
    masks={"full":np.ones(len(y),bool),"early":b<=2,"late":b>=4}
    return {k:metric(y[v],s[v]) for k,v in masks.items()}

def load():
    out={}
    for p in sorted(E74.glob("fold_*.npz")):
        with np.load(p) as z:
            q=str(z["target"].item()); out[q]={k:z[k].astype(float if k in ("lr_scaled","mlp_scaled") else z[k].dtype) for k in ("t","y","block","lr_scaled","mlp_scaled")}
    return out

def main():
    data=load(); assert len(data)==57
    diagnostic={}
    for q,d in data.items():
        lr,mlp=d["lr_scaled"],d["mlp_scaled"]; raw=.5*(lr+mlp)
        diagnostic[q]={"n":len(lr),"lr_mlp_corr":float(np.corrcoef(lr,mlp)[0,1]),"mean_abs_disagreement":float(np.mean(np.abs(lr-mlp))),"std_abs_disagreement":float(np.std(np.abs(lr-mlp))),"lr_std":float(np.std(lr)),"mlp_std":float(np.std(mlp)),"lr_drift":float(np.polyfit(np.arange(len(lr)),lr,1)[0]),"mlp_drift":float(np.polyfit(np.arange(len(mlp)),mlp,1)[0]),"components":{}}
        for name,s in {"lr":lr,"mlp":mlp,"raw_w0.50":raw}.items(): diagnostic[q]["components"][name]=evaluate(d["y"],d["block"],stage(2*s))
    folds=[]
    for i,q in enumerate(SUBJECTS,1):
        d=data[q]; tr=[p for p in SUBJECTS if p!=q]
        X=np.concatenate([features(data[p]["lr_scaled"],data[p]["mlp_scaled"]) for p in tr]); y=np.concatenate([data[p]["y"] for p in tr])
        gate=LogisticRegression(C=0.1,class_weight="balanced",max_iter=300,solver="liblinear",random_state=0).fit(X,y)
        xt=features(d["lr_scaled"],d["mlp_scaled"]); gate_score=gate.decision_function(xt)
        lr,mlp=d["lr_scaled"],d["mlp_scaled"]
        raws={"w0.50":2*.5*(lr+mlp),"w0.25":2*(.25*lr+.75*mlp),"mlp":2*mlp,"conditioned":gate_score}
        # One predeclared minimal iteration: shrink unstable gate output halfway toward fixed w=.5.
        raws["conditioned_shrink"]=.5*gate_score+.5*raws["w0.50"]
        scores={k:stage(v) for k,v in raws.items()}; metrics={k:evaluate(d["y"],d["block"],v) for k,v in scores.items()}
        np.savez_compressed(HERE/f"fold_{i:02d}_{q}.npz",target=np.array(q),train_subjects=np.array(tr),t=d["t"],y=d["y"],block=d["block"],lr_scaled=lr,mlp_scaled=mlp,gate_score=gate_score,**{f"score_{k}":v for k,v in scores.items()})
        folds.append({"fold":i,"target":q,"train_subjects":tr,"metrics":metrics,"gate_coef":gate.coef_[0].tolist(),"gate_intercept":float(gate.intercept_[0]),"elapsed_s":0})
        print(f"[{i}/57] {q} conditioned full={metrics['conditioned']['full']} w50={metrics['w0.50']['full']}",flush=True)
    aggregate={k:{part:{m:float(np.mean([r["metrics"][k][part][m] for r in folds if r["metrics"][k][part][m] is not None])) for m in METRICS} for part in PARTS} for k in ("w0.50","w0.25","mlp","conditioned","conditioned_shrink")}
    comp={}
    for k in aggregate:
        comp[k]={}
        for part in PARTS:
            comp[k][part]={m:{"improved":0,"regressed":0,"equal":0,"undefined":0} for m in METRICS}
            for r in folds:
                for m in METRICS:
                    a=r["metrics"][k][part][m]; b=r["metrics"]["w0.50"][part][m]; z="undefined" if a is None or b is None else ("improved" if a>b else "regressed" if a<b else "equal"); comp[k][part][m][z]+=1
    protocol={"experiment":"E76_conditioned_fusion","status":"complete","n_folds":57,"source":"E74 fold NPZ only; no src/data/old experiments modified","diagnostics":"per-subject unlabeled component disagreement, correlation, volatility and linear drift, plus labeled descriptive metrics","gate":"strict outer LOSO; inner training uses only other 56 cached component outputs and labels; LogisticRegression C=0.1; features=lr,mlp,abs_diff,causal rolling means/stds (window 120)","comparators":["fixed w=.5","fixed w=.25","pure MLP","conditioned_shrink (one minimal iteration)"],"iteration":"After first gate, one predeclared half-shrink toward w=.5 was evaluated; no further search.","postprocessing":"EMA1200 + forward HMM300","parts":{"full":"all","early":"block<=2","late":"block>=4"},"threshold":0,"diagnostic_subjects":57}
    (HERE/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    result={"protocol":protocol,"aggregate":aggregate,"relative_to_w0.50":comp,"diagnostics":diagnostic,"folds":folds}
    (HERE/"e76_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# E76 条件化融合（严格嵌套LOSO）","","基于E74完整57折缓存，仅使用每折 `LR_scaled`、`MLP_scaled`、标签和块；未修改 `src`、原始数据或旧实验。门控为训练折内拟合的极简Logistic stacking，输入仅为组件分数、绝对分歧及因果滚动均值/标准差；统一EMA1200+前向HMM300。额外只执行一次预先声明的最小迭代：门控分数与固定w=.5原始分数各半收缩。","","## 宏观指标","","|方法|阶段|AUC|F1|bACC|","|---|---|---:|---:|---:|"]
    for k in aggregate:
        for part in PARTS:
            a=aggregate[k][part]; lines.append(f"|{k}|{part}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["","## 相对固定w=.5逐人比较","","|方法|阶段|AUC改善/退步/持平|F1改善/退步/持平|bACC改善/退步/持平|","|---|---|---|---|---|"]
    for k in aggregate:
        if k=="w0.50": continue
        for part in PARTS:
            c=comp[k][part]; lines.append(f"|{k}|{part}|{c['auc']['improved']}/{c['auc']['regressed']}/{c['auc']['equal']}|{c['f1']['improved']}/{c['f1']['regressed']}/{c['f1']['equal']}|{c['bacc']['improved']}/{c['bacc']['regressed']}/{c['bacc']['equal']}|")
    lines += ["","## 诊断摘要","","每人诊断明细见 `e76_results.json` 的 `diagnostics`，包含LR/MLP相关性、绝对分歧均值/波动、分数波动和线性漂移，并给出组件描述性AUC/F1/bACC。","","## 结论","","本实验先完成可靠的线性门控；是否具备实质提升以全体宏观指标和后段bACC共同判断。若无明确提升，不再进行无休止搜索。"]
    (HERE/"报告_E76_条件化融合.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__": main()
