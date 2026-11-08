"""Strict nested LOSO conditioned fusion with subject-grouped inner OOF gating."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TESTS = ROOT / "test"
sys.path.insert(0, str(TESTS / "17_compliant"))
import loso as L
from multiscale import fit_mlp

def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
E42 = imp("e42_nested_loso", TESTS / "42_e17_loso_hmm" / "e42.py")
E61 = imp("e61_nested_loso", TESTS / "61_trainfold_initialization" / "e59_trainfold.py")

OUT = HERE
CACHE = OUT / "cache"
PARTS = ("full", "early", "late")
METRICS = ("auc", "f1", "bacc")
CONFIG = {"inner_folds": 7, "inner_seed_base": 76001, "lr_C": .001, "mlp_seeds": [0, 1, 2],
          "gate_C": .1, "gate_class_weight": "balanced", "gate_random_state": 0,
          "rolling_window": 120, "postprocessing": "EMA1200+forward_HMM300", "threshold": 0}

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config_hash(): return hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()

def causal_stats(x, w=120):
    x = np.asarray(x, float); cs = np.cumsum(np.r_[0., x]); cs2 = np.cumsum(np.r_[0., x*x]); n=np.arange(1,len(x)+1)
    lo=np.maximum(0,n-w); cnt=n-lo
    mean=(cs[n]-cs[lo])/cnt
    var=np.maximum((cs2[n]-cs2[lo])/cnt-mean*mean,0.)
    return mean, np.sqrt(var)

def gate_features(lr, mlp):
    d=np.abs(lr-mlp); lm,ls=causal_stats(lr); mm,ms=causal_stats(mlp); dm,ds=causal_stats(d)
    return np.column_stack([lr,mlp,d,lm,mm,dm,ls,ms,ds]).astype(np.float32)

def metric(y, s):
    y=np.asarray(y,int); s=np.asarray(s,float); p=s>0; n0=int((y==0).sum()); n1=int((y==1).sum())
    return {"n":len(y),"n_negative":n0,"n_positive":n1,
            "auc":float(roc_auc_score(y,s)) if n0 and n1 else None,
            "f1":float(f1_score(y,p,zero_division=0)) if len(y) else None,
            "bacc":float(.5*(p[y==1].mean()+(~p[y==0]).mean())) if n0 and n1 else None}

def evaluate(y,b,s):
    masks={"full":np.ones(len(y),bool),"early":b<=2,"late":b>=4}
    return {k:metric(y[v],s[v]) for k,v in masks.items()}

def stage(raw): return E61.hmm(E61.ema_center(np.asarray(raw,float),1200,None),300)

def subject_source(q): return L.SLID / f"sub-{q}.npz"

def outputs(models, Fq):
    flr, fms, sd_lr, sd_mlp = models
    lr = (flr(Fq) / sd_lr).detach().cpu().numpy()
    mlp = (torch.stack([f(Fq) for f in fms]).mean(0) / sd_mlp).detach().cpu().numpy()
    return lr, mlp

def fit_base(F, D, train):
    X=torch.cat([F[p] for p in train]); y=np.concatenate([D[p][1] for p in train])
    flr=L.fit_lr(X,y,CONFIG["lr_C"]); fms=[fit_mlp(X,y,seed=s) for s in CONFIG["mlp_seeds"]]
    lr_train=torch.cat([flr(F[p]) for p in train]); mlp_train=torch.cat([torch.stack([f(F[p]) for f in fms]).mean(0) for p in train])
    sd_lr=float(lr_train.std()); sd_mlp=float(mlp_train.std()); assert sd_lr>0 and sd_mlp>0
    del X,lr_train,mlp_train; torch.cuda.empty_cache()
    return flr,fms,sd_lr,sd_mlp

def split_inner(train, seed):
    perm=list(np.random.RandomState(seed).permutation(train)); return [perm[k::CONFIG["inner_folds"]] for k in range(CONFIG["inner_folds"])]

def run_outer(i,q,subs,D,F,source_hashes):
    t0=time.time(); T=[p for p in subs if p!=q]; assert q not in T and len(T)==56
    inner=split_inner(T, CONFIG["inner_seed_base"]+i)
    assert sorted(sum(inner,[]))==sorted(T) and len(set(sum(inner,[])))==len(T)
    oof_lr={}; oof_mlp={}; inner_rows=[]
    for j,V in enumerate(inner,1):
        tr=[p for p in T if p not in V]; assert not set(tr)&set(V) and q not in tr and q not in V
        models=fit_base(F,D,tr)
        for p in V: oof_lr[p],oof_mlp[p]=outputs(models,F[p])
        inner_rows.append({"inner_fold":j,"train_subjects":tr,"validation_subjects":V,"base_train_subjects":tr})
        del models; torch.cuda.empty_cache()
    assert set(oof_lr)==set(T)
    Xg=np.concatenate([gate_features(oof_lr[p],oof_mlp[p]) for p in T]); yg=np.concatenate([D[p][1] for p in T])
    gate=LogisticRegression(C=CONFIG["gate_C"],class_weight=CONFIG["gate_class_weight"],max_iter=300,solver="liblinear",random_state=0).fit(Xg,yg)
    models=fit_base(F,D,T); lr,mlp=outputs(models,F[q]); gate_score=gate.decision_function(gate_features(lr,mlp))
    raws={"w0.50":lr+mlp,"conditioned":gate_score,"conditioned_shrink":.5*gate_score+.5*(lr+mlp),"mlp":2*mlp}
    scores={k:stage(v) for k,v in raws.items()}; y,b=D[q][1],D[q][2]
    metrics={k:evaluate(y,b,v) for k,v in scores.items()}
    meta={"fold":i,"target":q,"outer_train_subjects":T,"inner_folds":inner_rows,"gate_train_subjects":T,
          "base_final_train_subjects":T,"source_sha256":source_hashes,"config":CONFIG,"config_sha256":config_hash(),
          "assertions":{"target_not_in_any_training":q not in T and all(q not in r["train_subjects"] for r in inner_rows),"inner_disjoint":all(not set(r["train_subjects"])&set(r["validation_subjects"]) for r in inner_rows),"all_T_once_oof":set(oof_lr)==set(T)},
          "gate_coef":gate.coef_[0].tolist(),"gate_intercept":float(gate.intercept_[0]),"metrics":metrics,"elapsed_s":round(time.time()-t0,2)}
    with np.load(subject_source(q)) as z:
        order=np.argsort(z["t"]); t=z["t"][order]
    np.savez_compressed(CACHE/f"fold_{i:02d}_{q}.npz",target=np.array(q),train_subjects=np.array(T),t=t,y=y,block=b,lr_scaled=lr,mlp_scaled=mlp,gate_score=gate_score,**{f"score_{k}":v for k,v in scores.items()})
    (CACHE/f"fold_{i:02d}_{q}.json").write_text(json.dumps(meta,ensure_ascii=False,indent=1),encoding="utf-8")
    print(f"[{i}/57] {q} conditioned={metrics['conditioned']['full']} elapsed={meta['elapsed_s']}s",flush=True)
    return meta

def aggregate(rows):
    out={}
    for method in ("w0.50","conditioned","conditioned_shrink","mlp"):
        out[method]={}
        for part in PARTS:
            out[method][part]={m:(float(np.mean([r["metrics"][method][part][m] for r in rows if r["metrics"][method][part][m] is not None])) if any(r["metrics"][method][part][m] is not None for r in rows) else None) for m in METRICS}
    return out

def report(result):
    lines=["# 嵌套 LOSO 条件化动态融合", "", "严格外层57折LOSO；每个目标被试完全留出。其余56人按确定性 seed=76001+外层折、7个被试组内层折生成基础LR/MLP OOF分数，再仅用这些OOF分数训练 Logistic gate。基础模型为E42固定多尺度特征、LR C=.001、MLP seed 0/1/2均值；尺度仅由对应训练被试估计；门控特征为lr、mlp、abs_diff及窗口120的严格因果滚动均值/std。后处理统一 EMA1200 + 前向HMM300，阈值0。", "", "## 聚合指标", "", "|方法|阶段|AUC|F1|bACC|", "|---|---|---:|---:|---:|"]
    for k,v in result["aggregate"].items():
        for p,a in v.items(): lines.append(f"|{k}|{p}|{a['auc']:.6f}|{a['f1']:.6f}|{a['bacc']:.6f}|")
    lines += ["", "## 与原E74/E76的关系与限制", "", "本实验不读取原E74/E76结果或组件缓存作为训练输入，独立重训所有外层基础模型并独立缓存；因此与原E74固定融合及原E76直接复用E74组件的数值不可视为同一缓存链路的逐点复现。conditioned_shrink为预声明的半幅收缩对照，未用目标被试标签调参。窗口统计严格只含当前及过去帧；外层目标标签只用于最终评价。", "", f"完成折数：{len(result['folds'])}/57；详细训练成员、源文件哈希、配置哈希和断言见 `cache/fold_*.json`。"]
    (OUT/"报告_嵌套LOSO动态融合.md").write_text("\n".join(lines),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--start",type=int,default=1); ap.add_argument("--end",type=int,default=57); a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    CACHE.mkdir(parents=True,exist_ok=True); subs,D=L.load(); assert len(subs)==57
    source_hashes={q:sha256(subject_source(q)) for q in subs}; source_hashes["e42.py"]=sha256(TESTS/"42_e17_loso_hmm"/"e42.py"); source_hashes["e61.py"]=sha256(TESTS/"61_trainfold_initialization"/"e59_trainfold.py")
    F=E42.build_features(subs,D); rows=[]
    for i in range(a.start,a.end+1):
        q=subs[i-1]; p=CACHE/f"fold_{i:02d}_{q}.json"
        rows.append(json.loads(p.read_text(encoding="utf-8")) if p.exists() else run_outer(i,q,subs,D,F,source_hashes))
    existing=[]
    for p in sorted(CACHE.glob("fold_*.json")): existing.append(json.loads(p.read_text(encoding="utf-8")))
    result={"protocol":"strict_nested_loso_dynamic_fusion","status":"complete" if len(existing)==57 else "partial","n_folds":len(existing),"aggregate":aggregate(existing),"folds":existing,"config":CONFIG,"config_sha256":config_hash(),"source_sha256":source_hashes}
    (OUT/"nested_loso_results.json").write_text(json.dumps(result,ensure_ascii=False,indent=1),encoding="utf-8"); report(result)
    print(json.dumps(result["aggregate"],ensure_ascii=False))
if __name__=="__main__": main()
