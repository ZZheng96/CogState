"""Database 口径历史上下文复现：严格因果历史，排除每位受试者前120窗。"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test'; DB = ROOT / 'database'
LOG = OUT / 'context_history_database.log'
RES = OUT / 'context_history_database_results.json'
OOF = OUT / 'context_history_database_oof.npz'
K0, SEED, NBOOT = 120, 0, 2000
A = (100, 200, 400); D = (50, 100, 200, 400, 800)
T0 = time.perf_counter()

class Tee:
    def __init__(self, path): self.f = path.open('w', encoding='utf-8')
    def write(self, x): sys.__stdout__.write(x); sys.__stdout__.flush(); self.f.write(x); self.f.flush()
    def flush(self): sys.__stdout__.flush(); self.f.flush()
    def close(self): self.f.close()

def log(x): print(f'[{time.perf_counter()-T0:8.1f}s] {x}', flush=True)

def positions(g):
    p = np.empty(len(g), int)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); p[ii] = np.arange(len(ii))
    return p

def causal(z, g, k):
    mu = np.empty_like(z); sd = np.empty_like(z)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); v = z[ii]; t = np.arange(len(ii)); lo = np.maximum(t-k, 0); n = t-lo
        cs = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v, 0)])
        cs2 = np.vstack([np.zeros((1, v.shape[1])), np.cumsum(v*v, 0)])
        m = np.divide(cs[t]-cs[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
        q = np.divide(cs2[t]-cs2[lo], n[:, None], out=np.zeros_like(v), where=n[:, None] > 0)
        mu[ii] = m; sd[ii] = np.sqrt(np.maximum(q-m*m, 0))
    return mu, sd

def smooth(v, g, k=5):
    out = np.full(len(v), np.nan)
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); x = v[ii]; good = np.isfinite(x); cs = np.cumsum(np.r_[0., np.where(good, x, 0.)]); ng = np.cumsum(np.r_[0, good.astype(int)]); t = np.arange(len(ii)); lo = np.maximum(t-k+1, 0); n = ng[t+1]-ng[lo]
        out[ii] = np.divide(cs[t+1]-cs[lo], n, out=np.full(len(ii), np.nan), where=n > 0)
    return out

def metrics(v, y, g, mask):
    m = mask & np.isfinite(v); r = {'n': int(m.sum()), 'AUC': float(roc_auc_score(y[m], v[m])), 'bACC': float(balanced_accuracy_score(y[m], v[m] >= .5))}
    vals = []
    for s in np.unique(g):
        ii = m & (g == s)
        if len(np.unique(y[ii])) == 2: vals.append(float(roc_auc_score(y[ii], v[ii])))
    r['subject_AUC_median'] = float(np.median(vals)); r['subject_AUC_lt_0.5_frac'] = float(np.mean(np.asarray(vals) < .5)); r['n_subject_auc'] = len(vals)
    return r

def direction(v, y, g, mask):
    out = {'R->T': [], 'T->R': []}
    for s in np.unique(g):
        ii = np.flatnonzero(g == s); cuts = np.flatnonzero(np.r_[True, np.diff(y[ii]) != 0, True]); blocks = [ii[a:b] for a,b in zip(cuts[:-1], cuts[1:])]
        for a,b in zip(blocks[:-1], blocks[1:]):
            z = np.r_[a,b]; z = z[mask[z] & np.isfinite(v[z])]
            if len(z) and len(np.unique(y[z])) == 2: out['R->T' if y[a[0]] == 0 else 'T->R'].append(roc_auc_score(y[z], v[z]))
    return {k: (float(np.mean(x)) if x else None) for k,x in out.items()}

def model(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=3, l2_regularization=1., random_state=0)

def run_oof(F, y, g, mask, splits, tag):
    p = np.full(len(y), np.nan)
    for fold,(tr,te) in enumerate(splits, 1):
        tr = tr[mask[tr]]; ev = te[mask[te]]; t = time.perf_counter(); m = model().fit(F[tr], y[tr]); p[ev] = m.predict_proba(F[ev])[:,1]
        log(f'{tag} fold={fold}/5 train={len(tr)} test={len(ev)} elapsed={time.perf_counter()-t:.1f}s')
    return p

def boot(a, b, y, g, mask):
    subs = np.unique(g); pools = [np.flatnonzero((g == s) & mask) for s in subs]; rng = np.random.default_rng(0); da=[]; db=[]
    for i in range(NBOOT):
        ix = np.concatenate([pools[j] for j in rng.integers(0,len(subs),len(subs))]); da.append(roc_auc_score(y[ix], a[ix])-roc_auc_score(y[ix], b[ix])); db.append(balanced_accuracy_score(y[ix], a[ix]>=.5)-balanced_accuracy_score(y[ix], b[ix]>=.5))
        if (i+1) % 200 == 0: log(f'bootstrap {i+1}/{NBOOT}')
    return {'dAUC': {'point': float(roc_auc_score(y[mask],a[mask])-roc_auc_score(y[mask],b[mask])), 'ci95': [float(np.percentile(da,2.5)),float(np.percentile(da,97.5))]}, 'dbACC': {'point': float(balanced_accuracy_score(y[mask],a[mask]>=.5)-balanced_accuracy_score(y[mask],b[mask]>=.5)), 'ci95': [float(np.percentile(db,2.5)),float(np.percentile(db,97.5))]}}

def block_perm(y, g, same, seed):
    rng=np.random.default_rng(seed); order=[]
    for s in np.unique(g):
        ii=np.flatnonzero(g==s); cuts=np.flatnonzero(np.r_[True,np.diff(y[ii])!=0,True]); bs=[ii[a:b] for a,b in zip(cuts[:-1],cuts[1:])]
        if same:
            labs=[y[b[0]] for b in bs]; out=list(bs)
            for lab in (0,1):
                q=[j for j,x in enumerate(labs) if x==lab]; z=list(rng.permutation(q))
                for j,k in zip(q,z): out[j]=bs[k]
            order.extend(out)
        else: order.extend([bs[j] for j in rng.permutation(len(bs))])
    return np.concatenate(order)

def main():
    tee=Tee(LOG); sys.stdout=tee
    try:
        if RES.exists() or OOF.exists(): raise FileExistsError('refusing to overwrite database artifacts')
        fs=[]; es=[]; ys=[]; gs=[]
        for p in sorted(DB.glob('windows/*/*.npz')):
            q=np.load(p); fs.append(q['f']); es.append(q['e']); ys.append(q['label']); gs.append(q['subject'])
        F0=np.hstack([np.vstack(fs),np.vstack(es)]).astype(np.float64); y=np.concatenate(ys).astype(int); g=np.concatenate(gs).astype(str)
        pos=positions(g); mask=pos>=K0
        keep=np.std(F0,0)>1e-8; F0=F0[:,keep]
        sys.path.insert(0,str(OUT)); from b_calib_variants import subject_quantile
        Z=subject_quantile(F0,g,K0); splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=SEED).split(Z,y,g))
        log(f'loaded database n={len(y)} subjects={len(np.unique(g))} features={F0.shape[1]} eval={mask.sum()} order=relax/task arrays per subject; raw x/highdim not used')
        oofs={}; raw={}
        with threadpool_limits(limits=4):
            for name,ks in [('A',A),('D',D)]:
                for k in ks:
                    mu,sd=causal(Z,g,k); oofs[f'{name}{k}']=run_oof((Z-mu)/np.maximum(sd,.1),y,g,mask,splits,f'{name}{k}')
                raw[name]=np.nanmean([oofs[f'{name}{k}'] for k in ks],axis=0)
                log(f'{name} averaged probabilities')
        smooths={x:smooth(raw[x],g,5) for x in ('A','D')}; results={'meta':{'data':'database/windows/*/*.npz f+e','n_windows':len(y),'n_subjects':len(np.unique(g)),'warmup_windows':K0,'eval_n':int(mask.sum()),'history_excludes_current':True,'folds':'StratifiedGroupKFold(5, shuffle=True, random_state=0)','A':list(A),'D':list(D),'bootstrap_subject_reps':NBOOT}}
        for name in ('A','D'):
            results[name]={'window':metrics(raw[name],y,g,mask),'smooth5':metrics(smooths[name],y,g,mask),'direction_smooth5':direction(smooths[name],y,g,mask)}
            log(f'{name} window AUC={results[name]["window"]["AUC"]:.4f} bACC={results[name]["window"]["bACC"]:.4f}; smooth5 AUC={results[name]["smooth5"]["AUC"]:.4f} bACC={results[name]["smooth5"]["bACC"]:.4f} R->T={results[name]["direction_smooth5"]["R->T"]} T->R={results[name]["direction_smooth5"]["T->R"]}')
        results['D_vs_A_bootstrap']=boot(smooths['D'],smooths['A'],y,g,mask)
        base=run_oof(Z,y,g,mask,splits,'baseline_q'); results['baseline_scale_gain_spearman']={}
        scales=[]; gains=[]; per=[]
        for s in np.unique(g):
            ii=mask&(g==s)
            if len(np.unique(y[ii]))==2:
                a=roc_auc_score(y[ii],smooths['D'][ii]); b=roc_auc_score(y[ii],base[ii]); scales.append(float(np.mean(np.std(F0[g==s],0)))); gains.append(float(a-b)); per.append({'subject':str(s),'AUC':a,'lt_0.5':bool(a<.5),'baseline_AUC':b,'gain':a-b,'scale':scales[-1]})
        results['per_subject']=per; results['baseline_scale_gain_spearman']={'rho':float(spearmanr(scales,gains).statistic),'p':float(spearmanr(scales,gains).pvalue),'definition':'scale=mean per-feature SD of database f+e; gain=D smooth5 subject AUC minus baseline q subject AUC'}
        audits={}
        for tag,perm in [('same_label_block_swap',block_perm(y,g,True,0)),('whole_block_random_reorder',block_perm(y,g,False,0))]:
            yp,gp,Zp=y[perm],g[perm],Z[perm]; mp=positions(gp)>=K0; mup,sdp=causal(Zp,gp,50); preds=[]
            for k in D: mup,sdp=causal(Zp,gp,k); preds.append(run_oof((Zp-mup)/np.maximum(sdp,.1),yp,gp,mp,splits,f'audit/{tag}/D{k}'))
            pp=np.nanmean(preds,axis=0); sp=smooth(pp,gp,5); audits[tag]={'permutation_changed':bool(not np.array_equal(perm,np.arange(len(y)))),'metrics':metrics(sp,yp,gp,mp),'direction':direction(sp,yp,gp,mp)}
        results['D_order_audit']=audits
        np.savez_compressed(OOF,y=y,groups=g,eval_mask=mask,**oofs,A_prob=raw['A'],D_prob=raw['D'],A_smooth5=smooths['A'],D_smooth5=smooths['D'],baseline_q=base)
        results['elapsed_s']=time.perf_counter()-T0; results['database_reaches_0.9']=bool(results['D']['smooth5']['AUC']>=.9 or results['A']['smooth5']['AUC']>=.9)
        RES.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8'); log(f'done elapsed={results["elapsed_s"]:.1f}s reaches_0.9={results["database_reaches_0.9"]}')
    finally: sys.stdout=tee; tee.close()
if __name__=='__main__': main()
