"""Small, reproducible subject-disjoint experiments on the supplied recordings."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys, json, time, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter, sosfilt, welch
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cogstate import load_trials, bad_intervals, valid_window, causal_features, _file_start
OUT = ROOT / 'test' / 'optimization_results.json'
START = time.time()

def save():
    results['elapsed_seconds'] = time.time() - START
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')

def lightweight(x, fs):
    # Work in microvolts; numerical floors must not dominate real EEG power.
    x = x * 1e6
    z = sosfilt(butter(4, [1, 45], fs=fs, btype='band', output='sos'), x - np.median(x))
    f, p = welch(z, fs=fs, nperseg=int(2 * fs), noverlap=int(fs))
    bands = [(1,4),(4,8),(8,13),(13,20),(20,30),(30,45)]
    bp = np.array([np.trapezoid(p[(f>=a)&(f<b)], f[(f>=a)&(f<b)]) for a,b in bands])
    rel = bp / max(bp.sum(), 1e-20)
    logbp = np.log(np.maximum(bp, 1e-20))
    q = p[(f>=1)&(f<45)]; q = q / max(q.sum(), 1e-20)
    var = max(np.var(z), 1e-20); d = np.diff(z); dd = np.diff(d)
    mobility = np.sqrt(np.var(d)/var)
    shape = [skew(z), kurtosis(z), mobility, np.sqrt(np.var(dd)/max(np.var(d),1e-20))/max(mobility,1e-20), np.mean(z[:-1]*z[1:])/var, -np.sum(q*np.log(np.maximum(q,1e-20))), np.sum(q*f[(f>=1)&(f<45)]), np.median(np.abs(z-np.median(z)))/np.sqrt(var)]
    amp = [np.log(var), np.log(max(np.ptp(z),1e-20)), np.log(max(np.quantile(z,.95)-np.quantile(z,.05),1e-20))]
    return np.nan_to_num(np.r_[logbp, rel, logbp-logbp.mean(), shape, amp]).astype('float32')

cache = np.load(ROOT / 'test' / 'features.npz')
X, y, groups = cache['X'], cache['y'], cache['groups'].astype(str)
results = {'status':'running', 'protocol':{'cv':'StratifiedGroupKFold', 'n_splits':5, 'shuffle':True, 'random_state':0, 'group':'subject', 'holdout':False, 'warning':'Exploratory model selection on the same CV; not an unbiased final generalization estimate.', 'normalization':'All learned scaling and feature selection fitted only on training subjects. No held-out-subject/session statistics or future samples used.', 'cache_sha256':hashlib.sha256((ROOT/'test'/'features.npz').read_bytes()).hexdigest()}, 'audit':[], 'candidates':{}}
print('Cache', X.shape, 'subjects', len(set(groups)), flush=True)
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X,y,groups))
valid_subjects = [set(groups[v]) for _,v in splits]
arrays = {k:[] for k in ['new','new_offset2','old_offset2']}
labels = {k:[] for k in arrays}; subjects = {k:[] for k in arrays}
rebuilt = []; rebuilt_y=[]; rebuilt_g=[]
for i, csv in enumerate(sorted((ROOT/'EEG_dataset').glob('*/*_events.csv')),1):
    bdf = next(csv.parent.glob('*.bdf')); sid=csv.parent.name
    raw=mne.io.read_raw_bdf(str(bdf), preload=True, verbose='ERROR')
    fs=float(raw.info['sfreq']); duration=raw.n_times/fs; bad=bad_intervals(raw)
    ev=pd.read_csv(csv); trials=load_trials(str(csv),str(bdf))
    timestamps=pd.to_datetime(ev.local_time, utc=True).astype('int64').to_numpy()/1e9-_file_start(str(bdf))
    row={'subject':sid, 'record_seconds':duration,'sfreq':fs,'channels':raw.ch_names,'bad_intervals':bad, 'events_min_seconds':float(timestamps.min()), 'events_max_seconds':float(timestamps.max()),'events_outside_record':int(np.sum((timestamps<0)|(timestamps>duration))), 'all_explicit_plus08':bool(ev.local_time.str.endswith('+08:00').all()), 'header_clock_minus_filename_seconds':raw.info['meas_date'].timestamp()-(_file_start(str(bdf))+8*3600), 'trial_label_counts':dict(Counter(str(t.label) for t in trials)), 'trials':[]}
    for tr in trials:
        trrow={'trial':tr.trial_id,'label':tr.label,'start':tr.start,'end':tr.end,'duration':tr.end-tr.start,'outside_record':tr.start<0 or tr.end>duration,'windows':{}}
        for offset in [0,2]:
            count=0
            for a in np.arange(tr.start+offset,tr.end-6+1e-6,6):
                z=a+6
                if not valid_window(a,z,bad,duration): continue
                s,e=int(round(a*fs)),int(round(z*fs)); x=raw.get_data(start=s,stop=e)[0]
                if len(x)!=e-s or not np.isfinite(x).all(): continue
                old=causal_features(x,fs)
                key='new' if offset==0 else 'new_offset2'
                arrays[key].append(lightweight(x,fs)); labels[key].append(tr.label); subjects[key].append(sid)
                if offset==0:
                    rebuilt.append(old); rebuilt_y.append(tr.label); rebuilt_g.append(sid)
                else:
                    arrays['old_offset2'].append(old); labels['old_offset2'].append(tr.label); subjects['old_offset2'].append(sid)
                count+=1
            trrow['windows'][str(offset)]=count
        row['trials'].append(trrow)
    raw.close(); results['audit'].append(row)
    print(f'Audit/extract {i}/57 subject={sid}, elapsed={time.time()-START:.1f}s',flush=True)
    save()
rebuilt=np.asarray(rebuilt)
results['cache_validation']={'shape':list(rebuilt.shape),'labels_equal':bool(np.array_equal(y,rebuilt_y)),'groups_equal':bool(np.array_equal(groups,rebuilt_g)), 'features_equal':bool(rebuilt.shape==X.shape and np.allclose(X,rebuilt,rtol=1e-5,atol=1e-10))}
assert all(results['cache_validation'][k] for k in ['labels_equal','groups_equal','features_equal']), 'Cache does not match current extraction'
for k in arrays:
    arrays[k]=np.asarray(arrays[k]); labels[k]=np.asarray(labels[k]); subjects[k]=np.asarray(subjects[k])
arrays['existing']=X; labels['existing']=y; subjects['existing']=groups
amplitude=[]
for sid in sorted(set(groups)):
    xx=arrays['new'][groups==sid]; yy=y[groups==sid]
    amplitude.append({'subject':sid,'median_log_variance':float(np.median(xx[:,-3])), 'task_minus_relax_median_log_variance':float(np.median(xx[yy==1,-3])-np.median(xx[yy==0,-3]))})
results['amplitude_bias']={'subjects':amplitude,'between_subject_sd_log_variance':float(np.std([r['median_log_variance'] for r in amplitude])), 'median_absolute_within_subject_class_difference':float(np.median([abs(r['task_minus_relax_median_log_variance']) for r in amplitude])), 'per_subject_robust_normalization':'Not evaluated for official target: unseen subjects have no training-fold subject statistics; full-session normalization would use future/test data.', 'session_limitation':'One recording per subject; subject and session effects cannot be separated.'}

def evaluate(name,key,model,cols=None):
    xx=arrays[key] if cols is None else arrays[key][:,cols]; yy=labels[key]; gg=subjects[key]
    pred=np.zeros(len(yy)); decisions=np.zeros(len(yy)); folds=[]; t=time.time()
    for fold, held in enumerate(valid_subjects,1):
        va=np.flatnonzero(np.isin(gg,list(held))); tr=np.flatnonzero(~np.isin(gg,list(held)))
        assert not set(gg[va]) & set(gg[tr])
        with threadpool_limits(limits=1):
            model.fit(xx[tr],yy[tr]); p=model.decision_function(xx[va]); yh=model.predict(xx[va])
        pred[va]=yh; decisions[va]=p
        folds.append({'fold':fold,'AUC':float(roc_auc_score(yy[va],p)),'bACC':float(balanced_accuracy_score(yy[va],yh)), 'validation_subjects':sorted(held), 'n_train':len(tr),'n_valid':len(va)})
        print(f'{name} fold {fold}/5 AUC={folds[-1]["AUC"]:.4f} bACC={folds[-1]["bACC"]:.4f} elapsed={time.time()-t:.1f}s',flush=True)
    result={'dataset':key,'n_samples':len(yy),'n_features':xx.shape[1], 'AUC':float(roc_auc_score(yy,decisions)),'bACC':float(balanced_accuracy_score(yy,pred)), 'fold_AUC_mean':float(np.mean([f['AUC'] for f in folds])), 'folds':folds,'seconds':time.time()-t,'subject_AUC_mean':float(np.mean([roc_auc_score(yy[gg==s],decisions[gg==s]) for s in set(gg) if len(set(yy[gg==s]))==2]))}
    results['candidates'][name]=result; save()

for c in [.001,.01,.1,1,10]:
    evaluate(f'existing_standard_logistic_C{c}','existing',make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1500,class_weight='balanced')))
evaluate('existing_robust_logistic_C01','existing',make_pipeline(RobustScaler(),LogisticRegression(C=.1,max_iter=1500,class_weight='balanced')))
evaluate('existing_select12_logistic','existing',make_pipeline(StandardScaler(),SelectKBest(f_classif,k=12),LogisticRegression(C=.1,max_iter=1500,class_weight='balanced')))
evaluate('existing_shrinkage_LDA','existing',make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')))
evaluate('existing_rbf_C1','existing',make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',class_weight='balanced',cache_size=512)))
# Predeclared shape-only subset excludes absolute power/amplitude and uses only this window.
shape_cols=list(range(6,26))
for key in ['new','new_offset2','old_offset2']:
    evaluate(key+'_logistic_C01',key,make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,class_weight='balanced')))
evaluate('new_shape_only_logistic','new',make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,class_weight='balanced')),shape_cols)
evaluate('new_shrinkage_LDA','new',make_pipeline(StandardScaler(),LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')))
evaluate('new_rbf_C1','new',make_pipeline(StandardScaler(),SVC(C=1,class_weight='balanced',cache_size=512)))
rows=results['candidates']; names=list(rows); pos=np.arange(len(names))
fig,ax=plt.subplots(figsize=(11,8)); ax.barh(pos-.18,[rows[n]['AUC'] for n in names],.35,label='OOF AUC'); ax.barh(pos+.18,[rows[n]['bACC'] for n in names],.35,label='OOF bACC'); ax.set_yticks(pos,names); ax.set_xlim(.45,1); ax.axvline(.9,color='red',linestyle='--'); ax.legend(); ax.set_title('Real EEG: subject-disjoint exploratory CV (not final test)'); fig.tight_layout(); fig.savefig(ROOT/'test'/'optimization_comparison.png',dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(12,4)); ax.plot([r['subject'] for r in amplitude],[r['median_log_variance'] for r in amplitude],'.-'); ax.tick_params(axis='x',rotation=90); ax.set_ylabel('Median log variance (microvolt squared)'); ax.set_title('Subject/session amplitude variability'); fig.tight_layout(); fig.savefig(ROOT/'test'/'optimization_amplitude.png',dpi=150); plt.close(fig)
results['status']='complete'; results['best_by_AUC']=max(rows,key=lambda k:rows[k]['AUC']); results['best_by_bACC']=max(rows,key=lambda k:rows[k]['bACC']); results['conclusion']='No evidence that 0.9 AUC/bACC is achievable under unseen-subject, causal 6-second constraints with these candidates. Repeated CV tuning is optimistic; no official-model replacement made.'; save()
print('COMPLETE', results['best_by_AUC'], rows[results['best_by_AUC']],flush=True)
