"""E56: raw-waveform counterfactual swaps, fixed position labels, held-out models."""
from pathlib import Path
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'test/17_compliant')]
from eeg_decoder import load_bdf, pair_trials, read_events
from infer import PostProcessor
from streaming import window_features, SCALES
import loso as L
from multiscale import fit_mlp


def normalize(f):
    x = torch.as_tensor(f, device='cuda', dtype=torch.float32)
    parts = []
    for w in SCALES:
        m, sd = L.run_stats_t(x, w)
        parts.append(((x-m)/sd).clamp(-10, 10))
    return torch.cat(parts, 1)


def metrics(y, s):
    if not len(y):
        return dict(n=0, auc=None, f1=None, bacc=None)
    both = len(np.unique(y)) == 2
    return dict(n=len(y), auc=float(roc_auc_score(y, s)) if both else None,
                f1=float(f1_score(y, s > 0, zero_division=0)),
                bacc=float(balanced_accuracy_score(y, s > 0)) if both else None)


def prepare(sid):
    bdf = next((ROOT/'EEG_dataset'/sid).glob('*.bdf'))
    rec = load_bdf(bdf)
    fs = rec['fs']
    x, valid = rec['signal_uv']*1e-6, rec['valid']
    trials = pair_trials(read_events(bdf.parent/f'{sid}_events.csv'))
    starts = np.array([round((t['start']-rec['start_time']).total_seconds()*fs) for t in trials])
    durations = np.array([int(np.floor((t['end']-t['start']).total_seconds())) for t in trials])
    phases = [t['phase'] for t in trials]
    cache = np.load(L.SLID/f'sub-{sid}.npz')
    order = np.argsort(cache['t'])
    f, y, blk, ts = [cache[k][order] for k in ('F', 'y', 'blk', 't')]
    # Reconstruct exact event-grid endpoints, avoiding float32 timestamp rounding errors.
    offsets = np.rint(ts-np.array([(trials[b]['start']-rec['start_time']).total_seconds() for b in blk])).astype(int)
    ends = starts[blk]+offsets*fs
    lookup = {int(e): i for i, e in enumerate(ends)}
    variants = {}
    maps = {}
    audit = []
    for name in ('original', 'early_late_same_label', 'within_trial_halves', 'early_late_cross_label'):
        source = np.arange(len(x))
        swaps = []
        if name == 'early_late_same_label':
            pairs = [(0, 4), (1, 5)]
        elif name == 'early_late_cross_label':
            pairs = [(0, 5), (1, 4)]
        else:
            pairs = []
        for a, b in pairs:
            assert (phases[a] == phases[b]) == (name == 'early_late_same_label')
            length = min(durations[a], durations[b])//6*6*fs
            swaps.append((int(starts[a]), int(starts[b]), int(length)))
        if name == 'within_trial_halves':
            for start, duration in zip(starts, durations):
                length = duration//12*6*fs
                swaps.append((int(start), int(start+length), int(length)))
        for a, b, n in swaps:
            source[a:a+n], source[b:b+n] = np.arange(b, b+n), np.arange(a, a+n)
        # Actual raw samples and validity are exchanged only in memory; labels never move.
        changed, good = x[source], valid[source]
        jumps = np.flatnonzero(np.diff(source) != 1)+1
        cs = np.r_[0, np.cumsum(~good)]
        keep = (cs[ends]-cs[ends-6*fs]) == 0
        for j in jumps:
            keep &= ~((ends-6*fs < j) & (ends > j))
        features, srcrows, rows = [], [], np.flatnonzero(keep)
        reused = 0
        for i in rows:
            end = ends[i]
            srcend = int(source[end-1]+1)
            row = lookup.get(srcend)
            if row is not None:
                features.append(f[row])
                reused += 1
            else:
                features.append(window_features(changed[end-6*fs:end], fs))
            srcrows.append(-1 if row is None else row)
        # Audit cached feature identity against raw windows (three evenly spaced frames).
        errors = []
        for k in np.linspace(0, len(rows)-1, 3).astype(int):
            i = rows[k]
            actual = window_features(changed[ends[i]-6*fs:ends[i]], fs)
            errors.append(float(np.max(np.abs(actual-np.asarray(features[k]))/(1+np.abs(actual)))))
        if max(errors) > 0.01:
            raise RuntimeError(f'{sid}/{name}: raw/cache mismatch {errors}')
        variants[name] = (rows, np.asarray(features, dtype=np.float32))
        maps[name] = (rows, np.asarray(srcrows))
        audit.append(dict(subject=sid, variant=name, swaps=swaps, fs=fs,
                          retained=len(rows), excluded=len(y)-len(rows), reused_features=reused,
                          raw_feature_relative_error=max(errors)))
    common = np.logical_and.reduce([np.isin(np.arange(len(y)), rows) for rows, _ in variants.values()])
    return variants, maps, y, blk, ts, common, audit


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    outcomes = pd.read_csv(ROOT/'test/54_nonstationarity_diag/outcome_subject_regions.csv', dtype={'subject':str})
    p = outcomes[(outcomes.model == 'dt1200_hmm') & outcomes.region.isin(['first600', 'late45'])].pivot(index=['subject','group'], columns='region', values='bacc')
    selected = sorted(set(['004','009','010','050']) | set(p[(p.index.get_level_values('group') == 'low') & (p.late45-p.first600 <= -.1)].index.get_level_values('subject')))
    print('Selected:', selected, flush=True)
    subs, data = L.load()
    train_features = {q: normalize(data[q][0]) for q in subs}
    results, audit, frame_tables = [], [], []
    for q in selected:
        t0 = time.time()
        variants, maps, yq, blk, ts, common, checks = prepare(q)
        audit.extend(checks)
        train = [s for s in subs if s != q]
        x = torch.cat([train_features[s] for s in train])
        y = np.concatenate([data[s][1] for s in train])
        lr = L.fit_lr(x, y, .001)
        mlps = [fit_mlp(x, y, seed=s) for s in range(3)]
        with torch.no_grad():
            std_lr = torch.cat([lr(train_features[s]) for s in train]).std()
            std_mlp = torch.cat([torch.stack([m(train_features[s]) for m in mlps]).mean(0) for s in train]).std()
        del x
        for name, (rows, features) in variants.items():
            z = normalize(features)
            with torch.no_grad():
                score = (lr(z)/std_lr + torch.stack([m(z) for m in mlps]).mean(0)/std_mlp).cpu().numpy()
            if name == 'original':
                reference = np.load(ROOT/'test/53_e17_post_loso/e53_scores.npz')[q]
                audit.append(dict(subject=q, reference='E53_retraining_comparison',
                                  score_mae=float(np.mean(np.abs(score-reference[rows]))),
                                  score_correlation=float(np.corrcoef(score, reference[rows])[0,1])))
            post = PostProcessor(detrend_halflife=1200)
            processed = np.array([post.update(float(s)) for s in score])
            masks = {'full':np.ones(len(rows), bool), 'early01':blk[rows]<2, 'late45':np.isin(blk[rows],[4,5]), 'first600': rows<600}
            masks.update({f'block{b}':blk[rows]==b for b in np.unique(blk)})
            for model, sc in [('raw',score), ('dt1200_hmm',processed)]:
                for region, mask in masks.items():
                    for support, usable in [('available',mask), ('common',mask & common[rows])]:
                        results.append(dict(subject=q, variant=name, model=model, region=region, support=support, **metrics(yq[rows][usable],sc[usable])))
            frame_tables.append(pd.DataFrame(dict(subject=q, variant=name, original_row=rows, time_s=ts[rows], block=blk[rows], label=yq[rows], source_row=maps[name][1], common=common[rows], raw_score=score, post_score=processed)))
            del z
        r = pd.DataFrame(results)
        print(q, r[(r.subject==q)&(r.model=='dt1200_hmm')&(r.region=='late45')&(r.support=='common')][['variant','auc','f1','bacc']].to_dict('records'), f'{time.time()-t0:.0f}s', flush=True)
        del lr, mlps
        torch.cuda.empty_cache()
    table = pd.DataFrame(results)
    macro = table.groupby(['variant','model','region','support'])[['auc','f1','bacc']].mean().reset_index()
    baseline = pd.read_csv(ROOT/'test/54_nonstationarity_diag/model_macro.csv')
    table.to_csv(HERE/'subject_metrics.csv', index=False)
    macro.to_csv(HERE/'macro_metrics.csv', index=False)
    pd.concat(frame_tables,ignore_index=True).to_csv(HERE/'frames.csv.gz',index=False,compression='gzip')
    payload = dict(subjects=selected, selection='E54 low-group late45 minus first600 bACC <= -0.10, union 004/009/010/050', baseline_e54=baseline[baseline.region=='full'].to_dict('records'), audit=audit, macro=macro.to_dict('records'), metrics=results)
    (HERE/'results.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),encoding='utf-8')
    report = ['# E56 前后段原始波形交换反事实实验','', '## 方法与证据边界',
              '**这不是严格NA因果证明。** 仅在内存中交换BDF波形与缺失掩码；时间位置标签保持不变。未修改原始数据和正式推理代码。',
              f'共{len(selected)}人：'+', '.join(selected)+'。预先按E54低组后段bACC下降至少0.10选择，另强制包含004/009/010/050；存在结果驱动选择和回归均值，不是独立验证。',
              '每人重新训练排除本人的E53配置LOSO模型：LR C=.001 + 三种子MLP；训练折尺度融合；GPU训练/推理。模型在四种条件间固定。本次重训不是直接复用E53权重，数值环境/训练随机性可能使基线不同；只用本次original做配对比较，E53分数相关性/MAE另存审计。',
              '主要交换：同标签块0↔4、1↔5，交换两段共同长度向下取整为6秒的连续原始波形；不等长尾部保持原位。块内控制：每个trial的前后等长半段交换，长度为6秒整数倍。跨标签负对照：0↔5、1↔4，故意违反内容标签对应关系，不用于NA正证据。',
              '每个窗口6秒、步长1秒，采用E53事件对齐网格；排除跨任何拼接点窗口和移入缺失样本窗口。主结果用四条件共同位置交集；available为敏感性分析。推理仍在各条件全部有效窗推进，不在分段重置标准化/HMM。',
              '无拼接的窗口与原始来源窗逐样本等价，因此复用其原始878维特征（每条件3个窗对照原始波形重算）；重新执行五尺度因果标准化、分类、dt1200/HMM，绝非搬运预测分数。frames.csv.gz含位置与来源行映射；results.json含样本级交换区间。',
              '本实验对齐E53有标签trial内评估流，未纳入无标签间隔；不等同正式连续全记录流式推理基准。早段early01为块0–1，first600另列兼容E54。单标签block的AUC/bACC不可定义，留空；F1仍报告。', '', '## 当前57人E54基线（非本实验选定子集）', '|模型|组|AUC|F1|bACC|','|---|---|---:|---:|---:|']
    for row in baseline[(baseline.region=='full')&(baseline.group=='all')].to_dict('records'):
        report.append(f"|{row['model']}|all|{row['auc']:.4f}|{row['f1']:.4f}|{row['bacc']:.4f}|")
    report += ['', '## 本实验选定被试宏平均（共同位置交集）','|模型|条件|区域|AUC|F1|bACC|','|---|---|---|---:|---:|---:|']
    for row in macro[(macro.support=='common')&macro.region.isin(['full','early01','late45'])].to_dict('records'):
        report.append(f"|{row['model']}|{row['variant']}|{row['region']}|{row['auc']:.4f}|{row['f1']:.4f}|{row['bacc']:.4f}|")
    report += ['', '## 每人后段（dt1200/HMM，共同位置）','|被试|条件|n|AUC|F1|bACC|','|---|---|---:|---:|---:|---:|']
    late = table[(table.model=='dt1200_hmm')&(table.support=='common')&(table.region=='late45')]
    for row in late.to_dict('records'):
        report.append(f"|{row['subject']}|{row['variant']}|{row['n']}|{row['auc']:.4f}|{row['f1']:.4f}|{row['bacc']:.4f}|")
    piv = late.pivot(index='subject',columns='variant',values='bacc')
    delta = piv.early_late_same_label-piv.original
    report += ['', '## 结果解释',f'同标签前后交换后，后段bACC提升人数{int((delta>0).sum())}/{len(delta)}，平均变化{delta.mean():+.4f}；详见上表，未作显著性筛选。',
              '若早段内容移入后段后性能改善、后段内容移入早段后性能下降，支持内容/状态依赖，而不是仅由绝对时间位置决定。但内容包含神经、肌电、接触和设备等混合因素，不能等同生理NA。',
              '若交换后仍随时间位置退化，支持在线历史/漂移或标签范式因素；本模型无显式时间输入，时间效应只能通过因果归一化和HMM历史等路径发生。',
              '若整体混乱，尤其块内控制也大幅损害，说明交换破坏上下文，不能解释为NA消失或存在。跨标签负对照下降主要是固定标签与波形类别错配，不能当作NA证据。',
              '六秒边界剔除避免直接拼接伪影，但不消除长期在线统计/HMM历史的变化；因此内容与适应历史并未完全正交。也未随机化实际人的认知状态或测量独立NA标记。',
              '', '## 复现与产物', '` .venv\\Scripts\\python.exe test/56_waveform_swap_na/waveform_swap.py `',
              'subject_metrics.csv含每人/分段/原始及后处理/共同与可用窗口的全部指标；macro_metrics.csv含宏平均；frames.csv.gz含逐帧标签、分数和来源；results.json含完整指标与交换审计。']
    (HERE/'报告_E56_波形交换NA.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('FINAL', macro[(macro.support=='common')&(macro.model=='dt1200_hmm')&macro.region.isin(['full','early01','late45'])].to_json(orient='records'),flush=True)


if __name__ == '__main__':
    main()
