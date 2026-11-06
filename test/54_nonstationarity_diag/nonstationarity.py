"""E54: label-free BDF signal diagnostics with a separate E53 outcome layer."""
from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from eeg_decoder import load_bdf, pair_trials, read_events
from infer import PostProcessor

SLID = ROOT / 'test/11_individual_20261102/sliding1'
BANDS = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
FEATURES = ['mean_uv', 'std_uv', 'rms_uv', 'ptp_uv', 'flat_fraction',
            *[f'log10_{b}_power' for b in BANDS], 'line50_ratio', 'line50_neighbor_ratio',
            'high45_100_ratio', 'spectral_centroid_hz', 'adjacent_mean_change_uv', 'adjacent_logpower_change']
QUALITY = ['valid_fraction', 'missing_fraction', 'frame_valid', 'model_valid']


def extract(x, valid, ends, fs):
    """No labels, phase or group are passed into signal feature extraction."""
    samples = ends[:, None] - np.arange(fs, 0, -1)[None, :]
    inside = (samples >= 0) & (samples < len(x))
    ix = np.clip(samples, 0, len(x) - 1)
    good = inside & valid[ix]
    frac = good.mean(1)
    out = pd.DataFrame({'valid_fraction': frac, 'missing_fraction': 1-frac, 'frame_valid': frac == 1})
    cs = np.r_[0, np.cumsum(~valid)]
    starts6 = ends - 6*fs
    out['model_valid'] = ((starts6 >= 0) & (ends <= len(x)) &
                          ((cs[np.clip(ends, 0, len(x))] - cs[np.clip(starts6, 0, len(x))]) == 0))
    ok = frac == 1
    z = x[ix[ok]]
    for col in FEATURES:
        out[col] = np.nan
    if not np.any(ok):
        return out
    out.loc[ok, 'mean_uv'] = z.mean(1)
    out.loc[ok, 'std_uv'] = z.std(1)
    out.loc[ok, 'rms_uv'] = np.sqrt(np.mean(z*z, axis=1))
    out.loc[ok, 'ptp_uv'] = np.ptp(z, axis=1)
    out.loc[ok, 'flat_fraction'] = np.mean(np.diff(z, axis=1) == 0, axis=1)
    f, p = periodogram(z, fs=fs, window='hann', detrend='constant', axis=1)
    power = lambda a, b: p[:, (f >= a) & (f < b)].sum(1) * (f[1]-f[0])
    total = power(1, 101) + 1e-20
    for band, (a, b) in BANDS.items():
        out.loc[ok, f'log10_{band}_power'] = np.log10(power(a, b) + 1e-20)
    out.loc[ok, 'line50_ratio'] = power(49, 52)/total
    out.loc[ok, 'line50_neighbor_ratio'] = (power(49, 52)/3)/(power(46, 49)/3 + power(52, 55)/3 + 1e-20)*2
    out.loc[ok, 'high45_100_ratio'] = power(45, 101)/total
    m = (f >= 1) & (f <= 100)
    out.loc[ok, 'spectral_centroid_hz'] = (p[:, m]*f[m]).sum(1)/(p[:, m].sum(1)+1e-20)
    out['adjacent_mean_change_uv'] = out.mean_uv.diff().abs()
    out['adjacent_logpower_change'] = out[[f'log10_{b}_power' for b in BANDS]].diff().abs().mean(1)
    return out


def metrics(y, s):
    pred = s > 0
    return {'auc': float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
            'f1': float(f1_score(y, pred, zero_division=0)),
            'bacc': float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else np.nan,
            'error_rate': float(np.mean(pred != y))}


def bh(df):
    df = df.copy()
    p = df.p.to_numpy()
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    q = np.full(len(p), np.nan)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(order)/np.arange(1, len(order)+1))[::-1])[::-1])
    df['q_bh'] = q
    return df


def main():
    groups = pd.read_csv(ROOT/'test/43_group_analysis/group_members.csv', dtype={'subject': str}).set_index('subject').group.to_dict()
    scores = np.load(ROOT/'test/53_e17_post_loso/e53_scores.npz')
    frames, summaries, outcomes, audits = [], [], [], []
    for bdf in sorted((ROOT/'EEG_dataset').glob('*/*.bdf')):
        sid = bdf.parent.name
        rec = load_bdf(bdf)
        trials = pair_trials(read_events(bdf.parent/f'{sid}_events.csv'))
        cache = np.load(SLID/f'sub-{sid}.npz')
        order = np.argsort(cache['t'])
        times, blocks, y = cache['t'][order], cache['blk'][order], cache['y'][order]
        ends_cache = np.rint(times*rec['fs']).astype(int)
        chunks = []
        for block, tr in enumerate(trials):
            start = (tr['start']-rec['start_time']).total_seconds()
            end = (tr['end']-rec['start_time']).total_seconds()
            ts = start + np.arange(1, int(np.floor(end-start+1e-6))+1)
            ends = np.rint(ts*rec['fs']).astype(int)
            df = extract(rec['signal_uv'], rec['valid'], ends, rec['fs'])
            df['subject'], df['group'], df['block'], df['end_s'] = sid, groups[sid], block, ts
            df['end_sample'] = ends
            df['model_valid'] &= ts >= start+6-1e-6
            chunks.append(df)
        df = pd.concat(chunks, ignore_index=True)
        # Cache times are float32; match actual event-grid times before rounding samples.
        index = pd.Index(df.end_s).get_indexer(times.astype(float), method='nearest', tolerance=0.001)
        assert np.all(index >= 0), f'{sid}: cache/event time mismatch'
        aligned = df.iloc[index].reset_index(drop=True)
        assert len(aligned) == len(scores[sid]) and np.all(aligned.block.to_numpy() == blocks)
        assert aligned.frame_valid.all()
        # Label use begins only here, after every signal feature has been computed.
        post = PostProcessor(detrend_halflife=1200)
        variants = {'raw': scores[sid], 'dt1200_hmm': np.array([post.update(float(s)) for s in scores[sid]])}
        selections = {'full': np.ones(len(y), bool), 'first600': np.arange(len(y)) < 600,
                      'late45': np.isin(blocks, [4, 5])}
        selections.update({f'block{k}': blocks == k for k in np.unique(blocks)})
        for region, mask in selections.items():
            a = aligned.loc[mask]
            row = {'subject': sid, 'group': groups[sid], 'region': region, 'n_frames': len(a)}
            row.update(a[FEATURES].median().to_dict())
            # Quality denominator includes missing frames, unlike cached model frames.
            if region == 'first600':
                q = df[df.end_s <= a.end_s.max()]
            elif region == 'late45':
                q = df[df.block.isin([4, 5])]
            elif region.startswith('block'):
                q = df[df.block == int(region[5:])]
            else:
                q = df
            row.update(q[QUALITY].mean().to_dict())
            row['n_candidate_frames'] = len(q)
            summaries.append(row)
            for model, score in variants.items():
                outcomes.append({'subject': sid, 'group': groups[sid], 'region': region, 'model': model,
                                 'n_frames': int(mask.sum()), **metrics(y[mask], score[mask])})
        audits.append({'subject': sid, 'n_samples': rec['n_samples'], 'bad_samples': int((~rec['valid']).sum()),
                       'candidate_frames': len(df), 'model_frames': len(aligned),
                       'candidate_model_valid': int(df.model_valid.sum()), 'trials': len(trials)})
        frames.append(df)
        print(f'{sid}: {len(df)} candidate / {len(aligned)} model frames', flush=True)

    summary = pd.DataFrame(summaries)
    outcome = pd.DataFrame(outcomes)
    # Independent inferential unit is the subject, never the individual second.
    cols = FEATURES + QUALITY
    early = summary[summary.region == 'first600'].set_index('subject')
    late = summary[summary.region == 'late45'].set_index('subject')
    delta = late[cols]-early[cols]
    delta['group'] = early.group
    rows = []
    for feature in cols:
        for group in ['high', 'low']:
            v = delta.loc[delta.group == group, feature].dropna().to_numpy()
            p = wilcoxon(v).pvalue if np.any(v != 0) else 1.0
            rows.append({'feature': feature, 'test': f'{group}_late_minus_early', 'n': len(v),
                         'effect_median': float(np.median(v)), 'p': float(p)})
        lo = delta.loc[delta.group == 'low', feature].dropna()
        hi = delta.loc[delta.group == 'high', feature].dropna()
        rows.append({'feature': feature, 'test': 'low_vs_high_change', 'n': len(lo)+len(hi),
                     'effect_median': float(lo.median()-hi.median()), 'p': float(mannwhitneyu(lo, hi).pvalue)})
    tests = bh(pd.DataFrame(rows))
    # Same protocol-position comparisons reduce state-composition confounding.
    same = []
    for a, b in [(0, 4), (1, 5)]:
        aa = summary[summary.region == f'block{a}'].set_index('subject')
        bb = summary[summary.region == f'block{b}'].set_index('subject')
        dd = bb[cols]-aa[cols]
        dd['group'] = aa.group
        for feature in cols:
            lo, hi = dd.loc[dd.group == 'low', feature].dropna(), dd.loc[dd.group == 'high', feature].dropna()
            same.append({'feature': feature, 'pair': f'{b}-{a}', 'low_median_change': float(lo.median()),
                         'high_median_change': float(hi.median()), 'p': float(mannwhitneyu(lo, hi).pvalue)})
    sensitivity = bh(pd.DataFrame(same))
    associations = []
    for model in variants:
        o = outcome[outcome.model == model]
        err = o[o.region == 'late45'].set_index('subject').error_rate-o[o.region == 'first600'].set_index('subject').error_rate
        delta[f'{model}_error_change'] = err
        for group in ['all', 'high', 'low']:
            d = delta if group == 'all' else delta[delta.group == group]
            for feature in cols:
                v = d[[feature, f'{model}_error_change']].dropna()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r, p = spearmanr(v.iloc[:, 0], v.iloc[:, 1])
                associations.append({'model': model, 'group': group, 'feature': feature, 'n': len(v), 'rho': float(r), 'p': float(p)})
    associations = bh(pd.DataFrame(associations))
    macro = outcome.groupby(['model', 'group', 'region'])[['auc', 'f1', 'bacc', 'error_rate']].mean().reset_index()
    all_macro = outcome.groupby(['model', 'region'])[['auc', 'f1', 'bacc', 'error_rate']].mean().reset_index()
    all_macro['group'] = 'all'
    macro = pd.concat([macro, all_macro], ignore_index=True)
    group_summary = summary.groupby(['group', 'region'])[cols].median().reset_index()
    result = {'subjects': len(audits), 'groups': pd.Series(groups).value_counts().to_dict(),
              'candidate_frames': sum(a['candidate_frames'] for a in audits),
              'model_frames': sum(a['model_frames'] for a in audits),
              'baseline': macro[macro.region == 'full'].to_dict('records'),
              'signal_tests': tests.to_dict('records'), 'associations': associations.to_dict('records'),
              'audit': audits, 'evidence': '探索性信号分布非平稳/误差关联；不证明生理来源或因果。'}
    report = ['# E54 无标签信号非平稳诊断', '', '## 结论与证据边界',
              '本实验直接读取57份BDF，未修改原始数据。信号特征不使用标签、预测或高低组；仅在提取后关联既有分组、时间块与LOSO误差。',
              '证据等级：一级为描述性信号变化，二级为被试层面的探索性统计关联；本分析最多支持这两级，不能达到生理机制验证或因果证据。',
              '高低组由既有模型表现事后划分，存在选择偏倚；不能把低组误差下降与组别差异当成独立验证。',
              '既有E53报告将疲劳/注意力/阻抗漂移直接认定为原因的表述超出了证据：这里没有疲劳、阻抗、眼动或肌电独立测量。', '',
              '## 当前模型基线（57折LOSO，逐被试宏平均）',
              '|模型|组|区域|AUC|F1|bACC|错误率|', '|---|---|---|---:|---:|---:|---:|']
    for r in macro[macro.region.isin(['full', 'first600', 'late45'])].to_dict('records'):
        report.append(f"|{r['model']}|{r['group']}|{r['region']}|{r['auc']:.4f}|{r['f1']:.4f}|{r['bacc']:.4f}|{r['error_rate']:.4f}|")
    report += ['', '## 信号变化（每人后段减前段，再汇总）',
               '主比较为前600个有效模型帧与零基块4–5（第5–6段）。以下列出全部组间变化检验，非按显著性筛选。',
               '|特征|低组变化中位数|高组变化中位数|组间p|全族BH q|', '|---|---:|---:|---:|---:|']
    for feature in cols:
        t = tests[(tests.feature == feature) & (tests.test == 'low_vs_high_change')].iloc[0]
        report.append(f"|{feature}|{delta.loc[delta.group == 'low', feature].median():.5g}|{delta.loc[delta.group == 'high', feature].median():.5g}|{t.p:.4g}|{t.q_bh:.4g}|")
    sig = tests[(tests.test == 'low_vs_high_change') & (tests.q_bh < .05)]
    assoc_sig = associations[associations.q_bh < .05]
    report += ['', f'组间漂移差异通过BH q<0.05的指标：{len(sig)}/{len(cols)}。误差变化关联通过BH的检验：{len(assoc_sig)}/{len(associations)}。',
               '## 误差关联（预先全列检验，以下展示低组两模型全部结果）',
               '|模型|特征|Spearman rho|p|BH q|', '|---|---|---:|---:|---:|']
    for r in associations[associations.group == 'low'].to_dict('records'):
        report.append(f"|{r['model']}|{r['feature']}|{r['rho']:.4f}|{r['p']:.4g}|{r['q_bh']:.4g}|")
    report += ['', '## 方法与解释限制',
               '- 从文件名毫秒时间及真实trial起止建立1秒帧；不假设每段300秒。每帧仅用末端之前1秒，模型为6秒窗/1秒步长，信号汇总匹配缓存模型有效帧。',
               '- BAD_ACQ_SKIP和非有限值均为无效，不插补。含缺失的1秒帧只计算有效/缺失率，幅值/频谱记缺失；有效率分母包括候选无效帧。前600帧质量分母覆盖到第600模型帧的实际时间，可能超过600秒。',
               '- 时域单位为微伏，功率为微伏平方，频段值为log10功率。Hann周期图1 Hz分辨率，delta=1–<4、theta=4–<8、alpha=8–<13、beta=13–<30、gamma=30–<45 Hz。50 Hz占比用49–51 Hz/1–100 Hz；邻域比比较46–48及52–54 Hz。',
               '- 相邻帧变化仅在同trial相邻有效1秒帧定义，遇缺失自然断开。PSD未陷波，便于检查工频；1秒频谱方差大且无法测<1 Hz活动。',
               '- 每人每区域信号用帧中位数，质量用候选帧均值；组内Wilcoxon、组间Mann–Whitney；主信号检验全族BH，敏感性比较单独BH，误差关联全族BH。秒级帧不当作独立样本。',
               '- sensitivity_same_position.csv比较块4−0、5−1，控制重复范式位置但不消除时间、行为、采集伪迹等混杂。标签只用于结果层AUC/F1/误差，不用于信号特征。',
               '- dt1200_hmm为每被试一个持续PostProcessor实例；绝不在每帧重新初始化。阈值为分数>0，未重新训练、调参或按标签选择特征。',
               '- 时间趋势、信号幅度/频谱变化不等同于疲劳或注意力变化；工频、电极、运动伪迹也能产生非平稳。无显著差异不等于稳定，且功效受57人及事后分组限制。',
               '- 本脚本为离线诊断，不是提交推理模块；区域汇总使用完整区域，仅用于研究。', '',
               '## 复现与产物', '` .venv\\Scripts\\python.exe test/54_nonstationarity_diag/nonstationarity.py `',
               'frames.csv.gz：全部候选1秒帧无标签特征；subject_regions.csv：被试/时间块汇总；group_regions.csv：组汇总；',
               'outcome_subject_regions.csv、model_macro.csv：结果层；subject_changes.csv：逐被试变化；signal_tests.csv、sensitivity_same_position.csv、error_associations.csv：统计；results.json：摘要和审计。']
    # Write deliverables only after all analysis has completed.
    pd.concat(frames, ignore_index=True).to_csv(HERE/'frames.csv.gz', index=False, compression='gzip')
    for name, table in [('subject_regions', summary), ('group_regions', group_summary), ('outcome_subject_regions', outcome),
                        ('model_macro', macro), ('signal_tests', tests), ('sensitivity_same_position', sensitivity),
                        ('error_associations', associations), ('subject_changes', delta.reset_index())]:
        table.to_csv(HERE/f'{name}.csv', index=False, encoding='utf-8-sig')
    clean = json.loads(json.dumps(result, default=lambda x: x.item() if hasattr(x, 'item') else str(x)))
    def finite(obj):
        if isinstance(obj, dict): return {k: finite(v) for k, v in obj.items()}
        if isinstance(obj, list): return [finite(v) for v in obj]
        if isinstance(obj, float) and not np.isfinite(obj): return None
        return obj
    (HERE/'results.json').write_text(json.dumps(finite(clean), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (HERE/'报告_E54_非平稳诊断.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(macro[macro.region.isin(['full', 'first600', 'late45'])].to_string(index=False), flush=True)
    print('SIGNAL BETWEEN-GROUP BH:', sig.to_string(index=False), flush=True)
    print('ASSOCIATION BH:', assoc_sig.to_string(index=False), flush=True)
    print('COUNTS', result['subjects'], result['candidate_frames'], result['model_frames'], flush=True)


if __name__ == '__main__':
    main()
