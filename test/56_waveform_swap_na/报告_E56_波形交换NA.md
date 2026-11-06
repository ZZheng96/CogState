# E56 前后段原始波形交换反事实实验

## 方法与证据边界
**这不是严格NA因果证明。** 仅在内存中交换BDF波形与缺失掩码；时间位置标签保持不变。未修改原始数据和正式推理代码。
共18人：002, 004, 006, 008, 009, 010, 012, 015, 021, 022, 023, 029, 037, 039, 042, 050, 053, 059。预先按E54低组后段bACC下降至少0.10选择，另强制包含004/009/010/050；存在结果驱动选择和回归均值，不是独立验证。
每人重新训练排除本人的E53配置LOSO模型：LR C=.001 + 三种子MLP；训练折尺度融合；GPU训练/推理。模型在四种条件间固定。本次重训不是直接复用E53权重，数值环境/训练随机性可能使基线不同；只用本次original做配对比较，E53分数相关性/MAE另存审计。
主要交换：同标签块0↔4、1↔5，交换两段共同长度向下取整为6秒的连续原始波形；不等长尾部保持原位。块内控制：每个trial的前后等长半段交换，长度为6秒整数倍。跨标签负对照：0↔5、1↔4，故意违反内容标签对应关系，不用于NA正证据。
每个窗口6秒、步长1秒，采用E53事件对齐网格；排除跨任何拼接点窗口和移入缺失样本窗口。主结果用四条件共同位置交集；available为敏感性分析。推理仍在各条件全部有效窗推进，不在分段重置标准化/HMM。
无拼接的窗口与原始来源窗逐样本等价，因此复用其原始878维特征（每条件3个窗对照原始波形重算）；重新执行五尺度因果标准化、分类、dt1200/HMM，绝非搬运预测分数。frames.csv.gz含位置与来源行映射；results.json含样本级交换区间。
本实验对齐E53有标签trial内评估流，未纳入无标签间隔；不等同正式连续全记录流式推理基准。早段early01为块0–1，first600另列兼容E54。单标签block的AUC/bACC不可定义，留空；F1仍报告。

## 当前57人E54基线（非本实验选定子集）
|模型|组|AUC|F1|bACC|
|---|---|---:|---:|---:|
|dt1200_hmm|all|0.8315|0.7742|0.7824|
|raw|all|0.8191|0.7481|0.7617|

## 本实验选定被试宏平均（共同位置交集）
|模型|条件|区域|AUC|F1|bACC|
|---|---|---|---:|---:|---:|
|dt1200_hmm|early_late_cross_label|early01|0.6897|0.6460|0.7090|
|dt1200_hmm|early_late_cross_label|full|0.5415|0.5444|0.5646|
|dt1200_hmm|early_late_cross_label|late45|0.2695|0.3463|0.3414|
|raw|early_late_cross_label|early01|0.6580|0.5885|0.6632|
|raw|early_late_cross_label|full|0.5282|0.5106|0.5415|
|raw|early_late_cross_label|late45|0.2853|0.3472|0.3487|
|dt1200_hmm|early_late_same_label|early01|0.6281|0.5100|0.6485|
|dt1200_hmm|early_late_same_label|full|0.5523|0.5522|0.5625|
|dt1200_hmm|early_late_same_label|late45|0.4999|0.5744|0.5141|
|raw|early_late_same_label|early01|0.5906|0.4597|0.6026|
|raw|early_late_same_label|full|0.5447|0.5193|0.5490|
|raw|early_late_same_label|late45|0.4997|0.5543|0.5159|
|dt1200_hmm|original|early01|0.8544|0.7852|0.8004|
|dt1200_hmm|original|full|0.6570|0.6103|0.6395|
|dt1200_hmm|original|late45|0.4270|0.3976|0.4556|
|raw|original|early01|0.8011|0.7055|0.7389|
|raw|original|full|0.6409|0.5752|0.6171|
|raw|original|late45|0.4262|0.3862|0.4514|
|dt1200_hmm|within_trial_halves|early01|0.6445|0.5739|0.6224|
|dt1200_hmm|within_trial_halves|full|0.5711|0.5277|0.5564|
|dt1200_hmm|within_trial_halves|late45|0.3707|0.3840|0.4033|
|raw|within_trial_halves|early01|0.6205|0.5417|0.6044|
|raw|within_trial_halves|full|0.5649|0.5065|0.5505|
|raw|within_trial_halves|late45|0.3920|0.3834|0.4200|

## 每人后段（dt1200/HMM，共同位置）
|被试|条件|n|AUC|F1|bACC|
|---|---|---:|---:|---:|---:|
|002|original|418|0.6808|0.5260|0.6467|
|002|early_late_same_label|418|0.7241|0.7441|0.6776|
|002|within_trial_halves|418|0.7502|0.7756|0.7847|
|002|early_late_cross_label|418|0.2831|0.4750|0.3591|
|004|original|406|0.5300|0.6573|0.6407|
|004|early_late_same_label|406|0.3146|0.3776|0.3427|
|004|within_trial_halves|406|0.0107|0.0401|0.0566|
|004|early_late_cross_label|406|0.1863|0.1618|0.2850|
|006|original|385|0.7196|0.6872|0.7009|
|006|early_late_same_label|385|0.3786|0.5075|0.3738|
|006|within_trial_halves|385|0.8970|0.8376|0.8268|
|006|early_late_cross_label|385|0.0607|0.1766|0.0916|
|008|original|470|0.2876|0.1184|0.3673|
|008|early_late_same_label|470|0.7011|0.5929|0.6118|
|008|within_trial_halves|470|0.1925|0.0411|0.2773|
|008|early_late_cross_label|470|0.0824|0.0000|0.1111|
|009|original|219|nan|0.0957|nan|
|009|early_late_same_label|219|nan|0.9908|nan|
|009|within_trial_halves|219|nan|0.5449|nan|
|009|early_late_cross_label|219|nan|0.5733|nan|
|010|original|510|0.4195|0.2559|0.4356|
|010|early_late_same_label|510|0.2596|0.4618|0.3954|
|010|within_trial_halves|510|0.6937|0.2899|0.5123|
|010|early_late_cross_label|510|0.5515|0.6444|0.6253|
|012|original|422|0.8405|0.7611|0.7277|
|012|early_late_same_label|422|0.9786|0.9006|0.8846|
|012|within_trial_halves|422|0.5152|0.5240|0.5333|
|012|early_late_cross_label|422|0.2048|0.2120|0.4919|
|015|original|397|0.0098|0.0215|0.1128|
|015|early_late_same_label|397|0.6252|0.7380|0.5854|
|015|within_trial_halves|397|0.0255|0.0619|0.0993|
|015|early_late_cross_label|397|0.2751|0.5257|0.3026|
|021|original|488|0.5159|0.5109|0.4942|
|021|early_late_same_label|488|0.3831|0.5083|0.3954|
|021|within_trial_halves|488|0.6372|0.5323|0.5107|
|021|early_late_cross_label|488|0.0293|0.2811|0.1835|
|022|original|550|0.6272|0.7344|0.6892|
|022|early_late_same_label|550|0.8675|0.8614|0.8482|
|022|within_trial_halves|550|0.8266|0.8084|0.7669|
|022|early_late_cross_label|550|0.7604|0.6572|0.6939|
|023|original|559|0.0420|0.3890|0.2419|
|023|early_late_same_label|559|0.2308|0.3746|0.3489|
|023|within_trial_halves|559|0.0677|0.3631|0.2222|
|023|early_late_cross_label|559|0.4846|0.5570|0.5278|
|029|original|530|0.1597|0.1603|0.2111|
|029|early_late_same_label|530|0.0459|0.0000|0.1512|
|029|within_trial_halves|530|0.0085|0.1661|0.0882|
|029|early_late_cross_label|530|0.6702|0.7285|0.7060|
|037|original|484|0.4279|0.3837|0.4326|
|037|early_late_same_label|484|0.3366|0.1907|0.3416|
|037|within_trial_halves|484|0.2464|0.2261|0.4342|
|037|early_late_cross_label|484|0.4965|0.4664|0.5055|
|039|original|544|0.6413|0.5980|0.5551|
|039|early_late_same_label|544|0.8895|0.8016|0.8162|
|039|within_trial_halves|544|0.2879|0.3469|0.3217|
|039|early_late_cross_label|544|0.1865|0.1121|0.3015|
|042|original|496|0.0673|0.0000|0.2500|
|042|early_late_same_label|496|0.1961|0.5246|0.4016|
|042|within_trial_halves|496|0.1700|0.1503|0.2462|
|042|early_late_cross_label|496|0.0259|0.2066|0.1346|
|050|original|484|0.0350|0.1277|0.0696|
|050|early_late_same_label|484|0.3952|0.5607|0.4952|
|050|within_trial_halves|484|0.4350|0.4199|0.4455|
|050|early_late_cross_label|484|0.0227|0.0000|0.0709|
|053|original|530|0.7469|0.7222|0.6963|
|053|early_late_same_label|530|0.6135|0.6571|0.5411|
|053|within_trial_halves|530|0.2970|0.5108|0.3943|
|053|early_late_cross_label|530|0.1063|0.2226|0.1271|
|059|original|545|0.5082|0.4074|0.4735|
|059|early_late_same_label|545|0.5580|0.5461|0.5298|
|059|within_trial_halves|545|0.2416|0.2725|0.3354|
|059|early_late_cross_label|545|0.1555|0.2323|0.2856|

## 结果解释
同标签前后交换后，后段bACC提升人数10/18，平均变化+0.0585；详见上表，未作显著性筛选。
若早段内容移入后段后性能改善、后段内容移入早段后性能下降，支持内容/状态依赖，而不是仅由绝对时间位置决定。但内容包含神经、肌电、接触和设备等混合因素，不能等同生理NA。
若交换后仍随时间位置退化，支持在线历史/漂移或标签范式因素；本模型无显式时间输入，时间效应只能通过因果归一化和HMM历史等路径发生。
若整体混乱，尤其块内控制也大幅损害，说明交换破坏上下文，不能解释为NA消失或存在。跨标签负对照下降主要是固定标签与波形类别错配，不能当作NA证据。
六秒边界剔除避免直接拼接伪影，但不消除长期在线统计/HMM历史的变化；因此内容与适应历史并未完全正交。也未随机化实际人的认知状态或测量独立NA标记。

## 复现与产物
` .venv\Scripts\python.exe test/56_waveform_swap_na/waveform_swap.py `
subject_metrics.csv含每人/分段/原始及后处理/共同与可用窗口的全部指标；macro_metrics.csv含宏平均；frames.csv.gz含逐帧标签、分数和来源；results.json含完整指标与交换审计。
