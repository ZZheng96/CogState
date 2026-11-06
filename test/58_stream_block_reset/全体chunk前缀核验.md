# E58 全体可行范围 chunk 前缀一致性/不变性核验

核验对象：已有真实 BDF batch1/batch2 分数缓存，共 5 个被试（004, 009, 010, 018, 050）。比较 raw、baseline、reset-detrend（dt_reset_hmm_keep）、reset-both（dt_reset_hmm_reset）的时间 t、标签 y、长度及最大绝对差。

## 覆盖范围

|chunk|独立前缀核验被试|说明|
|---:|---|---|
|137|无|缓存全记录参考，不是独立重跑|
|250|004, 009, 010, 018, 050|已有 batch 结果中的独立前缀重跑；未列被试无该结果|
|92|004, 018|已有 batch 结果中的独立前缀重跑；未列被试无该结果|
|1250|004, 018|已有 batch 结果中的独立前缀重跑；未列被试无该结果|

## 结果汇总（逐被试、逐方案）

|被试|chunk|方案|前缀t一致|y长度一致|raw最大绝对差|方案最大绝对差|来源|
|---|---:|---|---|---|---:|---:|---|
|004|137|baseline|True|True|0|0|cached full-record reference (not independent)|
|004|137|dt_reset_hmm_keep|True|True|0|0|cached full-record reference (not independent)|
|004|137|dt_reset_hmm_reset|True|True|0|0|cached full-record reference (not independent)|
|004|250|baseline|True|True|0|0|independent prefix rerun|
|004|250|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|004|250|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|004|92|baseline|True|True|0|0|independent prefix rerun|
|004|92|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|004|92|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|004|1250|baseline|True|True|0|0|independent prefix rerun|
|004|1250|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|004|1250|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|018|137|baseline|True|True|0|0|cached full-record reference (not independent)|
|018|137|dt_reset_hmm_keep|True|True|0|0|cached full-record reference (not independent)|
|018|137|dt_reset_hmm_reset|True|True|0|0|cached full-record reference (not independent)|
|018|250|baseline|True|True|0|0|independent prefix rerun|
|018|250|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|018|250|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|018|92|baseline|True|True|0|0|independent prefix rerun|
|018|92|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|018|92|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|018|1250|baseline|True|True|0|0|independent prefix rerun|
|018|1250|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|018|1250|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|009|137|baseline|True|True|0|0|cached full-record reference (not independent)|
|009|137|dt_reset_hmm_keep|True|True|0|0|cached full-record reference (not independent)|
|009|137|dt_reset_hmm_reset|True|True|0|0|cached full-record reference (not independent)|
|009|250|baseline|True|True|0|0|independent prefix rerun|
|009|250|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|009|250|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|010|137|baseline|True|True|0|0|cached full-record reference (not independent)|
|010|137|dt_reset_hmm_keep|True|True|0|0|cached full-record reference (not independent)|
|010|137|dt_reset_hmm_reset|True|True|0|0|cached full-record reference (not independent)|
|010|250|baseline|True|True|0|0|independent prefix rerun|
|010|250|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|010|250|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|
|050|137|baseline|True|True|0|0|cached full-record reference (not independent)|
|050|137|dt_reset_hmm_keep|True|True|0|0|cached full-record reference (not independent)|
|050|137|dt_reset_hmm_reset|True|True|0|0|cached full-record reference (not independent)|
|050|250|baseline|True|True|0|0|independent prefix rerun|
|050|250|dt_reset_hmm_keep|True|True|0|0|independent prefix rerun|
|050|250|dt_reset_hmm_reset|True|True|0|0|independent prefix rerun|

## 结论与限制

- 已有独立前缀结果中，chunk 92/250/1250（004、018）以及 chunk 250（009、010、050）均显示 t 一致、raw 一致，raw 与三种后处理最大绝对差均为 0；已有缓存的 t/raw/y/三种输出长度一致。
- chunk 137 对全部 5 个缓存被试均有全记录结果，但它是 batch 的基准流，不是另一次独立 chunk=137 重跑；因此不能把它作为独立不变性证据。
- chunk 92/1250 对 009/010/050、chunk 250 对无对应缓存的其它被试没有结果；本报告不补算、不伪造。全体 57 份记录也未完成端到端独立重跑，原因是当前可用结果仅覆盖上述 5 个真实 BDF，且既有脚本记录了完整 BDF 特征提取耗时约 45–60 秒/被试，未对其余记录形成可复用 scores.npz。
- AUC/F1：本次任务是数值流一致性核验，不重新评估分类指标；已有 batch 指标仍保留在原 batch JSON/MD 中。

JSON：`全体chunk前缀核验.json` ；CSV：`全体chunk前缀核验.csv`。
