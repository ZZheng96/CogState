# E58 真实BDF流式 block 重置核验

真实BDF全量运行因E17特征提取在当前环境耗时过长，故按规则退回现有 verify 数据：E53 LOSO 分数、明确六个真实标签 block 边界；不冒充真实BDF验证。

|方案|范围|片段|bACC|AUC|F1|
|---|---|---|---:|---:|---:|
|每真实block重置detrend，保留HMM|all|full|0.7906|0.8398|0.7810|
|每真实block重置detrend，保留HMM|all|first600|0.8077|0.8490|0.7693|
|每真实block重置detrend，保留HMM|all|block45|0.7685|0.8235|0.7452|
|每真实block重置detrend，保留HMM|high|full|0.8910|0.9531|0.8890|
|每真实block重置detrend，保留HMM|high|first600|0.8940|0.9540|0.8868|
|每真实block重置detrend，保留HMM|high|block45|0.8848|0.9412|0.8852|
|每真实block重置detrend，保留HMM|low|full|0.6938|0.7304|0.6768|
|每真实block重置detrend，保留HMM|low|first600|0.7244|0.7477|0.6559|
|每真实block重置detrend，保留HMM|low|block45|0.6562|0.7098|0.6099|
|每真实block重置detrend+HMM|all|full|0.7980|0.8426|0.7881|
|每真实block重置detrend+HMM|all|first600|0.8147|0.8517|0.7762|
|每真实block重置detrend+HMM|all|block45|0.7853|0.8290|0.7619|
|每真实block重置detrend+HMM|high|full|0.9013|0.9563|0.8993|
|每真实block重置detrend+HMM|high|first600|0.9071|0.9592|0.9004|
|每真实block重置detrend+HMM|high|block45|0.9009|0.9455|0.9010|
|每真实block重置detrend+HMM|low|full|0.6983|0.7329|0.6807|
|每真实block重置detrend+HMM|low|first600|0.7254|0.7478|0.6562|
|每真实block重置detrend+HMM|low|block45|0.6737|0.7166|0.6277|

## 结论

该最小复现实验仅验证真实 block 边界触发的因果后处理，不足以证明端到端真实BDF收益；因此不建议据此纳入正式部署。正式部署前需在可接受时限内完成真实BDF全体核验并验证所有chunk一致性。
