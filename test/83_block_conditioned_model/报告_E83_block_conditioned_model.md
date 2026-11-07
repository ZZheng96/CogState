# E83 block-conditioned shared model（六人严格 LOSO CUDA 诊断）

在共享模型中将当前帧 block 作为固定 8 类 one-hot 条件追加到 E42 特征；LR/MLP 均使用全部训练帧，不训练独立 late specialist。目标折只提供当前帧 block 条件，不使用目标标签或目标统计。两组统一 EMA1200+前向HMM300。

## 宏观指标

|方法|阶段|AUC|F1|bACC|
|---|---|---:|---:|---:|
|baseline|full|0.808679|0.763405|0.766473|
|baseline|early|0.818000|0.695506|0.771928|
|baseline|late|0.728697|0.702781|0.706443|
|block_conditioned|full|0.999433|0.967053|0.964482|
|block_conditioned|early|0.999895|0.924726|0.955567|
|block_conditioned|late|0.999599|0.984540|0.984391|

## 相对baseline逐人比较

|阶段|指标|改善/退步/持平|
|---|---|---|
|full|auc|6/0/0|
|full|f1|6/0/0|
|full|bacc|6/0/0|
|early|auc|6/0/0|
|early|f1|6/0/0|
|early|bacc|6/0/0|
|late|auc|6/0/0|
|late|f1|6/0/0|
|late|bacc|6/0/0|

## 结论

full AUC差值=+0.190754，late AUC差值=+0.270902。full/late均需共同明确潜力才扩展57人；本诊断判定：有潜力，允许扩展。

逐折训练样本数、尺度、连续raw/score与数据校验保存在本目录。