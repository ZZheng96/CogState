# E82 block>=4 late specialist（六人严格 LOSO 诊断）

late expert仅使用训练折block>=4帧训练；目标折仅在block>=4切换expert，early/block3保留baseline。两组均为E42特征、LR C=.001 + 三个MLP seed均值、训练折尺度、EMA1200+前向HMM300。

## 宏观指标

|方法|阶段|AUC|F1|bACC|
|---|---|---:|---:|---:|
|baseline|full|0.808679|0.763405|0.766473|
|baseline|early|0.818000|0.695506|0.771928|
|baseline|late|0.728697|0.702781|0.706443|
|late_specialist|full|0.787674|0.732007|0.744031|
|late_specialist|early|0.818000|0.695506|0.771928|
|late_specialist|late|0.665674|0.584937|0.640737|

## 相对baseline逐人比较

|阶段|指标|改善/退步/持平|
|---|---|---|
|full|auc|1/5/0|
|full|f1|1/5/0|
|full|bacc|1/5/0|
|early|auc|0/0/6|
|early|f1|0/0/6|
|early|bacc|0/0/6|
|late|auc|2/4/0|
|late|f1|1/5/0|
|late|bacc|1/5/0|

## 结论

full AUC差值=-0.021004，late AUC差值=-0.063023。full/late均需共同明确潜力才扩展57人；本诊断判定：无共同明确潜力，不扩展57人。

逐折训练样本数、尺度、连续raw/score与数据校验保存在本目录。