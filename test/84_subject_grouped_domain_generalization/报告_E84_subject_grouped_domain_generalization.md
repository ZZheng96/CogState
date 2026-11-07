# E84 被试分组域泛化（六人严格 LOSO CUDA）

比较低容量ERM与固定 lambda=0.1 的被试风险方差正则。特征、尺度和 EMA1200+前向HMM300 均按训练折或因果当前/过去输入计算；不使用 block/phase、目标统计、目标标签适配或未来数据。

## 指标

|方法|阶段|AUC|F1|bACC|
|---|---|---:|---:|---:|
|erm|full|0.783341|0.733594|0.745666|
|erm|early|0.777795|0.691378|0.762279|
|erm|late|0.738164|0.648468|0.683388|
|subject_risk_variance|full|0.785543|0.736552|0.748715|
|subject_risk_variance|early|0.781966|0.695860|0.766505|
|subject_risk_variance|late|0.741333|0.651844|0.686237|

## 逐折比较（方差正则相对ERM）

|阶段|指标|改善/退步/持平|
|---|---|---|
|full|auc|5/1/0|
|full|f1|5/1/0|
|full|bacc|5/1/0|
|early|auc|3/3/0|
|early|f1|5/1/0|
|early|bacc|5/1/0|
|late|auc|5/1/0|
|late|f1|3/1/2|
|late|bacc|3/1/2|

## 扩展判定

full 与 late 均要求相对ERM有共同明确潜力；六折判定：有潜力，允许继续57人。

逐折连续 raw/score、训练被试清单与数据校验保存在本目录。