# 方法 × 被试 AUC 矩阵（n×57）

协议：deploy=单窗6s跨被试无位置；history=依赖历史；diag=同人/oracle诊断。split=SGKF5rs0 表示与 5 折(shuffle,rs=0) 同一被试划分。

| 方法 | 协议 | 覆盖 | 均值 | 中位 | 最低 | 最高 | 说明 |
|---|---|---|---|---|---|---|---|
| ceiling/within_rand | diag | 57 | 0.833 | 0.850 | 0.642 | 0.934 | 同人随机切分 |
| audit/own_cv | diag | 57 | 0.833 | 0.850 | 0.641 | 0.934 | 本人方向(交叉拟合) |
| bottleneck/within_subject | diag | 57 | 0.816 | 0.819 | 0.623 | 0.941 | 同人 vs 跨人 |
| ceiling/merge_best | deploy | 57 | 0.692 | 0.703 | 0.307 | 0.885 | 历史融合最佳 |
| reliability/base | deploy | 57 | 0.671 | 0.694 | 0.359 | 0.879 | 收缩前的基线 LR |
| bank/consensus_all | deploy | 57 | 0.661 | 0.677 | 0.270 | 0.875 | 本轮同协议方法银行 |
| bank/lda_shrink | deploy | 57 | 0.652 | 0.669 | 0.331 | 0.849 | 本轮同协议方法银行 |
| bank/ridge | deploy | 57 | 0.652 | 0.679 | 0.334 | 0.853 | 本轮同协议方法银行 |
| eeg_only/lr_c001 | deploy | 57 | 0.651 | 0.669 | 0.333 | 0.851 | 本轮 LR 复现 |
| bank/lr_c001 | deploy | 57 | 0.651 | 0.669 | 0.333 | 0.851 | 本轮同协议方法银行 |
| polarity/base | deploy | 57 | 0.646 | 0.664 | 0.354 | 0.822 | 符号压力前的基线 |
| bank/lr_c1 | deploy | 57 | 0.645 | 0.664 | 0.355 | 0.822 | 本轮同协议方法银行 |
| bank/hgb | deploy | 57 | 0.645 | 0.654 | 0.247 | 0.869 | 本轮同协议方法银行 |
| audit/xs | deploy | 57 | 0.642 | 0.650 | 0.360 | 0.851 | 全局方向 |
| ceiling/order_free_xs | deploy | 57 | 0.642 | 0.650 | 0.360 | 0.851 | 天花板审计 |
| theory/own_loading_xs | deploy | 57 | 0.642 | 0.650 | 0.360 | 0.851 | 特征族能量份额分析 |
| bank/mlp256 | deploy | 57 | 0.636 | 0.654 | 0.296 | 0.823 | 本轮同协议方法银行 |
| bank/nystroem_svc | deploy | 57 | 0.635 | 0.651 | 0.310 | 0.852 | 本轮同协议方法银行 |
| highdim/large_mlp300 | deploy | 57 | 0.635 | 0.648 | 0.276 | 0.855 | 878维，内层选特征 |
| bottleneck/cross_subject | deploy | 57 | 0.635 | 0.648 | 0.276 | 0.855 | 同人 vs 跨人 |
| highdim/mlp300 | deploy | 57 | 0.635 | 0.659 | 0.297 | 0.874 | 878维，内层选特征 |
| highdim/logistic300_corr | deploy | 57 | 0.631 | 0.640 | 0.375 | 0.837 | 878维，内层选特征 |
| highdim/logistic600 | deploy | 57 | 0.629 | 0.629 | 0.386 | 0.841 | 878维，内层选特征 |
| highdim/logistic300 | deploy | 57 | 0.627 | 0.631 | 0.340 | 0.841 | 878维，内层选特征 |
| bank/rf | deploy | 57 | 0.624 | 0.646 | 0.252 | 0.856 | 本轮同协议方法银行 |
| bank/lr_stft | deploy | 57 | 0.622 | 0.646 | 0.323 | 0.834 | 本轮同协议方法银行 |
| highdim/logistic300_variance | deploy | 57 | 0.621 | 0.647 | 0.336 | 0.793 | 878维，内层选特征 |
| bank/lr_emg | deploy | 57 | 0.616 | 0.675 | 0.295 | 0.879 | 本轮同协议方法银行 |
| highdim/mlp82 | deploy | 57 | 0.616 | 0.637 | 0.317 | 0.856 | 878维，内层选特征 |
| bank/pca64_lr | deploy | 57 | 0.615 | 0.643 | 0.349 | 0.823 | 本轮同协议方法银行 |
| highdim/logistic150 | deploy | 57 | 0.614 | 0.625 | 0.337 | 0.824 | 878维，内层选特征 |
| nonstat/wavelet_timefreq_quality | deploy | 57 | 0.613 | 0.630 | 0.267 | 0.853 | 82维特征族消融 |
| highdim/logistic82 | deploy | 57 | 0.613 | 0.630 | 0.267 | 0.853 | 878维，内层选特征 |
| depo/delta_k32__hgb | history | 57 | 0.613 | 0.616 | 0.377 | 0.841 | 去位置化流式 |
| nonstat/timefreq_only | deploy | 57 | 0.611 | 0.639 | 0.281 | 0.850 | 50维特征族消融 |
| bank/pca16_lr | deploy | 57 | 0.610 | 0.640 | 0.330 | 0.845 | 本轮同协议方法银行 |
| bank/extratrees | deploy | 57 | 0.608 | 0.614 | 0.242 | 0.853 | 本轮同协议方法银行 |
| nonstat/selected30_logistic | deploy | 57 | 0.606 | 0.616 | 0.311 | 0.863 | 82维特征族消融 |
| bank/lr_hilbert | deploy | 57 | 0.605 | 0.617 | 0.378 | 0.808 | 本轮同协议方法银行 |
| bank/lr_dwt | deploy | 57 | 0.601 | 0.611 | 0.319 | 0.814 | 本轮同协议方法银行 |
| nonstat/quality53 | deploy | 57 | 0.599 | 0.590 | 0.351 | 0.845 | 53维特征族消融 |
| nonstat/raw44 | deploy | 57 | 0.597 | 0.597 | 0.356 | 0.829 | 44维特征族消融 |
| kalman/no_kalman | deploy | 57 | 0.593 | 0.625 | 0.265 | 0.853 | 每窗重置滤波器 |
| depo/current_only__hgb | deploy | 57 | 0.588 | 0.596 | 0.351 | 0.820 | 去位置化流式 |
| depo/current_only__lr | deploy | 57 | 0.588 | 0.592 | 0.396 | 0.719 | 去位置化流式 |
| bank/knn50_pca32 | deploy | 57 | 0.583 | 0.583 | 0.250 | 0.849 | 本轮同协议方法银行 |
| kalman/kalman | deploy | 57 | 0.582 | 0.604 | 0.282 | 0.842 | 每窗重置滤波器 |
| bank/gnb_pca32 | deploy | 57 | 0.571 | 0.575 | 0.217 | 0.839 | 本轮同协议方法银行 |
| depo/xs_full | history | 57 | 0.571 | 0.564 | 0.452 | 0.772 | 锚点/方向法 |
| depo/anchor_dir_64 | history | 57 | 0.568 | 0.563 | 0.324 | 0.824 | 锚点/方向法 |
| depo/anchor_128 | history | 57 | 0.539 | 0.518 | 0.425 | 0.692 | 锚点/方向法 |
| depo/anchor_oracle_64 | history | 57 | 0.536 | 0.522 | 0.418 | 0.720 | 锚点/方向法 |
| nonstat/wavelet_only | deploy | 57 | 0.535 | 0.546 | 0.273 | 0.809 | 23维特征族消融 |
| depo/anchor_64 | history | 57 | 0.535 | 0.518 | 0.416 | 0.661 | 锚点/方向法 |
| depo/anchor_32 | history | 57 | 0.517 | 0.507 | 0.422 | 0.651 | 锚点/方向法 |
| depo/anchor_16 | history | 57 | 0.513 | 0.504 | 0.419 | 0.641 | 锚点/方向法 |
| kalman/raw_cnn | deploy | 57 | 0.511 | 0.538 | 0.176 | 0.807 | 每窗重置滤波器 |
| kalman/raw_kalman_cnn | deploy | 57 | 0.511 | 0.538 | 0.176 | 0.807 | 每窗重置滤波器 |
| highdim/raw_cnn | deploy | 57 | 0.483 | 0.503 | 0.177 | 0.808 | 878维，内层选特征 |
| bottleneck/within_trial | diag | 56 | 0.406 | 0.415 | 0.108 | 0.827 | 按试次阻断的同人训练 |
