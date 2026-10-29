"""学习式因果记忆：单向 GRU 只读取当前与历史全局分数。

训练时随机截取最近 1--8 窗，禁止块序号、未来信息及目标受试者标签。
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from singlem_eval import smooth, boot

torch.set_num_threads(1)

class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(1, 8, batch_first=True)
        self.head = nn.Linear(8, 1)
    def forward(self, x):
        return self.head(self.rnn(x)[0]).squeeze(-1)

def stream_tensor(raw, ids):
    return torch.tensor(raw[ids], dtype=torch.float32)[None, :, None]

def main():
    tick = time.perf_counter(); torch.manual_seed(0); np.random.seed(0)
    d = np.load(OUT/'highdim_deep_features.npz'); y, g = d['y'].astype(int), d['groups']
    pos = np.empty(len(y), int)
    for s in np.unique(g):
        i = np.where(g == s)[0]; pos[i] = np.arange(len(i))
    mask = pos >= 120
    pool = {}
    for f in ('b_branch_oof.npz', 'b_calib_variants_oof.npz', 'b_calib2_oof.npz'):
        z = np.load(OUT/f); pool.update({k:z[k] for k in z.files if k not in ('y','groups')})
    keys = ['quant120_histgb','quant120w_histgb','quant120_svc','base_svc']
    raw = np.mean([rankdata(pool[k])/len(y) for k in keys], 0)
    base = smooth(raw, g, 5)
    streams = {s: np.where(g == s)[0] for s in np.unique(g)}
    oof = np.full(len(y), .5)
    for fold, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(raw, y, g), 1):
        torch.manual_seed(fold); net = Memory()
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        train_subjects = np.unique(g[tr])
        for epoch in range(30):
            net.train(); losses = []
            order = np.random.permutation(train_subjects)
            for s in order:
                ids = streams[s]; x = stream_tensor(raw, ids); target = torch.tensor(y[ids], dtype=torch.float32)[None]
                # 每轮随机历史长度；损失只用于当前末端窗，训练的是局部因果滤波而非实验时间表。
                length = int(np.random.randint(1, 9)); end = int(np.random.randint(length-1, len(ids)))
                sl = slice(end-length+1, end+1)
                opt.zero_grad(); loss = loss_fn(net(x[:, sl])[:, -1], target[:, sl][:, -1])
                loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step(); losses.append(float(loss))
            if (epoch+1) % 10 == 0:
                print(f'[fold {fold}/5] epoch {epoch+1}/30 loss={np.mean(losses):.4f} elapsed={time.perf_counter()-tick:.0f}s', flush=True)
        net.eval()
        with torch.no_grad():
            for s in np.unique(g[te]):
                ids = streams[s]
                oof[ids] = torch.sigmoid(net(stream_tensor(raw, ids))).numpy()[0]
        print(f'[fold {fold}/5] test AUC={roc_auc_score(y[te], oof[te]):.4f}', flush=True)
    auc = float(roc_auc_score(y[mask], oof[mask])); bacc = float(balanced_accuracy_score(y[mask], oof[mask] >= .5))
    from scipy.stats import norm
    from b_calib_variants import subject_quantile
    quant = norm.cdf(subject_quantile(oof[:, None], g, 120)[:, 0])
    quant_result = {'AUC': float(roc_auc_score(y[mask], quant[mask])),
                    'bACC': float(balanced_accuracy_score(y[mask], quant[mask] >= .5)),
                    'vs_p3': boot(quant, base, y, g, mask)}
    result = {'gru': {'AUC': auc, 'bACC': bacc, 'vs_p3': boot(oof, base, y, g, mask)},
              'gru_quantile120': quant_result,
              'protocol': {'history_windows': 'random 1-8 during training; unlimited forward state at test',
                           'input': 'P3 unsmoothed global score only', 'target_labels_used': False}}
    print(f'[result] GRU AUC={auc:.4f} bACC={bacc:.4f} Δ={result["gru"]["vs_p3"]["mean"]:+.4f}', flush=True)
    print(f'[result] GRU+quantile120 AUC={quant_result["AUC"]:.4f} bACC={quant_result["bACC"]:.4f} '
          f'Δ={quant_result["vs_p3"]["mean"]:+.4f} [{quant_result["vs_p3"]["p2.5"]:+.4f},{quant_result["vs_p3"]["p97.5"]:+.4f}]', flush=True)
    np.savez_compressed(OUT/'causal_gru_oof.npz', score=oof, y=y, groups=g)
    (OUT/'causal_gru_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__': main()
