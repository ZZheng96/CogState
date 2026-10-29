"""受试者对抗训练：用梯度反转层（GRL）逼迫特征表示丢失"受试者身份"。

动机（见报告 5.4）：许多特征受试者内 AUC 可达 0.69，但约 39% 的受试者方向相反，
单一全局方向无法同时适配。对抗训练试图找一个"与受试者无关、但保留 relax/task"的子空间。

对照：同构网络、同超参，但不带受试者头（普通 MLP），用于隔离对抗项的作用。
全部按受试者分组五折，300 维在训练折内筛选与标准化。
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

ROOT = Path(__file__).resolve().parents[1]; DB = ROOT/'database'; OUT = ROOT/'test'
sys.path[:0] = [str(OUT)]
torch.set_num_threads(4)


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha*grad, None


def grl(x, alpha):
    return _GRLFn.apply(x, alpha)


def make_net(dim, n_subj, use_adv):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(.2),
                                     nn.Linear(256, 128), nn.ReLU())
            self.head = nn.Linear(128, 1)
            self.adv = nn.Linear(128, n_subj) if use_adv else None
            self.alpha = 0.

        def forward(self, x):
            z = self.enc(x)
            out = [self.head(z).squeeze(1)]
            if self.adv is not None:
                out.append(self.adv(grl(z, self.alpha)))
            return out
    return Net()


def train_fold(Xtr, ytr, gtr, Xte, use_adv, n_subj, seed=0, epochs=18, lam=1.0):
    torch.manual_seed(seed)
    net = make_net(Xtr.shape[1], n_subj, use_adv)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-2)
    gmap = {s: i for i, s in enumerate(sorted(set(gtr)))}
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    gt = torch.tensor([gmap[s] for s in gtr], dtype=torch.long)
    pw = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    ce = nn.CrossEntropyLoss()
    n = len(ytr); bs = 256
    for ep in range(epochs):
        net.train()
        p = ep/max(epochs-1, 1)
        net.alpha = lam*(2/(1+np.exp(-10*p))-1)
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            opt.zero_grad()
            outs = net(Xt[b])
            loss = bce(outs[0], yt[b])
            if use_adv:
                loss = loss + ce(outs[1], gt[b])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32))[0]).numpy()


def main():
    tick = time.perf_counter()
    fs, es, ys, gs = [], [], [], []
    for p in sorted((DB/'windows').glob('*/*.npz')):
        d = np.load(p); fs.append(d['f']); es.append(d['e'])
        ys.append(d['label']); gs.append(d['subject'])
    F = np.vstack(fs).astype(np.float64); F = F[:, F.std(0) > 1e-8]
    X = np.hstack([F, np.vstack(es).astype(np.float64)])
    y = np.concatenate(ys).astype(int); g = np.concatenate(gs)
    splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g))
    n_subj = len(np.unique(g))
    print(f'[data] {X.shape} subjects={n_subj}', flush=True)

    res, oofs = {}, {}
    for name, adv, lam in (('mlp_plain', False, 0.), ('mlp_adv', True, 1.0), ('mlp_adv_weak', True, .3)):
        oof = np.full(len(y), np.nan)
        for fold, (tr, te) in enumerate(splits, 1):
            A, T = X[tr], X[te]
            sc = StandardScaler().fit(A)
            A, T = sc.transform(A), sc.transform(T)
            sel = SelectKBest(f_classif, k=300).fit(A, y[tr])
            A, T = sel.transform(A), sel.transform(T)
            oof[te] = train_fold(A, y[tr], g[tr], T, adv, n_subj, seed=fold, lam=lam)
            print(f'[{name}] fold={fold}/5 AUC={roc_auc_score(y[te], oof[te]):.4f} '
                  f'({time.perf_counter()-tick:.0f}s)', flush=True)
        r = {'AUC': float(roc_auc_score(y, oof)), 'bACC': float(balanced_accuracy_score(y, oof >= .5))}
        per = [roc_auc_score(y[g == s], oof[g == s]) for s in np.unique(g) if len(set(y[g == s])) == 2]
        r['subject_AUC'] = float(np.mean(per))
        res[name] = r; oofs[name] = oof
        print(f'[result] {name} AUC={r["AUC"]:.4f} bACC={r["bACC"]:.4f} subjAUC={r["subject_AUC"]:.4f}', flush=True)

    np.savez_compressed(OUT/'subject_adversarial_oof.npz', y=y, groups=g, **oofs)
    res['_meta'] = {'seconds': time.perf_counter()-tick, 'n_features': int(X.shape[1])}
    (OUT/'subject_adversarial_results.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
