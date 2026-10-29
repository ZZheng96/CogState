"""严格 subject-disjoint 的 6 秒 EEG 时频图视觉模型实验。

每个数据库窗口独立转换为当前 6 秒内的 log-STFT 图像；不使用顺序、时间位置、测试历史或测试被试校准。
输入三通道分别为 log 功率、时间差分和频率去均值图。模型只在外层训练被试上拟合。
"""
import copy, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DB, OUT = ROOT / 'database', ROOT / 'test'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load():
    xs, ys, gs = [], [], []
    for p in sorted((DB / 'windows').glob('*/*.npz')):
        d = np.load(p)
        xs.append(d['x'].astype(np.float32)); ys.append(d['label']); gs.append(d['subject'])
    return np.vstack(xs), np.concatenate(ys).astype(np.int64), np.concatenate(gs)


def make_images(x, device=DEVICE, batch=512):
    """GPU STFT; output NCHW float32."""
    out = []
    win = torch.hann_window(128, device=device)
    for i in range(0, len(x), batch):
        z = torch.from_numpy(x[i:i+batch]).to(device)
        z = (z - z.mean(1, keepdim=True)) / (z.std(1, keepdim=True) + 1e-6)
        st = torch.stft(z, n_fft=128, hop_length=32, win_length=128,
                        window=win, center=False, return_complex=True)
        p = torch.log1p(st.abs().square())[:, 1:24, :]
        p = (p - p.mean((1, 2), keepdim=True)) / (p.std((1, 2), keepdim=True) + 1e-5)
        dt = torch.diff(p, dim=2, prepend=p[:, :, :1])
        df = p - p.mean(1, keepdim=True)
        out.append(torch.stack((p, dt, df), 1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


class VisionEEG(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Dropout(.25),
        )
        self.head = nn.Linear(192, 1)

    def forward(self, x):
        return self.head(self.body(x)).ravel()


def predict(model, X, batch=512):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(X[i:i+batch]).to(DEVICE))).cpu().numpy())
    return np.concatenate(out)


def score(y, p, threshold=.5):
    return {'AUC': float(roc_auc_score(y, p)), 'bACC': float(balanced_accuracy_score(y, p >= threshold))}


def main():
    t0 = time.perf_counter()
    torch.set_float32_matmul_precision('high')
    x, y, g = load()
    print(f'[data] x={x.shape}, subjects={len(np.unique(g))}, device={DEVICE}', flush=True)
    images = make_images(x)
    print(f'[image] shape={images.shape}, params={sum(p.numel() for p in VisionEEG().parameters()):,}', flush=True)
    splits = StratifiedGroupKFold(5, shuffle=True, random_state=0).split(images, y, g)
    oof = np.full(len(y), np.nan); folds = []
    for fold, (tr0, te) in enumerate(splits, 1):
        # validation uses held-out subjects within outer training only
        gss = GroupShuffleSplit(n_splits=1, test_size=.15, random_state=100 + fold)
        tri, vi = next(gss.split(tr0, y[tr0], g[tr0]))
        tr, va = tr0[tri], tr0[vi]
        model = VisionEEG().to(DEVICE)
        pos = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos, device=DEVICE))
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
        rng = np.random.default_rng(1000 + fold)
        best_loss, best_state, stale, best_epoch = float('inf'), None, 0, 0
        for epoch in range(1, 31):
            model.train(); order = rng.permutation(tr)
            losses = []
            for j in range(0, len(order), 256):
                ids = order[j:j+256]
                xb = torch.from_numpy(images[ids]).to(DEVICE)
                yb = torch.from_numpy(y[ids]).float().to(DEVICE)
                opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
                losses.append(float(loss.detach().cpu()))
            vp = predict(model, images[va])
            vl = float(-np.mean(y[va] * np.log(np.clip(vp, 1e-7, 1)) + (1-y[va]) * np.log(np.clip(1-vp, 1e-7, 1))))
            if epoch == 1 or epoch % 5 == 0:
                print(f'[fold {fold}/5] epoch {epoch}/30 train_loss={np.mean(losses):.4f} val_loss={vl:.4f}', flush=True)
            if vl < best_loss - 1e-4:
                best_loss, best_state, stale, best_epoch = vl, copy.deepcopy(model.state_dict()), 0, epoch
            else: stale += 1
            if stale >= 6: break
        model.load_state_dict(best_state)
        p = predict(model, images[te]); oof[te] = p
        # Threshold is selected from validation subjects only; fixed .5 remains primary comparison.
        thresholds = np.linspace(.1, .9, 161)
        vt = thresholds[np.argmax([balanced_accuracy_score(y[va], vp >= q) for q in thresholds])]
        r = {'fold': fold, 'n_train': len(tr), 'n_test': len(te), 'best_epoch': best_epoch,
             'fixed_0.5': score(y[te], p), 'validation_threshold': float(vt),
             'threshold_on_test_for_diagnostic_only': score(y[te], p, vt)}
        folds.append(r); print(f'[fold {fold}/5] AUC={r["fixed_0.5"]["AUC"]:.4f} bACC={r["fixed_0.5"]["bACC"]:.4f}', flush=True)
    result = {'protocol': {'test_subject_isolation': True, 'test_history': False, 'window_position_schedule_r_t': False,
                           'test_subject_calibration': False, 'per_window_only': True, 'input_seconds': 6,
                           'stft_center': False, 'outer_cv': 'StratifiedGroupKFold(5, shuffle=True, random_state=0)',
                           'threshold_primary': 0.5, 'validation_threshold_not_used_for_model_selection': False},
              'model': {'name': 'VisionEEG', 'parameters': int(sum(p.numel() for p in VisionEEG().parameters())),
                        'input_shape': list(images.shape[1:]), 'trainable_parameters': int(sum(p.numel() for p in VisionEEG().parameters())),
                        'epochs_max': 30, 'batch_size': 256, 'lr': 2e-3, 'weight_decay': 2e-3, 'early_stop_patience': 6},
              'overall_fixed_0.5': score(y, oof), 'subject_count': int(len(np.unique(g))),
              'subject_fixed_0.5': {'AUC': float(np.mean([roc_auc_score(y[g==s], oof[g==s]) for s in np.unique(g)])),
                                    'bACC': float(np.mean([balanced_accuracy_score(y[g==s], oof[g==s]>=.5) for s in np.unique(g)]))},
              'folds': folds, '_meta': {'seconds': time.perf_counter()-t0, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}
    np.savez_compressed(OUT / 'timefreq_vision_window6_oof.npz', y=y, groups=g, pred=oof)
    (OUT / 'timefreq_vision_window6_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] AUC={result["overall_fixed_0.5"]["AUC"]:.4f} bACC={result["overall_fixed_0.5"]["bACC"]:.4f} total={time.perf_counter()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
