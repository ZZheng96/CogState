"""用预训练单通道 EEG 基础模型 SingLEM（Sukhbaatar et al., 2025）提取冻结表征。

预处理按官方仓库：0.5–50 Hz 带通、50 Hz 陷波、重采样到 128 Hz、µV×0.01。
所有滤波只在 6 s 窗内部进行，不跨窗，因此满足 6 s 感受野。
输出：每窗 6 个 1 s token × 16 维表征，另存 transformer 层平均 (128 维)。
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'test'
EXT = ROOT.parent/'ext'/'SingLEM'
sys.path.insert(0, str(EXT))
from SingLEM_model import Config, EEGEncoder


def preprocess(raw):
    x = raw.astype(np.float64)*1e6                    # V → µV
    x -= x.mean(1, keepdims=True)
    b, a = iirnotch(50., 30., 250.)
    x = filtfilt(b, a, x, axis=1)
    x = sosfiltfilt(butter(4, [.5, 50.], fs=250., btype='bandpass', output='sos'), x, axis=1)
    x = resample_poly(x, 64, 125, axis=1)             # 250 → 128 Hz, 1500 → 768
    x = np.clip(x, -100, 100)*.01
    return x.reshape(len(x), 6, 128).astype(np.float32)


def main():
    tick = time.perf_counter()
    d = np.load(OUT/'highdim_deep_features.npz')
    X = preprocess(d['raw'])
    print(f'[prep] {X.shape} ({time.perf_counter()-tick:.0f}s)', flush=True)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = Config()
    ck = torch.load(EXT/'SingLEM_checkpoints_singlem_downstream_excluded.pt', map_location=dev, weights_only=True)
    for k, v in ck['model_config'].items():
        setattr(cfg, k, v)
    cfg.mask_prob = 0.
    m = EEGEncoder(cfg).to(dev); m.load_state_dict(ck['encoder_state_dict']); m.eval()

    hid = {}
    m.TransformerEncoder.ln_final.register_forward_hook(lambda mod, i, o: hid.__setitem__('h', o))
    R, H = [], []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            r, _, _ = m(torch.from_numpy(X[i:i+512]).to(dev))
            R.append(r.cpu().numpy()); H.append(hid['h'].mean(1).cpu().numpy())
            if i % 4096 == 0:
                print(f'[embed] {i}/{len(X)} ({time.perf_counter()-tick:.0f}s)', flush=True)
    R, H = np.concatenate(R), np.concatenate(H)
    np.savez_compressed(OUT/'singlem_embed.npz', R=R.astype(np.float32), H=H.astype(np.float32))
    print(f'[done] R={R.shape} H={H.shape} ({time.perf_counter()-tick:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
