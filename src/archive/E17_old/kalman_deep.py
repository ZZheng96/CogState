from __future__ import annotations
import numpy as np
from cogstate import causal_features, FS

Q_DEFAULT = 1e-10
R_DEFAULT = 1e-8
P0_DEFAULT = R_DEFAULT


def kalman_filter_window(x, q=Q_DEFAULT, r=R_DEFAULT, p0=P0_DEFAULT):
    """Causal scalar random-walk filter, reset at the beginning of each window."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if not np.isfinite(q) or not np.isfinite(r) or not np.isfinite(p0) or q < 0 or r <= 0 or p0 < 0:
        raise ValueError("q/r/p0 must satisfy q>=0, r>0, p0>=0")
    if x.size == 0:
        return x.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Kalman input contains non-finite samples")
    state = float(x[0]); cov = float(p0); out = np.empty_like(x)
    for i, obs in enumerate(x):
        cov += q
        gain = cov / (cov + r)
        state += gain * (float(obs) - state)
        cov = (1.0 - gain) * cov
        out[i] = state
    return out.astype(np.float32)


def kalman_features(x, sfreq=FS, q=Q_DEFAULT, r=R_DEFAULT, p0=P0_DEFAULT):
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) < int(sfreq) or not np.isfinite(x).all():
        return None
    return causal_features(kalman_filter_window(x, q, r, p0), sfreq).astype(np.float32)


def _network_for_checkpoint(ck):
    from torch import nn
    model_name = str(ck.get('model', 'mlp_kalman'))
    if model_name.startswith('mlp'):
        net = nn.Sequential(nn.Linear(int(ck['input_dim']), 32), nn.ReLU(),
                            nn.Dropout(.15), nn.Linear(32, 16), nn.ReLU(),
                            nn.Linear(16, 1))
        mean = ck.get('mean', np.zeros(ck['input_dim'], dtype=np.float32))
        scale = ck.get('scale', np.ones(ck['input_dim'], dtype=np.float32))
    else:
        net = nn.Sequential(nn.Conv1d(1, 8, 9, padding=4), nn.ReLU(),
                            nn.AvgPool1d(4), nn.Conv1d(8, 16, 9, padding=4),
                            nn.ReLU(), nn.AvgPool1d(4), nn.Flatten(),
                            nn.Linear(16 * 15, 16), nn.ReLU(), nn.Dropout(.15),
                            nn.Linear(16, 1))
        mean, scale = np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)
    state = ck['state_dict']
    if any(key.startswith('net.') for key in state):
        state = {key[4:]: value for key, value in state.items()}
    net.load_state_dict(state)
    return net, mean, scale


def load_kalman_deep_checkpoint(path, device=None):
    import torch
    ck = torch.load(path, map_location=device or 'cpu', weights_only=False)
    net, mean, scale = _network_for_checkpoint(ck)
    net.to(device or 'cpu').eval()
    return ck, net, mean, scale


class KalmanDeepModel:
    """Streaming wrapper for a saved fixed-window causal model."""
    def __init__(self, net, mean, scale, use_kalman=True, q=Q_DEFAULT, r=R_DEFAULT,
                 p0=P0_DEFAULT, sfreq=FS, window=6.0, device=None, model=None):
        import torch
        if float(sfreq) != FS or float(window) != 6.0:
            raise ValueError('KalmanDeepModel supports exactly sfreq=250 Hz and window=6 s')
        self.net, self.mean, self.scale = net, np.asarray(mean), np.asarray(scale)
        self.use_kalman, self.q, self.r, self.p0 = bool(use_kalman), q, r, p0
        self.model = str(model or ('cnn' if any(isinstance(m, torch.nn.Conv1d) for m in net.modules()) else 'mlp'))
        if self.model.startswith('cnn'):
            self.input_dim = 250
        else:
            self.input_dim = int(np.asarray(mean).size)
        self.sfreq, self.window = FS, 6.0; self.buf = np.empty(0, dtype=np.float64)
        self.device = torch.device(device or next(net.parameters()).device)
        self.net.to(self.device).eval()

    @classmethod
    def from_checkpoint(cls, path, device=None):
        ck, net, mean, scale = load_kalman_deep_checkpoint(path, device=device)
        return cls(net, mean, scale, use_kalman=ck.get('use_kalman', False),
                   q=ck.get('q', Q_DEFAULT), r=ck.get('r', R_DEFAULT),
                   p0=ck.get('p0', P0_DEFAULT), sfreq=ck.get('sfreq', FS),
                   window=ck.get('window', 6.0), device=device, model=ck.get('model'))

    def reset(self):
        self.buf = np.empty(0, dtype=np.float64)

    def predict_proba(self, features):
        import torch
        z = ((np.asarray(features, dtype=np.float32) - self.mean) / self.scale).astype(np.float32)
        tensor = torch.from_numpy(z).to(self.device)
        if self.model.startswith('cnn'):
            tensor = tensor[:, None, :]
        with torch.no_grad():
            return torch.sigmoid(self.net(tensor)).cpu().numpy().reshape(-1, 1)

    def update(self, samples):
        x = np.asarray(samples, dtype=np.float64).ravel()
        if x.size == 0:
            return None
        if not np.isfinite(x).all():
            self.reset(); return None
        n = 1500
        self.buf = np.concatenate([self.buf, x])[-n:]
        if len(self.buf) < n: return None
        signal = kalman_filter_window(self.buf, self.q, self.r, self.p0) if self.use_kalman else self.buf.astype(np.float32)
        if self.model.startswith('cnn'):
            features = signal.reshape(250, 6).mean(axis=1)
        else:
            features = causal_features(signal, FS)
        return None if features is None else float(self.predict_proba(features[None])[0, 0])
