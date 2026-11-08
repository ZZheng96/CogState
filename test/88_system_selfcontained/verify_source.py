"""逐函数比对 system/algorithms 与原代码的源码（去空行/注释/文档串后）。"""
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / p) for p in ("src", "test/17_compliant", "test/03_deep", "test/08_diagnostics", "test/01_features")] + [str(ROOT)]
import loso, multiscale, streaming, cogstate, preprocessing, infer, eeg_decoder  # noqa: E401,E402
import highdim_deep_ablation as H, nonstationary_ablation as N, emg_v2  # noqa: E402
from system.algorithms import train_utils as T, e17 as E, io as I, features as F, postprocess as P  # noqa: E402

pairs = [(loso.run_stats_t, T.run_stats_t), (loso.fit_lr, T.fit_lr), (multiscale.fit_mlp, T.fit_mlp),
         (streaming._Scorer, E._Scorer), (streaming.StreamingInference, E.StreamingInference),
         (streaming.train_bundle, E.train_bundle), (streaming.window_features, F.window_features),
         (cogstate.load_trials, I.load_trials), (cogstate._file_start, I._file_start),
         (cogstate.valid_window, I.valid_window), (cogstate.bad_intervals, I.bad_intervals),
         (eeg_decoder.load_bdf, I.load_bdf), (eeg_decoder.parse_recording_start, I.parse_recording_start),
         (preprocessing.quality_metrics, F.quality_metrics), (preprocessing.causal_filter, F.causal_filter),
         (preprocessing.sos_filters, F.sos_filters),
         (H.statistics, F.statistics), (H.nonlinear, F.nonlinear), (H.highdim_features, F.highdim_features),
         (N.check_window, F.check_window), (N.wavelet_features, F.wavelet_features),
         (N.timefreq_features, F.timefreq_features), (N.spectrogram, F.spectrogram), (N.feature_blocks, F.feature_blocks),
         (emg_v2.features_v2, F.features_v2),
         (infer.PostProcessor, P.PostProcessor), (infer.ForwardHMM, P.ForwardHMM), (infer.CausalDetrend, P.CausalDetrend)]


def norm(f):
    out = []
    for line in inspect.getsource(f).splitlines():
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith('"""'):
            out.append(s.split("  #")[0].rstrip())
    return out


for a, b in pairs:
    x, y = norm(a), norm(b)
    d = [(u, v) for u, v in zip(x, y) if u != v]
    print(f"{a.__qualname__:22s}", "same" if not d and len(x) == len(y) else f"DIFF {len(x)} vs {len(y)}: {d[:4]}")
print("consts", streaming.SCALES == E.SCALES, streaming.WIN_S == E.WIN_S, streaming.C_LR == E.C_LR,
      streaming.SEEDS == E.SEEDS, cogstate.BANDS == F.BANDS, emg_v2.FINE == F.FINE, H.STAT_NAMES == F.STAT_NAMES,
      preprocessing.QUALITY_NAMES == F.QUALITY_NAMES, preprocessing.PreprocessingConfig() .__dict__ == F.PreprocessingConfig().__dict__)
