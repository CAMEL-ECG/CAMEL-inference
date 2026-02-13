from typing import Optional, List, Dict, Any
import numpy as np
import torch

from read_ecg import load_record

_LEAD_SYNONYMS: Dict[str, str] = {
    # Limb
    "I": "I", "II": "II", "III": "III",
    "DI": "I", "DII": "II", "DIII": "III",
    "MLII": "II",
    # Augmented
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    # Precordial
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    # Dataset-specific
    "ECG": "I",      # Apnea-ECG
    "ECG1": "I", "ECG2": "II", # AFDB
    "CM5": "V5", "D3": "V3", "D4": "V4",
    "CM2": "V2", "ML5": "V5",
    "VF": "VF",
}

class NormOp:
    def __init__(self, name: str, params: Dict[str, float] | None = None):
        self.name = name
        self.params = params or {}

def to_canonical_lead(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    s = name.strip().upper()
    if s in _LEAD_SYNONYMS:
        return _LEAD_SYNONYMS[s]
    if s in ("A VR", "A VL", "A VF"):
        return "a" + s.replace(" ", "")[1:]
    return None

def parse_pipeline(spec: Optional[str]) -> List[NormOp]:
    if not spec:
        return [NormOp("nonfinite_to_zero"), NormOp("clip", {})]
    ops: List[NormOp] = []
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        parts = token.split(':')
        name = parts[0].lower()
        if name == 'clip':
            mn = None; mx = None
            if len(parts) >= 2 and parts[1] != '':
                mn = float(parts[1])
            if len(parts) >= 3 and parts[2] != '':
                mx = float(parts[2])
            ops.append(NormOp('clip', {'min': mn, 'max': mx}))
        elif name == 'nonfinite_to_zero':
            ops.append(NormOp('nonfinite_to_zero'))
        else:
            raise ValueError(f"Unknown op '{name}'.")
    return ops

def apply_pre_ops(signal_1d: np.ndarray, ops: List[NormOp]) -> np.ndarray:
    x = signal_1d
    for op in ops:
        if op.name == 'nonfinite_to_zero':
            np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return x


def apply_post_ops(segments: np.ndarray, ops: List[NormOp], lead_name: str,
                   clip_stats: Optional[Dict[str, Dict[str, float]]] = None) -> np.ndarray:
    x = segments
    for op in ops:
        if op.name == 'clip':
            mn = op.params.get('min', None)
            mx = op.params.get('max', None)
            if mn is None and mx is None and clip_stats is not None and lead_name in clip_stats:
                mn = clip_stats[lead_name].get('clip_min', None)
                mx = clip_stats[lead_name].get('clip_max', None)
            if mn is None:
                mn = -np.inf
            if mx is None:
                mx = np.inf
            if mn > mx:
                mn, mx = mx, mn
            np.clip(x, mn, mx, out=x)
    return x.astype(np.float32, copy=False)

def _segment_data(ecg_signal: np.ndarray, raw_fs: int) -> np.ndarray:
    """
    Segment 1D signal sampled at ``raw_fs`` into 1-second clips and resample to 256 samples.
    Returns np.float32 array [N_segments, 256].
    """
    assert raw_fs > 0, f"raw sampling rate must be positive (got {raw_fs})"
    samples_per_second = int(raw_fs)
    n_samples = int(len(ecg_signal))
    if samples_per_second <= 0 or n_samples <= 0:
        return np.empty((0, 256), dtype=np.float32)

    # Pre-allocate upper bound
    n_full = n_samples // samples_per_second
    has_partial = (n_samples % samples_per_second) > 0
    max_segments = n_full + (1 if has_partial else 0)
    if max_segments == 0:
        return np.empty((0, 256), dtype=np.float32)

    out = np.empty((max_segments, 256), dtype=np.float32)
    new_idx = np.linspace(0, 1, num=256, dtype=np.float32)
    actual = 0
    for start in range(0, n_samples, samples_per_second):
        end = min(start + samples_per_second, n_samples)
        seg = ecg_signal[start:end]
        if seg.shape[0] < samples_per_second * 0.5:
            continue
        old_idx = np.linspace(0, 1, num=seg.shape[0], dtype=np.float32)
        out[actual] = np.interp(new_idx, old_idx, seg).astype(np.float32, copy=False)
        actual += 1
    return out[:actual]

def _apply_filters(signal_1d, fs: int):
    """
    Apply ECG signal filters: 50/60 Hz notch filters and 0.3 Hz high-pass filter.
    
    Removes powerline interference and baseline wander from ECG signals.
    Uses cascaded second-order sections (SOS) for numerical stability.
    """
    import numpy as np
    from scipy.signal import iirnotch, butter, sosfiltfilt, tf2sos

    x = np.asarray(signal_1d, dtype=np.float64)
    if x.size == 0 or fs <= 0:
        return x.astype('float32')

    # Design notch filters for powerline interference
    Q = 30.0
    nyq = fs / 2.0
    sos_filters = []
    for freq in (50.0, 60.0):
        if freq >= nyq:
            continue
        b, a = iirnotch(freq, Q, fs)
        sos_filters.append(tf2sos(b, a))
    
    # Add high-pass filter for baseline wander removal
    sos_hp = butter(N=2, Wn=0.3 / nyq, btype='highpass', output='sos')
    sos_filters.append(sos_hp)
    
    # Apply all filters in cascade
    sos = np.vstack(sos_filters)
    x = sosfiltfilt(sos, x)
    return x.astype('float32')

def get_waveform(device:torch.device, ecg_path:str, start_sec = None, end_sec = None, leads: Optional[List[str]] = None,
                 process:bool = False, norm: str = "nonfinite_to_zero,clip") -> Dict[str, Any]:
    """
    Run ECG preprocessing pipeline (Step II): raw → filter → segment → normalize → storage.
    
    Processes ECG records from manifest, applies signal filtering, segments into 1s windows,
    normalizes using computed clip stats, and writes to efficient storage format.
    
    Args:
        dataset: Dataset name
        p: file path
        clip_stats_path: Path to clip_stats json file
        process: Y/N process
        fs: Target sampling frequency
        leads: Optional list of leads to process
        norm: Normalization pipeline (e.g., "nonfinite_to_zero,clip")
    
    Returns:
        Dict with output paths: {output_dir, index, clip_stats}
    """

    # Parallel path: use per-worker shard/npy directories then aggregate
    ops = parse_pipeline(norm)

    # Convert to dict format
    ecg_dict = {}
    try:
        df, sig_names, original_fs = load_record(ecg_path, start_sec, end_sec, leads)
        
        # Process each lead in the record
        for i, raw_name in enumerate(sig_names):
            canon = to_canonical_lead(raw_name)
            if not canon:
                print('to_canonical_lead')
                continue
            x = df[:, i].astype('float32', copy=False)

            if process:
                x = apply_pre_ops(x, ops)  # Handle non-finite values
                x = _apply_filters(x, original_fs)  # Remove noise and baseline wander

            # Segment into 1s windows and normalize
            segs = _segment_data(x, original_fs)
            segs = apply_post_ops(segs, ops, canon)

            lead_tensor = torch.from_numpy(segs).to(torch.float32).to(device)
            if not (torch.any(lead_tensor.isnan())):
                lead_tensor = lead_tensor.nan_to_num()

            ecg_dict[canon] = lead_tensor
    
    except Exception as e:
        # Skip failed records
        print(e)

    return ecg_dict
