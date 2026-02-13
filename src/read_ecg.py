from typing import Optional
import numpy as np
import wfdb

def load_record(ecg_path, start_sec: Optional[int], end_sec: Optional[int], leads: Optional[list[str]]):
    record = wfdb.rdrecord(ecg_path)
    fs = record.fs
    lead_names = record.sig_name

    signal = record.p_signal # n_samples x n_leads
    if leads:
        kept_signals, kept_leads = [], []
        lead_to_idx = {name: i for i, name in enumerate(lead_names)}
        for l in leads:
            if l in lead_to_idx:
                kept_signals.append(signal[:, lead_to_idx[l]])
                kept_leads.append(l)
            else:
                print(f'Lead {l} does not exist. Skipping.')
        if not kept_signals:
            raise ValueError(f"None of the requested leads were found. requested={leads}, available={lead_names}")
        
        signal = np.stack(kept_signals, axis=1)
        lead_names = kept_leads

    # Optinally subsample the signal
    start_ind = 0 if start_sec is None else start_sec * fs
    end_ind = len(signal) if end_sec is None else end_sec * fs
    if end_ind > len(signal):
        print(f'ECG is {len(signal) / fs} seconds')
    signal = signal[start_ind:end_ind, :]

    return signal, lead_names, fs