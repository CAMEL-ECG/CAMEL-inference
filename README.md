# CAMEL Inference

Inference-only repository for running CAMEL ECG-language checkpoints.

Only `run_camel.py` is intended as a public entrypoint. Modules under `src/camel/` are internal implementation details and may change.

## Scope

- Inference only (no training workflow exposed)
- Checkpoint-driven execution from local `checkpoints/`
- CLI usage via `run_camel.py`

## Repository Layout

- `run_camel.py`: public inference CLI
- `src/camel/`: internal model, tokenizer, ECG packing, and loading utilities
- `checkpoints/`: local adapter/checkpoint files

## Requirements

- Python 3.10+
- CUDA-enabled PyTorch recommended for practical inference latency

## Install

```bash
conda create -n camel python=3.10 -y
conda activate camel
pip install -e .
```

## Checkpoints

Checkpoints must be downloaded with the repository script:

```bash
bash scripts/download_checkpoints.sh
```

## Usage

```bash
python run_camel.py \
  --mode forecast \
  --text "Forecast cardiac rhythm for the next 5 minutes." \
  --ecg /path/to/ecg \
  --device 0
```

Available modes:
- `base`
- `ecgbench`
- `forecast`

Required inputs:

- `--text`: prompt/query text
- `--ecg`: ECG input path/string consumed by the CLI

`--ecg` input formats:

- Path to CSV waveform data (for example: `/data/sample.csv`)
- Path to NumPy array files (`.npy` / `.npz`)
- WFDB record path/prefix
- Other dataset-specific ECG paths supported by the backend loader

Sampling flags:

- `--temperature`
- `--top-k`
- `--top-p`
- `--min-p`
- `--max-new-tokens`
