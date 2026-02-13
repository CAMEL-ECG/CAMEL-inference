# CAMEL Inference

Inference-only repository for running CAMEL ECG-language checkpoints.

Only `run_camel.py` is intended as a public entrypoint. Modules under `src/camel/` are internal implementation details and may change.

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

Checkpoints must be downloaded from huggingface `CAMEL-ECG/CAMEL` or with the repository script:

```bash
bash scripts/download_checkpoints.sh
```

## Usage

* CAMEL is available in three modes: 
  - `base`
  - `ecgbench`
  - `forecast`

  ```bash
  python run_camel.py \
    --mode forecast \
    --text "Forecast cardiac rhythm for the next 5 minutes." \
    --ecgs demo/08704_hr \
    --device cuda:0
  ```

  ```bash
  python run_camel.py \
    --mode base \
    --text "Compare the two ECG waveforms." \
    --ecgs demo/12585_hr demo/12646_hr \
    --device cuda:0
  ```

* Optionally, you can set start, end, and leads with `--ecgs-config`.

  ```bash
  python run_camel.py \
    --mode forecast \
    --text "Forecast cardiac rhythm for the next 5 minutes." \
    --ecgs demo/08704_hr \
    --ecg-configs "start:0;end:5;use_leads:I,II" \
    --device cuda:0
  ```

* Using `--text` and `--ecgs` defaults to text followed by the ecg in order. 
For arbitrary text/ECG interleaving use `--json`.
  ```bash
  python run_camel.py --mode base --json demo/example_prompt.json --device cuda:0
  ```

* Sampling flags:
  - `--temperature`
  - `--top-k`
  - `--top-p`
  - `--min-p`
  - `--max-new-tokens`

Implementation notes:
- ECG loading is currently implemented for WFDB-format inputs. To support additional formats, extend `src/read_ecg.py`.