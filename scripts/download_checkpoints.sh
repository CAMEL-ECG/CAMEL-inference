#!/usr/bin/env bash
set -euo pipefail

echo "Installing huggingface_hub if needed..."
python3 -m pip install -q --user huggingface_hub

echo "Downloading CAMEL checkpoints from Hugging Face..."
mkdir -p checkpoints

python3 - <<'PY'
import os, shutil
from huggingface_hub import hf_hub_download

repo = "CAMEL-ECG/CAMEL"
files = [
    "camel_base.pt",
    "camel_ecginstruct.pt",
    "camel_forecast.pt"
]

os.makedirs("checkpoints", exist_ok=True)

for f in files:
    print(f"Downloading {f}...")
    src = hf_hub_download(
        repo_id=repo,
        filename=f,
        repo_type="model"
    )
    dst = os.path.join("checkpoints", f)
    shutil.copy2(src, dst)
    print(f"Saved to {dst}")

print("All checkpoints downloaded.")
PY

echo "Done."
