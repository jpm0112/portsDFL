#!/usr/bin/env bash
# Auto-detect CUDA, create venv, install deps. Linux/Mac mirror of setup.ps1.
set -euo pipefail

index_url="https://download.pytorch.org/whl/cpu"
cuda_label="CPU"

if command -v nvidia-smi >/dev/null 2>&1; then
    cuda=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version:\s+[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)
    if [[ -n "${cuda:-}" ]]; then
        major=${cuda%.*}
        minor=${cuda#*.}
        if   (( major >= 13 )); then
            index_url="https://download.pytorch.org/whl/cu130"; cuda_label="CUDA 13.0"
        elif (( major == 12 )) && (( minor >= 6 )); then
            index_url="https://download.pytorch.org/whl/cu126"; cuda_label="CUDA 12.6"
        elif (( major == 12 )) && (( minor >= 4 )); then
            index_url="https://download.pytorch.org/whl/cu124"; cuda_label="CUDA 12.4"
        elif (( major == 12 )) && (( minor >= 1 )); then
            index_url="https://download.pytorch.org/whl/cu121"; cuda_label="CUDA 12.1"
        elif (( major == 11 )) && (( minor >= 8 )); then
            index_url="https://download.pytorch.org/whl/cu118"; cuda_label="CUDA 11.8"
        fi
        echo "Detected CUDA $cuda -> $cuda_label"
    fi
else
    echo "No NVIDIA GPU detected -> CPU wheels."
fi

[[ -d .venv ]] || python -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install --extra-index-url "$index_url" -r requirements.txt
.venv/bin/python -m pip install -e .

echo
echo "=== Setup complete ==="
echo "Activate venv:  source .venv/bin/activate"
echo "Verify GPU:     python scripts/check_gpu.py"
echo "Run tests:      pytest -q"
