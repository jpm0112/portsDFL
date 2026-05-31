"""Quick CUDA / PyTorch sanity check.

Run this once after `setup.ps1` to confirm the GPU is reachable from PyTorch.
Prints versions and the device name. Exits with code 1 if CUDA is not available.

Usage:
    python scripts/check_gpu.py
"""

import sys

import torch


def main() -> int:
    """Print PyTorch / CUDA / GPU info, return 0 if CUDA available else 1."""
    print(f"PyTorch:       {torch.__version__}")
    print(f"CUDA built:    {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Non-zero return code signals failure to the calling script/shell.
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA is not available. Models will run on CPU.")
        return 1

    print(f"GPU count:     {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")

    # Quick smoke test: tensor allocation and matmul
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    # GPU work is asynchronous; wait for it to finish before reporting.
    torch.cuda.synchronize()
    print(f"\nSmoke test:    {y.shape} on {y.device} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
