"""Quick CUDA / PyTorch sanity check.

Run this once after `setup.ps1` to confirm the GPU is reachable from PyTorch.
Prints versions and the device name. Exits with code 1 if CUDA is not available.

Usage:
    python scripts/check_gpu.py
"""

import sys  # gives us sys.exit() to end the program with a status code

import torch  # PyTorch: the deep-learning library we are checking for GPU support


# "-> int" is a type hint: it says this function returns an integer.
# Hints are notes for humans/tools only; Python does not enforce them.
def main() -> int:
    """Print PyTorch / CUDA / GPU info, return 0 if CUDA available else 1."""
    # f-strings: the f"..." prefix lets us drop {expressions} inside the text.
    # Each {...} below is replaced by the value it evaluates to when printed.
    print(f"PyTorch:       {torch.__version__}")   # installed PyTorch version
    print(f"CUDA built:    {torch.version.cuda}")  # CUDA version PyTorch was compiled with
    print(f"CUDA available: {torch.cuda.is_available()}")  # True if a usable GPU is found

    # If no GPU is usable, warn and return 1. A non-zero return code is the
    # standard way to signal "something is wrong" to the calling script/shell.
    if not torch.cuda.is_available():
        print("\nWARNING: CUDA is not available. Models will run on CPU.")
        return 1

    print(f"GPU count:     {torch.cuda.device_count()}")  # how many GPUs are visible
    # range(n) yields 0, 1, ... n-1, so this loops once per GPU by index.
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)  # hardware details for GPU i
        mem_gb = props.total_memory / (1024 ** 3)    # bytes -> GB (1024**3 = bytes per GB)
        # ":.1f" inside the f-string rounds mem_gb to 1 decimal place.
        print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")

    # Quick smoke test: tensor allocation and matmul
    x = torch.randn(1000, 1000, device="cuda")  # random 1000x1000 matrix placed on the GPU
    y = x @ x.T  # "@" is matrix multiply; x.T is the transpose of x
    # GPU work is asynchronous; synchronize() waits for it to actually finish
    # so the timing/result below reflects completed work.
    torch.cuda.synchronize()
    print(f"\nSmoke test:    {y.shape} on {y.device} OK")  # confirm shape + that it ran on GPU
    return 0  # 0 = success


# This block only runs when the file is executed directly (python check_gpu.py),
# not when it is imported by another module. sys.exit() ends the process and
# passes main()'s return value out as the exit code (0 = ok, 1 = problem).
if __name__ == "__main__":
    sys.exit(main())
