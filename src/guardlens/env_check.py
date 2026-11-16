"""Environment check: Python, PyTorch, CUDA and the GPU."""

import platform
import sys
import time


def check_env() -> int:
    """Print environment info and run a tiny GPU matmul. Returns 0 if CUDA works, else 1."""
    import torch

    print(f"Python        : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"PyTorch       : {torch.__version__}")
    print(f"Built for CUDA: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("CUDA available: NO")
        return 1

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    major, minor = torch.cuda.get_device_capability(device)
    arch = f"sm_{major}{minor}"
    print("CUDA available: yes")
    print(f"GPU           : {props.name}")
    print(f"VRAM          : {props.total_memory / 1024**3:.1f} GB")
    print(f"Compute arch  : {arch} (in this build: {arch in torch.cuda.get_arch_list()})")

    start = time.time()
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    c = (a @ b).sum().item()
    torch.cuda.synchronize()
    end = time.time()
    print(f"Elapsed time: {end-start:3.2f}s")
    print(f"GPU matmul    : ok (checksum {c:.2f})")
    return 0
