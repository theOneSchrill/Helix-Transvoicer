#!/usr/bin/env python3
"""
Helix Transvoicer - CUDA PyTorch Installation Script.

Run this script to install PyTorch with CUDA support for GPU acceleration.
"""

import subprocess
import sys


def main():
    print("=" * 60)
    print("Helix Transvoicer - CUDA Installation")
    print("=" * 60)
    print()

    # Detect CUDA version
    print("Checking for NVIDIA CUDA...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,cuda_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            info = result.stdout.strip()
            print(f"  Found NVIDIA GPU: {info}")
        else:
            print("  Warning: nvidia-smi not found or failed")
    except FileNotFoundError:
        print("  Warning: nvidia-smi not found - is NVIDIA driver installed?")

    print()
    print("Installing PyTorch with CUDA 12.1 support...")
    print("(This works with RTX 20xx, 30xx, 40xx series)")
    print()

    # Uninstall CPU-only torch first
    subprocess.run([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "torch", "torchaudio", "torchvision"
    ], capture_output=True)

    # Install CUDA version from PyTorch index
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print()
        print("CUDA 12.1 installation failed, trying CUDA 11.8...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        result = subprocess.run(cmd)

    print()
    print("=" * 60)

    if result.returncode == 0:
        # Verify installation
        print("Verifying installation...")
        verify_cmd = [
            sys.executable, "-c",
            "import torch; "
            "print(f'PyTorch version: {torch.__version__}'); "
            "print(f'CUDA available: {torch.cuda.is_available()}'); "
            "print(f'CUDA version: {torch.version.cuda}' if torch.cuda.is_available() else 'CUDA: Not available'); "
            "print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'GPU: None detected')"
        ]
        subprocess.run(verify_cmd)
        print()
        print("Installation complete! Restart Helix Transvoicer to use GPU.")
    else:
        print("Installation failed. Please install manually:")
        print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")

    print("=" * 60)


if __name__ == "__main__":
    main()
