#!/usr/bin/env python3
"""
Helix Transvoicer - CUDA PyTorch Installation Script.

Run this script to install PyTorch with CUDA support for GPU acceleration.
Requires Python 3.8-3.12 (PyTorch does not support Python 3.13 yet).
"""

import subprocess
import sys
import platform


def check_python_version():
    """Check if Python version is compatible with PyTorch CUDA."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major != 3 or version.minor < 8 or version.minor > 12:
        print()
        print("ERROR: PyTorch with CUDA requires Python 3.8-3.12")
        print(f"       You have Python {version.major}.{version.minor}")
        print()
        print("Please install Python 3.10, 3.11, or 3.12 from:")
        print("  https://www.python.org/downloads/")
        print()
        print("After installing, run this script with that Python version:")
        print("  py -3.11 install_cuda.py")
        print("  or")
        print("  python3.11 install_cuda.py")
        return False
    return True


def check_nvidia_gpu():
    """Check for NVIDIA GPU on Windows."""
    print("Checking for NVIDIA GPU...")

    # On Windows, try nvidia-smi from standard location
    nvidia_smi_paths = [
        "nvidia-smi",
        r"C:\Windows\System32\nvidia-smi.exe",
        r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
    ]

    for smi_path in nvidia_smi_paths:
        try:
            result = subprocess.run(
                [smi_path, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                print(f"  Found: {gpu_info}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    print("  Warning: Could not detect NVIDIA GPU via nvidia-smi")
    print("  Make sure NVIDIA drivers are installed from:")
    print("  https://www.nvidia.com/Download/index.aspx")
    return False


def uninstall_cpu_torch():
    """Uninstall CPU-only PyTorch if present."""
    print("Removing existing PyTorch installation...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio", "torchvision"],
        capture_output=True,
    )


def install_cuda_torch(cuda_version="cu118"):
    """Install PyTorch with CUDA support."""
    index_url = f"https://download.pytorch.org/whl/{cuda_version}"

    print(f"Installing PyTorch with CUDA ({cuda_version})...")
    print(f"Index URL: {index_url}")
    print()

    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchaudio",
        "--index-url", index_url,
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def verify_installation():
    """Verify PyTorch CUDA installation."""
    print()
    print("Verifying installation...")

    verify_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA compiled version: {torch.version.cuda or 'None (CPU only)'}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    print("SUCCESS! GPU acceleration is enabled.")
else:
    print()
    print("WARNING: CUDA is not available.")
    if torch.version.cuda:
        print("PyTorch has CUDA support, but no GPU was detected.")
        print("Make sure NVIDIA drivers are installed.")
    else:
        print("PyTorch was installed without CUDA support.")
"""

    result = subprocess.run([sys.executable, "-c", verify_script])
    return result.returncode == 0


def main():
    print("=" * 60)
    print("Helix Transvoicer - CUDA Installation")
    print("=" * 60)
    print()

    # Check Python version
    if not check_python_version():
        print("=" * 60)
        return 1

    print()

    # Check for NVIDIA GPU
    check_nvidia_gpu()

    print()

    # Uninstall existing torch
    uninstall_cpu_torch()

    print()

    # Try CUDA 11.8 first (more compatible)
    print("Attempting CUDA 11.8 installation (recommended for RTX 30xx)...")
    print()

    if install_cuda_torch("cu118"):
        verify_installation()
    else:
        print()
        print("CUDA 11.8 failed. Trying CUDA 12.1...")
        print()

        if install_cuda_torch("cu121"):
            verify_installation()
        else:
            print()
            print("ERROR: Installation failed.")
            print()
            print("Manual installation:")
            print("1. Make sure you have Python 3.10-3.12 (not 3.13)")
            print("2. Run: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print()
            return 1

    print()
    print("=" * 60)
    print("Restart Helix Transvoicer to use GPU acceleration.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
