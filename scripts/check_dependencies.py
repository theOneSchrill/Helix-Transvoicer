#!/usr/bin/env python3
"""
Helix Transvoicer - Dependency Checker

Checks for missing dependencies and offers to install them.
Run before starting the application.
"""

import subprocess
import sys
from pathlib import Path

# Core dependencies to check (module_name, pip_package, description)
DEPENDENCIES = [
    # Core ML
    ("torch", "torch", "PyTorch (Machine Learning)"),
    ("torchaudio", "torchaudio", "TorchAudio"),
    ("numpy", "numpy", "NumPy"),
    ("scipy", "scipy", "SciPy"),

    # Audio processing
    ("librosa", "librosa", "Librosa (Audio Analysis)"),
    ("soundfile", "soundfile", "SoundFile"),
    ("pydub", "pydub", "PyDub"),
    ("noisereduce", "noisereduce", "NoiseReduce"),

    # Voice processing
    ("parselmouth", "praat-parselmouth", "Parselmouth (Voice Conversion)"),

    # API framework
    ("fastapi", "fastapi", "FastAPI"),
    ("uvicorn", "uvicorn[standard]", "Uvicorn"),
    ("pydantic", "pydantic", "Pydantic"),
    ("pydantic_settings", "pydantic-settings", "Pydantic Settings"),
    ("aiofiles", "aiofiles", "AioFiles"),
    ("httpx", "httpx", "HTTPX"),

    # Configuration
    ("yaml", "pyyaml", "PyYAML"),
    ("dotenv", "python-dotenv", "Python-dotenv"),

    # UI
    ("customtkinter", "customtkinter", "CustomTkinter (UI)"),
    ("PIL", "pillow", "Pillow (Images)"),
    ("matplotlib", "matplotlib", "Matplotlib"),

    # Utilities
    ("tqdm", "tqdm", "TQDM (Progress Bars)"),
    ("rich", "rich", "Rich (Console Output)"),
]

# Optional dependencies
OPTIONAL_DEPENDENCIES = [
    ("tkinterdnd2", "tkinterdnd2", "TkinterDnD2 (Drag & Drop)"),
]


def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def get_missing_dependencies() -> list:
    """Get list of missing dependencies."""
    missing = []

    for module_name, pip_package, description in DEPENDENCIES:
        if not check_import(module_name):
            missing.append((module_name, pip_package, description))

    return missing


def get_missing_optional() -> list:
    """Get list of missing optional dependencies."""
    missing = []

    for module_name, pip_package, description in OPTIONAL_DEPENDENCIES:
        if not check_import(module_name):
            missing.append((module_name, pip_package, description))

    return missing


def install_packages(packages: list) -> bool:
    """Install packages using pip."""
    pip_packages = [pkg for _, pkg, _ in packages]

    print(f"\n  Installing {len(pip_packages)} package(s)...")
    print()

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--upgrade", "--quiet",
            *pip_packages
        ])
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] Installation failed: {e}")
        return False


def main():
    """Main entry point."""
    print()
    print("  " + "=" * 50)
    print("  Helix Transvoicer - Dependency Check")
    print("  " + "=" * 50)
    print()

    # Check for missing dependencies
    missing = get_missing_dependencies()
    missing_optional = get_missing_optional()

    if not missing and not missing_optional:
        print("  [OK] All dependencies are installed!")
        print()
        return 0

    # Show missing required dependencies
    if missing:
        print(f"  [!] Missing {len(missing)} required package(s):")
        print()
        for module_name, pip_package, description in missing:
            print(f"      - {description} ({pip_package})")
        print()

    # Show missing optional dependencies
    if missing_optional:
        print(f"  [i] Missing {len(missing_optional)} optional package(s):")
        print()
        for module_name, pip_package, description in missing_optional:
            print(f"      - {description} ({pip_package})")
        print()

    # Ask user
    if missing:
        print("  " + "-" * 50)
        response = input("  Install missing required packages? [Y/n]: ").strip().lower()

        if response in ("", "y", "yes", "j", "ja"):
            if install_packages(missing):
                print()
                print("  [OK] Required packages installed successfully!")
            else:
                print()
                print("  [ERROR] Some packages failed to install.")
                print("  Try running manually: pip install -r requirements.txt")
                return 1
        else:
            print()
            print("  [!] Skipping installation. App may not work correctly.")
            return 1

    # Ask about optional
    if missing_optional:
        print()
        response = input("  Install optional packages too? [y/N]: ").strip().lower()

        if response in ("y", "yes", "j", "ja"):
            if install_packages(missing_optional):
                print()
                print("  [OK] Optional packages installed!")
            else:
                print("  [!] Some optional packages failed (not critical).")

    print()
    print("  [OK] Dependency check complete!")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
