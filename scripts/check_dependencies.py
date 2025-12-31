#!/usr/bin/env python3
"""
Helix Transvoicer - Dependency Checker

Checks for missing dependencies and offers to install them.
Run before starting the application.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Suppress pydub warning during import check
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

# System tools (not pip packages)
SYSTEM_TOOLS = [
    ("ffmpeg", "FFmpeg (Audio/Video Processing)"),
]


def check_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def check_system_tool(tool_name: str) -> bool:
    """Check if a system tool is available."""
    return shutil.which(tool_name) is not None


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


def get_missing_system_tools() -> list:
    """Get list of missing system tools."""
    missing = []

    for tool_name, description in SYSTEM_TOOLS:
        if not check_system_tool(tool_name):
            missing.append((tool_name, description))

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


def install_ffmpeg_windows() -> bool:
    """Try to install ffmpeg on Windows using winget or choco."""
    print("\n  Attempting to install FFmpeg...")

    # Try winget first (Windows 10/11)
    if shutil.which("winget"):
        try:
            print("  Using winget...")
            subprocess.check_call(
                ["winget", "install", "Gyan.FFmpeg", "-e", "--silent"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("  [OK] FFmpeg installed via winget!")
            print("  [!] Please restart your terminal for changes to take effect.")
            return True
        except subprocess.CalledProcessError:
            pass

    # Try chocolatey
    if shutil.which("choco"):
        try:
            print("  Using chocolatey...")
            subprocess.check_call(
                ["choco", "install", "ffmpeg", "-y"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("  [OK] FFmpeg installed via chocolatey!")
            return True
        except subprocess.CalledProcessError:
            pass

    return False


def show_ffmpeg_instructions():
    """Show manual FFmpeg installation instructions."""
    print()
    print("  " + "-" * 50)
    print("  FFmpeg Installation Instructions:")
    print("  " + "-" * 50)
    print()
    print("  Option 1 - Using winget (Windows 10/11):")
    print("    winget install Gyan.FFmpeg")
    print()
    print("  Option 2 - Using chocolatey:")
    print("    choco install ffmpeg")
    print()
    print("  Option 3 - Manual download:")
    print("    1. Go to: https://www.gyan.dev/ffmpeg/builds/")
    print("    2. Download 'ffmpeg-release-essentials.zip'")
    print("    3. Extract to C:\\ffmpeg")
    print("    4. Add C:\\ffmpeg\\bin to your PATH")
    print()
    print("  After installing, restart your terminal.")
    print("  " + "-" * 50)


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
    missing_tools = get_missing_system_tools()

    all_ok = not missing and not missing_optional and not missing_tools

    if all_ok:
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

    # Show missing system tools
    if missing_tools:
        print(f"  [!] Missing {len(missing_tools)} system tool(s):")
        print()
        for tool_name, description in missing_tools:
            print(f"      - {description}")
        print()
        print("      These are NOT Python packages and need separate installation.")
        print()

    # Install Python packages
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

    # Ask about optional packages
    if missing_optional:
        print()
        response = input("  Install optional packages too? [y/N]: ").strip().lower()

        if response in ("y", "yes", "j", "ja"):
            if install_packages(missing_optional):
                print()
                print("  [OK] Optional packages installed!")
            else:
                print("  [!] Some optional packages failed (not critical).")

    # Handle missing system tools
    if missing_tools:
        for tool_name, description in missing_tools:
            if tool_name == "ffmpeg":
                print()
                response = input("  Try to install FFmpeg automatically? [Y/n]: ").strip().lower()

                if response in ("", "y", "yes", "j", "ja"):
                    if sys.platform == "win32":
                        if not install_ffmpeg_windows():
                            print("  [!] Automatic installation failed.")
                            show_ffmpeg_instructions()
                    else:
                        print("  [i] On Linux/Mac, install via package manager:")
                        print("      Ubuntu/Debian: sudo apt install ffmpeg")
                        print("      macOS: brew install ffmpeg")
                else:
                    show_ffmpeg_instructions()

    print()
    print("  [OK] Dependency check complete!")
    print()

    # Return success even if ffmpeg is missing (it's not critical for all features)
    return 0


if __name__ == "__main__":
    sys.exit(main())
