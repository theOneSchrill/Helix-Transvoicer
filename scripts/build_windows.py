#!/usr/bin/env python3
"""
Helix Transvoicer - Windows Build Script

This script builds the Windows executable using PyInstaller.
Run this script to create a standalone Windows application.

Usage:
    python scripts/build_windows.py

Requirements:
    - Python 3.10+
    - PyInstaller (pip install pyinstaller)
    - All project dependencies installed

The script will create:
    - dist/HelixTransvoicer/HelixTransvoicer.exe (main application)
    - dist/HelixTransvoicer/ (folder with all dependencies)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


# Configuration
APP_NAME = "HelixTransvoicer"
APP_VERSION = "1.0.0"
ICON_NAME = "app.ico"
MAIN_SCRIPT = "src/helix_transvoicer/main.py"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def check_requirements():
    """Check if all requirements are met."""
    print("Checking requirements...")

    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher required")
        sys.exit(1)
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor} - OK")

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  PyInstaller {PyInstaller.__version__} - OK")
    except ImportError:
        print("Error: PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)

    # Check customtkinter
    try:
        import customtkinter
        print(f"  customtkinter - OK")
    except ImportError:
        print("Error: customtkinter not installed. Run: pip install customtkinter")
        sys.exit(1)


def clean_build_dirs(project_root: Path):
    """Clean previous build directories."""
    print("\nCleaning previous builds...")

    dirs_to_clean = ["build", "dist", f"{APP_NAME}.spec"]
    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists():
            if dir_path.is_dir():
                shutil.rmtree(dir_path)
            else:
                dir_path.unlink()
            print(f"  Removed: {dir_name}")


def find_icon(project_root: Path) -> Path:
    """Find the application icon."""
    # Check multiple possible locations
    icon_locations = [
        project_root / "assets" / ICON_NAME,
        project_root / ICON_NAME,
        project_root / "src" / "helix_transvoicer" / "assets" / ICON_NAME,
        project_root / "resources" / ICON_NAME,
    ]

    for icon_path in icon_locations:
        if icon_path.exists():
            return icon_path

    return None


def get_hidden_imports():
    """Get list of hidden imports for PyInstaller."""
    return [
        # Torch and CUDA
        "torch",
        "torch.nn",
        "torch.optim",
        "torch.utils.data",
        "torchaudio",
        "torchaudio.transforms",

        # Audio processing
        "librosa",
        "soundfile",
        "numpy",
        "scipy",
        "scipy.signal",

        # UI
        "customtkinter",
        "PIL",
        "matplotlib",
        "matplotlib.backends.backend_tkagg",

        # API
        "fastapi",
        "uvicorn",
        "httpx",
        "pydantic",

        # Utilities
        "yaml",
        "dotenv",
        "tqdm",
        "rich",

        # TkinterDnD2 for drag & drop
        "tkinterdnd2",
    ]


def get_data_files(project_root: Path):
    """Get additional data files to include."""
    data_files = []

    # Include customtkinter themes
    try:
        import customtkinter
        ctk_path = Path(customtkinter.__file__).parent
        data_files.append((str(ctk_path), "customtkinter"))
    except ImportError:
        pass

    # Include tkinterdnd2 if available
    try:
        import tkinterdnd2
        dnd_path = Path(tkinterdnd2.__file__).parent
        data_files.append((str(dnd_path), "tkinterdnd2"))
    except ImportError:
        pass

    # Include assets folder if exists
    assets_dir = project_root / "assets"
    if assets_dir.exists():
        data_files.append((str(assets_dir), "assets"))

    return data_files


def build_executable(project_root: Path):
    """Build the Windows executable."""
    print("\nBuilding Windows executable...")

    # Change to project root
    os.chdir(project_root)

    # Find icon
    icon_path = find_icon(project_root)

    # Build PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--windowed",  # No console window
        "--onedir",    # Create directory with dependencies
        "--clean",     # Clean before build
        "--noconfirm", # Don't ask for confirmation
    ]

    # Add icon if found
    if icon_path:
        cmd.extend(["--icon", str(icon_path)])
        print(f"  Using icon: {icon_path}")
    else:
        print(f"  Warning: Icon not found. Place {ICON_NAME} in the assets folder.")

    # Add hidden imports
    for hidden_import in get_hidden_imports():
        cmd.extend(["--hidden-import", hidden_import])

    # Add data files
    for src, dest in get_data_files(project_root):
        cmd.extend(["--add-data", f"{src}{os.pathsep}{dest}"])

    # Add collect-all for important packages
    collect_packages = ["customtkinter", "torch", "torchaudio"]
    for pkg in collect_packages:
        cmd.extend(["--collect-all", pkg])

    # Main script
    cmd.append(MAIN_SCRIPT)

    print(f"\n  Running: {' '.join(cmd[:10])}...")

    # Run PyInstaller
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("\nError: Build failed!")
        sys.exit(1)


def create_launcher(project_root: Path):
    """Create a launcher script for the backend server."""
    dist_dir = project_root / "dist" / APP_NAME

    # Create backend launcher
    launcher_content = '''@echo off
echo Starting Helix Transvoicer Backend Server...
cd /d "%~dp0"
python -m uvicorn helix_transvoicer.backend.main:app --host 127.0.0.1 --port 8420
pause
'''

    launcher_path = dist_dir / "start_backend.bat"
    launcher_path.write_text(launcher_content)
    print(f"\n  Created: {launcher_path.name}")


def print_summary(project_root: Path):
    """Print build summary."""
    dist_dir = project_root / "dist" / APP_NAME
    exe_path = dist_dir / f"{APP_NAME}.exe"

    print("\n" + "=" * 60)
    print("BUILD COMPLETE!")
    print("=" * 60)
    print(f"\n  Application: {APP_NAME}")
    print(f"  Version: {APP_VERSION}")
    print(f"\n  Output directory: {dist_dir}")

    if exe_path.exists():
        size_mb = exe_path.stat().st_size / (1024 * 1024)
        print(f"  Executable size: {size_mb:.1f} MB")

    print("\n  To run the application:")
    print(f"    1. Navigate to: {dist_dir}")
    print(f"    2. Run: {APP_NAME}.exe")
    print("\n  Note: The backend server runs automatically with the app.")
    print("=" * 60)


def main():
    """Main build function."""
    print("=" * 60)
    print(f"  HELIX TRANSVOICER - Windows Build")
    print(f"  Version {APP_VERSION}")
    print("=" * 60)

    project_root = get_project_root()
    print(f"\nProject root: {project_root}")

    # Check requirements
    check_requirements()

    # Clean previous builds
    clean_build_dirs(project_root)

    # Build executable
    build_executable(project_root)

    # Create launcher scripts
    create_launcher(project_root)

    # Print summary
    print_summary(project_root)


if __name__ == "__main__":
    main()
