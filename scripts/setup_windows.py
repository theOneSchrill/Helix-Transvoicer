#!/usr/bin/env python3
"""
Helix Transvoicer - Windows Setup Script

This script sets up Helix Transvoicer on Windows 11.
Run this script after cloning the repository to install dependencies
and configure the application.

Usage:
    python scripts/setup_windows.py

What this script does:
    1. Checks Python version (requires 3.10+)
    2. Creates virtual environment (optional)
    3. Installs required dependencies
    4. Verifies CUDA/GPU availability
    5. Creates desktop shortcut (optional)
    6. Creates start menu entry (optional)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


APP_NAME = "Helix Transvoicer"
APP_VERSION = "1.0.0"
MIN_PYTHON_VERSION = (3, 10)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}{Colors.ENDC}\n")


def print_step(step: int, text: str):
    """Print a step."""
    print(f"{Colors.BLUE}[{step}]{Colors.ENDC} {text}")


def print_success(text: str):
    """Print success message."""
    print(f"  {Colors.GREEN}OK{Colors.ENDC} {text}")


def print_warning(text: str):
    """Print warning message."""
    print(f"  {Colors.YELLOW}WARNING{Colors.ENDC} {text}")


def print_error(text: str):
    """Print error message."""
    print(f"  {Colors.RED}ERROR{Colors.ENDC} {text}")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.resolve()


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    print_step(1, "Checking Python version...")

    version = sys.version_info
    required = f"{MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"

    if version >= MIN_PYTHON_VERSION:
        print_success(f"Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor} found, but {required}+ required")
        return False


def check_pip() -> bool:
    """Check if pip is available."""
    print_step(2, "Checking pip...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            pip_version = result.stdout.split()[1]
            print_success(f"pip {pip_version} - OK")
            return True
    except Exception as e:
        pass

    print_error("pip not found. Please install pip first.")
    return False


def install_dependencies(project_root: Path) -> bool:
    """Install project dependencies."""
    print_step(3, "Installing dependencies...")

    requirements_file = project_root / "requirements.txt"
    if not requirements_file.exists():
        print_error("requirements.txt not found")
        return False

    print("    This may take a few minutes...")

    # Upgrade pip first
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True,
    )

    # Install requirements
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        capture_output=False,
    )

    if result.returncode == 0:
        print_success("Dependencies installed successfully")
        return True
    else:
        print_error("Failed to install some dependencies")
        return False


def check_cuda() -> Optional[str]:
    """Check CUDA/GPU availability."""
    print_step(4, "Checking GPU/CUDA support...")

    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print_success(f"CUDA {cuda_version} available")
            print_success(f"GPU: {device_name}")
            return device_name
        else:
            print_warning("CUDA not available - will use CPU (slower)")
            return None
    except ImportError:
        print_warning("PyTorch not yet installed - GPU check skipped")
        return None


def install_package(project_root: Path) -> bool:
    """Install the helix_transvoicer package."""
    print_step(5, "Installing Helix Transvoicer package...")

    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(project_root)],
        capture_output=True,
    )

    if result.returncode == 0:
        print_success("Package installed in development mode")
        return True
    else:
        print_warning("Package installation skipped (run manually if needed)")
        return True  # Non-fatal


def create_run_script(project_root: Path):
    """Create a run.bat script for easy launching."""
    print_step(6, "Creating launcher script...")

    run_script = project_root / "run.bat"
    content = f'''@echo off
title {APP_NAME}
echo Starting {APP_NAME}...
echo.
python -m helix_transvoicer
pause
'''

    run_script.write_text(content)
    print_success(f"Created run.bat")


def create_desktop_shortcut(project_root: Path) -> bool:
    """Create a desktop shortcut (Windows only)."""
    print_step(7, "Creating desktop shortcut...")

    if sys.platform != "win32":
        print_warning("Desktop shortcuts only supported on Windows")
        return False

    try:
        import winreg
        from pathlib import Path

        # Get desktop path
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
        )
        desktop_path = Path(winreg.QueryValueEx(key, "Desktop")[0])
        winreg.CloseKey(key)

        # Create shortcut using PowerShell
        shortcut_path = desktop_path / f"{APP_NAME}.lnk"
        run_bat = project_root / "run.bat"
        icon_path = project_root / "assets" / "app.ico"

        ps_script = f'''
$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{run_bat}"
$Shortcut.WorkingDirectory = "{project_root}"
$Shortcut.Description = "{APP_NAME} - Voice Conversion & TTS"
'''
        if icon_path.exists():
            ps_script += f'$Shortcut.IconLocation = "{icon_path}"\n'
        ps_script += '$Shortcut.Save()'

        result = subprocess.run(
            ["powershell", "-Command", ps_script],
            capture_output=True,
        )

        if result.returncode == 0:
            print_success(f"Desktop shortcut created")
            return True
        else:
            print_warning("Could not create desktop shortcut")
            return False

    except Exception as e:
        print_warning(f"Desktop shortcut creation failed: {e}")
        return False


def print_instructions(project_root: Path, has_gpu: bool):
    """Print final instructions."""
    print_header("Setup Complete!")

    print(f"  {APP_NAME} v{APP_VERSION} has been set up successfully.\n")

    print(f"  {Colors.BOLD}To run the application:{Colors.ENDC}")
    print(f"    Option 1: Double-click run.bat")
    print(f"    Option 2: python -m helix_transvoicer")
    print(f"    Option 3: helix (if installed as package)")

    print(f"\n  {Colors.BOLD}Available commands:{Colors.ENDC}")
    print(f"    helix                    Full application (UI + backend)")
    print(f"    helix --server-only      API server only")
    print(f"    helix --ui-only          UI only (needs external server)")

    if has_gpu:
        print(f"\n  {Colors.GREEN}GPU acceleration enabled - training will be fast!{Colors.ENDC}")
    else:
        print(f"\n  {Colors.YELLOW}No GPU detected - running on CPU (slower){Colors.ENDC}")
        print(f"  For GPU support, install CUDA and reinstall PyTorch with CUDA support.")

    print(f"\n  {Colors.BOLD}Next steps:{Colors.ENDC}")
    print(f"    1. Run the application")
    print(f"    2. Add voice samples in Model Builder")
    print(f"    3. Train your custom voice model")
    print(f"    4. Convert or synthesize speech!")

    print(f"\n  {Colors.CYAN}Documentation: https://github.com/theOneSchrill/Helix-Transvoicer{Colors.ENDC}")
    print()


def main():
    """Main setup function."""
    # Enable ANSI colors on Windows
    if sys.platform == "win32":
        os.system("")

    print_header(f"{APP_NAME} - Windows Setup")

    project_root = get_project_root()
    print(f"  Project directory: {project_root}\n")

    # Check requirements
    if not check_python_version():
        print("\nSetup failed. Please install Python 3.10 or higher.")
        sys.exit(1)

    if not check_pip():
        sys.exit(1)

    # Install dependencies
    if not install_dependencies(project_root):
        print("\nSome dependencies failed to install. Check the errors above.")
        sys.exit(1)

    # Check GPU
    gpu = check_cuda()

    # Install package
    install_package(project_root)

    # Create launcher
    create_run_script(project_root)

    # Try to create desktop shortcut
    create_desktop_shortcut(project_root)

    # Print final instructions
    print_instructions(project_root, has_gpu=gpu is not None)


if __name__ == "__main__":
    main()
