@echo off
:: Helix Transvoicer - Windows 11 Launcher
:: Double-click to start the application

title Helix Transvoicer

:: Change to script directory
cd /d "%~dp0"

:: Check for virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo  [ERROR] Virtual environment not found!
    echo.
    echo  Please run setup.ps1 first:
    echo    1. Right-click setup.ps1
    echo    2. Select "Run with PowerShell"
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Check dependencies
echo.
echo  Checking dependencies...
python scripts\check_dependencies.py
if errorlevel 1 (
    echo.
    echo  [!] Dependency check failed. Continue anyway? [y/N]
    set /p CONTINUE=
    if /i not "%CONTINUE%"=="y" (
        if /i not "%CONTINUE%"=="yes" (
            exit /b 1
        )
    )
)

:: Start the application
echo.
echo  Starting Helix Transvoicer...
echo.

python -m helix_transvoicer.main %*

:: Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo  [ERROR] Application exited with an error.
    echo.
    pause
)
