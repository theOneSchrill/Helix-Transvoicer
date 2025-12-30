@echo off
:: Helix Transvoicer - Backend Server Only
:: Starts the API server without the UI

title Helix Transvoicer - API Server

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo  [ERROR] Run setup.ps1 first!
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo.
echo  Starting Helix Transvoicer API Server...
echo  API available at: http://127.0.0.1:8420
echo  API docs at: http://127.0.0.1:8420/docs
echo.
echo  Press Ctrl+C to stop the server.
echo.

python -m helix_transvoicer.main --server-only %*

pause
