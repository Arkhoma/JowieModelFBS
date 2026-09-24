@echo off
REM One-shot setup and launch for Windows.
REM
REM Creates a virtual environment, installs dependencies, downloads the
REM public data (~45 MB, no API key needed), and starts the server.
REM Safe to re-run: every step is skipped if already done.

setlocal
cd /d "%~dp0"

echo.
echo === CFB Rankings ===
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo Install Python 3.11 or newer from https://python.org
    echo Be sure to tick "Add Python to PATH" in the installer.
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv || exit /b 1
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet || exit /b 1

if not exist "data\raw\schedules" (
    echo.
    echo Downloading data ^(~45 MB, one time^)...
    ".venv\Scripts\python.exe" tools\fetch_mirror.py || exit /b 1
)

echo.
echo Starting server at http://127.0.0.1:8421
echo Press Ctrl+C to stop.
echo.
start "" "http://127.0.0.1:8421"
".venv\Scripts\python.exe" -m uvicorn app.main:app --port 8421

endlocal
