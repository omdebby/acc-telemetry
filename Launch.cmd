@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" gui.py %*
) else if exist "%USERPROFILE%\ACC-Telemetry\.venv\Scripts\python.exe" (
  "%USERPROFILE%\ACC-Telemetry\.venv\Scripts\python.exe" gui.py %*
) else (
  python gui.py %*
)
if errorlevel 1 (
  echo.
  echo Could not start. See README.md for setup instructions.
  pause
)
