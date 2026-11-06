@echo off
setlocal
cd /d "%~dp0"
set PORT=8765
set URL=http://127.0.0.1:%PORT%/

rem Server already running: just open the browser
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" >nul
if %errorlevel%==0 (
  echo Viewer already running at %URL%
  start "" "%URL%"
  exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found. Create the venv first.
  pause
  exit /b 1
)

echo Starting EEG viewer at %URL%  (close this window to stop)
".venv\Scripts\python.exe" "vis\viewer_server.py" --port %PORT%
if errorlevel 1 pause
