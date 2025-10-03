@echo off
setlocal
REM Wrapper to launch the PowerShell starter with sane defaults (port 8040)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-app.ps1" %*
endlocal
