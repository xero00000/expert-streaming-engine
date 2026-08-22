@echo off
setlocal
where py.exe >nul 2>nul
if %errorlevel% equ 0 (
  py.exe -3 "%~dp0ese" %*
) else (
  python.exe "%~dp0ese" %*
)
exit /b %errorlevel%
