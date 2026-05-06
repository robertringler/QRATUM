@echo off
REM ============================================================
REM  qratum-boot.cmd — boot QRATUM as an OS-style stack.
REM
REM  Stages:
REM    POST    : verify python + state directory
REM    INIT    : spawn the headless kernel daemon
REM    SHELL   : launch the QRATUM Desktop shell (Tauri)
REM    [opt]   : --with-ue5 also launches the UE5 simulation
REM ============================================================
setlocal enableextensions enabledelayedexpansion

set "REPO_ROOT=%~dp0"
set "REPO_ROOT=%REPO_ROOT:~0,-1%"
set "STATE_DIR=%LOCALAPPDATA%\QRATUM\bridge"
set "DAEMON=%REPO_ROOT%\qratum_kernel\qratum_kernel_daemon.py"
set "DESKTOP_EXE=%REPO_ROOT%\dist\desktop\qratum-desktop\QRATUM Desktop.exe"
set "WITH_UE5=0"

:parse
if "%~1"=="" goto post
if /i "%~1"=="--with-ue5" set "WITH_UE5=1"
if /i "%~1"=="--halt" goto halt
shift
goto parse

:post
echo [QRATUM] POST  ...
where python >nul 2>&1
if errorlevel 1 (
    echo [QRATUM] POST FAIL: python not on PATH
    exit /b 1
)
if not exist "%DAEMON%" (
    echo [QRATUM] POST FAIL: kernel daemon missing: %DAEMON%
    exit /b 1
)
if not exist "%STATE_DIR%" mkdir "%STATE_DIR%" >nul 2>&1
echo [QRATUM] POST  OK

echo [QRATUM] INIT  spawning kernel daemon ...
start "QRATUM Kernel" /B python "%DAEMON%" --hz 60 --quiet
REM wait briefly for state.json to appear so the desktop has data on first paint
set /a tries=0
:wait_state
if exist "%STATE_DIR%\state.json" goto state_ready
set /a tries+=1
if %tries% GEQ 50 goto state_ready
ping -n 1 127.0.0.1 >nul
goto wait_state
:state_ready
echo [QRATUM] INIT  kernel publishing to %STATE_DIR%\state.json

if "%WITH_UE5%"=="1" (
    if exist "%REPO_ROOT%\soi\unreal_bridge\Run.bat" (
        echo [QRATUM] launching UE5 simulation ...
        start "QRATUM UE5" cmd /c "%REPO_ROOT%\soi\unreal_bridge\Run.bat"
    ) else (
        echo [QRATUM] UE5 Run.bat not present — skipping
    )
)

if exist "%DESKTOP_EXE%" (
    echo [QRATUM] SHELL launching desktop: %DESKTOP_EXE%
    start "QRATUM Desktop" "%DESKTOP_EXE%"
) else (
    echo [QRATUM] SHELL desktop binary not built; build with:
    echo         cd qratum_desktop ^&^& npm run build
)

echo [QRATUM] boot complete. State: %STATE_DIR%\state.json
echo [QRATUM] To halt: %~nx0 --halt
exit /b 0

:halt
echo [QRATUM] halting kernel daemons ...
for /f "tokens=2 delims=," %%P in ('tasklist /FI "WINDOWTITLE eq QRATUM Kernel*" /FO CSV /NH 2^>nul') do (
    set "PID=%%~P"
    taskkill /PID !PID! /T /F >nul 2>&1
)
taskkill /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq QRATUM Kernel*" /F >nul 2>&1
echo [QRATUM] halt complete.
exit /b 0
