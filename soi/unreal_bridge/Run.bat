@echo off
REM ============================================================
REM  QRATUM IntentOS — UE5 launcher
REM
REM  Usage:
REM     Run.bat              - open editor (interactive)
REM     Run.bat /headless    - cooked headless run, deterministic
REM     Run.bat /build       - rebuild C++ module only
REM ============================================================
setlocal

set UE_ROOT=C:\Program Files\Epic Games\UE_5.7
set PROJECT=%~dp0IntentOS.uproject
set EDITOR=%UE_ROOT%\Engine\Binaries\Win64\UnrealEditor.exe
set CMD=%UE_ROOT%\Engine\Binaries\Win64\UnrealEditor-Cmd.exe
set BUILD=%UE_ROOT%\Engine\Build\BatchFiles\Build.bat

if /I "%~1"=="/build" goto BUILD
if /I "%~1"=="/headless" goto HEADLESS

REM Default: launch the editor
echo [QRATUM/UE5] launching editor on %PROJECT%
"%EDITOR%" "%PROJECT%"
goto END

:BUILD
echo [QRATUM/UE5] building QRATUM module ...
call "%BUILD%" IntentOSEditor Win64 Development -Project="%PROJECT%" -WaitMutex
goto END

:HEADLESS
echo [QRATUM/UE5] headless tick run (Ctrl+C to stop)
"%CMD%" "%PROJECT%" -unattended -nopause -nullrhi -log
goto END

:END
endlocal
