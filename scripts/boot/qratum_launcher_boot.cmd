@echo off
title QRATUM Launcher
cd /d C:\Users\rober\QRATUM-1
echo ============================================================
echo  QRATUM CONTROL PLANE  -  v1.0 wire + autonomy v1.1 + console v1.2 + arbitration v1.0
echo ============================================================
"C:\Users\rober\AppData\Local\Programs\Python\Python312\python.exe" qratum_launcher\qratum_launcher.py
echo.
echo [QRATUM] launcher exited with code %ERRORLEVEL%
pause
