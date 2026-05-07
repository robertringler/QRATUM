@echo off
title QRATUM Kernel
cd /d C:\Users\rober\QRATUM-1
echo ============================================================
echo  QRATUM KERNEL DAEMON  -  60 Hz tick
echo ============================================================
"C:\Users\rober\AppData\Local\Programs\Python\Python312\python.exe" qratum_kernel\qratum_kernel_daemon.py --hz 60
echo.
echo [QRATUM] kernel exited with code %ERRORLEVEL%
pause
