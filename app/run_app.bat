@echo off
title Infigo FaceIntel Server Launcher
echo ===================================================
echo     Infigo FaceIntel - Automatic Launcher
echo ===================================================
echo.

:: 1. Navigate to application folder
cd /d "%~dp0"

:: 2. Start Virtual CCTV Server (Port 8081)
echo [1/3] Starting Virtual CCTV Camera Server...
start "Infigo CCTV Server" /min "C:\Users\hp\AppData\Local\Python\pythoncore-3.14-64\python.exe" virtual_cctv_server.py

:: 3. Start Surveillance AI Monitoring Engine
echo [2/3] Starting Background Surveillance AI Engine...
start "Infigo Surveillance Engine" /min "C:\Users\hp\AppData\Local\Python\pythoncore-3.14-64\python.exe" run_surveillance.py

:: 4. Start Main Flask Web Application (Port 5000)
echo [3/3] Starting Main Flask Web Application...
start "Infigo Web App" /min "C:\Users\hp\AppData\Local\Python\pythoncore-3.14-64\python.exe" app.py

:: 5. Wait 3 seconds and open Web Dashboard in default browser
echo.
echo Launching Web Dashboard in browser...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

echo.
echo ===================================================
echo  All services are running in background!
echo  Web Dashboard: http://127.0.0.1:5000
echo ===================================================
pause
