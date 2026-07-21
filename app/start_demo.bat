@echo off
REM ── Infigo FaceIntel — Demo Startup ──
REM Starts all three services, each in its own window:
REM   1. Virtual CCTV Camera Server (camera API on port 8081)
REM   2. CCTV Surveillance Service  (auto attendance from camera feeds)
REM   3. Web Dashboard              (http://127.0.0.1:5000)

cd /d "%~dp0"

echo Starting Virtual CCTV Camera Server (port 8081)...
start "CCTV Camera API" cmd /k python virtual_cctv_server.py

timeout /t 3 /nobreak >nul

echo Starting Surveillance Service...
start "Surveillance Service" cmd /k python run_surveillance.py

echo Starting Web Dashboard (port 5000)...
start "Web Dashboard" cmd /k python app.py

timeout /t 8 /nobreak >nul
start http://127.0.0.1:5000/live_cameras

echo.
echo All services started!
echo   Camera API   : http://127.0.0.1:8081/
echo   Dashboard    : http://127.0.0.1:5000/
echo   Live Cameras : http://127.0.0.1:5000/live_cameras
