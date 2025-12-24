@echo off
REM HealthSync AI - Start Application
REM This script starts both backend and frontend in separate windows

echo.
echo ========================================
echo   Starting HealthSync AI
echo ========================================
echo.

REM Check if setup was done
if not exist backend\venv (
    echo [ERROR] Backend not set up. Please run SETUP.bat first!
    pause
    exit /b 1
)

if not exist frontend\web\node_modules (
    echo [ERROR] Frontend not set up. Please run SETUP.bat first!
    pause
    exit /b 1
)

if not exist backend\.env (
    echo [ERROR] Backend .env file not found. Please configure it!
    pause
    exit /b 1
)

echo [INFO] Starting backend server...
start "HealthSync Backend" cmd /k "cd backend && venv\Scripts\activate && python main.py"

timeout /t 3 >nul

echo [INFO] Starting frontend server...
start "HealthSync Frontend" cmd /k "cd frontend\web && npm run dev"

echo.
echo ========================================
echo   HealthSync AI Started!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Press any key to close this window...
echo (Backend and Frontend will keep running)
echo.

pause
