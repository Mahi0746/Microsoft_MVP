@echo off
REM HealthSync AI - Automated Setup Script for Windows
REM This script will set up both backend and frontend with minimal manual intervention

echo.
echo ========================================
echo   HealthSync AI - Automated Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.11+ from python.org
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Node.js is not installed or not in PATH
    echo Please install Node.js 18+ from nodejs.org
    pause
    exit /b 1
)

echo [OK] Python found
echo [OK] Node.js found
echo.

REM ========================================
REM BACKEND SETUP
REM ========================================

echo ========================================
echo   Setting up Backend
echo ========================================
echo.

cd backend

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo [4/5] Setting up environment file...
if not exist .env (
    copy .env.example .env >nul
    echo [INFO] Created .env file - Please edit it with your configuration
) else (
    echo [INFO] .env file already exists - skipping
)

echo [5/5] Backend setup complete!
echo.

REM ========================================
REM FRONTEND SETUP
REM ========================================

cd ..\frontend\web

echo ========================================
echo   Setting up Frontend
echo ========================================
echo.

echo [1/3] Installing Node.js dependencies...
call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    cd ..\..
    pause
    exit /b 1
)

echo [2/3] Setting up environment file...
if not exist .env.local (
    copy .env.example .env.local >nul
    echo [INFO] Created .env.local file
) else (
    echo [INFO] .env.local file already exists - skipping
)

echo [3/3] Frontend setup complete!
echo.

cd ..\..

REM ========================================
REM FINAL INSTRUCTIONS
REM ========================================

echo ========================================
echo   Setup Complete! 
echo ========================================
echo.
echo IMPORTANT: Before starting the application:
echo.
echo 1. Edit backend\.env file:
echo    - Set a strong SECRET_KEY (min 32 characters)
echo    - Configure MONGODB_URL (local or Atlas)
echo    - Add API keys for AI services (optional for auth)
echo.
echo 2. Ensure MongoDB is running:
echo    - Local: Start MongoDB service
echo    - Atlas: Verify connection string in .env
echo.
echo 3. Start the application:
echo    - Backend:  cd backend ^&^& venv\Scripts\activate ^&^& python main.py
echo    - Frontend: cd frontend\web ^&^& npm run dev
echo.
echo 4. Access the application:
echo    - Frontend: http://localhost:3000
echo    - Backend:  http://localhost:8000
echo    - API Docs: http://localhost:8000/docs
echo.
echo For quick start guide, see QUICKSTART.md
echo For detailed setup, see SETUP_GUIDE.md
echo.

pause
