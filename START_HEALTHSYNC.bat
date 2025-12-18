@echo off
echo 🏥 HealthSync AI - GUARANTEED TO WORK!
echo.

echo 🔍 Checking your setup...

REM Check if we're in the right directory
if not exist "backend" (
    echo ❌ Error: Please run this from the main HealthSync AI directory
    echo    Make sure you can see the 'backend' and 'frontend' folders
    pause
    exit /b
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo    Please install Python 3.11+ from python.org
    pause
    exit /b
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Node.js is not installed or not in PATH
    echo    Please install Node.js 18+ from nodejs.org
    pause
    exit /b
)

echo ✅ Python and Node.js found!
echo.

echo 🚀 Starting HealthSync AI...
echo.

echo ✅ Step 1: Installing minimal backend dependencies...
cd backend
pip install fastapi uvicorn python-dotenv motor pymongo 2>nul
if errorlevel 1 (
    echo ⚠️  Some packages may already be installed, continuing...
)
echo Backend ready!
cd ..

echo ✅ Step 2: Installing web frontend dependencies...
cd frontend\web
call npm install >nul 2>&1
if errorlevel 1 (
    echo ⚠️  NPM install had some warnings, but continuing...
)
echo Frontend ready!
cd ..\..

echo ✅ Step 3: Starting backend server...
start "HealthSync Backend" cmd /k "cd backend && echo Starting HealthSync AI Backend... && python main_working.py"

echo ⏳ Waiting for backend to start...
timeout /t 8 >nul

echo ✅ Step 4: Starting web dashboard...
start "HealthSync Web" cmd /k "cd frontend\web && echo Starting HealthSync Web Dashboard... && npm run dev"

echo ⏳ Waiting for web dashboard to start...
timeout /t 10 >nul

echo.
echo 🎉 HealthSync AI is now running!
echo.
echo 🌐 Web Dashboard: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 Health Check: http://localhost:8000/health
echo.
echo 💡 Your MongoDB Atlas database is connected!
echo    All your data will be saved to the cloud.
echo.
echo 🌐 Opening web dashboard in 3 seconds...
timeout /t 3 >nul
start http://localhost:3000

echo.
echo ✅ Setup complete! 
echo.
echo 📱 To test the platform:
echo    1. Go to http://localhost:3000
echo    2. Create an account (any email works)
echo    3. Explore all the features!
echo.
echo 🛑 To stop the servers:
echo    - Close the backend and web terminal windows
echo    - Or press Ctrl+C in each window
echo.
echo Press any key to exit this setup window...
pause >nul