@echo off
echo 🚀 HealthSync AI - Quick Start (Simplified Mode)
echo This will run the platform with demo data - no API keys needed!
echo.

echo ✅ Step 1: Installing minimal backend dependencies...
cd backend
pip install fastapi uvicorn python-dotenv
echo Backend dependencies installed!
cd ..

echo ✅ Step 2: Installing web frontend dependencies...
cd frontend\web
call npm install
echo Web frontend dependencies installed!
cd ..\..

echo ✅ Step 3: Starting simplified backend API...
start "HealthSync Backend" cmd /k "cd backend && python main_simple.py"
timeout /t 3

echo ✅ Step 4: Starting web dashboard...
start "HealthSync Web" cmd /k "cd frontend\web && npm run dev"
timeout /t 5

echo.
echo 🎉 HealthSync AI is starting up in DEMO MODE!
echo.
echo 📱 Web Dashboard: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 API Health Check: http://localhost:8000/health
echo.
echo 💡 This is running with demo data. To get full functionality:
echo    1. Get free API keys from the platforms mentioned in .env
echo    2. Update the .env file with your real keys
echo    3. Run the full backend with: python main.py
echo.
echo 🌐 Opening web dashboard...
timeout /t 3
start http://localhost:3000

echo.
echo ✅ Setup complete! Check the opened browser window.
echo Press any key to exit...
pause