@echo off
echo 🍃 HealthSync AI - MongoDB Atlas Quick Start
echo This version uses MongoDB Atlas cloud database (FREE)
echo.

echo ⚠️  IMPORTANT: Make sure you have set up MongoDB Atlas first!
echo    📖 Read MONGODB_ATLAS_SETUP.md for instructions
echo    🔗 Or go to: https://www.mongodb.com/atlas
echo.

set /p continue="Have you set up MongoDB Atlas and updated .env file? (y/n): "
if /i not "%continue%"=="y" (
    echo.
    echo 📖 Please follow these steps:
    echo    1. Read MONGODB_ATLAS_SETUP.md
    echo    2. Create free MongoDB Atlas account
    echo    3. Update MONGODB_URL in .env file
    echo    4. Run this script again
    echo.
    pause
    exit /b
)

echo.
echo ✅ Step 1: Installing backend dependencies...
cd backend
pip install fastapi uvicorn python-dotenv motor pymongo
echo Backend dependencies installed!
cd ..

echo ✅ Step 2: Installing web frontend dependencies...
cd frontend\web
call npm install
echo Web frontend dependencies installed!
cd ..\..

echo ✅ Step 3: Starting MongoDB Atlas backend...
start "HealthSync Atlas Backend" cmd /k "cd backend && python main_atlas.py"
timeout /t 5

echo ✅ Step 4: Starting web dashboard...
start "HealthSync Web" cmd /k "cd frontend\web && npm run dev"
timeout /t 5

echo.
echo 🎉 HealthSync AI with MongoDB Atlas is starting up!
echo.
echo 📱 Web Dashboard: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 API Health Check: http://localhost:8000/health
echo 🍃 Database: MongoDB Atlas (Cloud)
echo.
echo 💡 Features with MongoDB Atlas:
echo    ✅ Real user registration and login
echo    ✅ Persistent data storage in cloud
echo    ✅ Voice AI session history
echo    ✅ AR scan records
echo    ✅ Appointment booking
echo    ✅ Future simulation storage
echo    ✅ Analytics and reporting
echo.
echo 🌐 Opening web dashboard...
timeout /t 3
start http://localhost:3000

echo.
echo ✅ Setup complete! Your data is now stored in MongoDB Atlas.
echo Press any key to exit...
pause