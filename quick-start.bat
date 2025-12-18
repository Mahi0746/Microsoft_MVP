@echo off
echo 🚀 Starting HealthSync AI - Quick Setup
echo.

echo ✅ Step 1: Starting databases...
docker-compose up -d postgres redis mongodb

echo ✅ Step 2: Installing backend dependencies...
cd backend
pip install fastapi uvicorn python-dotenv pydantic groq replicate requests
cd ..

echo ✅ Step 3: Installing web frontend dependencies...
cd frontend\web
npm install
cd ..\..

echo ✅ Step 4: Starting backend API...
start cmd /k "cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

echo ✅ Step 5: Starting web dashboard...
start cmd /k "cd frontend\web && npm run dev"

echo.
echo 🎉 HealthSync AI is starting up!
echo.
echo 📱 Web Dashboard: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 API Health Check: http://localhost:8000/health
echo.
echo Press any key to continue...
pause