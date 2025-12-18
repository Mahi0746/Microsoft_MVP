@echo off
echo 🏥 HealthSync AI - Backend Server
echo ================================
echo.

REM Check if we're in the right directory
if not exist "backend" (
    echo ❌ Error: Please run this from the main HealthSync AI directory
    pause
    exit /b
)

echo 🚀 Starting HealthSync AI Backend...
echo    📍 URL: http://localhost:8000
echo    📚 API Docs: http://localhost:8000/docs
echo.

cd backend
python -m uvicorn main_complete:app --reload --host 0.0.0.0 --port 8000