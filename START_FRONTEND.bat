@echo off
echo 🌐 HealthSync AI - Frontend Dashboard
echo ====================================
echo.

REM Check if we're in the right directory
if not exist "frontend" (
    echo ❌ Error: Please run this from the main HealthSync AI directory
    pause
    exit /b
)

REM Check if web directory exists
if not exist "frontend\web" (
    echo ❌ Error: frontend\web directory not found
    pause
    exit /b
)

echo 🚀 Starting HealthSync AI Web Dashboard...
echo    📍 URL: http://localhost:3000
echo    🔗 Backend: http://localhost:8000
echo    📁 Directory: frontend\web
echo.

cd frontend\web
npm run dev