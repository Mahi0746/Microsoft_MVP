@echo off
echo 🧪 Testing HealthSync AI Platform
echo.

echo ✅ Testing API Health...
curl -f http://localhost:8000/health
echo.

echo ✅ Testing Web Dashboard...
curl -f http://localhost:3000
echo.

echo ✅ Testing API Documentation...
start http://localhost:8000/docs

echo ✅ Testing Web Dashboard...
start http://localhost:3000

echo.
echo 🎉 Platform test completed!
echo Check the opened browser windows to verify everything is working.
pause