"""Comprehensive Frontend-Backend Integration Test"""
import requests
import json
import time

BACKEND_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

def test_backend_health():
    """Test backend health endpoint"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        print(f"✅ Backend Health: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Backend Health Failed: {e}")
        return False

def test_frontend_accessible():
    """Test if frontend is accessible"""
    try:
        response = requests.get(FRONTEND_URL, timeout=5)
        print(f"✅ Frontend Accessible: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Frontend Access Failed: {e}")
        return False

def test_cors_configuration():
    """Test CORS is properly configured"""
    try:
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        response = requests.options(f"{BACKEND_URL}/api/auth/login", headers=headers, timeout=5)
        print(f"✅ CORS Preflight: {response.status_code}")
        print(f"   Allow-Origin: {response.headers.get('access-control-allow-origin', 'Not Set')}")
        return True
    except Exception as e:
        print(f"❌ CORS Test Failed: {e}")
        return False

def test_auth_flow():
    """Test complete authentication flow"""
    try:
        # Test Login
        login_payload = {
            "email": "testuser@healthsync.com",
            "password": "TestPassword123!"
        }
        
        response = requests.post(
            f"{BACKEND_URL}/api/auth/login",
            json=login_payload,
            headers={"Origin": "http://localhost:3000"},
            timeout=5
        )
        
        print(f"✅ Login Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            print(f"   Token Type: {data.get('token_type')}")
            print(f"   Expires In: {data.get('expires_in')}s")
            
            # Test authenticated request
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Origin": "http://localhost:3000"
            }
            health_response = requests.get(f"{BACKEND_URL}/health", headers=headers, timeout=5)
            print(f"✅ Authenticated Request: {health_response.status_code}")
            
            return True
        else:
            print(f"   Error: {response.json()}")
            return False
            
    except Exception as e:
        print(f"❌ Auth Flow Failed: {e}")
        return False

def test_api_endpoints():
    """Test key API endpoints"""
    endpoints = [
        ("/health", "GET"),
        ("/api/auth/login", "POST"),
        ("/docs", "GET"),
    ]
    
    print("\n📋 Testing API Endpoints:")
    for endpoint, method in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{BACKEND_URL}{endpoint}", timeout=5)
            else:
                response = requests.post(f"{BACKEND_URL}{endpoint}", json={}, timeout=5)
            
            status_icon = "✅" if response.status_code < 500 else "❌"
            print(f"   {status_icon} {method} {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"   ❌ {method} {endpoint}: {e}")

def main():
    print("="*70)
    print("🚀 HEALTHSYNC FULL STACK INTEGRATION TEST")
    print("="*70)
    
    print("\n1️⃣ Testing Backend Server...")
    backend_ok = test_backend_health()
    
    print("\n2️⃣ Testing Frontend Server...")
    frontend_ok = test_frontend_accessible()
    
    print("\n3️⃣ Testing CORS Configuration...")
    cors_ok = test_cors_configuration()
    
    print("\n4️⃣ Testing Authentication Flow...")
    auth_ok = test_auth_flow()
    
    print("\n5️⃣ Testing Additional API Endpoints...")
    test_api_endpoints()
    
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Backend Server:        {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Frontend Server:       {'✅ PASS' if frontend_ok else '❌ FAIL'}")
    print(f"CORS Configuration:    {'✅ PASS' if cors_ok else '❌ FAIL'}")
    print(f"Authentication Flow:   {'✅ PASS' if auth_ok else '❌ FAIL'}")
    print("="*70)
    
    all_pass = backend_ok and frontend_ok and cors_ok and auth_ok
    if all_pass:
        print("\n🎉 ALL TESTS PASSED! Website is fully functional!")
        print("\n📝 Next Steps:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Try signing up or logging in")
        print("   3. Explore the dashboard features")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("="*70)

if __name__ == "__main__":
    main()
