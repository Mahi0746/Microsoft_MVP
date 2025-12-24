"""Simple test script to verify authentication works without Supabase"""
import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_signup():
    """Test user signup"""
    payload = {
        "email": "testuser@healthsync.com",
        "password": "TestPassword123!",
        "first_name": "Test",
        "last_name": "User",
        "role": "patient"
    }
    
    response = requests.post(f"{BASE_URL}/auth/signup", json=payload)
    print("\n=== SIGNUP TEST ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json() if response.status_code == 200 else None

def test_login():
    """Test user login"""
    payload = {
        "email": "testuser@healthsync.com",
        "password": "TestPassword123!"
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=payload)
    print("\n=== LOGIN TEST ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.json() if response.status_code == 200 else None

def test_protected_route(access_token):
    """Test accessing a protected route with JWT token"""
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    
    response = requests.get("http://localhost:8000/health", headers=headers)
    print("\n=== PROTECTED ROUTE TEST ===")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

if __name__ == "__main__":
    print("🚀 Testing HealthSync Authentication (MongoDB + JWT)")
    print("=" * 60)
    
    # Test signup
    signup_result = test_signup()
    
    # Test login
    login_result = test_login()
    
    # Test protected route if login succeeded
    if login_result and "access_token" in login_result:
        test_protected_route(login_result["access_token"])
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
