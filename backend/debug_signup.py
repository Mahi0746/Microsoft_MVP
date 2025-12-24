import requests
import json

API_URL = "http://localhost:8000/api/auth"

def test_signup():
    # 1. Signup as patient
    email = "testpatient_debug@example.com"
    password = "Password123"
    payload = {
        "first_name": "Test",
        "last_name": "Patient",
        "email": email,
        "password": password,
        "role": "patient",
        "age": 30,
        "gender": "male"
    }
    
    print(f"Sending signup request for {email} with role 'patient'...")
    try:
        response = requests.post(f"{API_URL}/signup", json=payload)
        if response.status_code != 200:
            print(f"Signup failed: {response.text}")
            return

        data = response.json()
        print(f"Signup response role: {data['user']['role']}")
        
        if data['user']['role'] != 'patient':
            print("CRITICAL FAILURE: Created user has wrong role!")
        else:
            print("SUCCESS: Backend correctly assigned 'patient' role.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_signup()
