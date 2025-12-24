import requests
import json
import sys

BASE_URL = "http://localhost:8000"

# 1. Login to get token
def login():
    print("Logging in...")
    try:
        response = requests.post(f"{BASE_URL}/api/auth/login", data={
            "username": "test@example.com", 
            "password": "Password123!"
        })
        if response.status_code == 200:
            return response.json()["access_token"]
        # Try to register if login fails? Or assume user exists from previous steps.
        # Let's try registering a dedicated test user.
        print("Login failed, trying registration...")
        reg_response = requests.post(f"{BASE_URL}/api/auth/signup", json={
            "email": "dungeonmaster@test.com",
            "password": "Password123!",
            "first_name": "Dungeon",
            "last_name": "Master"
        })
        if reg_response.status_code == 200:
            return reg_response.json()["access_token"]
        print(f"Registration failed: {reg_response.text}")
        return None
    except Exception as e:
        print(f"Auth error: {e}")
        return None

# 2. Start Adventure
def start_adventure(token):
    print("\nStarting Adventure...")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.post(
            f"{BASE_URL}/api/therapy-game/start-adventure",
            json={"game_type": "shoulder_rehabilitation", "difficulty": 3},
            headers=headers
        )
        if response.status_code == 200:
            data = response.json()
            print("Adventure Started Successfully!")
            print(f"Story: {data.get('narrative_history', [''])[0]}")
            print(f"Exercise: {data.get('current_exercise', {})}")
            return True
        else:
            print(f"Failed to start adventure: {response.text}")
            return False
    except Exception as e:
        print(f"Adventure error: {e}")
        return False

# Main execution
token = login()
if token:
    start_adventure(token)
else:
    # If login fails, we might mock headers or local run if we had direct access, 
    # but for integration test we need real auth.
    # If 'dungeonmaster@test.com' fails, user might need to adjust creds.
    pass
