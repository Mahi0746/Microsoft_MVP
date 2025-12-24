import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_memory():
    print("1. Starting new session...")
    try:
        response = requests.post(f"{BASE_URL}/api/voice/start-session", json={"user_id": "test_memory_user"})
        if response.status_code != 200:
            print(f"Failed to start session: {response.text}")
            return
        
        session_data = response.json()
        session_id = session_data["session_id"]
        print(f"Session ID: {session_id}")
        
        print("\n2. Sending first message: 'I have a sharp pain in my left knee.'")
        msg1 = {
            "text": "I have a sharp pain in my left knee.",
            "user_id": "test_memory_user_cmd",
            "session_id": session_id,
            "context": {"role": "patient"}
        }
        res1 = requests.post(f"{BASE_URL}/api/voice/command", json=msg1)
        data1 = res1.json()
        result1 = data1.get('result', {})
        print(f"AI Response 1: {result1.get('message')}")
        
        # Artificial delay
        time.sleep(1)
        
        print("\n3. Sending follow-up: 'How should I treat it?' (Context: 'it' = knee pain)")
        msg2 = {
            "text": "How should I treat it?",
            "user_id": "test_memory_user_cmd",
            "session_id": session_id,
            "context": {"role": "patient"}
        }
        res2 = requests.post(f"{BASE_URL}/api/voice/command", json=msg2)
        data2 = res2.json()
        result2 = data2.get('result', {})
        response_text = result2.get('message', '')
        print(f"AI Response 2: {response_text}")
        
        # Check for keywords
        keywords = ["knee", "pain", "joint", "ice", "rest", "elevation"]
        if any(k in response_text.lower() for k in keywords):
            print("\nSUCCESS: The AI remembered the context!")
        else:
            print("\nFAILURE: The AI did not seem to remember the context.")
            
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_memory()
