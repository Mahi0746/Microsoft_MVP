"""
Debug script to test file upload
"""
import requests
import io

API_URL = "http://localhost:8000"

print("Testing file upload to backend...")
print("=" * 60)

# Test 1: Test endpoint
print("\n1. Testing /api/voice/test-upload endpoint...")
try:
    audio_data = b'\x1a\x45\xdf\xa3' + b'\x00' * 5000
    
    files = {
        'audio_file': ('test.webm', io.BytesIO(audio_data), 'audio/webm')
    }
    
    data = {
        'session_id': 'test_session_123',
        'user_id': 'test_user_456'
    }
    
    response = requests.post(
        f"{API_URL}/api/voice/test-upload",
        files=files,
        data=data
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        print("✅ Test upload works!")
    else:
        print("❌ Test upload failed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Real endpoint
print("\n2. Testing /api/voice/send-audio endpoint...")
try:
    # First start a session
    session_response = requests.post(
        f"{API_URL}/api/voice/start-session",
        json={
            "user_id": "test_user",
            "symptoms": []
        }
    )
    
    if session_response.status_code != 200:
        print(f"❌ Failed to start session: {session_response.status_code}")
        print(f"Response: {session_response.text}")
    else:
        session_data = session_response.json()
        session_id = session_data.get('session_id')
        print(f"✅ Session started: {session_id}")
        
        # Now try to upload audio
        audio_data = b'\x1a\x45\xdf\xa3' + b'\x00' * 5000
        
        files = {
            'audio_file': ('recording.webm', io.BytesIO(audio_data), 'audio/webm')
        }
        
        data = {
            'session_id': session_id,
            'user_id': 'test_user'
        }
        
        print(f"\nUploading with:")
        print(f"  session_id: {session_id}")
        print(f"  user_id: test_user")
        print(f"  audio size: {len(audio_data)} bytes")
        
        response = requests.post(
            f"{API_URL}/api/voice/send-audio",
            files=files,
            data=data
        )
        
        print(f"\nStatus: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Audio upload works!")
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"❌ Audio upload failed")
            print(f"Response: {response.text}")
            
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
