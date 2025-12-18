"""
Test script to verify audio upload and processing
"""
import requests
import io

# Test configuration
API_URL = "http://localhost:8000"

def test_start_session():
    """Test starting a voice session"""
    print("🧪 Testing voice session start...")
    
    response = requests.post(
        f"{API_URL}/api/voice/start-session",
        json={
            "user_id": "test_user",
            "symptoms": ["headache", "fever"]
        }
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Session started: {data.get('session_id')}")
        print(f"AI Response: {data.get('ai_response')}")
        return data.get('session_id')
    else:
        print(f"❌ Failed to start session")
        return None

def test_audio_upload(session_id):
    """Test audio file upload"""
    print("\n🧪 Testing audio upload...")
    
    # Create a small dummy audio file (WebM format header)
    # This is just for testing the upload mechanism
    dummy_audio = b'\x1a\x45\xdf\xa3' + b'\x00' * 1000  # WebM header + padding
    
    files = {
        'audio_file': ('test_recording.webm', io.BytesIO(dummy_audio), 'audio/webm')
    }
    
    data = {
        'session_id': session_id,
        'user_id': 'test_user'
    }
    
    response = requests.post(
        f"{API_URL}/api/voice/send-audio",
        files=files,
        data=data
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Audio processed successfully")
        print(f"Conversation: {result.get('conversation')}")
        print(f"AI Response: {result.get('ai_response')}")
    else:
        print(f"❌ Failed to process audio")
        print(f"Error: {response.text}")

def test_text_message(session_id):
    """Test text message"""
    print("\n🧪 Testing text message...")
    
    response = requests.post(
        f"{API_URL}/api/voice/send-message",
        json={
            "session_id": session_id,
            "message": "I have a headache and feel tired"
        }
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Message processed successfully")
        print(f"AI Response: {result.get('ai_response')}")
    else:
        print(f"❌ Failed to process message")
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("=" * 60)
    print("HealthSync AI - Voice API Test")
    print("=" * 60)
    print(f"Testing API at: {API_URL}")
    print()
    
    try:
        # Test 1: Start session
        session_id = test_start_session()
        
        if session_id:
            # Test 2: Upload audio
            test_audio_upload(session_id)
            
            # Test 3: Send text message
            test_text_message(session_id)
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to backend server")
        print("Make sure the backend is running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")
