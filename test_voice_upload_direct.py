"""
Direct test of voice audio upload endpoint
"""
import requests
import io

API_URL = "http://localhost:8000"

def test_voice_upload():
    """Test voice audio upload with minimal data"""
    
    print("=" * 60)
    print("Testing Voice Audio Upload")
    print("=" * 60)
    
    # Step 1: Start session
    print("\n1. Starting voice session...")
    try:
        response = requests.post(
            f"{API_URL}/api/voice/start-session",
            json={
                "user_id": "test_user_123",
                "symptoms": ["test symptom"]
            }
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to start session: {response.status_code}")
            print(f"Response: {response.text}")
            return
        
        data = response.json()
        session_id = data.get("session_id")
        print(f"✅ Session started: {session_id}")
        print(f"AI Response: {data.get('ai_response')[:100]}...")
        
    except Exception as e:
        print(f"❌ Error starting session: {e}")
        return
    
    # Step 2: Upload audio
    print("\n2. Uploading audio file...")
    try:
        # Create a dummy audio file (WebM header + some data)
        audio_data = b'\x1a\x45\xdf\xa3' + b'\x00' * 5000  # WebM magic number + padding
        
        files = {
            'audio_file': ('test_recording.webm', io.BytesIO(audio_data), 'audio/webm')
        }
        
        data = {
            'session_id': session_id,
            'user_id': 'test_user_123'
        }
        
        print(f"Sending request with:")
        print(f"  - session_id: {session_id}")
        print(f"  - user_id: test_user_123")
        print(f"  - audio_file: {len(audio_data)} bytes")
        
        response = requests.post(
            f"{API_URL}/api/voice/send-audio",
            files=files,
            data=data
        )
        
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Audio processed successfully!")
            print(f"\nConversation:")
            for msg in result.get('conversation', []):
                print(f"  {msg['role']}: {msg['message'][:100]}...")
        else:
            print(f"❌ Failed to process audio")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error uploading audio: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        test_voice_upload()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to backend")
        print("Make sure the backend is running: python backend/main_complete.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
