#!/usr/bin/env python3
"""
Test script for Groq voice integration
Run this to test if Groq API is working properly
"""

import asyncio
import os
from config_flexible import settings

async def test_groq_text():
    """Test Groq text completion"""
    print("🧪 Testing Groq Text Completion...")
    
    if not settings.has_real_groq_key():
        print("❌ No real Groq API key found. Using demo mode.")
        print(f"Current key: {settings.groq_api_key[:20]}...")
        return False
    
    try:
        from groq import Groq
        
        client = Groq(api_key=settings.groq_api_key)
        
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": "You are a helpful medical AI assistant."},
                {"role": "user", "content": "I have a headache and feel tired. What could this be?"}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        print("✅ Groq Text API working!")
        print(f"Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Groq Text API error: {e}")
        return False

async def test_groq_whisper():
    """Test Groq Whisper (speech-to-text)"""
    print("\n🧪 Testing Groq Whisper (Speech-to-Text)...")
    
    if not settings.has_real_groq_key():
        print("❌ No real Groq API key found. Using demo mode.")
        return False
    
    # Create a simple test audio file (you can replace this with a real audio file)
    test_audio_path = "test_audio.wav"
    
    if not os.path.exists(test_audio_path):
        print("⚠️ No test audio file found. Skipping Whisper test.")
        print("   To test Whisper, place a .wav audio file named 'test_audio.wav' in the backend directory.")
        return False
    
    try:
        from groq import Groq
        
        client = Groq(api_key=settings.groq_api_key)
        
        with open(test_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=file,
                model="whisper-large-v3",
                response_format="text",
                language="en"
            )
        
        print("✅ Groq Whisper API working!")
        print(f"Transcription: {transcription}")
        return True
        
    except Exception as e:
        print(f"❌ Groq Whisper API error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🏥 HealthSync AI - Groq Integration Test")
    print("=" * 50)
    
    # Test configuration
    print(f"Environment: {settings.environment}")
    print(f"Groq Model: {settings.groq_model}")
    print(f"Demo Mode: {settings.is_demo_mode()}")
    print()
    
    # Run tests
    text_success = await test_groq_text()
    whisper_success = await test_groq_whisper()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Text Completion: {'✅ PASS' if text_success else '❌ FAIL'}")
    print(f"   Speech-to-Text:  {'✅ PASS' if whisper_success else '❌ FAIL'}")
    
    if text_success:
        print("\n🎉 Groq integration is working! Your voice AI doctor is ready.")
    else:
        print("\n⚠️  To enable real AI functionality:")
        print("   1. Get a free Groq API key from: https://console.groq.com")
        print("   2. Update GROQ_API_KEY in your .env file")
        print("   3. Restart the backend server")

if __name__ == "__main__":
    asyncio.run(main())