"""
Quick test to verify Groq API key works
"""
import os
from dotenv import load_dotenv

# Load .env from root directory
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

print("=" * 60)
print("Testing Groq API Key")
print("=" * 60)
print(f"API Key found: {groq_key[:20]}..." if groq_key else "No API key found")
print(f"Key starts with 'gsk_': {groq_key.startswith('gsk_') if groq_key else False}")
print()

if groq_key and groq_key.startswith('gsk_'):
    print("Testing Groq API connection...")
    try:
        from groq import Groq
        
        client = Groq(api_key=groq_key)
        
        # Test chat completion
        print("\n1. Testing Chat Completion...")
        response = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, I am working!' in exactly those words."}
            ],
            max_tokens=50
        )
        
        result = response.choices[0].message.content
        print(f"✅ Chat API works! Response: {result}")
        
        print("\n" + "=" * 60)
        print("✅ Groq API key is VALID and WORKING!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error testing Groq API: {e}")
        print("\nPossible issues:")
        print("1. API key might be invalid or expired")
        print("2. Network connection issue")
        print("3. Groq service might be down")
        print("\nGet a new key from: https://console.groq.com")
else:
    print("❌ Invalid or missing Groq API key")
    print("\nTo fix:")
    print("1. Go to https://console.groq.com")
    print("2. Create an API key")
    print("3. Update .env file: GROQ_API_KEY=gsk_your_key_here")
