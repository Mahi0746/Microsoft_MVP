
import asyncio
import sys
import os

# Add parent dir to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.voice_agent_service import VoiceAgentService
from config_flexible import settings

# Mock Groq Key if missing (for test purposes to avoid crashing if user hasn't set it)
if not settings.groq_api_key:
    print("⚠️ WARNING: No GROQ_API_KEY found in env or config.")
    print("   Please set it to run standard tests, or this test might fail/skip.")

async def test_agent():
    print("🚀 Initializing Voice Agent Service...")
    VoiceAgentService.initialize()
    
    # print("\n🧪 Test 1: Navigation Command")
    # print("   User: 'Navigate to my profile'")
    # try:
    #     result = await VoiceAgentService.process_voice_command("Navigate to my profile")
    #     print(f"   Agent: {result}")
    # except Exception as e:
    #     print(f"   ❌ Failed: {e}")

    # print("\n🧪 Test 2: Medication Query")
    # print("   User: 'Show my medications'")
    # try:
    #     result = await VoiceAgentService.process_voice_command("Show my medications")
    #     print(f"   Agent: {result}")
    # except Exception as e:
    #     print(f"   ❌ Failed: {e}")

    # print("\n🧪 Test 3: General Health Question")
    # print("   User: 'I have a headache, what should I do?'")
    # try:
    #     result = await VoiceAgentService.process_voice_command("I have a headache, what should I do?")
    #     print(f"   Agent: {result}")
    # except Exception as e:
    #     print(f"   ❌ Failed: {e}")

    # print("\n🧪 Test 4: Scans Query")
    # print("   User: 'Open the scan document'")
    # try:
    #     result = await VoiceAgentService.process_voice_command("Open the scan document")
    #     print(f"   Agent: {result}")
    # except Exception as e:
    #     print(f"   ❌ Failed: {e}")
        
    print("\n🧪 Test 5: Daily Health Plan")
    print("   User: 'What is my exercise plan for today?'")
    try:
        result = await VoiceAgentService.process_voice_command("What is my exercise plan for today?")
        print(f"   Agent: {result}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(test_agent())
