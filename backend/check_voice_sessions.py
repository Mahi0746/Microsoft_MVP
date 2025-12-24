#!/usr/bin/env python3
"""
Check Voice Sessions in Database
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from services.mongodb_atlas_service import get_mongodb_service


async def check_voice_sessions():
    """Check what voice sessions exist"""
    
    print("\n📊 CHECKING VOICE SESSIONS IN DATABASE")
    print("="*70)
    
    mongodb = await get_mongodb_service()
    if not mongodb or not mongodb.client:
        print("❌ Failed to connect to MongoDB")
        return
    
    # Get all voice sessions
    sessions = await mongodb.database.voice_sessions.find({}).to_list(length=100)
    
    print(f"\n📋 Total voice sessions found: {len(sessions)}")
    
    if sessions:
        for i, session in enumerate(sessions, 1):
            print(f"\n{i}. Session:")
            print(f"   User ID: {session.get('user_id', 'N/A')}")
            print(f"   Session ID: {session.get('session_id', session.get('_id', 'N/A'))}")
            print(f"   Created: {session.get('created_at', 'N/A')}")
            print(f"   Transcript: {session.get('transcript', 'N/A')[:80]}...")
            print(f"   Conversation: {len(session.get('conversation', []))} messages")
    else:
        print("\n❌ No voice sessions found in database")
        print("\nThe voice sessions from `add_medical_data.py` might not have been saved.")
        print("Voice chat history will only show sessions created through the Voice Doctor page.")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(check_voice_sessions())
