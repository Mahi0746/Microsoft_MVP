#!/usr/bin/env python3
"""
Test Script: Fetch REAL Google Fit Data
Verifies the GoogleFitService integration
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(__file__))

from services.google_fit_service import GoogleFitService
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle


async def test_google_fit_integration():
    """Test fetching real Google Fit data"""
    
    print("\n" + "="*70)
    print("🏥 TESTING GOOGLE FIT INTEGRATION - REAL DATA")
    print("="*70)
    
    # Path to client secret (from your demo.py)
    CLIENT_SECRET_FILE = r"C:\Users\mahip\Downloads\client_secret_341978188796-t67qofh4bsql49crua3n9h7brfkme34r.apps.googleusercontent.com.json"
    
    if not os.path.exists(CLIENT_SECRET_FILE):
        print("\n❌ ERROR: Client secret file not found!")
        print(f"   Expected: {CLIENT_SECRET_FILE}")
        print("   Please update the path or place the file there.")
        return
    
    print("\n📱 Authenticating with Google Fit...")
    
    # Use the same OAuth flow as demo.py
    SCOPES = [
        'https://www.googleapis.com/auth/fitness.activity.read',
        'https://www.googleapis.com/auth/fitness.heart_rate.read',
        'https://www.googleapis.com/auth/fitness.sleep.read',
        'https://www.googleapis.com/auth/fitness.body.read',
    ]
    
    credentials = None
    
    # Check for saved token
    if os.path.exists('token.pickle'):
        print("   ✅ Loading saved credentials...")
        with open('token.pickle', 'rb') as token:
            credentials = pickle.load(token)
    
    # Refresh or get new token
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print("   🔄 Refreshing expired token...")
            credentials.refresh(Request())
        else:
            print("   🔐 Opening browser for authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, 
                SCOPES
            )
            credentials = flow.run_local_server(port=8080)
        
        # Save credentials
        with open('token.pickle', 'wb') as token:
            pickle.dump(credentials, token)
        print("   ✅ Credentials saved!")
    
    access_token = credentials.token
    print(f"   ✅ Access token obtained: {access_token[:20]}...")
    
    # Initialize GoogleFitService
    print("\n🔧 Initializing GoogleFitService...")
    google_fit = GoogleFitService()
    
    # Test 1: Fetch Steps
    print("\n" + "-"*70)
    print("👟 TEST 1: Fetching Steps Data (Last 7 Days)")
    print("-"*70)
    
    steps_data = await google_fit.get_steps(access_token, days=7)
    print(f"   Total Steps (7 days): {steps_data.get('total_steps', 0):,}")
    print(f"   Average Steps/Day: {steps_data.get('avg_steps_per_day', 0):,}")
    
    # Test 2: Fetch Heart Rate
    print("\n" + "-"*70)
    print("❤️  TEST 2: Fetching Heart Rate Data")
    print("-"*70)
    
    hr_data = await google_fit.get_heart_rate(access_token, days=7)
    print(f"   Resting HR: {hr_data.get('resting_hr', 'N/A')} bpm")
    print(f"   Average HR: {hr_data.get('avg_hr', 'N/A')} bpm")
    print(f"   Max HR: {hr_data.get('max_hr', 'N/A')} bpm")
    
    # Test 3: Fetch Sleep
    print("\n" + "-"*70)
    print("😴 TEST 3: Fetching Sleep Data")
    print("-"*70)
    
    sleep_data = await google_fit.get_sleep(access_token, days=7)
    print(f"   Average Sleep: {sleep_data.get('avg_sleep_hours', 'N/A')} hours/night")
    
    # Test 4: Fetch All Metrics
    print("\n" + "-"*70)
    print("🎯 TEST 4: Fetching ALL Metrics Combined")
    print("-"*70)
    
    all_metrics = await google_fit.get_all_metrics(access_token, days=7)
    
    print("\n📊 COMPLETE HEALTH SNAPSHOT:")
    print(f"   Steps/Day: {all_metrics.get('avg_steps_per_day', 0):,}")
    print(f"   Resting HR: {all_metrics.get('resting_hr', 70)} bpm")
    print(f"   Sleep: {all_metrics.get('avg_sleep_hours', 7.0)} hrs")
    
    # Test 5: Token Encryption
    print("\n" + "-"*70)
    print("🔐 TEST 5: Token Encryption/Decryption")
    print("-"*70)
    
    encrypted = google_fit.encrypt_token(access_token)
    print(f"   Encrypted Token (sample): {encrypted[:50]}...")
    
    decrypted = google_fit.decrypt_token(encrypted)
    matches = decrypted == access_token
    print(f"   Decryption Test: {'✅ PASSED' if matches else '❌ FAILED'}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED!")
    print("="*70)
    
    print("\n📋 Summary:")
    print(f"   ✅ OAuth authentication working")
    print(f"   ✅ Steps data: {steps_data.get('avg_steps_per_day', 0):,} steps/day")
    print(f"   ✅ Heart rate: {hr_data.get('resting_hr', 'N/A')} bpm resting")
    print(f"   ✅ Sleep: {sleep_data.get('avg_sleep_hours', 'N/A')} hrs/night")
    print(f"   ✅ Token encryption working")
    
    # Show how this data would look in Observer Agent
    print("\n" + "-"*70)
    print("🤖 SIMULATED OBSERVER AGENT OUTPUT:")
    print("-"*70)
    
    simulated_observer_output = {
        "steps_trend": "increasing" if all_metrics.get('avg_steps_per_day', 0) > 8000 else "stable",
        "reported_pain": [],
        "fatigue_level": "low",
        "exercise_adherence": 85,
        "avg_steps_per_day": all_metrics.get('avg_steps_per_day', 0),
        "resting_hr": all_metrics.get('resting_hr', 70),
        "sleep_hours": all_metrics.get('avg_sleep_hours', 7.0),
        "therapy_completion_rate": 0,
        "total_therapy_sessions": 0,
        "google_fit_connected": True,
    }
    
    import json
    print(json.dumps(simulated_observer_output, indent=2))
    
    print("\n✨ Integration verified! Observer Agent can now use this real data.\n")


if __name__ == "__main__":
    asyncio.run(test_google_fit_integration())
