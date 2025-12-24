#!/usr/bin/env python3
"""
Display Medical Data for tanisha1@gmail.com
Shows all stored health information
"""

import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from services.mongodb_atlas_service import get_mongodb_service


async def display_user_data():
    """Display all medical data for tanisha1@gmail.com"""
    
    print("\n" + "="*70)
    print("📊 MEDICAL DATA FOR tanisha1@gmail.com")
    print("="*70)
    
    mongodb = await get_mongodb_service()
    if not mongodb or not mongodb.client:
        print("❌ Failed to connect to MongoDB")
        return
    
    user_email = "hemil@gmail.com"
    user = await mongodb.database.users.find_one({"email": user_email})
    
    if not user:
        print(f"❌ User {user_email} not found")
        return
    
    user_id = user["user_id"]
    
    print(f"\n👤 USER PROFILE")
    print("-" * 70)
    print(f"   Name: {user.get('firstName')} {user.get('lastName')}")
    print(f"   Email: {user.get('email')}")
    print(f"   User ID: {user_id}")
    print(f"   Role: {user.get('role')}")
    print(f"   Phone: {user.get('phone')}")
    
    # Medical Conditions
    conditions = user.get('medical_conditions', [])
    print(f"\n💊 MEDICAL CONDITIONS ({len(conditions)})")
    print("-" * 70)
    if conditions:
        for i, condition in enumerate(conditions, 1):
            print(f"   {i}. {condition}")
    else:
        print("   (none)")
    
    # Prescriptions
    prescriptions = user.get('current_prescriptions', [])
    print(f"\n💉 CURRENT PRESCRIPTIONS ({len(prescriptions)})")
    print("-" * 70)
    if prescriptions:
        for i, rx in enumerate(prescriptions, 1):
            print(f"\n   {i}. {rx.get('drug')}")
            print(f"      Dose: {rx.get('dose')}")
            print(f"      Frequency: {rx.get('frequency')}")
            print(f"      Time: {rx.get('time')}")
            if 'note' in rx:
                print(f"      Note: {rx.get('note')}")
    else:
        print("   (none)")
    
    # Consultations
    print(f"\n🩺 CONSULTATION NOTES")
    print("-" * 70)
    consultations = await mongodb.database.consultations.find({
        "user_id": user_id
    }).to_list(length=100)
    
    if consultations:
        for i, consult in enumerate(consultations, 1):
            date = consult.get('consultation_date', 'N/A')
            if hasattr(date, 'isoformat'):
                date = date.strftime('%Y-%m-%d')
            print(f"\n   {i}. Date: {date}")
            print(f"      Doctor: {consult.get('doctor_name')}")
            print(f"      Notes: {consult.get('notes')}")
            restrictions = consult.get('exercise_restrictions', [])
            if restrictions:
                print(f"      Restrictions: {', '.join(restrictions)}")
    else:
        print("   (none)")
    
    # Voice Sessions
    print(f"\n🎤 VOICE CHAT LOGS")
    print("-" * 70)
    voice_sessions = await mongodb.database.voice_sessions.find({
        "user_id": user_id
    }).sort("created_at", -1).to_list(length=5)
    
    if voice_sessions:
        for i, session in enumerate(voice_sessions, 1):
            created = session.get('created_at', 'N/A')
            if hasattr(created, 'isoformat'):
                created = created.strftime('%Y-%m-%d %H:%M')
            print(f"\n   {i}. {created}")
            transcript = session.get('transcript', '')
            if len(transcript) > 100:
                transcript = transcript[:100] + "..."
            print(f"      \"{transcript}\"")
    else:
        print("   (none)")
    
    # Therapy Sessions
    print(f"\n🏋️ THERAPY SESSION ADHERENCE")
    print("-" * 70)
    therapy_sessions = await mongodb.database.therapy_sessions.find({
        "user_id": user_id
    }).to_list(length=100)
    
    if therapy_sessions:
        total = len(therapy_sessions)
        completed = sum(1 for s in therapy_sessions if s.get('completed'))
        completion_rate = (completed / total * 100) if total > 0 else 0
        
        print(f"   Total Sessions: {total}")
        print(f"   Completed: {completed}")
        print(f"   Completion Rate: {completion_rate:.0f}%")
        
        # Show pain progression
        pain_levels = [(s.get('created_at'), s.get('pain_level', 0)) 
                      for s in therapy_sessions if 'pain_level' in s]
        if pain_levels:
            pain_levels.sort(key=lambda x: x[0])
            print(f"\n   Pain Level Progression:")
            for date, pain in pain_levels[:5]:
                if hasattr(date, 'isoformat'):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                print(f"      {date_str}: {pain}/10")
    else:
        print("   (none)")
    
    # Daily Plans
    print(f"\n📅 DAILY PLANS GENERATED")
    print("-" * 70)
    daily_plans = await mongodb.database.daily_agent_memory.find({
        "user_id": user_id
    }).sort("date", -1).to_list(length=5)
    
    if daily_plans:
        for i, plan in enumerate(daily_plans, 1):
            print(f"\n   {i}. Date: {plan.get('date')}")
            daily_plan = plan.get('daily_plan', {})
            print(f"      Exercises: {len(daily_plan.get('exercises', []))}")
            print(f"      Medications: {len(daily_plan.get('medication_reminders', []))}")
            
            reflection = plan.get('reflection', {})
            if reflection:
                print(f"      Adherence Score: {reflection.get('adherence_score', 0)}%")
    else:
        print("   (none)")
    
    print("\n" + "="*70)
    print("✅ DATA DISPLAY COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(display_user_data())
