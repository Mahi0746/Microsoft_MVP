#!/usr/bin/env python3
"""
Demo Data Populator for Daily Medical Plan System
Adds medical conditions, prescriptions, consultations, and therapy data
"""

import asyncio
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from services.mongodb_atlas_service import get_mongodb_service


async def populate_demo_data():
    """Add demo medical data for tanisha1@gmail.com"""
    
    print("\n" + "="*70)
    print("📊 POPULATING DEMO MEDICAL DATA")
    print("="*70)
    
    # Connect to MongoDB
    mongodb = await get_mongodb_service()
    if not mongodb or not mongodb.client:
        print("❌ Failed to connect to MongoDB")
        return
    
    # Find user
    user_email = "tanisha1@gmail.com"
    user = await mongodb.database.users.find_one({"email": user_email})
    
    if not user:
        print(f"❌ User {user_email} not found. Please sign up first.")
        return
    
    user_id = user["user_id"]
    print(f"\n✅ Found user: {user_email} (ID: {user_id})")
    
    # 1. Add Medical Conditions & Prescriptions
    print("\n1️⃣ Adding medical conditions and prescriptions...")
    await mongodb.database.users.update_one(
        {"user_id": user_id},
        {"$set": {
            "medical_conditions": [
                "Type 2 Diabetes",
                "Hypertension",
                "Chronic shoulder pain"
            ],
            "current_prescriptions": [
                {
                    "drug": "Metformin",
                    "dose": "500mg",
                    "frequency": "2x daily",
                    "time": "morning and evening",
                    "prescribed_date": (datetime.utcnow() - timedelta(days=30)).isoformat()
                },
                {
                    "drug": "Lisinopril",
                    "dose": "10mg",
                    "frequency": "1x daily",
                    "time": "morning",
                    "prescribed_date": (datetime.utcnow() - timedelta(days=60)).isoformat()
                },
                {
                    "drug": "Ibuprofen",
                    "dose": "200mg",
                    "frequency": "as needed",
                    "time": "with food",
                    "prescribed_date": (datetime.utcnow() - timedelta(days=10)).isoformat()
                }
            ]
        }}
    )
    print("   ✅ Added 3 medical conditions")
    print("   ✅ Added 3 prescriptions (Metformin, Lisinopril, Ibuprofen)")
    
    # 2. Add Consultation Notes
    print("\n2️⃣ Adding consultation notes...")
    consultations = [
        {
            "user_id": user_id,
            "doctor_name": "Dr. Sarah Johnson",
            "consultation_date": datetime.utcnow() - timedelta(days=7),
            "notes": "Patient doing well on current diabetes medication. Blood sugar levels stable. Continue current regimen.",
            "exercise_restrictions": ["Avoid high-impact exercises"],
            "created_at": datetime.utcnow() - timedelta(days=7)
        },
        {
            "user_id": user_id,
            "doctor_name": "Dr. Michael Chen",
            "consultation_date": datetime.utcnow() - timedelta(days=14),
            "notes": "Shoulder pain improving with physical therapy. Recommended gentle stretching exercises daily. Follow up in 2 weeks.",
            "exercise_restrictions": ["No overhead lifting", "Avoid heavy shoulder work"],
            "created_at": datetime.utcnow() - timedelta(days=14)
        }
    ]
    
    # Create collections if they don't exist
    for doc in consultations:
        try:
            await mongodb.database.consultations.insert_one(doc)
        except Exception as e:
            print(f"   ⚠️ Skipping consultation: {e}")
    
    print(f"   ✅ Added consultation records")
    
    # 3. Add Voice Session Chat Logs with Symptoms
    print("\n3️⃣ Adding voice chat logs with symptoms...")
    voice_sessions = [
        {
            "user_id": user_id,
            "transcript": "I've been experiencing some shoulder pain after yesterday's workout. It's not severe, maybe a 4 out of 10, but it's noticeable when I raise my arm.",
            "created_at": datetime.utcnow() - timedelta(days=2),
            "session_duration": 45
        },
        {
            "user_id": user_id,
            "transcript": "Feeling much better today. The shoulder pain has reduced significantly. I did the stretching exercises you recommended and they really helped.",
            "created_at": datetime.utcnow() - timedelta(days=1),
            "session_duration": 32
        },
        {
            "user_id": user_id,
            "transcript": "I had a bit of fatigue yesterday afternoon. Might be related to my blood sugar levels. I've been monitoring them and they're mostly stable.",
            "created_at": datetime.utcnow() - timedelta(hours=12),
            "session_duration": 28
        }
    ]
    
    await mongodb.database.voice_sessions.insert_many(voice_sessions)
    print(f"   ✅ Added {len(voice_sessions)} voice chat sessions with symptom mentions")
    
    # 4. Add Therapy Session Data
    print("\n4️⃣ Adding therapy session adherence data...")
    therapy_sessions = []
    for i in range(7):
        # Simulate 7 days of therapy sessions
        day_offset = i + 1
        completed = i % 3 != 0  # Some sessions not completed
        pain_level = 5 - i if i < 5 else 2  # Pain decreasing over time
        
        therapy_sessions.append({
            "user_id": user_id,
            "exercise_name": "Shoulder mobility routine",
            "completed": completed,
            "pain_level": pain_level,
            "duration_minutes": 15 if completed else 0,
            "created_at": datetime.utcnow() - timedelta(days=day_offset)
        })
    
    await mongodb.database.therapy_sessions.insert_many(therapy_sessions)
    completed_count = sum(1 for s in therapy_sessions if s['completed'])
    print(f"   ✅ Added {len(therapy_sessions)} therapy sessions")
    print(f"   ✅ Completion rate: {completed_count}/{len(therapy_sessions)} ({completed_count/len(therapy_sessions)*100:.0f}%)")
    
    # 5. Summary
    print("\n" + "="*70)
    print("✅ DEMO DATA POPULATED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n📋 Summary for {user_email}:")
    print(f"   • Medical Conditions: 3 (Diabetes, Hypertension, Shoulder pain)")
    print(f"   • Prescriptions: 3 medications")
    print(f"   • Consultations: 2 doctor visits")
    print(f"   • Voice Chats: 3 sessions with symptoms")
    print(f"   • Therapy Sessions: {len(therapy_sessions)} sessions")
    
    print("\n🚀 Now run the Observer Agent to see it collect this data!")
    print(f"   Test endpoint: GET /api/agents/observer/test")
    print(f"   Generate plan: POST /api/agents/trigger-daily-planning")
    print()


if __name__ == "__main__":
    asyncio.run(populate_demo_data())
