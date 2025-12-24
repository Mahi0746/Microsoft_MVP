#!/usr/bin/env python3
"""
Add Medical Data to Existing User
Updates tanisha1@gmail.com with health conditions and prescriptions
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from services.mongodb_atlas_service import get_mongodb_service


async def add_medical_data():
    """Add medical conditions and prescriptions to tanisha1@gmail.com"""
    
    print("\n📊 Adding Medical Data to User")
    print("="*60)
    
    mongodb = await get_mongodb_service()
    if not mongodb or not mongodb.client:
        print("❌ Failed to connect to MongoDB")
        return
    
    user_id = "3e5dfdc2-df86-46a0-ab20-65fa0368ea3e"
    
    # Update user document with medical data
    result = await mongodb.database.users.update_one(
        {"user_id": user_id},
        {"$set": {
            "medical_conditions": [
                "Type 2 Diabetes",
                "Hypertension (High Blood Pressure)",
                "Chronic Shoulder Pain"
            ],
            "current_prescriptions": [
                {
                    "drug": "Metformin",
                    "dose": "500mg",
                    "frequency": "2x daily",
                    "time": ["8:00 AM", "8:00 PM"],
                    "prescribed_date": (datetime.utcnow() - timedelta(days=30)).isoformat()
                },
                {
                    "drug": "Lisinopril",
                    "dose": "10mg",
                    "frequency": "1x daily",
                    "time": ["8:00 AM"],
                    "prescribed_date": (datetime.utcnow() - timedelta(days=60)).isoformat()
                },
                {
                    "drug": "Ibuprofen",
                    "dose": "200mg",
                    "frequency": "as needed",
                    "time": ["with food"],
                    "prescribed_date": (datetime.utcnow() - timedelta(days=10)).isoformat(),
                    "note": "For shoulder pain"
                }
            ],
            "last_updated": datetime.utcnow().isoformat()
        }}
    )
    
    if result.modified_count > 0:
        print("\n✅ Successfully added medical data!")
        print("\n📋 Added:")
        print("   • Medical Conditions:")
        print("     - Type 2 Diabetes")
        print("     - Hypertension (High Blood Pressure)")
        print("     - Chronic Shoulder Pain")
        print("\n   • Prescriptions:")
        print("     - Metformin 500mg (2x daily)")
        print("     - Lisinopril 10mg (1x daily)")
        print("     - Ibuprofen 200mg (as needed)")
        
        print("\n🚀 Now you can:")
        print("   1. Go to http://localhost:3000/ai-coach")
        print("   2. Click 'Generate Today's Plan'")
        print("   3. See medication reminders and safe exercises!")
    else:
        print("⚠️ No changes made (data might already exist)")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    asyncio.run(add_medical_data())
