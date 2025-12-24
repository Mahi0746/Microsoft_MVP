#!/usr/bin/env python3
"""
Reset Daily Plans - Clear all generated plans and related data
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from services.mongodb_atlas_service import get_mongodb_service


async def reset_all_plans():
    """Delete all daily plans, reminders, and tasks"""
    
    print("\n🔄 RESETTING ALL DAILY PLANS")
    print("="*60)
    
    mongodb = await get_mongodb_service()
    if not mongodb or not mongodb.client:
        print("❌ Failed to connect to MongoDB")
        return
    
    # Delete all daily plans
    print("\n1️⃣ Deleting daily_agent_memory...")
    result1 = await mongodb.database.daily_agent_memory.delete_many({})
    print(f"   ✅ Deleted {result1.deleted_count} daily plans")
    
    # Delete all medication reminders
    print("\n2️⃣ Deleting user_reminders...")
    result2 = await mongodb.database.user_reminders.delete_many({})
    print(f"   ✅ Deleted {result2.deleted_count} medication reminders")
    
    # Delete all exercise tasks
    print("\n3️⃣ Deleting user_tasks...")
    result3 = await mongodb.database.user_tasks.delete_many({})
    print(f"   ✅ Deleted {result3.deleted_count} exercise tasks")
    
    # Summary
    total_deleted = result1.deleted_count + result2.deleted_count + result3.deleted_count
    
    print("\n" + "="*60)
    print("✅ RESET COMPLETE!")
    print("="*60)
    print(f"\nTotal items deleted: {total_deleted}")
    print("\n🚀 System ready for fresh demo!")
    print("   • All daily plans cleared")
    print("   • All reminders cleared")
    print("   • All tasks cleared")
    print("\nYou can now generate a new daily plan from scratch.")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(reset_all_plans())
