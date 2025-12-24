"""Seed sample data for HealthSync AI development and testing."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import uuid
from services.db_service import DatabaseService
from config import settings


async def seed_doctors():
    """Seed sample doctors into the database."""
    
    sample_doctors = [
        {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "first_name": "Dr. Sarah",
            "last_name": "Johnson",
            "specialization": "Cardiology",
            "sub_specializations": ["Heart Disease", "Hypertension", "Arrhythmia"],
            "years_experience": 15,
            "rating": 4.8,
            "total_reviews": 127,
            "base_consultation_fee": 150.0,
            "bio": "Board-certified cardiologist with expertise in preventive cardiology and heart disease management.",
            "languages": ["English", "Spanish"],
            "is_verified": True,
            "is_accepting_patients": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "first_name": "Dr. Michael",
            "last_name": "Chen",
            "specialization": "Dermatology",
            "sub_specializations": ["Skin Cancer", "Acne Treatment", "Cosmetic Dermatology"],
            "years_experience": 12,
            "rating": 4.9,
            "total_reviews": 203,
            "base_consultation_fee": 120.0,
            "bio": "Experienced dermatologist specializing in medical and cosmetic skin care.",
            "languages": ["English", "Mandarin"],
            "is_verified": True,
            "is_accepting_patients": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "first_name": "Dr. Emily",
            "last_name": "Rodriguez",
            "specialization": "Pediatrics",
            "sub_specializations": ["General Pediatrics", "Immunizations", "Child Development"],
            "years_experience": 10,
            "rating": 4.7,
            "total_reviews": 156,
            "base_consultation_fee": 100.0,
            "bio": "Compassionate pediatrician dedicated to children's health and wellbeing.",
            "languages": ["English", "Spanish", "Portuguese"],
            "is_verified": True,
            "is_accepting_patients": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "first_name": "Dr. James",
            "last_name": "Williams",
            "specialization": "Orthopedics",
            "sub_specializations": ["Sports Medicine", "Joint Replacement", "Spine Surgery"],
            "years_experience": 18,
            "rating": 4.6,
            "total_reviews": 89,
            "base_consultation_fee": 175.0,
            "bio": "Orthopedic surgeon with focus on minimally invasive procedures.",
            "languages": ["English"],
            "is_verified": True,
            "is_accepting_patients": True,
            "created_at": datetime.utcnow()
        },
        {
            "id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "first_name": "Dr. Priya",
            "last_name": "Patel",
            "specialization": "Psychiatry",
            "sub_specializations": ["Anxiety Disorders", "Depression", "ADHD"],
            "years_experience": 8,
            "rating": 4.9,
            "total_reviews": 174,
            "base_consultation_fee": 130.0,
            "bio": "Psychiatrist specializing in evidence-based treatment for mental health conditions.",
            "languages": ["English", "Hindi", "Gujarati"],
            "is_verified": True,
            "is_accepting_patients": True,
            "created_at": datetime.utcnow()
        }
    ]
    
    for doctor in sample_doctors:
        try:
            # Check if doctor already exists
            existing = await DatabaseService.mongodb_find_one("doctors", {"first_name": doctor["first_name"], "last_name": doctor["last_name"]})
            if not existing:
                await DatabaseService.mongodb_insert_one("doctors", doctor)
                print(f"✓ Seeded doctor: {doctor['first_name']} {doctor['last_name']} ({doctor['specialization']})")
            else:
                print(f"- Skipped existing doctor: {doctor['first_name']} {doctor['last_name']}")
        except Exception as e:
            print(f"✗ Failed to seed doctor {doctor['first_name']} {doctor['last_name']}: {e}")


async def seed_sample_scans():
    """Seed sample AR scan history for the test user."""
    
    test_user_id = "2c965a92-cbfa-46d7-889b-b97cd5f4fe54"  # testuser@healthsync.com
    
    sample_scans = [
        {
            "scan_id": str(uuid.uuid4()),
            "user_id": test_user_id,
            "scan_type": "skin_analysis",
            "timestamp": datetime.utcnow() - timedelta(days=7),
            "confidence_score": 0.85,
            "medical_assessment": {
                "urgency_level": "routine",
                "findings": ["Healthy skin tone", "No visible abnormalities"],
                "recommendations": ["Maintain good skincare routine", "Use sunscreen daily"]
            },
            "ar_overlay": {},
            "follow_up_required": False,
            "created_at": datetime.utcnow() - timedelta(days=7)
        },
        {
            "scan_id": str(uuid.uuid4()),
            "user_id": test_user_id,
            "scan_type": "prescription_ocr",
            "timestamp": datetime.utcnow() - timedelta(days=3),
            "confidence_score": 0.92,
            "medical_assessment": {
                "urgency_level": "routine",
                "findings": ["Prescription text extracted successfully"],
                "recommendations": ["Verify with pharmacist before use"]
            },
            "ar_overlay": {},
            "follow_up_required": False,
            "created_at": datetime.utcnow() - timedelta(days=3)
        },
        {
            "scan_id": str(uuid.uuid4()),
            "user_id": test_user_id,
            "scan_type": "vitals_estimation",
            "timestamp": datetime.utcnow() - timedelta(days=1),
            "confidence_score": 0.78,
            "medical_assessment": {
                "urgency_level": "routine",
                "findings": ["Heart rate: ~72 bpm", "Estimated blood pressure: Normal range"],
                "recommendations": ["Continue healthy lifestyle", "Regular exercise"]
            },
            "ar_overlay": {},
            "follow_up_required": False,
            "created_at": datetime.utcnow() - timedelta(days=1)
        }
    ]
    
    for scan in sample_scans:
        try:
            existing = await DatabaseService.mongodb_find_one("ar_scans", {"scan_id": scan["scan_id"]})
            if not existing:
                await DatabaseService.mongodb_insert_one("ar_scans", scan)
                print(f"✓ Seeded scan: {scan['scan_type']} (confidence: {scan['confidence_score']})")
            else:
                print(f"- Skipped existing scan: {scan['scan_id']}")
        except Exception as e:
            print(f"✗ Failed to seed scan {scan['scan_id']}: {e}")


async def main():
    """Main seeding function."""
    
    print("=" * 60)
    print("HealthSync AI - Sample Data Seeder")
    print("=" * 60)
    print()
    
    try:
        # Initialize database
        print("Initializing database connection...")
        await DatabaseService.initialize()
        print("✓ Database connected\n")
        
        # Seed doctors
        print("Seeding doctors...")
        await seed_doctors()
        print()
        
        # Seed sample scans
        print("Seeding sample AR scans...")
        await seed_sample_scans()
        print()
        
        print("=" * 60)
        print("✓ Sample data seeding completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Seeding failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await DatabaseService.close()


if __name__ == "__main__":
    asyncio.run(main())
