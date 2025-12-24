"""
User Onboarding Routes - Lifestyle Assessment

5-question onboarding to determine user's health profile
Generates realistic simulated data for Observer Agent
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import Dict
from datetime import datetime

from api.middleware.auth import get_current_user
from services.smart_health_simulator import SmartHealthSimulator
from services.mongodb_atlas_service import get_mongodb_service

router = APIRouter(prefix="/onboarding", tags=["Onboarding"])


class LifestyleAnswers(BaseModel):
    """5-question lifestyle assessment"""
    activity: int  # 1-4: Mostly sitting -> Very active
    exercise: int  # 1-4: Rarely -> 5+ times/week
    sleep: int  # 1-4: Poor -> Excellent
    stress: int  # 1-4: Very high -> Very relaxed
    diet: int  # 1-4: Fast food -> Very clean


@router.post("/lifestyle-assessment")
async def submit_lifestyle_assessment(
    answers: LifestyleAnswers,
    current_user: dict = Depends(get_current_user)
):
    """
    Submit 5-question lifestyle assessment
    Generates user's health profile and simulated data
    """
    user_id = current_user["user_id"]
    
    try:
        # Convert to dict
        answers_dict = {
            'activity': answers.activity,
            'exercise': answers.exercise,
            'sleep': answers.sleep,
            'stress': answers.stress,
            'diet': answers.diet
        }
        
        # Determine profile
        profile = SmartHealthSimulator.determine_profile_from_answers(answers_dict)
        
        # Generate health data
        health_data = SmartHealthSimulator.generate_health_data(profile, answers_dict)
        
        # Stor in MongoDB
        mongodb = await get_mongodb_service()
        if mongodb and mongodb.client:
            # Store lifestyle profile
            await mongodb.database.users.update_one(
                {"user_id": user_id},
                {"$set": {
                    "lifestyle_profile": profile,
                    "lifestyle_answers": answers_dict,
                    "simulated_health_data": health_data,
                    "onboarding_completed": True,
                    "onboarding_completed_at": datetime.utcnow()
                }}
            )
        
        return {
            "success": True,
            "profile": profile,
            "health_snapshot": {
                "avg_steps": health_data["avg_steps_per_day"],
                "sleep_hours": health_data["avg_sleep_hours"],
                "resting_hr": health_data["resting_hr"],
                "stress_level": health_data["stress_level"]
            },
            "message": f"Profile set to: {profile.upper()}"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process assessment: {str(e)}"
        )


@router.get("/status")
async def get_onboarding_status(
    current_user: dict = Depends(get_current_user)
):
    """Check if user has completed onboarding"""
    user_id = current_user["user_id"]
    
    try:
        mongodb = await get_mongodb_service()
        if not mongodb or not mongodb.client:
            return {"completed": False}
        
        user = await mongodb.database.users.find_one({"user_id": user_id})
        
        return {
            "completed": user.get("onboarding_completed", False) if user else False,
            "profile": user.get("lifestyle_profile") if user else None
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
