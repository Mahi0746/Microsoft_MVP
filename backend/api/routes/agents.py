"""
AI Health Agents API Routes

Endpoints for the multi-agent health management system.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

# Import auth dependencies
from api.middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["AI Health Agents"])


# ============= Request/Response Models =============

class TriggerPlanningRequest(BaseModel):
    """Request to trigger daily planning cycle"""
    force: bool = Field(False, description="Force cycle even if already generated today")


class AgentCycleResponse(BaseModel):
    """Response from agent cycle execution"""
    user_id: str
    success: bool
    cycle_started_at: str
    cycle_completed_at: Optional[str]
    stages: Dict[str, Any]
    error: Optional[str] = None


class HealthPlanResponse(BaseModel):
    """Current week's health plan"""
    user_id: str
    week_start: str
    intensity_level: str
    focus: str
    target_sessions: int
    recovery_exercises: bool
    plan_note: str
    status: str


class AgentHistoryResponse(BaseModel):
    """Historical agent decisions"""
    user_id: str
    total_weeks: int
    history: List[Dict[str, Any]]


# ============= Endpoints =============

@router.post("/trigger-daily-planning", response_model=AgentCycleResponse)
async def trigger_daily_planning(
    request: TriggerPlanningRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Manually trigger the multi-agent DAILY planning cycle.
    
    This runs the full agent flow:
    1. Observer collects medical data (checks today_generated)
    2. Planner generates daily plan
    3. Safety validates plan
    4. Action executes plan (medication reminders, exercises)
    5. Reflection runs end-of-day
    """
    user_id = current_user.get("user_id")
    
    try:
        logger.info(f"Triggering DAILY agent cycle for user {user_id}")
        
        from services.agents import get_orchestrator
        
        orchestrator = get_orchestrator()
        result = await orchestrator.run_daily_cycle(user_id)
        
        return AgentCycleResponse(**result)
        
    except Exception as e:
        logger.error(f"Agent cycle failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent cycle failed: {str(e)}"
        )


@router.get("/daily-plan")
async def get_current_daily_plan(
    current_user: dict = Depends(get_current_user)
):
    """
    Get today's AI-generated daily health plan.
    """
    user_id = current_user.get("user_id")
    
    try:
        from services.mongodb_atlas_service import get_mongodb_service
        from datetime import datetime
        
        mongodb = await get_mongodb_service()
        if not mongodb or not mongodb.client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        today = datetime.utcnow().date().isoformat()
        
        # Get today's plan from daily_agent_memory
        plan_doc = await mongodb.database.daily_agent_memory.find_one(
            {"user_id": user_id, "date": today}
        )
        
        if not plan_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No plan found for today. Trigger planning first."
            )
        
        return {
            "user_id": user_id,
            "date": plan_doc.get("date"),
            "daily_plan": plan_doc.get("daily_plan", {}),
            "safety_approval": plan_doc.get("safety_approval", {}),
            "reflection": plan_doc.get("reflection", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get daily plan for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/history")
async def get_agent_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """
    Get historical daily agent decisions.
    """
    user_id = current_user.get("user_id")
    
    try:
        from services.mongodb_atlas_service import get_mongodb_service
        
        mongodb = await get_mongodb_service()
        if not mongodb or not mongodb.client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        # Get daily agent memory records
        cursor = mongodb.database.daily_agent_memory.find(
            {"user_id": user_id}
        ).sort("date", -1).limit(limit)
        
        history = []
        async for record in cursor:
            record["_id"] = str(record["_id"])
            history.append(record)
        
        return {
            "user_id": user_id,
            "total_days": len(history),
            "history": history
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent history for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/override-plan")
async def override_weekly_plan(
    override_plan: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    """
    Allow manual override of AI-generated plan (for doctors/admins).
    
    Safety note: This bypasses AI safety checks. Use carefully.
    """
    user_id = current_user.get("user_id")
    user_role = current_user.get("role", "patient")
    
    # Only doctors and admins can override
    if user_role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors and admins can override AI plans"
        )
    
    try:
        from services.mongodb_atlas_service import get_mongodb_service
        
        mongodb = await get_mongodb_service()
        if not mongodb or not mongodb.client:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not available"
            )
        
        # Create override plan document
        override_doc = {
            "user_id": override_plan.get("target_user_id", user_id),
            "week_start": datetime.utcnow().isoformat(),
            "intensity_level": override_plan.get("intensity_level", "maintain"),
            "focus": override_plan.get("focus", "balanced"),
            "target_sessions": override_plan.get("target_sessions", 3),
            "recovery_exercises": override_plan.get("recovery_exercises", False),
            "status": "active",
            "plan_note": f"Manual override by {user_role} {user_id}: {override_plan.get('note', '')}",
            "is_override": True,
            "overridden_by": user_id,
            "created_at": datetime.utcnow()
        }
        
        # Deactivate existing plans
        await mongodb.database.weekly_health_plans.update_many(
            {"user_id": override_doc["user_id"], "status": "active"},
            {"$set": {"status": "superseded"}}
        )
        
        # Insert override
        await mongodb.database.weekly_health_plans.insert_one(override_doc)
        
        logger.info(f"Plan overridden for user {override_doc['user_id']} by {user_role} {user_id}")
        
        return {
            "success": True,
            "message": "Plan override successful",
            "override_id": str(override_doc["_id"])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan override failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/observer/test")
async def test_observer(
    current_user: dict = Depends(get_current_user)
):
    """Test endpoint to run Observer agent only - now for daily medical data"""
    user_id = current_user.get("user_id")
    
    try:
        from services.agents import get_observer_agent
        
        observer = get_observer_agent()
        data = await observer.observe(user_id)
        
        return {
            "success": True,
            "observed_data": data
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
