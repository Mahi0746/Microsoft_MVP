# HealthSync AI - Pain-to-Game Therapy Routes
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel, validator
import structlog

from config import settings
from api.middleware.auth import get_current_user
from api.middleware.rate_limit import rate_limit_general, rate_limit_image
from services.therapy_game_service import TherapyGameService, ExerciseType, DifficultyLevel
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class StartSessionRequest(BaseModel):
    exercise_type: str
    difficulty: str
    duration_minutes: int
    pain_level_before: int
    
    @validator('exercise_type')
    def validate_exercise_type(cls, v):
        try:
            ExerciseType(v)
            return v
        except ValueError:
            raise ValueError(f'Invalid exercise type: {v}')
    
    @validator('difficulty')
    def validate_difficulty(cls, v):
        try:
            DifficultyLevel(v)
            return v
        except ValueError:
            raise ValueError(f'Invalid difficulty level: {v}')
    
    @validator('duration_minutes')
    def validate_duration(cls, v):
        if not (1 <= v <= 60):
            raise ValueError('Duration must be between 1 and 60 minutes')
        return v
    
    @validator('pain_level_before')
    def validate_pain_level(cls, v):
        if not (0 <= v <= 10):
            raise ValueError('Pain level must be between 0 and 10')
        return v


class MotionFrameRequest(BaseModel):
    session_id: str
    frame_data: str  # Base64 encoded image
    timestamp: float
    
    @validator('frame_data')
    def validate_frame_data(cls, v):
        if not v or len(v) < 100:
            raise ValueError('Valid frame data is required')
        return v


class CompleteSessionRequest(BaseModel):
    session_id: str
    pain_level_after: int
    user_feedback: Optional[str] = None
    
    @validator('pain_level_after')
    def validate_pain_level(cls, v):
        if not (0 <= v <= 10):
            raise ValueError('Pain level must be between 0 and 10')
        return v


class SessionResponse(BaseModel):
    session_id: str
    exercise_config: Dict[str, Any]
    target_repetitions: int
    instructions: Dict[str, Any]
    tracking_points: List[str]
    estimated_calories: float
    potential_points: int


class MotionAnalysisResponse(BaseModel):
    success: bool
    movement_analysis: Optional[Dict[str, Any]] = None
    feedback: Optional[Dict[str, Any]] = None
    progress: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SessionResultResponse(BaseModel):
    session_result: Dict[str, Any]
    summary: Dict[str, Any]
    achievements: List[str]
    recommendations: List[str]


class UserProgressResponse(BaseModel):
    total_sessions: int
    total_points: int
    total_calories: float
    total_exercise_time: int
    achievements: List[str]
    exercise_stats: Dict[str, Any]
    pain_trend: str
    consistency_score: float


# =============================================================================
# SESSION MANAGEMENT ROUTES
# =============================================================================

@router.post("/session/start", response_model=SessionResponse)
@rate_limit_general
async def start_therapy_session(
    request: Request,
    session_request: StartSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Start a new pain-to-game therapy session."""
    
    try:
        # Check if therapy game is enabled
        if not settings.enable_therapy_game:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Therapy game service is currently disabled"
            )
        
        # Start therapy session
        session_data = await TherapyGameService.start_therapy_session(
            current_user["user_id"],
            session_request.exercise_type,
            session_request.difficulty,
            session_request.duration_minutes,
            session_request.pain_level_before
        )
        
        logger.info(
            "Therapy session started",
            user_id=current_user["user_id"],
            session_id=session_data["session_id"],
            exercise_type=session_request.exercise_type,
            difficulty=session_request.difficulty
        )
        
        return SessionResponse(**session_data)
        
    except ValueError as e:
        logger.warning("Invalid therapy session request", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to start therapy session", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start therapy session"
        )


@router.post("/session/motion", response_model=MotionAnalysisResponse)
@rate_limit_image
async def process_motion_frame(
    request: Request,
    motion_request: MotionFrameRequest,
    current_user: dict = Depends(get_current_user)
):
    """Process motion tracking frame for real-time analysis."""
    
    try:
        # Process motion frame
        result = await TherapyGameService.process_motion_frame(
            motion_request.session_id,
            motion_request.frame_data,
            motion_request.timestamp
        )
        
        return MotionAnalysisResponse(**result)
        
    except ValueError as e:
        logger.warning("Invalid motion frame", session_id=motion_request.session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Motion frame processing failed", session_id=motion_request.session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Motion frame processing failed"
        )


@router.post("/session/complete", response_model=SessionResultResponse)
@rate_limit_general
async def complete_therapy_session(
    request: Request,
    complete_request: CompleteSessionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Complete therapy session and get results."""
    
    try:
        # Complete therapy session
        result = await TherapyGameService.complete_therapy_session(
            complete_request.session_id,
            complete_request.pain_level_after,
            complete_request.user_feedback
        )
        
        logger.info(
            "Therapy session completed",
            user_id=current_user["user_id"],
            session_id=complete_request.session_id,
            pain_reduction=result["summary"]["pain_reduction"]
        )
        
        return SessionResultResponse(**result)
        
    except ValueError as e:
        logger.warning("Invalid session completion request", session_id=complete_request.session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to complete therapy session", session_id=complete_request.session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete therapy session"
        )


@router.get("/progress", response_model=UserProgressResponse)
@rate_limit_general
async def get_user_progress(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get user's therapy progress and statistics."""
    
    try:
        progress_data = await TherapyGameService.get_user_progress(current_user["user_id"])
        
        if "error" in progress_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=progress_data["error"]
            )
        
        if "message" in progress_data:
            # No progress found - return empty progress
            return UserProgressResponse(
                total_sessions=0,
                total_points=0,
                total_calories=0.0,
                total_exercise_time=0,
                achievements=[],
                exercise_stats={},
                pain_trend="no_data",
                consistency_score=0.0
            )
        
        user_progress = progress_data["user_progress"]
        trends = progress_data["trends"]
        
        return UserProgressResponse(
            total_sessions=user_progress.get("total_sessions", 0),
            total_points=user_progress.get("total_points", 0),
            total_calories=user_progress.get("total_calories", 0.0),
            total_exercise_time=user_progress.get("total_exercise_time", 0),
            achievements=user_progress.get("achievements", []),
            exercise_stats=user_progress.get("exercise_stats", {}),
            pain_trend=trends.get("pain_trend", "no_data"),
            consistency_score=trends.get("consistency_score", 0.0)
        )
        
    except Exception as e:
        logger.error("Failed to get user progress", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user progress"
        )


# =============================================================================
# EXERCISE CONFIGURATION ROUTES
# =============================================================================

@router.get("/exercises")
@rate_limit_general
async def get_available_exercises(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get list of available therapy exercises."""
    
    try:
        exercises = []
        
        for exercise_type, config in TherapyGameService.EXERCISE_CONFIGS.items():
            exercises.append({
                "type": exercise_type.value,
                "name": config["name"],
                "description": config["description"],
                "target_muscles": config["target_muscles"],
                "pain_conditions": config["pain_conditions"],
                "duration_range": config["duration_range"],
                "repetition_range": config["repetition_range"],
                "difficulty_levels": [level.value for level in DifficultyLevel]
            })
        
        return {
            "exercises": exercises,
            "total_count": len(exercises)
        }
        
    except Exception as e:
        logger.error("Failed to get available exercises", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve available exercises"
        )


@router.get("/exercises/{exercise_type}")
@rate_limit_general
async def get_exercise_details(
    request: Request,
    exercise_type: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific exercise."""
    
    try:
        # Validate exercise type
        try:
            exercise_enum = ExerciseType(exercise_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid exercise type: {exercise_type}"
            )
        
        config = TherapyGameService.EXERCISE_CONFIGS[exercise_enum]
        
        # Get user's history with this exercise
        user_stats = await DatabaseService.mongodb_find_one(
            "user_therapy_progress",
            {"user_id": current_user["user_id"]}
        )
        
        exercise_stats = {}
        if user_stats and exercise_type in user_stats.get("exercise_stats", {}):
            exercise_stats = user_stats["exercise_stats"][exercise_type]
        
        return {
            "exercise": {
                "type": exercise_type,
                "name": config["name"],
                "description": config["description"],
                "target_muscles": config["target_muscles"],
                "pain_conditions": config["pain_conditions"],
                "duration_range": config["duration_range"],
                "repetition_range": config["repetition_range"],
                "tracking_points": config["tracking_points"],
                "movement_threshold": config["movement_threshold"],
                "accuracy_threshold": config["accuracy_threshold"]
            },
            "user_stats": exercise_stats,
            "difficulty_levels": [
                {
                    "level": level.value,
                    "description": _get_difficulty_description(level)
                }
                for level in DifficultyLevel
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get exercise details", exercise_type=exercise_type, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exercise details"
        )


# =============================================================================
# SESSION HISTORY ROUTES
# =============================================================================

@router.get("/sessions/history")
@rate_limit_general
async def get_session_history(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    exercise_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get user's therapy session history."""
    
    try:
        # Build query filter
        query_filter = {
            "user_id": current_user["user_id"],
            "status": "completed"
        }
        
        if exercise_type:
            try:
                ExerciseType(exercise_type)  # Validate exercise type
                query_filter["exercise_type"] = exercise_type
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid exercise type: {exercise_type}"
                )
        
        # Get sessions
        sessions = await DatabaseService.mongodb_find_many(
            "therapy_sessions",
            query_filter,
            limit=limit,
            skip=offset,
            sort=[("start_time", -1)]
        )
        
        # Get total count
        total_count = await DatabaseService.mongodb_count_documents(
            "therapy_sessions",
            query_filter
        )
        
        # Format session data
        formatted_sessions = []
        for session in sessions:
            session_result = session.get("session_result", {})
            formatted_sessions.append({
                "session_id": session["session_id"],
                "exercise_type": session["exercise_type"],
                "difficulty": session["difficulty"],
                "start_time": session["start_time"],
                "end_time": session.get("end_time"),
                "duration_minutes": session_result.get("duration_minutes", 0),
                "completed_repetitions": session_result.get("completed_repetitions", 0),
                "target_repetitions": session["target_repetitions"],
                "accuracy_score": session_result.get("accuracy_score", 0.0),
                "pain_level_before": session["pain_level_before"],
                "pain_level_after": session_result.get("pain_level_after", 0),
                "calories_burned": session_result.get("calories_burned", 0.0),
                "points_earned": session_result.get("points_earned", 0),
                "achievements_unlocked": session_result.get("achievements_unlocked", [])
            })
        
        return {
            "sessions": formatted_sessions,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(formatted_sessions) < total_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session history", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session history"
        )


@router.get("/sessions/{session_id}")
@rate_limit_general
async def get_session_details(
    request: Request,
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific session."""
    
    try:
        session = await DatabaseService.mongodb_find_one(
            "therapy_sessions",
            {
                "session_id": session_id,
                "user_id": current_user["user_id"]
            }
        )
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        # Format detailed session data
        session_data = {
            "session_id": session["session_id"],
            "exercise_type": session["exercise_type"],
            "difficulty": session["difficulty"],
            "start_time": session["start_time"],
            "end_time": session.get("end_time"),
            "status": session["status"],
            "config": session["config"],
            "target_repetitions": session["target_repetitions"],
            "pain_level_before": session["pain_level_before"],
            "performance_metrics": session.get("performance_metrics", {}),
            "user_feedback": session.get("user_feedback")
        }
        
        # Add session result if completed
        if session["status"] == "completed":
            session_data["session_result"] = session.get("session_result", {})
        
        # Add tracking data summary (last 10 frames)
        tracking_data = session.get("tracking_data", [])
        if tracking_data:
            session_data["tracking_summary"] = {
                "total_frames": len(tracking_data),
                "recent_frames": tracking_data[-10:] if len(tracking_data) > 10 else tracking_data
            }
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get session details", session_id=session_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session details"
        )


# =============================================================================
# ACHIEVEMENTS AND GAMIFICATION ROUTES
# =============================================================================

@router.get("/achievements")
@rate_limit_general
async def get_achievements(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get user's achievements and available achievements."""
    
    try:
        # Get user progress
        progress = await DatabaseService.mongodb_find_one(
            "user_therapy_progress",
            {"user_id": current_user["user_id"]}
        )
        
        user_achievements = progress.get("achievements", []) if progress else []
        
        # Format achievement data
        achievements_data = []
        for achievement_id, achievement_info in TherapyGameService.ACHIEVEMENT_SYSTEM.items():
            achievements_data.append({
                "id": achievement_id,
                "name": achievement_info["name"],
                "description": achievement_info["description"],
                "points": achievement_info["points"],
                "unlocked": achievement_id in user_achievements,
                "unlock_date": None  # Could be added to track when unlocked
            })
        
        # Calculate achievement stats
        total_achievements = len(TherapyGameService.ACHIEVEMENT_SYSTEM)
        unlocked_achievements = len(user_achievements)
        completion_percentage = (unlocked_achievements / total_achievements) * 100 if total_achievements > 0 else 0
        
        total_achievement_points = 0
        for achievement_id in user_achievements:
            if achievement_id in TherapyGameService.ACHIEVEMENT_SYSTEM:
                total_achievement_points += TherapyGameService.ACHIEVEMENT_SYSTEM[achievement_id]["points"]
        
        return {
            "achievements": achievements_data,
            "stats": {
                "total_achievements": total_achievements,
                "unlocked_achievements": unlocked_achievements,
                "completion_percentage": round(completion_percentage, 1),
                "total_achievement_points": total_achievement_points
            }
        }
        
    except Exception as e:
        logger.error("Failed to get achievements", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve achievements"
        )


# =============================================================================
# ANALYTICS AND INSIGHTS ROUTES
# =============================================================================

@router.get("/analytics/pain-trends")
@rate_limit_general
async def get_pain_trends(
    request: Request,
    days: int = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get pain level trends and analytics."""
    
    try:
        # Get user progress
        progress = await DatabaseService.mongodb_find_one(
            "user_therapy_progress",
            {"user_id": current_user["user_id"]}
        )
        
        if not progress:
            return {
                "message": "No pain tracking data available",
                "trends": []
            }
        
        pain_tracking = progress.get("pain_tracking", [])
        
        # Filter by date range
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_tracking = [
            entry for entry in pain_tracking
            if entry["date"] >= cutoff_date
        ]
        
        # Calculate trends
        trends = []
        pain_reductions = []
        
        for entry in recent_tracking:
            pain_reduction = entry["before"] - entry["after"]
            pain_reductions.append(pain_reduction)
            
            trends.append({
                "date": entry["date"],
                "pain_before": entry["before"],
                "pain_after": entry["after"],
                "reduction": pain_reduction,
                "exercise_type": entry["exercise_type"]
            })
        
        # Calculate statistics
        stats = {}
        if pain_reductions:
            stats = {
                "average_reduction": round(sum(pain_reductions) / len(pain_reductions), 2),
                "best_reduction": max(pain_reductions),
                "worst_reduction": min(pain_reductions),
                "total_sessions": len(pain_reductions),
                "improvement_rate": len([r for r in pain_reductions if r > 0]) / len(pain_reductions) * 100
            }
        
        return {
            "trends": trends,
            "stats": stats,
            "period_days": days,
            "data_points": len(trends)
        }
        
    except Exception as e:
        logger.error("Failed to get pain trends", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pain trends"
        )


@router.get("/analytics/exercise-performance")
@rate_limit_general
async def get_exercise_performance(
    request: Request,
    exercise_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get exercise performance analytics."""
    
    try:
        # Get user progress
        progress = await DatabaseService.mongodb_find_one(
            "user_therapy_progress",
            {"user_id": current_user["user_id"]}
        )
        
        if not progress:
            return {
                "message": "No exercise performance data available",
                "performance": {}
            }
        
        exercise_stats = progress.get("exercise_stats", {})
        
        # Filter by exercise type if specified
        if exercise_type:
            try:
                ExerciseType(exercise_type)  # Validate exercise type
                if exercise_type in exercise_stats:
                    exercise_stats = {exercise_type: exercise_stats[exercise_type]}
                else:
                    exercise_stats = {}
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid exercise type: {exercise_type}"
                )
        
        # Format performance data
        performance_data = {}
        for ex_type, stats in exercise_stats.items():
            performance_data[ex_type] = {
                "total_sessions": stats.get("sessions", 0),
                "total_repetitions": stats.get("total_reps", 0),
                "average_accuracy": round(stats.get("avg_accuracy", 0.0) * 100, 1),
                "best_accuracy": round(stats.get("best_accuracy", 0.0) * 100, 1),
                "average_reps_per_session": round(
                    stats.get("total_reps", 0) / max(stats.get("sessions", 1), 1), 1
                )
            }
        
        return {
            "performance": performance_data,
            "total_exercise_types": len(performance_data),
            "filter_applied": exercise_type is not None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get exercise performance", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve exercise performance"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _get_difficulty_description(difficulty: DifficultyLevel) -> str:
    """Get description for difficulty level."""
    
    descriptions = {
        DifficultyLevel.BEGINNER: "Perfect for those new to exercise or with limited mobility",
        DifficultyLevel.INTERMEDIATE: "Suitable for regular exercisers with good mobility",
        DifficultyLevel.ADVANCED: "Challenging exercises for experienced users",
        DifficultyLevel.REHABILITATION: "Gentle exercises designed for injury recovery"
    }
    
    return descriptions.get(difficulty, "Standard difficulty level")

# =============================================================================
# LEADERBOARD AND GAMIFICATION ROUTES
# =============================================================================

@router.get("/leaderboard")
@rate_limit_general
async def get_leaderboard(
    request: Request,
    timeframe: str = "weekly",  # weekly, monthly, all_time
    exercise_type: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user)
):
    """Get therapy game leaderboard."""
    
    try:
        # Validate timeframe
        valid_timeframes = ["weekly", "monthly", "all_time"]
        if timeframe not in valid_timeframes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}"
            )
        
        # Validate exercise type if provided
        if exercise_type:
            try:
                ExerciseType(exercise_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid exercise type: {exercise_type}"
                )
        
        # Get leaderboard data
        leaderboard = await TherapyGameService.get_leaderboard(
            timeframe=timeframe,
            exercise_type=exercise_type,
            limit=limit,
            requesting_user_id=current_user["user_id"]
        )
        
        return leaderboard
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get leaderboard", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve leaderboard"
        )


@router.get("/leaderboard/personal")
@rate_limit_general
async def get_personal_ranking(
    request: Request,
    timeframe: str = "weekly",
    current_user: dict = Depends(get_current_user)
):
    """Get user's personal ranking and nearby competitors."""
    
    try:
        ranking = await TherapyGameService.get_personal_ranking(
            user_id=current_user["user_id"],
            timeframe=timeframe
        )
        
        return ranking
        
    except Exception as e:
        logger.error("Failed to get personal ranking", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve personal ranking"
        )


@router.get("/levels")
@rate_limit_general
async def get_level_system(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get level system information and user's current level."""
    
    try:
        # Get user's current progress
        progress_data = await TherapyGameService.get_user_progress(current_user["user_id"])
        
        if "error" in progress_data:
            total_points = 0
        else:
            user_progress = progress_data.get("user_progress", {})
            total_points = user_progress.get("total_points", 0)
        
        # Calculate current level
        current_level = TherapyGameService._calculate_user_level(total_points)
        current_level_info = TherapyGameService.LEVEL_SYSTEM.get(current_level, {})
        
        # Get next level info
        next_level = current_level + 1
        next_level_info = TherapyGameService.LEVEL_SYSTEM.get(next_level, {})
        
        # Calculate progress to next level
        points_needed = 0
        progress_percentage = 100.0
        
        if next_level_info:
            points_needed = next_level_info["points_required"] - total_points
            current_level_points = current_level_info.get("points_required", 0)
            next_level_points = next_level_info["points_required"]
            
            if next_level_points > current_level_points:
                progress_in_level = total_points - current_level_points
                level_point_range = next_level_points - current_level_points
                progress_percentage = (progress_in_level / level_point_range) * 100
        
        return {
            "current_level": {
                "level": current_level,
                "name": current_level_info.get("name", "Unknown"),
                "color": current_level_info.get("color", "#8BC34A"),
                "multiplier": current_level_info.get("multiplier", 1.0),
                "points_required": current_level_info.get("points_required", 0)
            },
            "next_level": {
                "level": next_level,
                "name": next_level_info.get("name", "Max Level"),
                "color": next_level_info.get("color", "#FFD700"),
                "multiplier": next_level_info.get("multiplier", 2.0),
                "points_required": next_level_info.get("points_required", 0)
            } if next_level_info else None,
            "user_stats": {
                "total_points": total_points,
                "points_needed_for_next": max(0, points_needed),
                "progress_percentage": min(100.0, max(0.0, progress_percentage))
            },
            "all_levels": [
                {
                    "level": level,
                    "name": info["name"],
                    "points_required": info["points_required"],
                    "multiplier": info["multiplier"],
                    "color": info["color"],
                    "unlocked": total_points >= info["points_required"]
                }
                for level, info in TherapyGameService.LEVEL_SYSTEM.items()
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get level system", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve level system"
        )


@router.get("/streaks")
@rate_limit_general
async def get_user_streaks(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get user's exercise streaks and consistency data."""
    
    try:
        streaks = await TherapyGameService.get_user_streaks(current_user["user_id"])
        
        return streaks
        
    except Exception as e:
        logger.error("Failed to get user streaks", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user streaks"
        )


@router.get("/challenges")
@rate_limit_general
async def get_daily_challenges(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get daily and weekly challenges for the user."""
    
    try:
        challenges = await TherapyGameService.get_daily_challenges(current_user["user_id"])
        
        return challenges
        
    except Exception as e:
        logger.error("Failed to get challenges", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve challenges"
        )


@router.post("/challenges/{challenge_id}/complete")
@rate_limit_general
async def complete_challenge(
    request: Request,
    challenge_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Mark a challenge as completed."""
    
    try:
        result = await TherapyGameService.complete_challenge(
            user_id=current_user["user_id"],
            challenge_id=challenge_id
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to complete challenge", challenge_id=challenge_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete challenge"
        )


# =============================================================================
# PAIN DETECTION AND ADAPTATION ROUTES
# =============================================================================

@router.get("/pain-detection/config")
@rate_limit_general
async def get_pain_detection_config(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get pain detection configuration and thresholds."""
    
    try:
        config = {
            "facial_action_units": TherapyGameService.PAIN_DETECTION_CONFIG["facial_action_units"],
            "pain_thresholds": TherapyGameService.PAIN_DETECTION_CONFIG["pain_thresholds"],
            "adaptation_rules": TherapyGameService.PAIN_DETECTION_CONFIG["adaptation_rules"],
            "enabled": settings.enable_therapy_game
        }
        
        return config
        
    except Exception as e:
        logger.error("Failed to get pain detection config", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pain detection configuration"
        )


@router.get("/game-mechanics")
@rate_limit_general
async def get_game_mechanics(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get game mechanics and scoring system."""
    
    try:
        return {
            "game_mechanics": TherapyGameService.GAME_MECHANICS,
            "achievement_system": TherapyGameService.ACHIEVEMENT_SYSTEM,
            "level_system": TherapyGameService.LEVEL_SYSTEM,
            "scoring_explanation": {
                "base_points": "10 points per repetition",
                "accuracy_bonus": "Up to 50% bonus for perfect form",
                "streak_bonus": "10% bonus for consecutive days",
                "pain_reduction_bonus": "20 points for reducing pain",
                "perfect_session_bonus": "100 points for 100% accuracy",
                "level_multiplier": "20% bonus per level"
            }
        }
        
    except Exception as e:
        logger.error("Failed to get game mechanics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve game mechanics"
        )