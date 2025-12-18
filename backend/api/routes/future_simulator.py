# HealthSync AI - Future-You Simulator API Routes
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Request, File, UploadFile, Form
from pydantic import BaseModel, validator
import structlog

from config import settings
from api.middleware.auth import get_current_user
from api.middleware.rate_limit import rate_limit_general, rate_limit_image
from services.future_simulator_service import FutureSimulatorService
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ImageUploadResponse(BaseModel):
    success: bool
    image_id: Optional[str] = None
    file_path: Optional[str] = None
    signed_url: Optional[str] = None
    file_size: Optional[int] = None
    dimensions: Optional[str] = None
    error: Optional[str] = None


class AgeProgressionRequest(BaseModel):
    image_path: str
    target_age_years: int
    current_age: Optional[int] = None
    
    @validator('target_age_years')
    def validate_target_age(cls, v):
        if v < 5 or v > 50:
            raise ValueError('Target age must be between 5 and 50 years')
        return v


class HealthProjectionRequest(BaseModel):
    target_age_years: int
    lifestyle_scenario: str = "current"
    
    @validator('target_age_years')
    def validate_target_age(cls, v):
        if v < 5 or v > 50:
            raise ValueError('Target age must be between 5 and 50 years')
        return v
    
    @validator('lifestyle_scenario')
    def validate_lifestyle_scenario(cls, v):
        allowed_scenarios = ['improved', 'current', 'declined']
        if v not in allowed_scenarios:
            raise ValueError(f'Lifestyle scenario must be one of: {", ".join(allowed_scenarios)}')
        return v


class FutureSimulationRequest(BaseModel):
    target_age_years: int
    lifestyle_scenario: str = "current"
    current_age: Optional[int] = None
    
    @validator('target_age_years')
    def validate_target_age(cls, v):
        if v < 5 or v > 50:
            raise ValueError('Target age must be between 5 and 50 years')
        return v
    
    @validator('lifestyle_scenario')
    def validate_lifestyle_scenario(cls, v):
        allowed_scenarios = ['improved', 'current', 'declined']
        if v not in allowed_scenarios:
            raise ValueError(f'Lifestyle scenario must be one of: {", ".join(allowed_scenarios)}')
        return v


# Response Models
class AgeProgressionResponse(BaseModel):
    success: bool
    progression_id: Optional[str] = None
    original_image_url: Optional[str] = None
    aged_image_url: Optional[str] = None
    target_age_years: Optional[int] = None
    generation_prompt: Optional[str] = None
    error: Optional[str] = None


class HealthProjectionResponse(BaseModel):
    success: bool
    projection_id: Optional[str] = None
    target_age_years: Optional[int] = None
    lifestyle_scenario: Optional[str] = None
    life_expectancy: Optional[float] = None
    condition_projections: Optional[Dict[str, Any]] = None
    lifestyle_impact: Optional[Dict[str, Any]] = None
    health_narrative: Optional[Dict[str, str]] = None
    recommendations: Optional[List[str]] = None
    error: Optional[str] = None


class FutureSimulationResponse(BaseModel):
    success: bool
    simulation_id: Optional[str] = None
    age_progression: Optional[AgeProgressionResponse] = None
    health_projections: Optional[HealthProjectionResponse] = None
    combined_analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class SimulationHistoryResponse(BaseModel):
    progression_id: str
    health_projection_id: Optional[str]
    target_age_years: int
    lifestyle_scenario: Optional[str]
    life_expectancy: Optional[float]
    original_image_url: Optional[str]
    aged_image_url: Optional[str]
    created_at: datetime


# =============================================================================
# IMAGE UPLOAD & VALIDATION
# =============================================================================

@router.post("/upload-image", response_model=ImageUploadResponse)
@rate_limit_image
async def upload_future_simulation_image(
    request: Request,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload and validate user image for future simulation."""
    
    try:
        # Validate file type
        if not file.content_type or file.content_type not in settings.allowed_image_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: {', '.join(settings.allowed_image_types)}"
            )
        
        # Read file data
        file_data = await file.read()
        
        # Validate file size
        if len(file_data) > settings.max_file_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {settings.max_file_size_mb}MB"
            )
        
        # Upload and validate image
        upload_result = await FutureSimulatorService.upload_and_validate_image(
            image_data=file_data,
            user_id=current_user["user_id"],
            content_type=file.content_type
        )
        
        if upload_result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=upload_result["error"]
            )
        
        logger.info(
            "Image uploaded for future simulation",
            user_id=current_user["user_id"],
            image_id=upload_result.get("image_id"),
            file_size=upload_result.get("file_size")
        )
        
        return ImageUploadResponse(**upload_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image upload failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Image upload failed"
        )


# =============================================================================
# AGE PROGRESSION
# =============================================================================

@router.post("/age-progression", response_model=AgeProgressionResponse)
@rate_limit_image
async def generate_age_progression(
    request: Request,
    progression_request: AgeProgressionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate AI-powered age progression of user image."""
    
    try:
        # Verify image exists and belongs to user
        image_record = await DatabaseService.execute_query(
            """
            SELECT file_path, user_id 
            FROM user_images 
            WHERE file_path = $1 AND user_id = $2 AND purpose = 'future_simulation'
            """,
            progression_request.image_path,
            current_user["user_id"],
            fetch_one=True
        )
        
        if not image_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found or access denied"
            )
        
        # Generate age progression
        progression_result = await FutureSimulatorService.generate_age_progression(
            image_path=progression_request.image_path,
            target_age_years=progression_request.target_age_years,
            user_id=current_user["user_id"],
            current_age=progression_request.current_age
        )
        
        if progression_result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=progression_result["error"]
            )
        
        logger.info(
            "Age progression generated",
            user_id=current_user["user_id"],
            progression_id=progression_result.get("progression_id"),
            target_age=progression_request.target_age_years
        )
        
        return AgeProgressionResponse(**progression_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Age progression failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Age progression generation failed"
        )


# =============================================================================
# HEALTH PROJECTIONS
# =============================================================================

@router.post("/health-projections", response_model=HealthProjectionResponse)
@rate_limit_general
async def generate_health_projections(
    request: Request,
    projection_request: HealthProjectionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate comprehensive health projections for future age."""
    
    try:
        # Generate health projections
        projection_result = await FutureSimulatorService.generate_health_projections(
            user_id=current_user["user_id"],
            target_age_years=projection_request.target_age_years,
            lifestyle_scenario=projection_request.lifestyle_scenario
        )
        
        if projection_result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=projection_result["error"]
            )
        
        logger.info(
            "Health projections generated",
            user_id=current_user["user_id"],
            projection_id=projection_result.get("projection_id"),
            target_age=projection_request.target_age_years,
            scenario=projection_request.lifestyle_scenario
        )
        
        return HealthProjectionResponse(**projection_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Health projections failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health projections generation failed"
        )


# =============================================================================
# COMPLETE FUTURE SIMULATION
# =============================================================================

@router.post("/complete-simulation", response_model=FutureSimulationResponse)
@rate_limit_image
async def generate_complete_future_simulation(
    request: Request,
    simulation_request: FutureSimulationRequest,
    image_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Generate complete future simulation with age progression and health projections."""
    
    try:
        # Step 1: Upload and validate image
        file_data = await image_file.read()
        
        upload_result = await FutureSimulatorService.upload_and_validate_image(
            image_data=file_data,
            user_id=current_user["user_id"],
            content_type=image_file.content_type
        )
        
        if upload_result.get("error"):
            return FutureSimulationResponse(
                success=False,
                error=f"Image upload failed: {upload_result['error']}"
            )
        
        # Step 2: Generate age progression
        progression_result = await FutureSimulatorService.generate_age_progression(
            image_path=upload_result["file_path"],
            target_age_years=simulation_request.target_age_years,
            user_id=current_user["user_id"],
            current_age=simulation_request.current_age
        )
        
        # Step 3: Generate health projections
        projection_result = await FutureSimulatorService.generate_health_projections(
            user_id=current_user["user_id"],
            target_age_years=simulation_request.target_age_years,
            lifestyle_scenario=simulation_request.lifestyle_scenario
        )
        
        # Step 4: Combine results and create comprehensive analysis
        combined_analysis = await _create_combined_analysis(
            progression_result,
            projection_result,
            simulation_request
        )
        
        # Create simulation record
        simulation_record = await DatabaseService.execute_query(
            """
            INSERT INTO future_simulations (user_id, progression_id, projection_id, 
                                          target_age_years, lifestyle_scenario, created_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            RETURNING id
            """,
            current_user["user_id"],
            progression_result.get("progression_id"),
            projection_result.get("projection_id"),
            simulation_request.target_age_years,
            simulation_request.lifestyle_scenario,
            fetch_one=True
        )
        
        logger.info(
            "Complete future simulation generated",
            user_id=current_user["user_id"],
            simulation_id=simulation_record["id"],
            target_age=simulation_request.target_age_years
        )
        
        return FutureSimulationResponse(
            success=True,
            simulation_id=simulation_record["id"],
            age_progression=AgeProgressionResponse(**progression_result) if progression_result.get("success") else None,
            health_projections=HealthProjectionResponse(**projection_result) if projection_result.get("success") else None,
            combined_analysis=combined_analysis
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Complete simulation failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Complete future simulation failed"
        )


async def _create_combined_analysis(
    progression_result: Dict[str, Any],
    projection_result: Dict[str, Any],
    simulation_request: FutureSimulationRequest
) -> Dict[str, Any]:
    """Create combined analysis of age progression and health projections."""
    
    try:
        analysis = {
            "simulation_summary": {
                "target_age_years": simulation_request.target_age_years,
                "lifestyle_scenario": simulation_request.lifestyle_scenario,
                "has_age_progression": progression_result.get("success", False),
                "has_health_projections": projection_result.get("success", False)
            }
        }
        
        # Add visual health effects mapping
        if projection_result.get("success") and projection_result.get("condition_projections"):
            visual_effects = []
            
            for condition, data in projection_result["condition_projections"].items():
                if data.get("visual_effects") and data.get("probability", 0) > 0.4:
                    visual_effects.append({
                        "condition": condition,
                        "probability": data["probability"],
                        "effects": data["visual_effects"]
                    })
            
            analysis["visual_health_effects"] = visual_effects
        
        # Add lifestyle comparison
        if projection_result.get("success") and projection_result.get("lifestyle_impact"):
            lifestyle_impact = projection_result["lifestyle_impact"]
            
            analysis["lifestyle_comparison"] = {
                "current_scenario": simulation_request.lifestyle_scenario,
                "potential_benefits": lifestyle_impact.get("potential_benefits", {}),
                "key_recommendations": lifestyle_impact.get("recommendations", [])[:5]  # Top 5
            }
        
        # Add overall health score
        if projection_result.get("success"):
            life_expectancy = projection_result.get("life_expectancy", 78)
            condition_probs = projection_result.get("condition_projections", {})
            
            # Calculate overall health score (0-100)
            health_score = 100
            
            for condition, data in condition_probs.items():
                if condition in ["diabetes", "heart_disease", "cancer"]:
                    prob = data.get("probability", 0)
                    health_score -= prob * 20  # Reduce score based on risk
            
            health_score = max(0, min(100, health_score))
            
            analysis["health_score"] = {
                "overall_score": round(health_score, 1),
                "life_expectancy": life_expectancy,
                "score_interpretation": (
                    "Excellent" if health_score >= 80 else
                    "Good" if health_score >= 60 else
                    "Fair" if health_score >= 40 else
                    "Needs Improvement"
                )
            }
        
        return analysis
        
    except Exception as e:
        logger.error("Combined analysis creation failed", error=str(e))
        return {"error": "Failed to create combined analysis"}


# =============================================================================
# SIMULATION HISTORY
# =============================================================================

@router.get("/history", response_model=List[SimulationHistoryResponse])
@rate_limit_general
async def get_simulation_history(
    request: Request,
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get user's future simulation history."""
    
    try:
        simulations = await FutureSimulatorService.get_user_simulations(
            user_id=current_user["user_id"],
            limit=limit
        )
        
        return [SimulationHistoryResponse(**sim) for sim in simulations]
        
    except Exception as e:
        logger.error("Failed to get simulation history", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get simulation history"
        )


@router.get("/simulation/{simulation_id}")
@rate_limit_general
async def get_simulation_details(
    request: Request,
    simulation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific simulation."""
    
    try:
        # Get simulation record
        simulation = await DatabaseService.execute_query(
            """
            SELECT fs.*, ap.original_image_path, ap.aged_image_path, ap.target_age_years as prog_age,
                   hp.life_expectancy, hp.lifestyle_scenario, hp.projections_data
            FROM future_simulations fs
            LEFT JOIN age_progressions ap ON fs.progression_id = ap.id
            LEFT JOIN health_projections hp ON fs.projection_id = hp.id
            WHERE fs.id = $1 AND fs.user_id = $2
            """,
            simulation_id,
            current_user["user_id"],
            fetch_one=True
        )
        
        if not simulation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Simulation not found"
            )
        
        # Generate signed URLs for images
        original_image_url = None
        aged_image_url = None
        
        if simulation["original_image_path"]:
            # For now, use direct file path (update to use MongoDB GridFS or cloud storage)
            original_image_url = f"/uploads/{simulation['original_image_path']}" if simulation.get("original_image_path") else None
        
        if simulation.get("aged_image_path"):
            # For now, use direct file path (update to use MongoDB GridFS or cloud storage)
            aged_image_url = f"/uploads/{simulation['aged_image_path']}"
        
        # Parse projections data
        projections_data = {}
        if simulation["projections_data"]:
            try:
                projections_data = json.loads(simulation["projections_data"])
            except json.JSONDecodeError:
                pass
        
        return {
            "simulation_id": simulation["id"],
            "target_age_years": simulation["target_age_years"],
            "lifestyle_scenario": simulation["lifestyle_scenario"],
            "life_expectancy": float(simulation["life_expectancy"]) if simulation["life_expectancy"] else None,
            "original_image_url": original_image_url,
            "aged_image_url": aged_image_url,
            "projections_data": projections_data,
            "created_at": simulation["created_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get simulation details", simulation_id=simulation_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get simulation details"
        )


# =============================================================================
# COMPARISON & ANALYTICS
# =============================================================================

@router.post("/compare-scenarios")
@rate_limit_general
async def compare_lifestyle_scenarios(
    request: Request,
    target_age_years: int,
    current_user: dict = Depends(get_current_user)
):
    """Compare health projections across different lifestyle scenarios."""
    
    try:
        scenarios = ["improved", "current", "declined"]
        comparisons = {}
        
        for scenario in scenarios:
            projection_result = await FutureSimulatorService.generate_health_projections(
                user_id=current_user["user_id"],
                target_age_years=target_age_years,
                lifestyle_scenario=scenario
            )
            
            if projection_result.get("success"):
                comparisons[scenario] = {
                    "life_expectancy": projection_result.get("life_expectancy"),
                    "condition_projections": projection_result.get("condition_projections", {}),
                    "recommendations": projection_result.get("recommendations", [])
                }
        
        # Calculate differences
        scenario_differences = {}
        if "current" in comparisons and "improved" in comparisons:
            current_life = comparisons["current"]["life_expectancy"]
            improved_life = comparisons["improved"]["life_expectancy"]
            
            scenario_differences["improvement_benefit"] = {
                "life_expectancy_gain": round(improved_life - current_life, 1) if current_life and improved_life else 0,
                "risk_reductions": {}
            }
            
            # Calculate risk reductions
            for condition in ["diabetes", "heart_disease", "cancer"]:
                current_risk = comparisons["current"]["condition_projections"].get(condition, {}).get("probability", 0)
                improved_risk = comparisons["improved"]["condition_projections"].get(condition, {}).get("probability", 0)
                
                scenario_differences["improvement_benefit"]["risk_reductions"][condition] = {
                    "current_risk": round(current_risk, 3),
                    "improved_risk": round(improved_risk, 3),
                    "risk_reduction": round(current_risk - improved_risk, 3)
                }
        
        logger.info(
            "Lifestyle scenarios compared",
            user_id=current_user["user_id"],
            target_age=target_age_years,
            scenarios_compared=len(comparisons)
        )
        
        return {
            "target_age_years": target_age_years,
            "scenario_comparisons": comparisons,
            "scenario_differences": scenario_differences,
            "recommendation": "Choose the 'improved' lifestyle scenario for the best health outcomes"
        }
        
    except Exception as e:
        logger.error("Scenario comparison failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lifestyle scenario comparison failed"
        )


# =============================================================================
# HEALTH INSIGHTS
# =============================================================================

@router.get("/health-insights")
@rate_limit_general
async def get_personalized_health_insights(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get personalized health insights based on user's simulation history."""
    
    try:
        # Get user's recent simulations
        recent_simulations = await FutureSimulatorService.get_user_simulations(
            user_id=current_user["user_id"],
            limit=5
        )
        
        if not recent_simulations:
            return {
                "message": "No simulations found. Create your first future simulation to get personalized insights!",
                "insights": []
            }
        
        # Analyze patterns in simulations
        insights = []
        
        # Life expectancy trend
        life_expectancies = [sim["life_expectancy"] for sim in recent_simulations if sim["life_expectancy"]]
        if len(life_expectancies) >= 2:
            trend = "improving" if life_expectancies[0] > life_expectancies[-1] else "declining"
            insights.append({
                "type": "life_expectancy_trend",
                "title": f"Life Expectancy Trend: {trend.title()}",
                "description": f"Your projected life expectancy is {trend} based on recent simulations.",
                "action": "Continue healthy habits" if trend == "improving" else "Consider lifestyle improvements"
            })
        
        # Most common target age
        target_ages = [sim["target_age_years"] for sim in recent_simulations]
        if target_ages:
            most_common_age = max(set(target_ages), key=target_ages.count)
            insights.append({
                "type": "planning_horizon",
                "title": f"Planning Focus: {most_common_age} Years Ahead",
                "description": f"You're most interested in your health {most_common_age} years from now.",
                "action": "Consider both short-term and long-term health goals"
            })
        
        # Lifestyle scenario preferences
        scenarios = [sim["lifestyle_scenario"] for sim in recent_simulations if sim["lifestyle_scenario"]]
        if scenarios:
            most_common_scenario = max(set(scenarios), key=scenarios.count)
            insights.append({
                "type": "lifestyle_focus",
                "title": f"Lifestyle Interest: {most_common_scenario.title()} Scenario",
                "description": f"You frequently explore the '{most_common_scenario}' lifestyle scenario.",
                "action": "Try comparing different scenarios to see potential benefits"
            })
        
        return {
            "total_simulations": len(recent_simulations),
            "insights": insights,
            "recommendations": [
                "Regular simulations help track your health trajectory",
                "Compare different lifestyle scenarios to make informed decisions",
                "Use insights to set realistic health goals"
            ]
        }
        
    except Exception as e:
        logger.error("Health insights generation failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate health insights"
        )