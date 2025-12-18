# HealthSync AI - Health Management Routes
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from pydantic import BaseModel, validator
import structlog

from config import settings
from api.middleware.auth import get_current_user, require_role
from api.middleware.rate_limit import rate_limit_ml, rate_limit_general
from services.db_service import DatabaseService
from services.ai_service import AIService
from services.ml_service import MLModelService
from services.family_graph_service import FamilyGraphService


logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class HealthMetricCreate(BaseModel):
    metric_type: str
    value: float
    unit: str
    measured_at: Optional[datetime] = None
    source: str = "manual"
    notes: Optional[str] = None
    
    @validator('metric_type')
    def validate_metric_type(cls, v):
        allowed_types = [
            'blood_pressure', 'heart_rate', 'weight', 'bmi', 
            'blood_sugar', 'temperature', 'oxygen_saturation'
        ]
        if v not in allowed_types:
            raise ValueError(f'Metric type must be one of: {", ".join(allowed_types)}')
        return v
    
    @validator('value')
    def validate_value(cls, v):
        if v < 0 or v > 1000:
            raise ValueError('Value must be between 0 and 1000')
        return v


class SymptomCreate(BaseModel):
    description: str
    severity: str = "medium"
    duration_hours: Optional[int] = None
    location_on_body: Optional[str] = None
    triggers: Optional[List[str]] = None
    
    @validator('severity')
    def validate_severity(cls, v):
        allowed_severities = ['low', 'medium', 'high', 'critical']
        if v not in allowed_severities:
            raise ValueError(f'Severity must be one of: {", ".join(allowed_severities)}')
        return v
    
    @validator('description')
    def validate_description(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Description must be at least 5 characters')
        return v.strip()


class FamilyMemberCreate(BaseModel):
    relation: str
    name: Optional[str] = None
    birth_year: Optional[int] = None
    death_year: Optional[int] = None
    health_conditions: List[str] = []
    age_of_onset: Optional[Dict[str, int]] = None
    
    @validator('relation')
    def validate_relation(cls, v):
        allowed_relations = [
            'father', 'mother', 'brother', 'sister', 'son', 'daughter',
            'grandfather_paternal', 'grandmother_paternal', 
            'grandfather_maternal', 'grandmother_maternal',
            'uncle_paternal', 'aunt_paternal', 'uncle_maternal', 'aunt_maternal',
            'cousin', 'spouse'
        ]
        if v not in allowed_relations:
            raise ValueError(f'Relation must be one of: {", ".join(allowed_relations)}')
        return v
    
    @validator('birth_year')
    def validate_birth_year(cls, v):
        if v and (v < 1900 or v > datetime.now().year):
            raise ValueError('Birth year must be between 1900 and current year')
        return v


class PredictionRequest(BaseModel):
    health_data: Dict[str, Any]
    include_family_history: bool = True
    
    @validator('health_data')
    def validate_health_data(cls, v):
        required_fields = ['age', 'gender']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Health data must include: {", ".join(required_fields)}')
        return v


# Response Models
class HealthMetricResponse(BaseModel):
    id: str
    metric_type: str
    value: float
    unit: str
    measured_at: datetime
    source: str
    notes: Optional[str]
    created_at: datetime


class SymptomResponse(BaseModel):
    id: str
    description: str
    severity: str
    duration_hours: Optional[int]
    location_on_body: Optional[str]
    triggers: Optional[List[str]]
    ai_assessment: Optional[Dict[str, Any]]
    timestamp: datetime
    created_at: datetime


class PredictionResponse(BaseModel):
    id: str
    disease: str
    probability: float
    confidence_score: float
    risk_factors: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime
    expires_at: datetime


class HealthSummaryResponse(BaseModel):
    user_id: str
    health_metrics: List[HealthMetricResponse]
    recent_symptoms: List[SymptomResponse]
    current_predictions: List[PredictionResponse]
    family_risk_factors: Dict[str, float]
    last_updated: datetime


# =============================================================================
# HEALTH METRICS ROUTES
# =============================================================================

@router.post("/metrics", response_model=HealthMetricResponse)
@rate_limit_general
async def create_health_metric(
    request: Request,
    metric_data: HealthMetricCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a new health metric measurement."""
    
    try:
        # Set measured_at to now if not provided
        if not metric_data.measured_at:
            metric_data.measured_at = datetime.utcnow()
        
        # Insert health metric
        metric_id = await DatabaseService.execute_query(
            """
            INSERT INTO health_metrics (user_id, metric_type, value, unit, measured_at, source, notes)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            """,
            current_user["user_id"],
            metric_data.metric_type,
            metric_data.value,
            metric_data.unit,
            metric_data.measured_at,
            metric_data.source,
            metric_data.notes,
            fetch_one=True
        )
        
        # Get the created metric
        metric = await DatabaseService.execute_query(
            """
            SELECT id, metric_type, value, unit, measured_at, source, notes, created_at
            FROM health_metrics 
            WHERE id = $1
            """,
            metric_id["id"],
            fetch_one=True
        )
        
        # Store health event in MongoDB
        await DatabaseService.store_health_event(
            current_user["user_id"],
            "health_metric",
            {
                "metric_type": metric_data.metric_type,
                "value": metric_data.value,
                "unit": metric_data.unit,
                "source": metric_data.source
            }
        )
        
        logger.info(
            "Health metric created",
            user_id=current_user["user_id"],
            metric_type=metric_data.metric_type,
            value=metric_data.value
        )
        
        return HealthMetricResponse(**dict(metric))
        
    except Exception as e:
        logger.error("Failed to create health metric", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create health metric"
        )


@router.get("/metrics", response_model=List[HealthMetricResponse])
@rate_limit_general
async def get_health_metrics(
    request: Request,
    current_user: dict = Depends(get_current_user),
    metric_type: Optional[str] = Query(None),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """Get user's health metrics with optional filtering."""
    
    try:
        # Build query with optional filtering
        where_clause = "WHERE user_id = $1"
        params = [current_user["user_id"]]
        
        if metric_type:
            where_clause += " AND metric_type = $2"
            params.append(metric_type)
        
        query = f"""
            SELECT id, metric_type, value, unit, measured_at, source, notes, created_at
            FROM health_metrics 
            {where_clause}
            ORDER BY measured_at DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """
        params.extend([limit, offset])
        
        metrics = await DatabaseService.execute_query(query, *params, fetch_all=True)
        
        return [HealthMetricResponse(**dict(metric)) for metric in metrics]
        
    except Exception as e:
        logger.error("Failed to get health metrics", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get health metrics"
        )


# =============================================================================
# SYMPTOMS ROUTES
# =============================================================================

@router.post("/symptoms", response_model=SymptomResponse)
@rate_limit_general
async def create_symptom(
    request: Request,
    symptom_data: SymptomCreate,
    current_user: dict = Depends(get_current_user)
):
    """Report a new symptom."""
    
    try:
        # Insert symptom
        symptom_id = await DatabaseService.execute_query(
            """
            INSERT INTO symptoms (user_id, description, severity, duration_hours, location_on_body, triggers)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            current_user["user_id"],
            symptom_data.description,
            symptom_data.severity,
            symptom_data.duration_hours,
            symptom_data.location_on_body,
            symptom_data.triggers,
            fetch_one=True
        )
        
        # Get the created symptom
        symptom = await DatabaseService.execute_query(
            """
            SELECT id, description, severity, duration_hours, location_on_body, 
                   triggers, ai_assessment, timestamp, created_at
            FROM symptoms 
            WHERE id = $1
            """,
            symptom_id["id"],
            fetch_one=True
        )
        
        # Store health event in MongoDB
        await DatabaseService.store_health_event(
            current_user["user_id"],
            "symptom_report",
            {
                "description": symptom_data.description,
                "severity": symptom_data.severity,
                "duration_hours": symptom_data.duration_hours,
                "location": symptom_data.location_on_body
            }
        )
        
        logger.info(
            "Symptom reported",
            user_id=current_user["user_id"],
            severity=symptom_data.severity,
            description=symptom_data.description[:50]
        )
        
        return SymptomResponse(**dict(symptom))
        
    except Exception as e:
        logger.error("Failed to create symptom", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create symptom"
        )


@router.get("/symptoms", response_model=List[SymptomResponse])
@rate_limit_general
async def get_symptoms(
    request: Request,
    current_user: dict = Depends(get_current_user),
    severity: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
    offset: int = Query(0, ge=0)
):
    """Get user's symptoms with optional filtering."""
    
    try:
        # Build query with optional filtering
        where_clause = "WHERE user_id = $1"
        params = [current_user["user_id"]]
        
        if severity:
            where_clause += " AND severity = $2"
            params.append(severity)
        
        query = f"""
            SELECT id, description, severity, duration_hours, location_on_body,
                   triggers, ai_assessment, timestamp, created_at
            FROM symptoms 
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """
        params.extend([limit, offset])
        
        symptoms = await DatabaseService.execute_query(query, *params, fetch_all=True)
        
        return [SymptomResponse(**dict(symptom)) for symptom in symptoms]
        
    except Exception as e:
        logger.error("Failed to get symptoms", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get symptoms"
        )


# =============================================================================
# PREDICTIONS ROUTES
# =============================================================================

@router.get("/predictions", response_model=List[PredictionResponse])
@rate_limit_general
async def get_health_predictions(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get current health predictions for user."""
    
    try:
        predictions = await DatabaseService.execute_query(
            """
            SELECT id, disease, probability, confidence_score, risk_factors, 
                   recommendations, created_at, expires_at
            FROM predictions 
            WHERE user_id = $1 AND expires_at > NOW()
            ORDER BY probability DESC
            """,
            current_user["user_id"],
            fetch_all=True
        )
        
        return [PredictionResponse(**dict(pred)) for pred in predictions]
        
    except Exception as e:
        logger.error("Failed to get predictions", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get predictions"
        )


@router.post("/predictions", response_model=List[PredictionResponse])
@rate_limit_ml
async def generate_health_predictions(
    request: Request,
    prediction_request: PredictionRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate new health predictions using ML models."""
    
    try:
        # Get user's health data
        health_summary = await DatabaseService.get_user_health_summary(current_user["user_id"])
        
        # Combine with provided health data
        combined_data = {**health_summary, **prediction_request.health_data}
        
        # Get family history if requested
        if prediction_request.include_family_history:
            family_graph = await DatabaseService.mongodb_find_one(
                "family_graph",
                {"user_id": current_user["user_id"]}
            )
            if family_graph:
                combined_data["family_history"] = family_graph.get("inherited_risks", {})
        
        # Generate predictions using ML service
        predictions = await MLModelService.predict_disease_risks(
            current_user["user_id"],
            combined_data,
            prediction_request.include_family_history
        )
        
        # Store predictions in database (already handled by ML service)
        stored_predictions = []
        
        for disease, prediction_data in predictions.items():
            # Get stored prediction from database
            prediction = await DatabaseService.execute_query(
                """
                SELECT id, disease, probability, confidence_score, risk_factors,
                       recommendations, created_at, expires_at
                FROM predictions 
                WHERE user_id = $1 AND disease = $2 AND expires_at > NOW()
                ORDER BY created_at DESC
                LIMIT 1
                """,
                current_user["user_id"],
                disease,
                fetch_one=True
            )
            
            if prediction:
                stored_predictions.append(PredictionResponse(**dict(prediction)))
        
        # Store prediction event in MongoDB
        await DatabaseService.store_health_event(
            current_user["user_id"],
            "prediction_update",
            {
                "diseases_predicted": list(predictions.keys()),
                "prediction_count": len(predictions),
                "include_family_history": prediction_request.include_family_history
            }
        )
        
        logger.info(
            "Health predictions generated",
            user_id=current_user["user_id"],
            prediction_count=len(predictions)
        )
        
        return stored_predictions
        
    except Exception as e:
        logger.error("Failed to generate predictions", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate predictions"
        )


# =============================================================================
# FAMILY HEALTH ROUTES
# =============================================================================

@router.get("/family-graph")
@rate_limit_general
async def get_family_health_graph(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get user's family health graph."""
    
    try:
        # Use family graph service to get or create graph
        family_graph = await FamilyGraphService.create_family_graph(current_user["user_id"])
        
        return family_graph
        
    except Exception as e:
        logger.error("Failed to get family graph", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get family health graph"
        )


@router.post("/family-member")
@rate_limit_general
async def add_family_member(
    request: Request,
    member_data: FamilyMemberCreate,
    current_user: dict = Depends(get_current_user)
):
    """Add a family member to the health graph."""
    
    try:
        # Use family graph service to add member
        result = await FamilyGraphService.add_family_member(
            current_user["user_id"],
            member_data.dict()
        )
        
        logger.info(
            "Family member added",
            user_id=current_user["user_id"],
            relation=member_data.relation,
            conditions=len(member_data.health_conditions)
        )
        
        return result
        
    except Exception as e:
        logger.error("Failed to add family member", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add family member"
        )


# =============================================================================
# ADVANCED ML ROUTES
# =============================================================================

@router.get("/family-insights")
@rate_limit_general
async def get_family_health_insights(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive family health insights and genetic risk analysis."""
    
    try:
        insights = await FamilyGraphService.get_family_health_insights(current_user["user_id"])
        
        logger.info("Family insights generated", user_id=current_user["user_id"])
        
        return insights
        
    except Exception as e:
        logger.error("Failed to get family insights", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get family health insights"
        )


@router.post("/retrain-model/{disease}")
@rate_limit_ml
async def retrain_disease_model(
    request: Request,
    disease: str,
    current_user: dict = Depends(require_role("admin"))
):
    """Retrain ML model for specific disease (admin only)."""
    
    try:
        result = await MLModelService.retrain_model(disease)
        
        logger.info("Model retrained", disease=disease, admin_user=current_user["user_id"])
        
        return result
        
    except Exception as e:
        logger.error("Failed to retrain model", disease=disease, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrain model"
        )


@router.get("/model-performance/{disease}")
@rate_limit_general
async def get_model_performance(
    request: Request,
    disease: str,
    current_user: dict = Depends(require_role("doctor"))
):
    """Get ML model performance metrics (doctors only)."""
    
    try:
        performance = await MLModelService.get_model_performance(disease)
        
        return performance
        
    except Exception as e:
        logger.error("Failed to get model performance", disease=disease, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get model performance"
        )


@router.get("/health-twin")
@rate_limit_ml
async def get_health_twin_data(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive health twin data with ML predictions."""
    
    try:
        # Get user's complete health profile
        health_summary = await DatabaseService.get_user_health_summary(current_user["user_id"])
        
        # Get family graph
        family_graph = await FamilyGraphService.create_family_graph(current_user["user_id"])
        
        # Generate ML predictions
        predictions = await MLModelService.predict_disease_risks(
            current_user["user_id"],
            health_summary,
            include_family_history=True
        )
        
        # Get family insights
        family_insights = await FamilyGraphService.get_family_health_insights(current_user["user_id"])
        
        health_twin = {
            "user_id": current_user["user_id"],
            "health_profile": health_summary,
            "disease_predictions": predictions,
            "family_genetics": {
                "inherited_risks": family_graph.get("inherited_risks", {}),
                "family_patterns": family_insights.get("family_patterns", {}),
                "genetic_insights": family_insights.get("genetic_insights", [])
            },
            "risk_timeline": family_insights.get("risk_timeline", {}),
            "personalized_recommendations": family_insights.get("recommendations", []),
            "comparative_analysis": family_insights.get("comparative_analysis", {}),
            "last_updated": datetime.utcnow(),
            "confidence_score": 0.85
        }
        
        logger.info("Health twin data generated", user_id=current_user["user_id"])
        
        return health_twin
        
    except Exception as e:
        logger.error("Failed to get health twin data", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get health twin data"
        )


# =============================================================================
# HEALTH SUMMARY ROUTE
# =============================================================================

@router.get("/summary", response_model=HealthSummaryResponse)
@rate_limit_general
async def get_health_summary(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive health summary for user."""
    
    try:
        # Get health summary from cache or database
        summary_data = await DatabaseService.get_user_health_summary(current_user["user_id"])
        
        # Convert to response format
        health_metrics = [
            HealthMetricResponse(**metric) for metric in summary_data.get("health_metrics", [])
        ]
        
        recent_symptoms = await DatabaseService.execute_query(
            """
            SELECT id, description, severity, duration_hours, location_on_body,
                   triggers, ai_assessment, timestamp, created_at
            FROM symptoms 
            WHERE user_id = $1
            ORDER BY timestamp DESC
            LIMIT 10
            """,
            current_user["user_id"],
            fetch_all=True
        )
        
        current_predictions = [
            PredictionResponse(**pred) for pred in summary_data.get("predictions", [])
        ]
        
        # Get family risk factors
        family_graph = summary_data.get("family_graph", {})
        family_risk_factors = family_graph.get("inherited_risks", {})
        
        return HealthSummaryResponse(
            user_id=current_user["user_id"],
            health_metrics=health_metrics[:10],  # Last 10 metrics
            recent_symptoms=[SymptomResponse(**dict(s)) for s in recent_symptoms],
            current_predictions=current_predictions,
            family_risk_factors=family_risk_factors,
            last_updated=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error("Failed to get health summary", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get health summary"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _calculate_inherited_risks(family_members: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate inherited disease risks based on family history."""
    
    risks = {}
    condition_counts = {}
    
    # Count conditions by family member type
    for member in family_members:
        relation = member.get("relation", "")
        conditions = member.get("health_conditions", [])
        
        # Weight by relationship closeness
        weight = 1.0
        if relation in ["father", "mother"]:
            weight = 0.5  # 50% genetic similarity
        elif relation in ["brother", "sister"]:
            weight = 0.5  # 50% genetic similarity
        elif relation in ["grandfather_paternal", "grandmother_paternal", 
                         "grandfather_maternal", "grandmother_maternal"]:
            weight = 0.25  # 25% genetic similarity
        elif relation in ["uncle_paternal", "aunt_paternal", 
                         "uncle_maternal", "aunt_maternal"]:
            weight = 0.125  # 12.5% genetic similarity
        else:
            weight = 0.1  # Other relations
        
        for condition in conditions:
            if condition not in condition_counts:
                condition_counts[condition] = 0
            condition_counts[condition] += weight
    
    # Convert counts to risk probabilities
    for condition, count in condition_counts.items():
        # Simple risk calculation (would be more sophisticated in production)
        base_risk = 0.1  # 10% base population risk
        genetic_factor = min(count, 1.0)  # Cap at 100%
        risks[condition] = min(base_risk + (genetic_factor * 0.4), 0.8)  # Max 80% risk
    
    return risks