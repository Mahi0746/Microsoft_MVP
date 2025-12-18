# HealthSync AI - AR Medical Scanner Routes
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Request, File, UploadFile
from pydantic import BaseModel, validator
import structlog

from config import settings
from api.middleware.auth import get_current_user, require_role
from api.middleware.rate_limit import rate_limit_image, rate_limit_general
from services.ar_scanner_service import ARMedicalScannerService
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ARScanRequest(BaseModel):
    scan_type: str
    image_data: str  # Base64 encoded image
    scan_metadata: Optional[Dict[str, Any]] = {}
    
    @validator('scan_type')
    def validate_scan_type(cls, v):
        allowed_types = [
            'skin_analysis', 'wound_assessment', 'rash_detection',
            'eye_examination', 'posture_analysis', 'vitals_estimation',
            'prescription_ocr', 'medical_report_ocr', 'pill_identification', 'medical_device_scan'
        ]
        if v not in allowed_types:
            raise ValueError(f'Scan type must be one of: {", ".join(allowed_types)}')
        return v
    
    @validator('image_data')
    def validate_image_data(cls, v):
        if not v or len(v) < 100:
            raise ValueError('Valid image data is required')
        return v


class ARScanResponse(BaseModel):
    scan_id: str
    user_id: str
    scan_type: str
    timestamp: datetime
    confidence_score: float
    medical_assessment: Dict[str, Any]
    ar_overlay: Dict[str, Any]
    recommendations: List[str]
    follow_up_required: bool


class ScanHistoryResponse(BaseModel):
    scan_id: str
    scan_type: str
    timestamp: datetime
    confidence_score: float
    urgency_level: str
    findings_count: int
    follow_up_required: bool


class ScanAnalyticsResponse(BaseModel):
    total_scans: int
    scan_types: Dict[str, int]
    average_confidence: float
    urgency_distribution: Dict[str, int]
    latest_scan: Optional[Dict[str, Any]]


# =============================================================================
# AR SCANNING ROUTES
# =============================================================================

@router.post("/scan", response_model=ARScanResponse)
@rate_limit_image
async def perform_ar_scan(
    request: Request,
    scan_request: ARScanRequest,
    current_user: dict = Depends(get_current_user)
):
    """Perform AR medical scan with AI analysis."""
    
    try:
        # Check if AR scanner is enabled
        if not settings.enable_ar_scanner:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AR scanner service is currently disabled"
            )
        
        # Process the AR scan
        scan_result = await ARMedicalScannerService.process_ar_scan(
            current_user["user_id"],
            scan_request.scan_type,
            scan_request.image_data,
            scan_request.scan_metadata
        )
        
        logger.info(
            "AR scan completed",
            user_id=current_user["user_id"],
            scan_type=scan_request.scan_type,
            scan_id=scan_result["scan_id"],
            confidence=scan_result["confidence_score"]
        )
        
        return ARScanResponse(
            scan_id=scan_result["scan_id"],
            user_id=scan_result["user_id"],
            scan_type=scan_result["scan_type"],
            timestamp=scan_result["timestamp"],
            confidence_score=scan_result["confidence_score"],
            medical_assessment=scan_result["medical_assessment"],
            ar_overlay=scan_result["ar_overlay"],
            recommendations=scan_result["recommendations"],
            follow_up_required=scan_result["follow_up_required"]
        )
        
    except ValueError as e:
        logger.warning("Invalid AR scan request", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("AR scan failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AR scan processing failed"
        )


@router.post("/scan/upload", response_model=ARScanResponse)
@rate_limit_image
async def upload_and_scan(
    request: Request,
    scan_type: str,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """Upload image file and perform AR scan."""
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
        
        # Check file size (max 10MB)
        if file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File size must be less than 10MB"
            )
        
        # Read and encode image
        image_bytes = await file.read()
        import base64
        image_data = base64.b64encode(image_bytes).decode()
        
        # Create scan request
        scan_request = ARScanRequest(
            scan_type=scan_type,
            image_data=image_data,
            scan_metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "file_size": file.size,
                "upload_method": "file_upload"
            }
        )
        
        # Process scan
        scan_result = await ARMedicalScannerService.process_ar_scan(
            current_user["user_id"],
            scan_request.scan_type,
            scan_request.image_data,
            scan_request.scan_metadata
        )
        
        logger.info(
            "File upload scan completed",
            user_id=current_user["user_id"],
            filename=file.filename,
            scan_type=scan_type,
            scan_id=scan_result["scan_id"]
        )
        
        return ARScanResponse(**scan_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File upload scan failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload and scan failed"
        )


# =============================================================================
# SCAN HISTORY AND ANALYTICS
# =============================================================================

@router.get("/history", response_model=List[ScanHistoryResponse])
@rate_limit_general
async def get_scan_history(
    request: Request,
    current_user: dict = Depends(get_current_user),
    scan_type: Optional[str] = None,
    limit: int = 20
):
    """Get user's AR scan history."""
    
    try:
        scans = await ARMedicalScannerService.get_scan_history(
            current_user["user_id"],
            scan_type,
            limit
        )
        
        # Convert to response format
        scan_history = []
        for scan in scans:
            scan_history.append(ScanHistoryResponse(
                scan_id=str(scan["_id"]),
                scan_type=scan["scan_type"],
                timestamp=scan["timestamp"],
                confidence_score=scan["confidence_score"],
                urgency_level=scan.get("medical_assessment", {}).get("urgency_level", "routine"),
                findings_count=len(scan.get("medical_assessment", {}).get("findings", [])),
                follow_up_required=scan.get("follow_up_required", False)
            ))
        
        return scan_history
        
    except Exception as e:
        logger.error("Failed to get scan history", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get scan history"
        )


@router.get("/analytics", response_model=ScanAnalyticsResponse)
@rate_limit_general
async def get_scan_analytics(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get analytics for user's AR scans."""
    
    try:
        analytics = await ARMedicalScannerService.get_scan_analytics(current_user["user_id"])
        
        if "error" in analytics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=analytics["error"]
            )
        
        return ScanAnalyticsResponse(**analytics)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get scan analytics", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get scan analytics"
        )


@router.get("/scan/{scan_id}")
@rate_limit_general
async def get_scan_details(
    request: Request,
    scan_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed information about a specific scan."""
    
    try:
        # Get scan from database
        scan = await DatabaseService.mongodb_find_one(
            "ar_scans",
            {"_id": scan_id, "user_id": current_user["user_id"]}
        )
        
        if not scan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan not found"
            )
        
        # Remove sensitive data
        scan.pop("ai_analysis", None)  # Remove raw AI analysis
        
        return scan
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get scan details", scan_id=scan_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get scan details"
        )


# =============================================================================
# SCAN COMPARISON AND TRACKING
# =============================================================================

@router.get("/compare/{scan_id1}/{scan_id2}")
@rate_limit_general
async def compare_scans(
    request: Request,
    scan_id1: str,
    scan_id2: str,
    current_user: dict = Depends(get_current_user)
):
    """Compare two scans for progress tracking."""
    
    try:
        # Get both scans
        scan1 = await DatabaseService.mongodb_find_one(
            "ar_scans",
            {"_id": scan_id1, "user_id": current_user["user_id"]}
        )
        
        scan2 = await DatabaseService.mongodb_find_one(
            "ar_scans",
            {"_id": scan_id2, "user_id": current_user["user_id"]}
        )
        
        if not scan1 or not scan2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="One or both scans not found"
            )
        
        # Ensure scans are of the same type
        if scan1["scan_type"] != scan2["scan_type"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot compare scans of different types"
            )
        
        # Generate comparison
        comparison = {
            "scan_type": scan1["scan_type"],
            "scan1": {
                "id": scan_id1,
                "timestamp": scan1["timestamp"],
                "confidence": scan1["confidence_score"],
                "findings": scan1.get("medical_assessment", {}).get("findings", [])
            },
            "scan2": {
                "id": scan_id2,
                "timestamp": scan2["timestamp"],
                "confidence": scan2["confidence_score"],
                "findings": scan2.get("medical_assessment", {}).get("findings", [])
            },
            "time_difference_days": (scan2["timestamp"] - scan1["timestamp"]).days,
            "progress_analysis": _analyze_scan_progress(scan1, scan2),
            "recommendations": _generate_progress_recommendations(scan1, scan2)
        }
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Scan comparison failed", scan_id1=scan_id1, scan_id2=scan_id2, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scan comparison failed"
        )


# =============================================================================
# SCAN TYPES AND CAPABILITIES
# =============================================================================

@router.get("/scan-types")
@rate_limit_general
async def get_scan_types(request: Request):
    """Get available AR scan types and their capabilities."""
    
    try:
        scan_types = []
        
        for scan_type, config in ARMedicalScannerService.SCAN_TYPES.items():
            scan_types.append({
                "type": scan_type,
                "name": scan_type.replace("_", " ").title(),
                "description": config["description"],
                "ai_model": config["ai_model"],
                "confidence_threshold": config["confidence_threshold"],
                "estimated_processing_time": config["processing_time"],
                "supported_conditions": _get_supported_conditions(scan_type)
            })
        
        return {
            "available_scan_types": scan_types,
            "total_types": len(scan_types),
            "service_status": "active" if settings.enable_ar_scanner else "disabled"
        }
        
    except Exception as e:
        logger.error("Failed to get scan types", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get scan types"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _analyze_scan_progress(scan1: Dict[str, Any], scan2: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze progress between two scans."""
    
    progress = {
        "confidence_change": scan2["confidence_score"] - scan1["confidence_score"],
        "findings_change": len(scan2.get("medical_assessment", {}).get("findings", [])) - 
                          len(scan1.get("medical_assessment", {}).get("findings", [])),
        "overall_trend": "stable"
    }
    
    # Determine overall trend
    if progress["confidence_change"] > 0.1 and progress["findings_change"] <= 0:
        progress["overall_trend"] = "improving"
    elif progress["confidence_change"] < -0.1 or progress["findings_change"] > 0:
        progress["overall_trend"] = "worsening"
    
    # Scan-specific progress analysis
    scan_type = scan1["scan_type"]
    
    if scan_type == "wound_assessment":
        # Analyze wound healing progress
        wound1 = scan1.get("ai_analysis", {}).get("wound_measurements", {})
        wound2 = scan2.get("ai_analysis", {}).get("wound_measurements", {})
        
        if wound1 and wound2:
            area_change = wound2.get("area_cm2", 0) - wound1.get("area_cm2", 0)
            progress["wound_area_change"] = area_change
            
            if area_change < -0.5:
                progress["healing_status"] = "good_healing"
            elif area_change > 0.5:
                progress["healing_status"] = "poor_healing"
            else:
                progress["healing_status"] = "stable"
    
    elif scan_type == "posture_analysis":
        # Analyze posture improvement
        score1 = scan1.get("ai_analysis", {}).get("overall_score", 5.0)
        score2 = scan2.get("ai_analysis", {}).get("overall_score", 5.0)
        
        progress["posture_score_change"] = score2 - score1
        
        if progress["posture_score_change"] > 1.0:
            progress["posture_trend"] = "significant_improvement"
        elif progress["posture_score_change"] > 0.5:
            progress["posture_trend"] = "improvement"
        elif progress["posture_score_change"] < -0.5:
            progress["posture_trend"] = "deterioration"
        else:
            progress["posture_trend"] = "stable"
    
    return progress


def _generate_progress_recommendations(scan1: Dict[str, Any], scan2: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on scan progress."""
    
    recommendations = []
    progress = _analyze_scan_progress(scan1, scan2)
    
    if progress["overall_trend"] == "improving":
        recommendations.append("Continue current treatment approach - showing positive progress")
    elif progress["overall_trend"] == "worsening":
        recommendations.append("Consider adjusting treatment plan - condition may be worsening")
        recommendations.append("Schedule follow-up with healthcare provider")
    else:
        recommendations.append("Maintain current monitoring schedule")
    
    # Scan-specific recommendations
    scan_type = scan1["scan_type"]
    
    if scan_type == "wound_assessment":
        healing_status = progress.get("healing_status", "stable")
        
        if healing_status == "poor_healing":
            recommendations.extend([
                "Wound healing appears slower than expected",
                "Consider professional wound care evaluation",
                "Ensure proper wound care hygiene"
            ])
        elif healing_status == "good_healing":
            recommendations.append("Wound healing progressing well - continue current care")
    
    elif scan_type == "posture_analysis":
        posture_trend = progress.get("posture_trend", "stable")
        
        if posture_trend == "improvement":
            recommendations.append("Posture improvements detected - continue exercises")
        elif posture_trend == "deterioration":
            recommendations.extend([
                "Posture appears to be worsening",
                "Review ergonomic setup and exercise routine",
                "Consider physical therapy consultation"
            ])
    
    return recommendations


def _get_supported_conditions(scan_type: str) -> List[str]:
    """Get list of conditions supported by scan type."""
    
    condition_map = {
        "skin_analysis": ["acne", "eczema", "psoriasis", "melanoma", "dermatitis", "rosacea"],
        "wound_assessment": ["wound_healing", "infection_detection", "tissue_analysis"],
        "rash_detection": ["contact_dermatitis", "allergic_reactions", "viral_rashes"],
        "eye_examination": ["conjunctivitis", "stye", "pterygium", "basic_screening"],
        "posture_analysis": ["forward_head", "rounded_shoulders", "scoliosis", "kyphosis"],
        "vitals_estimation": ["heart_rate", "respiratory_rate", "stress_indicators"]
    }
    
    return condition_map.get(scan_type, [])