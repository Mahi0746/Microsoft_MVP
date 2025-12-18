# HealthSync AI - Doctor Marketplace Routes
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends, Request, Query
from pydantic import BaseModel, validator
import structlog

from config import settings
from api.middleware.auth import get_current_user, require_doctor, require_role
from api.middleware.rate_limit import rate_limit_general
from services.db_service import DatabaseService
from services.ai_service import AIService
from services.marketplace_service import MarketplaceService
from api.routes.websocket import notify_new_appointment_bid, notify_appointment_accepted


logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class DoctorSearchRequest(BaseModel):
    specialization: Optional[str] = None
    location: Optional[str] = None
    max_price: Optional[float] = None
    min_rating: Optional[float] = None
    availability_date: Optional[str] = None
    
    @validator('min_rating')
    def validate_rating(cls, v):
        if v is not None and (v < 0 or v > 5):
            raise ValueError('Rating must be between 0 and 5')
        return v


class AppointmentRequest(BaseModel):
    symptoms_summary: str
    urgency_level: str = "medium"
    preferred_date: Optional[datetime] = None
    consultation_type: str = "video"
    max_budget: Optional[float] = None
    
    @validator('symptoms_summary')
    def validate_symptoms(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Symptoms summary must be at least 10 characters')
        return v.strip()
    
    @validator('urgency_level')
    def validate_urgency(cls, v):
        allowed_levels = ['low', 'medium', 'high', 'critical']
        if v not in allowed_levels:
            raise ValueError(f'Urgency level must be one of: {", ".join(allowed_levels)}')
        return v
    
    @validator('consultation_type')
    def validate_consultation_type(cls, v):
        allowed_types = ['video', 'audio', 'chat', 'in_person']
        if v not in allowed_types:
            raise ValueError(f'Consultation type must be one of: {", ".join(allowed_types)}')
        return v


class AppointmentBid(BaseModel):
    appointment_id: str
    bid_amount: float
    estimated_duration: int = 30
    available_slots: List[str] = []
    message: Optional[str] = None
    
    @validator('bid_amount')
    def validate_bid_amount(cls, v):
        if v <= 0 or v > 1000:
            raise ValueError('Bid amount must be between 0 and 1000')
        return v
    
    @validator('estimated_duration')
    def validate_duration(cls, v):
        if v < 15 or v > 180:
            raise ValueError('Duration must be between 15 and 180 minutes')
        return v


class AppointmentReview(BaseModel):
    appointment_id: str
    rating: int
    review_text: Optional[str] = None
    is_anonymous: bool = False
    
    @validator('rating')
    def validate_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('Rating must be between 1 and 5')
        return v


# Response Models
class DoctorProfile(BaseModel):
    id: str
    user_id: str
    first_name: str
    last_name: str
    specialization: str
    sub_specializations: List[str]
    years_experience: int
    rating: float
    total_reviews: int
    base_consultation_fee: float
    bio: Optional[str]
    languages: List[str]
    is_verified: bool
    is_accepting_patients: bool
    # Enhanced matching fields
    match_confidence: Optional[float] = None
    match_specialty: Optional[str] = None
    match_reasoning: Optional[str] = None
    consultation_available: Optional[bool] = None


class AppointmentResponse(BaseModel):
    id: str
    patient_id: str
    doctor_id: Optional[str]
    status: str
    symptoms_summary: str
    urgency_level: str
    preferred_date: Optional[datetime]
    consultation_type: str
    final_fee: Optional[float]
    scheduled_at: Optional[datetime]
    created_at: datetime


class BidResponse(BaseModel):
    id: str
    appointment_id: str
    doctor_id: str
    doctor_name: str
    specialization: str
    bid_amount: float
    estimated_duration: int
    available_slots: List[str]
    message: Optional[str]
    is_selected: bool
    created_at: datetime


# =============================================================================
# DOCTOR SEARCH & DISCOVERY
# =============================================================================

@router.get("/search", response_model=List[DoctorProfile])
@rate_limit_general
async def search_doctors(
    request: Request,
    specialization: Optional[str] = Query(None),
    location: Optional[str] = Query(None),
    max_price: Optional[float] = Query(None),
    min_rating: Optional[float] = Query(None),
    limit: int = Query(20, le=50),
    offset: int = Query(0, ge=0),
    current_user: dict = Depends(get_current_user)
):
    """Search for doctors based on criteria."""
    
    try:
        # Build dynamic query
        where_conditions = ["d.is_verified = true", "d.is_accepting_patients = true", "u.is_active = true"]
        params = []
        param_count = 0
        
        if specialization:
            param_count += 1
            where_conditions.append(f"d.specialization ILIKE ${param_count}")
            params.append(f"%{specialization}%")
        
        if max_price:
            param_count += 1
            where_conditions.append(f"d.base_consultation_fee <= ${param_count}")
            params.append(max_price)
        
        if min_rating:
            param_count += 1
            where_conditions.append(f"d.rating >= ${param_count}")
            params.append(min_rating)
        
        # Add limit and offset
        param_count += 1
        limit_param = param_count
        param_count += 1
        offset_param = param_count
        params.extend([limit, offset])
        
        query = f"""
            SELECT d.id, d.user_id, u.first_name, u.last_name, d.specialization,
                   d.sub_specializations, d.years_experience, d.rating, d.total_reviews,
                   d.base_consultation_fee, d.bio, d.languages, d.is_verified,
                   d.is_accepting_patients
            FROM doctors d
            JOIN users u ON d.user_id = u.id
            WHERE {' AND '.join(where_conditions)}
            ORDER BY d.rating DESC, d.total_reviews DESC
            LIMIT ${limit_param} OFFSET ${offset_param}
        """
        
        doctors = await DatabaseService.execute_query(query, *params, fetch_all=True)
        
        return [DoctorProfile(**dict(doctor)) for doctor in doctors]
        
    except Exception as e:
        logger.error("Doctor search failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Doctor search failed"
        )


@router.post("/match-specialists")
@rate_limit_general
async def match_specialists_for_symptoms(
    request: Request,
    appointment_request: AppointmentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Enhanced AI-powered specialist matching with comprehensive analysis."""
    
    try:
        # Get patient demographics for better matching
        patient_info = await DatabaseService.execute_query(
            "SELECT date_of_birth, gender FROM users WHERE id = $1",
            current_user["user_id"],
            fetch_one=True
        )
        
        patient_age = None
        patient_gender = None
        
        if patient_info:
            if patient_info["date_of_birth"]:
                today = datetime.now().date()
                birth_date = patient_info["date_of_birth"]
                patient_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            patient_gender = patient_info["gender"]
        
        # Enhanced symptom analysis with MarketplaceService
        specialist_analysis = await MarketplaceService.analyze_symptoms_for_specialists(
            appointment_request.symptoms_summary,
            appointment_request.urgency_level,
            patient_age,
            patient_gender
        )
        
        # Find matching doctors with enhanced criteria
        matching_doctors = await MarketplaceService.find_matching_doctors(
            specialist_analysis,
            max_budget=appointment_request.max_budget,
            consultation_type=appointment_request.consultation_type
        )
        
        # Convert to response format
        doctor_profiles = []
        for doctor in matching_doctors:
            profile = DoctorProfile(**{k: v for k, v in doctor.items() if k in DoctorProfile.__fields__})
            # Add match metadata
            profile.match_confidence = doctor.get("match_confidence", 0.5)
            profile.match_specialty = doctor.get("match_specialty", "")
            profile.match_reasoning = doctor.get("match_reasoning", "")
            doctor_profiles.append(profile)
        
        logger.info(
            "Enhanced specialist matching completed",
            user_id=current_user["user_id"],
            primary_specialists=len(specialist_analysis.get("primary_specialists", [])),
            matched_doctors=len(doctor_profiles),
            urgency=specialist_analysis.get("urgency_assessment", {}).get("level", "unknown")
        )
        
        return {
            "analysis": specialist_analysis,
            "matching_doctors": doctor_profiles,
            "match_summary": {
                "total_matches": len(doctor_profiles),
                "top_specialties": [spec["specialty"] for spec in specialist_analysis.get("primary_specialists", [])[:3]],
                "urgency_level": specialist_analysis.get("urgency_assessment", {}).get("level", "medium"),
                "recommended_timeframe": specialist_analysis.get("urgency_assessment", {}).get("time_frame", "within 2-3 days")
            }
        }
        
    except Exception as e:
        logger.error("Enhanced specialist matching failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Specialist matching failed"
        )


@router.get("/{doctor_id}/profile", response_model=DoctorProfile)
@rate_limit_general
async def get_doctor_profile(
    request: Request,
    doctor_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed doctor profile."""
    
    try:
        doctor = await DatabaseService.execute_query(
            """
            SELECT d.id, d.user_id, u.first_name, u.last_name, d.specialization,
                   d.sub_specializations, d.years_experience, d.rating, d.total_reviews,
                   d.base_consultation_fee, d.bio, d.languages, d.is_verified,
                   d.is_accepting_patients
            FROM doctors d
            JOIN users u ON d.user_id = u.id
            WHERE d.id = $1 AND d.is_verified = true
            """,
            doctor_id,
            fetch_one=True
        )
        
        if not doctor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doctor not found"
            )
        
        return DoctorProfile(**dict(doctor))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get doctor profile", doctor_id=doctor_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get doctor profile"
        )


# =============================================================================
# APPOINTMENT MANAGEMENT
# =============================================================================

@router.post("/appointments", response_model=AppointmentResponse)
@rate_limit_general
async def create_appointment_request(
    request: Request,
    appointment_request: AppointmentRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create a new appointment request for bidding."""
    
    try:
        # Create appointment record
        appointment_id = await DatabaseService.execute_query(
            """
            INSERT INTO appointments (patient_id, symptoms_summary, urgency_level, 
                                    preferred_date, consultation_type, status)
            VALUES ($1, $2, $3, $4, $5, 'pending')
            RETURNING id
            """,
            current_user["user_id"],
            appointment_request.symptoms_summary,
            appointment_request.urgency_level,
            appointment_request.preferred_date,
            appointment_request.consultation_type,
            fetch_one=True
        )
        
        # Get the created appointment
        appointment = await DatabaseService.execute_query(
            """
            SELECT id, patient_id, doctor_id, status, symptoms_summary, urgency_level,
                   preferred_date, consultation_type, final_fee, scheduled_at, created_at
            FROM appointments
            WHERE id = $1
            """,
            appointment_id["id"],
            fetch_one=True
        )
        
        # Notify matching doctors (simplified - would use more sophisticated matching)
        await _notify_matching_doctors(
            appointment_id["id"],
            appointment_request.symptoms_summary,
            appointment_request.urgency_level
        )
        
        logger.info(
            "Appointment request created",
            user_id=current_user["user_id"],
            appointment_id=appointment_id["id"],
            urgency=appointment_request.urgency_level
        )
        
        return AppointmentResponse(**dict(appointment))
        
    except Exception as e:
        logger.error("Failed to create appointment", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create appointment request"
        )


@router.post("/appointments/bid", response_model=BidResponse)
@require_doctor
@rate_limit_general
async def submit_appointment_bid(
    request: Request,
    bid_request: AppointmentBid,
    current_user: dict = Depends(get_current_user)
):
    """Submit a bid for an appointment (doctors only)."""
    
    try:
        # Get doctor ID
        doctor = await DatabaseService.execute_query(
            "SELECT id FROM doctors WHERE user_id = $1",
            current_user["user_id"],
            fetch_one=True
        )
        
        if not doctor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doctor profile not found"
            )
        
        # Check if appointment exists and is open for bidding
        appointment = await DatabaseService.execute_query(
            "SELECT id, patient_id, status FROM appointments WHERE id = $1",
            bid_request.appointment_id,
            fetch_one=True
        )
        
        if not appointment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appointment not found"
            )
        
        if appointment["status"] not in ["pending", "bidding"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Appointment is not open for bidding"
            )
        
        # Check if doctor already bid
        existing_bid = await DatabaseService.execute_query(
            "SELECT id FROM appointment_bids WHERE appointment_id = $1 AND doctor_id = $2",
            bid_request.appointment_id,
            doctor["id"],
            fetch_one=True
        )
        
        if existing_bid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You have already submitted a bid for this appointment"
            )
        
        # Create bid
        bid_id = await DatabaseService.execute_query(
            """
            INSERT INTO appointment_bids (appointment_id, doctor_id, bid_amount, 
                                        estimated_duration, available_slots, message)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            bid_request.appointment_id,
            doctor["id"],
            bid_request.bid_amount,
            bid_request.estimated_duration,
            bid_request.available_slots,
            bid_request.message,
            fetch_one=True
        )
        
        # Update appointment status to bidding
        await DatabaseService.execute_query(
            "UPDATE appointments SET status = 'bidding' WHERE id = $1 AND status = 'pending'",
            bid_request.appointment_id
        )
        
        # Get bid details with doctor info
        bid_details = await DatabaseService.execute_query(
            """
            SELECT ab.id, ab.appointment_id, ab.doctor_id, ab.bid_amount, 
                   ab.estimated_duration, ab.available_slots, ab.message, 
                   ab.is_selected, ab.created_at,
                   u.first_name || ' ' || u.last_name as doctor_name,
                   d.specialization
            FROM appointment_bids ab
            JOIN doctors d ON ab.doctor_id = d.id
            JOIN users u ON d.user_id = u.id
            WHERE ab.id = $1
            """,
            bid_id["id"],
            fetch_one=True
        )
        
        # Notify patient of new bid
        await notify_new_appointment_bid(
            appointment["patient_id"],
            bid_request.appointment_id,
            bid_details["doctor_name"],
            bid_request.bid_amount
        )
        
        logger.info(
            "Appointment bid submitted",
            doctor_id=doctor["id"],
            appointment_id=bid_request.appointment_id,
            bid_amount=bid_request.bid_amount
        )
        
        return BidResponse(**dict(bid_details))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit bid", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit bid"
        )


@router.get("/appointments/{appointment_id}/bids", response_model=List[BidResponse])
@rate_limit_general
async def get_appointment_bids(
    request: Request,
    appointment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all bids for an appointment (patient only)."""
    
    try:
        # Verify patient owns the appointment
        appointment = await DatabaseService.execute_query(
            "SELECT patient_id FROM appointments WHERE id = $1",
            appointment_id,
            fetch_one=True
        )
        
        if not appointment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appointment not found"
            )
        
        if appointment["patient_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get all bids for the appointment
        bids = await DatabaseService.execute_query(
            """
            SELECT ab.id, ab.appointment_id, ab.doctor_id, ab.bid_amount, 
                   ab.estimated_duration, ab.available_slots, ab.message, 
                   ab.is_selected, ab.created_at,
                   u.first_name || ' ' || u.last_name as doctor_name,
                   d.specialization
            FROM appointment_bids ab
            JOIN doctors d ON ab.doctor_id = d.id
            JOIN users u ON d.user_id = u.id
            WHERE ab.appointment_id = $1
            ORDER BY ab.bid_amount ASC, ab.created_at ASC
            """,
            appointment_id,
            fetch_all=True
        )
        
        return [BidResponse(**dict(bid)) for bid in bids]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get appointment bids", appointment_id=appointment_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get appointment bids"
        )


@router.put("/appointments/{appointment_id}/accept-bid/{bid_id}")
@rate_limit_general
async def accept_appointment_bid(
    request: Request,
    appointment_id: str,
    bid_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Accept a bid for an appointment (patient only)."""
    
    try:
        # Verify patient owns the appointment
        appointment = await DatabaseService.execute_query(
            "SELECT patient_id, status FROM appointments WHERE id = $1",
            appointment_id,
            fetch_one=True
        )
        
        if not appointment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appointment not found"
            )
        
        if appointment["patient_id"] != current_user["user_id"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        if appointment["status"] != "bidding":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Appointment is not in bidding status"
            )
        
        # Get bid details
        bid = await DatabaseService.execute_query(
            """
            SELECT ab.doctor_id, ab.bid_amount, ab.estimated_duration,
                   d.user_id as doctor_user_id,
                   u.first_name || ' ' || u.last_name as doctor_name
            FROM appointment_bids ab
            JOIN doctors d ON ab.doctor_id = d.id
            JOIN users u ON d.user_id = u.id
            WHERE ab.id = $1 AND ab.appointment_id = $2
            """,
            bid_id,
            appointment_id,
            fetch_one=True
        )
        
        if not bid:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Bid not found"
            )
        
        # Start transaction to update appointment and bid
        async with DatabaseService.postgres_transaction() as conn:
            # Update appointment
            await conn.execute(
                """
                UPDATE appointments 
                SET doctor_id = $1, status = 'confirmed', final_fee = $2
                WHERE id = $3
                """,
                bid["doctor_id"],
                bid["bid_amount"],
                appointment_id
            )
            
            # Mark bid as selected
            await conn.execute(
                "UPDATE appointment_bids SET is_selected = true WHERE id = $1",
                bid_id
            )
        
        # Notify doctor that bid was accepted
        await notify_appointment_accepted(
            bid["doctor_user_id"],
            appointment_id,
            current_user.get("first_name", "Patient")
        )
        
        logger.info(
            "Appointment bid accepted",
            patient_id=current_user["user_id"],
            appointment_id=appointment_id,
            doctor_id=bid["doctor_id"],
            final_fee=bid["bid_amount"]
        )
        
        return {
            "message": "Bid accepted successfully",
            "appointment_id": appointment_id,
            "doctor_name": bid["doctor_name"],
            "final_fee": bid["bid_amount"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to accept bid", appointment_id=appointment_id, bid_id=bid_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to accept bid"
        )


# =============================================================================
# APPOINTMENT REVIEWS
# =============================================================================

@router.post("/appointments/{appointment_id}/review")
@rate_limit_general
async def submit_appointment_review(
    request: Request,
    appointment_id: str,
    review_request: AppointmentReview,
    current_user: dict = Depends(get_current_user)
):
    """Submit a review for a completed appointment."""
    
    try:
        # Verify appointment exists and user participated
        appointment = await DatabaseService.execute_query(
            """
            SELECT patient_id, doctor_id, status
            FROM appointments 
            WHERE id = $1
            """,
            appointment_id,
            fetch_one=True
        )
        
        if not appointment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appointment not found"
            )
        
        if appointment["status"] != "completed":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only review completed appointments"
            )
        
        # Determine reviewer and reviewee
        if current_user["user_id"] == appointment["patient_id"]:
            # Patient reviewing doctor
            reviewer_id = appointment["patient_id"]
            reviewee_id = await DatabaseService.execute_query(
                "SELECT user_id FROM doctors WHERE id = $1",
                appointment["doctor_id"],
                fetch_one=True
            )
            reviewee_id = reviewee_id["user_id"] if reviewee_id else None
        elif current_user["role"] == "doctor":
            # Doctor reviewing patient (less common)
            doctor = await DatabaseService.execute_query(
                "SELECT id FROM doctors WHERE user_id = $1",
                current_user["user_id"],
                fetch_one=True
            )
            if doctor and doctor["id"] == appointment["doctor_id"]:
                reviewer_id = current_user["user_id"]
                reviewee_id = appointment["patient_id"]
            else:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied"
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        if not reviewee_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid appointment data"
            )
        
        # Check if review already exists
        existing_review = await DatabaseService.execute_query(
            """
            SELECT id FROM appointment_reviews 
            WHERE appointment_id = $1 AND reviewer_id = $2
            """,
            appointment_id,
            reviewer_id,
            fetch_one=True
        )
        
        if existing_review:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Review already submitted for this appointment"
            )
        
        # Create review
        review_id = await DatabaseService.execute_query(
            """
            INSERT INTO appointment_reviews (appointment_id, reviewer_id, reviewee_id, 
                                           rating, review_text, is_anonymous)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            appointment_id,
            reviewer_id,
            reviewee_id,
            review_request.rating,
            review_request.review_text,
            review_request.is_anonymous,
            fetch_one=True
        )
        
        logger.info(
            "Appointment review submitted",
            reviewer_id=reviewer_id,
            reviewee_id=reviewee_id,
            appointment_id=appointment_id,
            rating=review_request.rating
        )
        
        return {
            "message": "Review submitted successfully",
            "review_id": review_id["id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit review", appointment_id=appointment_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit review"
        )


@router.get("/appointments/{appointment_id}/reviews")
@rate_limit_general
async def get_appointment_reviews(
    request: Request,
    appointment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get reviews for an appointment."""
    
    try:
        # Get reviews for the appointment
        reviews = await DatabaseService.execute_query(
            """
            SELECT ar.id, ar.rating, ar.review_text, ar.is_anonymous, ar.created_at,
                   CASE 
                       WHEN ar.is_anonymous THEN 'Anonymous'
                       ELSE u.first_name || ' ' || u.last_name
                   END as reviewer_name,
                   u2.first_name || ' ' || u2.last_name as reviewee_name
            FROM appointment_reviews ar
            JOIN users u ON ar.reviewer_id = u.id
            JOIN users u2 ON ar.reviewee_id = u2.id
            WHERE ar.appointment_id = $1
            ORDER BY ar.created_at DESC
            """,
            appointment_id,
            fetch_all=True
        )
        
        return [dict(review) for review in reviews]
        
    except Exception as e:
        logger.error("Failed to get reviews", appointment_id=appointment_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get reviews"
        )


# =============================================================================
# DOCTOR DASHBOARD
# =============================================================================

@router.get("/dashboard/appointments")
@require_doctor
@rate_limit_general
async def get_doctor_appointments(
    request: Request,
    current_user: dict = Depends(get_current_user),
    status_filter: Optional[str] = Query(None),
    limit: int = Query(20, le=50),
    offset: int = Query(0, ge=0)
):
    """Get doctor's appointments (doctor dashboard)."""
    
    try:
        # Get doctor ID
        doctor = await DatabaseService.execute_query(
            "SELECT id FROM doctors WHERE user_id = $1",
            current_user["user_id"],
            fetch_one=True
        )
        
        if not doctor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doctor profile not found"
            )
        
        # Build query with optional status filter
        where_clause = "WHERE a.doctor_id = $1"
        params = [doctor["id"]]
        
        if status_filter:
            where_clause += " AND a.status = $2"
            params.append(status_filter)
            limit_offset_params = [limit, offset]
        else:
            limit_offset_params = [limit, offset]
        
        query = f"""
            SELECT a.id, a.patient_id, a.status, a.symptoms_summary, a.urgency_level,
                   a.preferred_date, a.consultation_type, a.final_fee, a.scheduled_at,
                   a.created_at, u.first_name || ' ' || u.last_name as patient_name
            FROM appointments a
            JOIN users u ON a.patient_id = u.id
            {where_clause}
            ORDER BY a.created_at DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
        """
        
        appointments = await DatabaseService.execute_query(
            query, 
            *params, 
            *limit_offset_params, 
            fetch_all=True
        )
        
        return [dict(appointment) for appointment in appointments]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get doctor appointments", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get appointments"
        )


@router.get("/bid-recommendations/{appointment_id}")
@require_doctor
@rate_limit_general
async def get_bid_recommendations(
    request: Request,
    appointment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get AI-powered bid recommendations for doctors."""
    
    try:
        # Get doctor ID
        doctor = await DatabaseService.execute_query(
            "SELECT id FROM doctors WHERE user_id = $1",
            current_user["user_id"],
            fetch_one=True
        )
        
        if not doctor:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Doctor profile not found"
            )
        
        # Get appointment details
        appointment = await DatabaseService.execute_query(
            """
            SELECT urgency_level, consultation_type, status
            FROM appointments 
            WHERE id = $1
            """,
            appointment_id,
            fetch_one=True
        )
        
        if not appointment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appointment not found"
            )
        
        if appointment["status"] not in ["pending", "bidding"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Appointment is not open for bidding"
            )
        
        # Get bid recommendations
        recommendations = await MarketplaceService.calculate_bid_recommendations(
            appointment_id,
            doctor["id"],
            appointment["urgency_level"],
            appointment["consultation_type"]
        )
        
        if "error" in recommendations:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=recommendations["error"]
            )
        
        logger.info(
            "Bid recommendations generated",
            doctor_id=doctor["id"],
            appointment_id=appointment_id,
            recommended_bid=recommendations.get("recommended_bid")
        )
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get bid recommendations", appointment_id=appointment_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get bid recommendations"
        )


@router.get("/appointments/{appointment_id}/analytics")
@rate_limit_general
async def get_appointment_analytics(
    request: Request,
    appointment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive appointment analytics."""
    
    try:
        # Verify user has access to this appointment
        appointment = await DatabaseService.execute_query(
            "SELECT patient_id, doctor_id FROM appointments WHERE id = $1",
            appointment_id,
            fetch_one=True
        )
        
        if not appointment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Appointment not found"
            )
        
        # Check if user is patient or doctor
        has_access = False
        if appointment["patient_id"] == current_user["user_id"]:
            has_access = True
        elif current_user["role"] == "doctor":
            doctor = await DatabaseService.execute_query(
                "SELECT id FROM doctors WHERE user_id = $1",
                current_user["user_id"],
                fetch_one=True
            )
            if doctor and appointment["doctor_id"] == doctor["id"]:
                has_access = True
        
        if not has_access:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get analytics
        analytics = await MarketplaceService.get_appointment_analytics(appointment_id)
        
        if "error" in analytics:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=analytics["error"]
            )
        
        return analytics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get appointment analytics", appointment_id=appointment_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get appointment analytics"
        )


@router.get("/marketplace/stats")
@rate_limit_general
async def get_marketplace_stats(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    """Get marketplace statistics and trends."""
    
    try:
        # Get overall marketplace stats
        stats = await DatabaseService.execute_query(
            """
            SELECT 
                COUNT(DISTINCT d.id) as total_doctors,
                COUNT(DISTINCT CASE WHEN d.is_accepting_patients THEN d.id END) as active_doctors,
                AVG(d.rating) as avg_doctor_rating,
                COUNT(DISTINCT a.id) as total_appointments,
                COUNT(DISTINCT CASE WHEN a.status = 'completed' THEN a.id END) as completed_appointments,
                AVG(CASE WHEN a.final_fee IS NOT NULL THEN a.final_fee END) as avg_appointment_fee
            FROM doctors d
            CROSS JOIN appointments a
            WHERE d.created_at > NOW() - INTERVAL '30 days'
            AND a.created_at > NOW() - INTERVAL '30 days'
            """,
            fetch_one=True
        )
        
        # Get specialty distribution
        specialty_stats = await DatabaseService.execute_query(
            """
            SELECT d.specialization, COUNT(*) as doctor_count,
                   AVG(d.rating) as avg_rating, AVG(d.base_consultation_fee) as avg_fee
            FROM doctors d
            WHERE d.is_verified = true
            GROUP BY d.specialization
            ORDER BY doctor_count DESC
            LIMIT 10
            """,
            fetch_all=True
        )
        
        # Get recent bidding trends
        bidding_trends = await DatabaseService.execute_query(
            """
            SELECT DATE(ab.created_at) as bid_date, COUNT(*) as daily_bids,
                   AVG(ab.bid_amount) as avg_bid_amount
            FROM appointment_bids ab
            WHERE ab.created_at > NOW() - INTERVAL '7 days'
            GROUP BY DATE(ab.created_at)
            ORDER BY bid_date DESC
            """,
            fetch_all=True
        )
        
        # Get urgency level distribution
        urgency_stats = await DatabaseService.execute_query(
            """
            SELECT a.urgency_level, COUNT(*) as appointment_count,
                   AVG(CASE WHEN a.final_fee IS NOT NULL THEN a.final_fee END) as avg_fee
            FROM appointments a
            WHERE a.created_at > NOW() - INTERVAL '30 days'
            GROUP BY a.urgency_level
            """,
            fetch_all=True
        )
        
        return {
            "overview": {
                "total_doctors": stats["total_doctors"],
                "active_doctors": stats["active_doctors"],
                "avg_doctor_rating": round(float(stats["avg_doctor_rating"] or 0), 2),
                "total_appointments": stats["total_appointments"],
                "completed_appointments": stats["completed_appointments"],
                "avg_appointment_fee": round(float(stats["avg_appointment_fee"] or 0), 2),
                "completion_rate": round(
                    (stats["completed_appointments"] / max(stats["total_appointments"], 1)) * 100, 1
                )
            },
            "specialties": [
                {
                    "specialty": spec["specialization"],
                    "doctor_count": spec["doctor_count"],
                    "avg_rating": round(float(spec["avg_rating"]), 2),
                    "avg_fee": round(float(spec["avg_fee"]), 2)
                }
                for spec in specialty_stats
            ],
            "bidding_trends": [
                {
                    "date": trend["bid_date"].isoformat(),
                    "daily_bids": trend["daily_bids"],
                    "avg_bid_amount": round(float(trend["avg_bid_amount"]), 2)
                }
                for trend in bidding_trends
            ],
            "urgency_distribution": [
                {
                    "urgency_level": urgency["urgency_level"],
                    "appointment_count": urgency["appointment_count"],
                    "avg_fee": round(float(urgency["avg_fee"] or 0), 2)
                }
                for urgency in urgency_stats
            ]
        }
        
    except Exception as e:
        logger.error("Failed to get marketplace stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get marketplace statistics"
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def _notify_matching_doctors(
    appointment_id: str,
    symptoms_summary: str,
    urgency_level: str
):
    """Enhanced notification system for matching doctors."""
    
    try:
        # Use enhanced MarketplaceService for better matching
        specialist_analysis = await MarketplaceService.analyze_symptoms_for_specialists(
            symptoms_summary,
            urgency_level
        )
        
        # Get top matching doctors
        matching_doctors = await MarketplaceService.find_matching_doctors(
            specialist_analysis,
            consultation_type="video"  # Default for notifications
        )
        
        # Notify top matches based on confidence and urgency
        notification_count = 15 if urgency_level in ["high", "critical"] else 10
        
        for doctor in matching_doctors[:notification_count]:
            try:
                # Create notification record
                await DatabaseService.execute_query(
                    """
                    INSERT INTO doctor_notifications (doctor_user_id, appointment_id, 
                                                    notification_type, match_confidence, 
                                                    match_specialty, urgency_level)
                    VALUES ($1, $2, 'new_appointment', $3, $4, $5)
                    """,
                    doctor["user_id"],
                    appointment_id,
                    doctor.get("match_confidence", 0.5),
                    doctor.get("match_specialty", "general"),
                    urgency_level
                )
                
                # Send real-time notification via WebSocket
                await notify_new_appointment_bid(
                    doctor["user_id"],
                    appointment_id,
                    f"New {urgency_level} urgency appointment",
                    doctor.get("match_confidence", 0.5)
                )
                
                logger.info(
                    "Doctor notified of matching appointment",
                    doctor_user_id=doctor["user_id"],
                    appointment_id=appointment_id,
                    match_confidence=doctor.get("match_confidence"),
                    specialty=doctor.get("match_specialty")
                )
                
            except Exception as notification_error:
                logger.warning(
                    "Failed to notify individual doctor",
                    doctor_user_id=doctor["user_id"],
                    error=str(notification_error)
                )
        
        logger.info(
            "Doctor notification process completed",
            appointment_id=appointment_id,
            total_notified=len(matching_doctors[:notification_count]),
            urgency_level=urgency_level
        )
        
    except Exception as e:
        logger.error("Failed to notify matching doctors", appointment_id=appointment_id, error=str(e))