# HealthSync AI - Authentication Routes
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr, validator
import structlog
from passlib.context import CryptContext

from config import settings
from api.middleware.auth import (
    create_access_token, create_refresh_token, verify_token,
    get_current_user, require_role
)
from api.middleware.rate_limit import rate_limit_general
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)
router = APIRouter()
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UserSignupRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    role: str = "patient"
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        allowed_roles = ['patient', 'doctor']
        if v not in allowed_roles:
            raise ValueError(f'Role must be one of: {", ".join(allowed_roles)}')
        return v


class DoctorSignupRequest(UserSignupRequest):
    role: str = "doctor"
    license_number: str
    specialization: str
    years_experience: int = 0
    base_consultation_fee: Optional[float] = None
    
    @validator('license_number')
    def validate_license(cls, v):
        if len(v) < 5:
            raise ValueError('License number must be at least 5 characters')
        return v
    
    @validator('specialization')
    def validate_specialization(cls, v):
        if len(v) < 2:
            raise ValueError('Specialization is required')
        return v


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserProfileUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None


# Response Models
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseModel):
    id: str
    email: str
    role: str
    first_name: str
    last_name: str
    is_active: bool
    created_at: datetime
    profile_complete: bool


# =============================================================================
# AUTHENTICATION ROUTES
# =============================================================================

@router.post("/signup", response_model=TokenResponse)
@rate_limit_general
async def signup_user(request: Request, user_data: UserSignupRequest):
    """Register a new user account."""
    
    try:
        # Check if user already exists
        existing_user = await DatabaseService.mongodb_find_one(
            "users",
            {"email": user_data.email}
        )
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Hash password
        hashed_password = pwd_context.hash(user_data.password)
        
        # Create user record in MongoDB
        import uuid
        user_id = str(uuid.uuid4())
        
        user_document = {
            "_id": user_id,
            "user_id": user_id,
            "email": user_data.email,
            "role": user_data.role,
            "first_name": user_data.first_name,
            "last_name": user_data.last_name,
            "date_of_birth": user_data.date_of_birth,
            "gender": user_data.gender,
            "phone": user_data.phone,
            "password_hash": hashed_password,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await DatabaseService.mongodb_insert_one("users", user_document)
        
        # Generate tokens
        token_data = {"sub": user_id, "email": user_data.email, "role": user_data.role}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        logger.info(
            "User registered successfully",
            user_id=user_id,
            email=user_data.email,
            role=user_data.role
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("User registration failed", error=str(e), email=user_data.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/signup/doctor", response_model=TokenResponse)
@rate_limit_general
async def signup_doctor(request: Request, doctor_data: DoctorSignupRequest):
    """Register a new doctor account."""
    
    try:
        # Check if user already exists
        existing_user = await DatabaseService.mongodb_find_one(
            "users",
            {"email": doctor_data.email}
        )
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Check if license number is already used
        existing_license = await DatabaseService.mongodb_find_one(
            "doctors",
            {"license_number": doctor_data.license_number}
        )
        
        if existing_license:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="License number already registered"
            )
        
        # Hash password
        hashed_password = pwd_context.hash(doctor_data.password)
        
        # Create user record in MongoDB
        import uuid
        user_id = str(uuid.uuid4())
        
        user_document = {
            "_id": user_id,
            "user_id": user_id,
            "email": doctor_data.email,
            "role": "doctor",
            "first_name": doctor_data.first_name,
            "last_name": doctor_data.last_name,
            "date_of_birth": doctor_data.date_of_birth,
            "gender": doctor_data.gender,
            "phone": doctor_data.phone,
            "password_hash": hashed_password,
            "is_active": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await DatabaseService.mongodb_insert_one("users", user_document)
        
        # Create doctor profile
        doctor_document = {
            "user_id": user_id,
            "license_number": doctor_data.license_number,
            "specialization": doctor_data.specialization,
            "years_experience": doctor_data.years_experience,
            "base_consultation_fee": doctor_data.base_consultation_fee,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await DatabaseService.mongodb_insert_one("doctors", doctor_document)
        
        # Generate tokens
        token_data = {"sub": user_id, "email": doctor_data.email, "role": "doctor"}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        logger.info(
            "Doctor registered successfully",
            user_id=str(user_id),
            email=doctor_data.email,
            specialization=doctor_data.specialization
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Doctor registration failed", error=str(e), email=doctor_data.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
@rate_limit_general
async def login_user(request: Request, login_data: UserLoginRequest):
    """Authenticate user and return tokens."""
    
    try:
        # Get user by email from MongoDB
        user = await DatabaseService.mongodb_find_one(
            "users",
            {"email": login_data.email}
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user is active
        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is disabled"
            )
        
        # Verify password
        if not pwd_context.verify(login_data.password, user.get("password_hash", "")):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Generate tokens
        token_data = {
            "sub": str(user.get("user_id") or user.get("_id")),
            "email": user["email"],
            "role": user["role"]
        }
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        # Update last login
        await DatabaseService.mongodb_update_one(
            "users",
            {"user_id": user.get("user_id") or user.get("_id")},
            {"$set": {"updated_at": datetime.utcnow(), "last_login": datetime.utcnow()}}
        )
        
        logger.info(
            "User logged in successfully",
            user_id=str(user.get("user_id") or user.get("_id")),
            email=user["email"],
            role=user["role"]
        )
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e), email=login_data.email)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenResponse)
@rate_limit_general
async def refresh_token(request: Request, refresh_data: TokenRefreshRequest):
    """Refresh access token using refresh token."""
    
    try:
        # Verify refresh token
        payload = verify_token(refresh_data.refresh_token)
        
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get current user info from MongoDB
        user = await DatabaseService.mongodb_find_one(
            "users",
            {"user_id": user_id}
        )
        
        if not user or not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Generate new tokens
        token_data = {
            "sub": str(user.get("user_id") or user.get("_id")),
            "email": user["email"],
            "role": user["role"]
        }
        access_token = create_access_token(token_data)
        new_refresh_token = create_refresh_token(token_data)
        
        logger.info("Token refreshed successfully", user_id=user_id)
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=new_refresh_token,
            expires_in=settings.jwt_access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(request: Request, current_user: dict = Depends(get_current_user)):
    """Get current user information."""
    
    try:
        # Get detailed user info from MongoDB
        user = await DatabaseService.mongodb_find_one(
            "users",
            {"user_id": current_user["user_id"]}
        )
        
        if user:
            user["profile_complete"] = bool(user.get("first_name") and user.get("last_name"))
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Convert MongoDB document to response format
        user_response = {
            "id": str(user.get("user_id") or user.get("_id")),
            "email": user["email"],
            "role": user["role"],
            "first_name": user.get("first_name", ""),
            "last_name": user.get("last_name", ""),
            "is_active": user.get("is_active", True),
            "created_at": user.get("created_at", datetime.utcnow()),
            "profile_complete": user.get("profile_complete", False)
        }
        return UserResponse(**user_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get user info", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    request: Request,
    profile_data: UserProfileUpdate,
    current_user: dict = Depends(get_current_user)
):
    """Update user profile information."""
    
    try:
        # Build update document
        update_data = {}
        for field, value in profile_data.dict(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value
        
        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )
        
        # Add updated_at
        update_data["updated_at"] = datetime.utcnow()
        
        # Update user in MongoDB
        await DatabaseService.mongodb_update_one(
            "users",
            {"user_id": current_user["user_id"]},
            {"$set": update_data}
        )
        
        # Get updated user
        updated_user = await DatabaseService.mongodb_find_one(
            "users",
            {"user_id": current_user["user_id"]}
        )
        
        if updated_user:
            updated_user["profile_complete"] = bool(updated_user.get("first_name") and updated_user.get("last_name"))
        
        logger.info(
            "User profile updated",
            user_id=current_user["user_id"],
            updated_fields=list(profile_data.dict(exclude_unset=True).keys())
        )
        
        # Convert MongoDB document to response format
        user_response = {
            "id": str(updated_user.get("user_id") or updated_user.get("_id")),
            "email": updated_user["email"],
            "role": updated_user["role"],
            "first_name": updated_user.get("first_name", ""),
            "last_name": updated_user.get("last_name", ""),
            "is_active": updated_user.get("is_active", True),
            "created_at": updated_user.get("created_at", datetime.utcnow()),
            "profile_complete": updated_user.get("profile_complete", False)
        }
        return UserResponse(**user_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profile update failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.post("/change-password")
async def change_password(
    request: Request,
    password_data: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user)
):
    """Change user password."""
    
    try:
        # Get user with password hash from MongoDB
        user = await DatabaseService.mongodb_find_one(
            "users",
            {"user_id": current_user["user_id"]}
        )
        
        if not user or not user.get("password_hash"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User password not found"
            )
        
        # Verify current password
        if not pwd_context.verify(password_data.current_password, user["password_hash"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_password_hash = pwd_context.hash(password_data.new_password)
        
        # Update password in MongoDB
        await DatabaseService.mongodb_update_one(
            "users",
            {"user_id": current_user["user_id"]},
            {"$set": {"password_hash": new_password_hash, "updated_at": datetime.utcnow()}}
        )
        
        logger.info("Password changed successfully", user_id=current_user["user_id"])
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Password change failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/logout")
async def logout_user(request: Request, current_user: dict = Depends(get_current_user)):
    """Logout user (invalidate tokens)."""
    
    try:
        # In a production system, you would add the token to a blacklist
        # For now, we'll just log the logout event
        
        logger.info("User logged out", user_id=current_user["user_id"])
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error("Logout failed", user_id=current_user["user_id"], error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )