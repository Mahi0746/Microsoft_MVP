"""
Google Fit API Routes

Endpoints for Google Fit OAuth flow and connection management.
"""

from fastapi import APIRouter, HTTPException, status, Depends, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional
import logging
from datetime import datetime, timedelta

from api.middleware.auth import get_current_user
from services.google_fit_service import get_google_fit_service
from services.mongodb_atlas_service import get_mongodb_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/google-fit", tags=["Google Fit"])


class GoogleFitStatus(BaseModel):
    """Google Fit connection status response"""
    connected: bool
    scopes: Optional[list] = None
    last_synced: Optional[str] = None


@router.get("/auth-url")
async def get_auth_url(
    current_user: dict = Depends(get_current_user)
):
    """
    Get Google Fit OAuth authorization URL
    
    Returns the URL to redirect user to for Google Fit authorization
    """
    try:
        google_fit = get_google_fit_service()
        
        # Use user_id as state for CSRF protection
        state = current_user["user_id"]
        
        auth_url = google_fit.get_authorization_url(state=state)
        
        return {
            "auth_url": auth_url,
            "message": "Redirect user to this URL to authorize Google Fit access"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate auth URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate authorization URL"
        )


@router.get("/callback")
async def oauth_callback(
    code: str,
    state: Optional[str] = None,
    error: Optional[str] = None
):
    """
    Handle Google OAuth callback
    
    This endpoint receives the authorization code from Google
    and exchanges it for access/refresh tokens
    """
    if error:
        logger.error(f"OAuth error: {error}")
        # Redirect to frontend with error
        return RedirectResponse(url=f"http://localhost:3000/settings?google_fit_error={error}")
    
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing authorization code"
        )
    
    try:
        google_fit = get_google_fit_service()
        mongodb = await get_mongodb_service()
        
        # Exchange code for tokens
        tokens = await google_fit.exchange_code_for_tokens(code)
        
        # Encrypt tokens before storage
        encrypted_access = google_fit.encrypt_token(tokens['access_token'])
        encrypted_refresh = google_fit.encrypt_token(tokens.get('refresh_token', ''))
        
        # Calculate token expiry
        expires_at = datetime.utcnow() + timedelta(seconds=tokens['expires_in'])
        
        # Store in user's document
        user_id = state  # State contains user_id
        
        update_data = {
            "google_fit": {
                "connected": True,
                "access_token": encrypted_access,
                "refresh_token": encrypted_refresh,
                "token_expiry": expires_at,
                "scopes": google_fit.SCOPES,
                "connected_at": datetime.utcnow()
            }
        }
        
        await mongodb.database.users.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        
        logger.info(f"Successfully connected Google Fit for user {user_id}")
        
        # Redirect to frontend success page
        return RedirectResponse(url="http://localhost:3000/settings?google_fit_success=true")
        
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse(url=f"http://localhost:3000/settings?google_fit_error=connection_failed")


@router.get("/status", response_model=GoogleFitStatus)
async def get_connection_status(
    current_user: dict = Depends(get_current_user)
):
    """
    Check if user has connected Google Fit
    """
    user_id = current_user["user_id"]
    
    try:
        mongodb = await get_mongodb_service()
        
        user = await mongodb.database.users.find_one({"user_id": user_id})
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        google_fit_data = user.get("google_fit", {})
        
        return GoogleFitStatus(
            connected=google_fit_data.get("connected", False),
            scopes=google_fit_data.get("scopes"),
            last_synced=google_fit_data.get("connected_at", "").isoformat() if google_fit_data.get("connected_at") else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get connection status"
        )


@router.delete("/disconnect")
async def disconnect_google_fit(
    current_user: dict = Depends(get_current_user)
):
    """
    Disconnect Google Fit from user account
    """
    user_id = current_user["user_id"]
    
    try:
        mongodb = await get_mongodb_service()
        
        # Remove Google Fit data from user document
        await mongodb.database.users.update_one(
            {"user_id": user_id},
            {"$unset": {"google_fit": ""}}
        )
        
        logger.info(f"Disconnected Google Fit for user {user_id}")
        
        return {
            "success": True,
            "message": "Google Fit disconnected successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to disconnect: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disconnect Google Fit"
        )
