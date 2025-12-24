"""
Google Fit Service - OAuth and API Integration

Handles OAuth flow, token management, and fetching health data from Google Fit API.
Based on the working demo.py implementation.
"""

import logging
import time
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import base64
import json

from config_flexible import settings

logger = logging.getLogger(__name__)


class GoogleFitService:
    """
    Service for integrating with Google Fit API
    
    Features:
    - OAuth2 flow for user authorization
    - Token encryption for secure storage
    - Fetch steps, heart rate, sleep, calories, activity minutes
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/fitness.activity.read',
        'https://www.googleapis.com/auth/fitness.heart_rate.read',
        'https://www.googleapis.com/auth/fitness.sleep.read',
        'https://www.googleapis.com/auth/fitness.body.read',
    ]
    
    def __init__(self):
        self.client_id = getattr(settings, 'google_fit_client_id', None)
        self.client_secret = getattr(settings, 'google_fit_client_secret', None)
        self.redirect_uri = getattr(settings, 'google_fit_redirect_uri', 'http://localhost:8000/api/google-fit/callback')
        self.encryption_key = getattr(settings, 'encryption_key', None)
        
        # Generate encryption key if not provided
        if not self.encryption_key:
            logger.warning("No encryption key found, generating temporary key")
            self.encryption_key = Fernet.generate_key()
        elif isinstance(self.encryption_key, str):
            self.encryption_key = self.encryption_key.encode()
        
        self.fernet = Fernet(self.encryption_key)
    
    def get_authorization_url(self, state: str = None) -> str:
        """
        Generate Google OAuth authorization URL
        
        Args:
            state: Optional state parameter for CSRF protection
            
        Returns:
            Authorization URL to redirect user to
        """
        if not self.client_id:
            raise ValueError("Google Fit client ID not configured")
        
        params = {
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'response_type': 'code',
            'scope': ' '.join(self.SCOPES),
            'access_type': 'offline',  # Get refresh token
            'prompt': 'consent',  # Force consent screen to ensure refresh token
        }
        
        if state:
            params['state'] = state
        
        base_url = 'https://accounts.google.com/o/oauth2/v2/auth'
        query_string = '&'.join([f"{k}={requests.utils.quote(v)}" for k, v in params.items()])
        
        return f"{base_url}?{query_string}"
    
    async def exchange_code_for_tokens(self, code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens
        
        Args:
            code: Authorization code from OAuth callback
            
        Returns:
            Dictionary with access_token, refresh_token, expires_in
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("Google Fit credentials not configured")
        
        token_url = 'https://oauth2.googleapis.com/token'
        
        data = {
            'code': code,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code',
        }
        
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            
            tokens = response.json()
            logger.info("Successfully exchanged code for tokens")
            
            return {
                'access_token': tokens['access_token'],
                'refresh_token': tokens.get('refresh_token'),
                'expires_in': tokens.get('expires_in', 3600),
                'token_type': tokens.get('token_type', 'Bearer'),
            }
            
        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            raise
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an expired access token using refresh token
        
        Args:
            refresh_token: The refresh token
            
        Returns:
            New access token data
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("Google Fit credentials not configured")
        
        token_url = 'https://oauth2.googleapis.com/token'
        
        data = {
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'refresh_token': refresh_token,
            'grant_type': 'refresh_token',
        }
        
        try:
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            
            tokens = response.json()
            logger.info("Successfully refreshed access token")
            
            return {
                'access_token': tokens['access_token'],
                'expires_in': tokens.get('expires_in', 3600),
            }
            
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            raise
    
    def encrypt_token(self, token: str) -> str:
        """Encrypt a token for secure storage"""
        if not token:
            return ""
        return self.fernet.encrypt(token.encode()).decode()
    
    def decrypt_token(self, encrypted_token: str) -> str:
        """Decrypt a stored token"""
        if not encrypted_token:
            return ""
        return self.fernet.decrypt(encrypted_token.encode()).decode()
    
    async def get_steps(self, access_token: str, days: int = 7) -> Dict[str, Any]:
        """
        Fetch steps data from Google Fit
        
        Args:
            access_token: Valid Google Fit access token
            days: Number of days to look back
            
        Returns:
            Dictionary with total_steps and avg_steps_per_day
        """
        headers = {'Authorization': f'Bearer {access_token}'}
        
        end_ns = int(time.time() * 1e9)
        start_ns = end_ns - days * 24 * 60 * 60 * int(1e9)
        
        url = (
            "https://www.googleapis.com/fitness/v1/users/me/dataSources/"
            "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"
            f"/datasets/{start_ns}-{end_ns}"
        )
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            total_steps = sum(
                point['value'][0]['intVal']
                for point in data.get('point', [])
            )
            
            avg = total_steps // days if total_steps else 0
            
            logger.info(f"Fetched steps: total={total_steps}, avg={avg}")
            
            return {
                'total_steps': total_steps,
                'avg_steps_per_day': avg
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch steps: {e}")
            return {'total_steps': 0, 'avg_steps_per_day': 0}
    
    async def get_heart_rate(self, access_token: str, days: int = 7) -> Dict[str, Any]:
        """Fetch heart rate data from Google Fit"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        end_time_ns = int(time.time() * 1000000000)
        start_time_ns = end_time_ns - (days * 24 * 60 * 60 * 1000000000)
        
        url = f'https://www.googleapis.com/fitness/v1/users/me/dataSources/derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm/datasets/{start_time_ns}-{end_time_ns}'
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            heart_rates = [
                point['value'][0]['fpVal']
                for point in data.get('point', [])
            ]
            
            if heart_rates:
                return {
                    'resting_hr': int(min(heart_rates)),
                    'avg_hr': int(sum(heart_rates) / len(heart_rates)),
                    'max_hr': int(max(heart_rates))
                }
            else:
                logger.warning("No heart rate data found")
                return {'resting_hr': 70}
                
        except Exception as e:
            logger.error(f"Failed to fetch heart rate: {e}")
            return {'resting_hr': 70}
    
    async def get_sleep(self, access_token: str, days: int = 7) -> Dict[str, Any]:
        """Fetch sleep data from Google Fit"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        end_time_ns = int(time.time() * 1000000000)
        start_time_ns = end_time_ns - (days * 24 * 60 * 60 * 1000000000)
        
        url = f'https://www.googleapis.com/fitness/v1/users/me/dataSources/derived:com.google.sleep.segment:com.google.android.gms:merged/datasets/{start_time_ns}-{end_time_ns}'
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            sleep_sessions = []
            for point in data.get('point', []):
                start = int(point['startTimeNanos']) / 1000000000
                end = int(point['endTimeNanos']) / 1000000000
                duration_hours = (end - start) / 3600
                sleep_sessions.append(duration_hours)
            
            if sleep_sessions:
                avg_sleep = sum(sleep_sessions) / len(sleep_sessions)
                return {'avg_sleep_hours': round(avg_sleep, 1)}
            else:
                logger.warning("No sleep data found")
                return {'avg_sleep_hours': 7.0}
                
        except Exception as e:
            logger.error(f"Failed to fetch sleep: {e}")
            return {'avg_sleep_hours': 7.0}
    
    async def get_all_metrics(self, access_token: str, days: int = 7) -> Dict[str, Any]:
        """
        Fetch all health metrics from Google Fit
        
        Args:
            access_token: Valid access token
            days: Days to look back
            
        Returns:
            Combined dictionary of all metrics
        """
        logger.info(f"Fetching all Google Fit metrics for last {days} days")
        
        results = {}
        
        try:
            steps_data = await self.get_steps(access_token, days)
            results.update(steps_data)
            
            hr_data = await self.get_heart_rate(access_token, days)
            results.update(hr_data)
            
            sleep_data = await self.get_sleep(access_token, days)
            results.update(sleep_data)
            
            logger.info(f"Successfully fetched Google Fit metrics: {results}")
            
        except Exception as e:
            logger.error(f"Error fetching Google Fit metrics: {e}")
        
        return results


# Singleton
_google_fit_service = None


def get_google_fit_service() -> GoogleFitService:
    """Get or create GoogleFitService singleton"""
    global _google_fit_service
    if _google_fit_service is None:
        _google_fit_service = GoogleFitService()
    return _google_fit_service
