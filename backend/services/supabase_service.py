# HealthSync AI - Supabase Service Integration
import asyncio
from typing import Optional, Dict, Any, List
from supabase import create_client, Client
from gotrue import SyncGoTrueClient
import structlog
import httpx
from datetime import datetime

from config import settings


logger = structlog.get_logger(__name__)


class SupabaseService:
    """Supabase service for authentication, database, and storage."""
    
    _client: Optional[Client] = None
    _auth_client: Optional[SyncGoTrueClient] = None
    
    @classmethod
    def initialize(cls):
        """Initialize Supabase client."""
        
        try:
            cls._client = create_client(
                settings.supabase_url,
                settings.supabase_anon_key
            )
            
            cls._auth_client = cls._client.auth
            
            logger.info("Supabase client initialized")
            
        except Exception as e:
            logger.error("Failed to initialize Supabase client", error=str(e))
            raise
    
    @classmethod
    def get_client(cls) -> Client:
        """Get Supabase client instance."""
        
        if not cls._client:
            cls.initialize()
        
        return cls._client
    
    @classmethod
    def get_auth_client(cls) -> SyncGoTrueClient:
        """Get Supabase auth client."""
        
        if not cls._auth_client:
            cls.initialize()
        
        return cls._auth_client
    
    # =============================================================================
    # AUTHENTICATION METHODS
    # =============================================================================
    
    @classmethod
    async def sign_up_user(
        cls,
        email: str,
        password: str,
        user_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sign up user with Supabase Auth."""
        
        try:
            auth_client = cls.get_auth_client()
            
            response = auth_client.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": user_metadata or {}
                }
            })
            
            if response.user:
                logger.info("User signed up with Supabase", user_id=response.user.id, email=email)
                
                return {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "email_confirmed": response.user.email_confirmed_at is not None,
                    "created_at": response.user.created_at,
                    "access_token": response.session.access_token if response.session else None,
                    "refresh_token": response.session.refresh_token if response.session else None
                }
            else:
                raise Exception("User creation failed")
                
        except Exception as e:
            logger.error("Supabase signup failed", email=email, error=str(e))
            raise
    
    @classmethod
    async def sign_in_user(cls, email: str, password: str) -> Dict[str, Any]:
        """Sign in user with Supabase Auth."""
        
        try:
            auth_client = cls.get_auth_client()
            
            response = auth_client.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user and response.session:
                logger.info("User signed in with Supabase", user_id=response.user.id, email=email)
                
                return {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at,
                    "user_metadata": response.user.user_metadata
                }
            else:
                raise Exception("Authentication failed")
                
        except Exception as e:
            logger.error("Supabase signin failed", email=email, error=str(e))
            raise
    
    @classmethod
    async def refresh_session(cls, refresh_token: str) -> Dict[str, Any]:
        """Refresh user session with Supabase."""
        
        try:
            auth_client = cls.get_auth_client()
            
            response = auth_client.refresh_session(refresh_token)
            
            if response.session:
                logger.info("Session refreshed with Supabase", user_id=response.user.id)
                
                return {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "access_token": response.session.access_token,
                    "refresh_token": response.session.refresh_token,
                    "expires_at": response.session.expires_at
                }
            else:
                raise Exception("Session refresh failed")
                
        except Exception as e:
            logger.error("Supabase session refresh failed", error=str(e))
            raise
    
    @classmethod
    async def verify_token(cls, access_token: str) -> Dict[str, Any]:
        """Verify access token with Supabase."""
        
        try:
            auth_client = cls.get_auth_client()
            
            response = auth_client.get_user(access_token)
            
            if response.user:
                return {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "email_confirmed": response.user.email_confirmed_at is not None,
                    "user_metadata": response.user.user_metadata,
                    "app_metadata": response.user.app_metadata
                }
            else:
                raise Exception("Token verification failed")
                
        except Exception as e:
            logger.error("Token verification failed", error=str(e))
            raise
    
    @classmethod
    async def sign_out_user(cls, access_token: str) -> bool:
        """Sign out user from Supabase."""
        
        try:
            auth_client = cls.get_auth_client()
            
            # Set the session first
            auth_client.set_session(access_token, "")  # Empty refresh token
            
            response = auth_client.sign_out()
            
            logger.info("User signed out from Supabase")
            return True
            
        except Exception as e:
            logger.error("Supabase signout failed", error=str(e))
            return False
    
    @classmethod
    async def update_user_metadata(
        cls,
        access_token: str,
        user_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update user metadata in Supabase."""
        
        try:
            auth_client = cls.get_auth_client()
            
            # Set the session first
            auth_client.set_session(access_token, "")
            
            response = auth_client.update_user({
                "data": user_metadata
            })
            
            if response.user:
                logger.info("User metadata updated", user_id=response.user.id)
                
                return {
                    "user_id": response.user.id,
                    "email": response.user.email,
                    "user_metadata": response.user.user_metadata
                }
            else:
                raise Exception("Metadata update failed")
                
        except Exception as e:
            logger.error("User metadata update failed", error=str(e))
            raise
    
    # =============================================================================
    # DATABASE METHODS
    # =============================================================================
    
    @classmethod
    async def insert_record(
        cls,
        table: str,
        data: Dict[str, Any],
        access_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Insert record into Supabase table."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            response = client.table(table).insert(data).execute()
            
            if response.data:
                logger.debug("Record inserted", table=table, record_id=response.data[0].get("id"))
                return response.data[0]
            else:
                raise Exception(f"Insert failed: {response}")
                
        except Exception as e:
            logger.error("Supabase insert failed", table=table, error=str(e))
            raise
    
    @classmethod
    async def select_records(
        cls,
        table: str,
        columns: str = "*",
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        access_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Select records from Supabase table."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            query = client.table(table).select(columns)
            
            # Apply filters
            if filters:
                for key, value in filters.items():
                    if isinstance(value, dict):
                        # Handle complex filters like {"gte": 18}
                        for op, val in value.items():
                            if op == "eq":
                                query = query.eq(key, val)
                            elif op == "neq":
                                query = query.neq(key, val)
                            elif op == "gt":
                                query = query.gt(key, val)
                            elif op == "gte":
                                query = query.gte(key, val)
                            elif op == "lt":
                                query = query.lt(key, val)
                            elif op == "lte":
                                query = query.lte(key, val)
                            elif op == "like":
                                query = query.like(key, val)
                            elif op == "ilike":
                                query = query.ilike(key, val)
                            elif op == "in":
                                query = query.in_(key, val)
                    else:
                        # Simple equality filter
                        query = query.eq(key, value)
            
            # Apply ordering
            if order_by:
                if order_by.startswith("-"):
                    query = query.order(order_by[1:], desc=True)
                else:
                    query = query.order(order_by)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            
            return response.data or []
            
        except Exception as e:
            logger.error("Supabase select failed", table=table, error=str(e))
            raise
    
    @classmethod
    async def update_record(
        cls,
        table: str,
        record_id: str,
        data: Dict[str, Any],
        access_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update record in Supabase table."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            response = client.table(table).update(data).eq("id", record_id).execute()
            
            if response.data:
                logger.debug("Record updated", table=table, record_id=record_id)
                return response.data[0]
            else:
                raise Exception(f"Update failed: {response}")
                
        except Exception as e:
            logger.error("Supabase update failed", table=table, record_id=record_id, error=str(e))
            raise
    
    @classmethod
    async def delete_record(
        cls,
        table: str,
        record_id: str,
        access_token: Optional[str] = None
    ) -> bool:
        """Delete record from Supabase table."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            response = client.table(table).delete().eq("id", record_id).execute()
            
            logger.debug("Record deleted", table=table, record_id=record_id)
            return True
            
        except Exception as e:
            logger.error("Supabase delete failed", table=table, record_id=record_id, error=str(e))
            return False
    
    # =============================================================================
    # STORAGE METHODS
    # =============================================================================
    
    @classmethod
    async def upload_file(
        cls,
        bucket: str,
        file_path: str,
        file_data: bytes,
        content_type: Optional[str] = None,
        access_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload file to Supabase Storage."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            response = client.storage.from_(bucket).upload(
                path=file_path,
                file=file_data,
                file_options={
                    "content-type": content_type or "application/octet-stream"
                }
            )
            
            if response.get("error"):
                raise Exception(f"Upload failed: {response['error']}")
            
            # Get public URL
            public_url = client.storage.from_(bucket).get_public_url(file_path)
            
            logger.info("File uploaded to Supabase Storage", bucket=bucket, path=file_path)
            
            return {
                "path": file_path,
                "public_url": public_url,
                "bucket": bucket
            }
            
        except Exception as e:
            logger.error("Supabase file upload failed", bucket=bucket, path=file_path, error=str(e))
            raise
    
    @classmethod
    async def download_file(
        cls,
        bucket: str,
        file_path: str,
        access_token: Optional[str] = None
    ) -> bytes:
        """Download file from Supabase Storage."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            response = client.storage.from_(bucket).download(file_path)
            
            if isinstance(response, bytes):
                return response
            else:
                raise Exception(f"Download failed: {response}")
                
        except Exception as e:
            logger.error("Supabase file download failed", bucket=bucket, path=file_path, error=str(e))
            raise
    
    @classmethod
    async def delete_file(
        cls,
        bucket: str,
        file_path: str,
        access_token: Optional[str] = None
    ) -> bool:
        """Delete file from Supabase Storage."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            response = client.storage.from_(bucket).remove([file_path])
            
            if response.get("error"):
                raise Exception(f"Delete failed: {response['error']}")
            
            logger.info("File deleted from Supabase Storage", bucket=bucket, path=file_path)
            return True
            
        except Exception as e:
            logger.error("Supabase file delete failed", bucket=bucket, path=file_path, error=str(e))
            return False
    
    @classmethod
    async def get_signed_url(
        cls,
        bucket: str,
        file_path: str,
        expires_in: int = 3600,
        access_token: Optional[str] = None
    ) -> str:
        """Get signed URL for private file access."""
        
        try:
            client = cls.get_client()
            
            # Set auth header if token provided
            if access_token:
                client.auth.set_session(access_token, "")
            
            response = client.storage.from_(bucket).create_signed_url(
                path=file_path,
                expires_in=expires_in
            )
            
            if response.get("error"):
                raise Exception(f"Signed URL creation failed: {response['error']}")
            
            return response["signedURL"]
            
        except Exception as e:
            logger.error("Supabase signed URL creation failed", bucket=bucket, path=file_path, error=str(e))
            raise
    
    # =============================================================================
    # REALTIME METHODS
    # =============================================================================
    
    @classmethod
    def create_realtime_channel(
        cls,
        channel_name: str,
        table: str = None,
        filter_conditions: Optional[Dict[str, str]] = None
    ):
        """Create realtime channel for live updates."""
        
        try:
            client = cls.get_client()
            
            channel = client.channel(channel_name)
            
            if table:
                # Subscribe to table changes
                postgres_changes = {
                    "event": "*",
                    "schema": "public",
                    "table": table
                }
                
                if filter_conditions:
                    postgres_changes["filter"] = filter_conditions
                
                channel.on_postgres_changes(postgres_changes, lambda payload: None)
            
            # Subscribe to the channel
            channel.subscribe()
            
            logger.info("Realtime channel created", channel=channel_name, table=table)
            
            return channel
            
        except Exception as e:
            logger.error("Realtime channel creation failed", channel=channel_name, error=str(e))
            raise
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    @classmethod
    async def health_check(cls) -> Dict[str, Any]:
        """Check Supabase service health."""
        
        try:
            # Test database connection
            response = await cls.select_records("users", "id", limit=1)
            
            # Test auth service
            auth_client = cls.get_auth_client()
            
            return {
                "status": "healthy",
                "database": "connected",
                "auth": "available",
                "storage": "available",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Supabase health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @classmethod
    async def get_service_stats(cls) -> Dict[str, Any]:
        """Get Supabase service statistics."""
        
        try:
            # Get user count
            users = await cls.select_records("users", "count(*)")
            
            # Get recent activity (last 24 hours)
            recent_users = await cls.select_records(
                "users",
                "count(*)",
                filters={"created_at": {"gte": (datetime.utcnow() - timedelta(days=1)).isoformat()}}
            )
            
            return {
                "total_users": users[0].get("count", 0) if users else 0,
                "recent_signups": recent_users[0].get("count", 0) if recent_users else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get Supabase stats", error=str(e))
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }