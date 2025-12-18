# HealthSync AI - Database Service (MongoDB Atlas Only)
import asyncio
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
import structlog

from config import settings


logger = structlog.get_logger(__name__)


class DatabaseService:
    """Centralized database service for MongoDB Atlas and Redis."""
    
    # Connection pools
    _mongodb_client: Optional[AsyncIOMotorClient] = None
    _redis_client: Optional[redis.Redis] = None
    
    @classmethod
    async def initialize(cls):
        """Initialize all database connections."""
        
        logger.info("Initializing database connections")
        
        try:
            # Initialize MongoDB client (Primary Database)
            await cls._init_mongodb()
            logger.info("MongoDB client initialized")
            
            # Initialize Redis client (Optional)
            await cls._init_redis()
            logger.info("Redis client initialized (optional)")
            
            # Test all connections
            await cls.health_check()
            logger.info("All database connections verified")
            
        except Exception as e:
            logger.error("Failed to initialize database connections", error=str(e))
            raise
    
    
    @classmethod
    async def _init_mongodb(cls):
        """Initialize MongoDB client."""
        
        try:
            # Skip if using localhost or demo MongoDB URL
            if "localhost" in settings.mongodb_url and "mongodb+srv://" not in settings.mongodb_url:
                logger.warning("MongoDB Atlas not configured, using local MongoDB or skipping")
                # Try to connect anyway for local development
                pass
            
            cls._mongodb_client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=10,
                minPoolSize=2,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=10000  # Increased timeout
            )
            
            # Test connection
            await cls._mongodb_client.admin.command('ping')
            logger.info("MongoDB connection established successfully")
            
        except Exception as e:
            logger.error("MongoDB connection failed", error=str(e))
            logger.warning("App will continue but MongoDB features may not work")
            # Don't raise - allow app to continue
            cls._mongodb_client = None
    
    @classmethod
    async def _init_redis(cls):
        """Initialize Redis client."""
        
        try:
            cls._redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
                retry_on_timeout=True,
                socket_connect_timeout=5
            )
            
            # Test connection
            await cls._redis_client.ping()
            logger.info("Redis connection established successfully")
            
        except Exception as e:
            logger.warning("Redis connection failed, caching disabled", error=str(e))
            # Don't raise - caching is optional
            cls._redis_client = None
    
    @classmethod
    async def close(cls):
        """Close all database connections."""
        
        logger.info("Closing database connections")
        
        try:
            # Close MongoDB client
            if cls._mongodb_client:
                cls._mongodb_client.close()
                cls._mongodb_client = None
                logger.info("MongoDB client closed")
            
            # Close Redis client
            if cls._redis_client:
                await cls._redis_client.close()
                cls._redis_client = None
                logger.info("Redis client closed")
                
        except Exception as e:
            logger.error("Error closing database connections", error=str(e))
    
    @classmethod
    async def health_check(cls):
        """Check health of all database connections."""
        
        health_status = {}
        
        # Check MongoDB (Required)
        if cls._mongodb_client:
            try:
                await cls._mongodb_client.admin.command('ping')
                health_status["mongodb"] = "healthy"
            except Exception as e:
                health_status["mongodb"] = f"error: {str(e)}"
                raise Exception(f"MongoDB connection failed: {str(e)}")
        else:
            health_status["mongodb"] = "not_configured"
            raise Exception("MongoDB client not initialized")
        
        # Check Redis (Optional)
        if cls._redis_client:
            try:
                pong = await cls._redis_client.ping()
                health_status["redis"] = "healthy" if pong else "unhealthy"
            except Exception as e:
                health_status["redis"] = f"error: {str(e)}"
        else:
            health_status["redis"] = "not_configured"
        
        return health_status
    
    # =============================================================================
    # MONGODB METHODS
    # =============================================================================
    
    @classmethod
    def get_mongodb_database(cls):
        """Get MongoDB database instance."""
        
        if not cls._mongodb_client:
            raise Exception("MongoDB client not initialized")
        
        return cls._mongodb_client[settings.mongodb_database]
    
    @classmethod
    def get_mongodb_collection(cls, collection_name: str):
        """Get MongoDB collection instance."""
        
        db = cls.get_mongodb_database()
        return db[collection_name]
    
    @classmethod
    async def mongodb_find_one(
        cls,
        collection_name: str,
        filter_dict: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None
    ) -> Optional[Dict[str, Any]]:
        """Find one document in MongoDB collection."""
        
        collection = cls.get_mongodb_collection(collection_name)
        return await collection.find_one(filter_dict, projection)
    
    @classmethod
    async def mongodb_find_many(
        cls,
        collection_name: str,
        filter_dict: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        limit: Optional[int] = None,
        sort: Optional[List[tuple]] = None
    ) -> List[Dict[str, Any]]:
        """Find multiple documents in MongoDB collection."""
        
        collection = cls.get_mongodb_collection(collection_name)
        cursor = collection.find(filter_dict, projection)
        
        if sort:
            cursor = cursor.sort(sort)
        
        if limit:
            cursor = cursor.limit(limit)
        
        return await cursor.to_list(length=limit)
    
    @classmethod
    async def mongodb_insert_one(
        cls,
        collection_name: str,
        document: Dict[str, Any]
    ) -> str:
        """Insert one document into MongoDB collection."""
        
        collection = cls.get_mongodb_collection(collection_name)
        result = await collection.insert_one(document)
        return str(result.inserted_id)
    
    @classmethod
    async def mongodb_update_one(
        cls,
        collection_name: str,
        filter_dict: Dict[str, Any],
        update_dict: Dict[str, Any],
        upsert: bool = False
    ) -> bool:
        """Update one document in MongoDB collection."""
        
        collection = cls.get_mongodb_collection(collection_name)
        result = await collection.update_one(filter_dict, update_dict, upsert=upsert)
        return result.modified_count > 0
    
    @classmethod
    async def mongodb_delete_one(
        cls,
        collection_name: str,
        filter_dict: Dict[str, Any]
    ) -> bool:
        """Delete one document from MongoDB collection."""
        
        collection = cls.get_mongodb_collection(collection_name)
        result = await collection.delete_one(filter_dict)
        return result.deleted_count > 0
    
    # =============================================================================
    # REDIS METHODS
    # =============================================================================
    
    @classmethod
    def get_redis_client(cls) -> redis.Redis:
        """Get Redis client instance."""
        
        if not cls._redis_client:
            raise Exception("Redis client not initialized")
        
        return cls._redis_client
    
    @classmethod
    async def redis_get(cls, key: str) -> Optional[str]:
        """Get value from Redis."""
        
        client = cls.get_redis_client()
        return await client.get(key)
    
    @classmethod
    async def redis_set(
        cls,
        key: str,
        value: str,
        expire: Optional[int] = None
    ) -> bool:
        """Set value in Redis with optional expiration."""
        
        client = cls.get_redis_client()
        return await client.set(key, value, ex=expire)
    
    @classmethod
    async def redis_delete(cls, key: str) -> bool:
        """Delete key from Redis."""
        
        client = cls.get_redis_client()
        return await client.delete(key) > 0
    
    @classmethod
    async def redis_exists(cls, key: str) -> bool:
        """Check if key exists in Redis."""
        
        client = cls.get_redis_client()
        return await client.exists(key) > 0
    
    @classmethod
    async def redis_expire(cls, key: str, seconds: int) -> bool:
        """Set expiration for Redis key."""
        
        client = cls.get_redis_client()
        return await client.expire(key, seconds)
    
    # =============================================================================
    # CACHING UTILITIES
    # =============================================================================
    
    @classmethod
    async def get_cached_or_fetch(
        cls,
        cache_key: str,
        fetch_function,
        expire_seconds: int = 3600,
        *args,
        **kwargs
    ) -> Any:
        """Get data from cache or fetch and cache it."""
        
        # Try to get from cache first
        cached_value = await cls.redis_get(cache_key)
        
        if cached_value is not None:
            logger.debug("Cache hit", key=cache_key)
            # Deserialize if needed (assuming JSON for now)
            import json
            try:
                return json.loads(cached_value)
            except json.JSONDecodeError:
                return cached_value
        
        # Cache miss - fetch data
        logger.debug("Cache miss", key=cache_key)
        
        if asyncio.iscoroutinefunction(fetch_function):
            data = await fetch_function(*args, **kwargs)
        else:
            data = fetch_function(*args, **kwargs)
        
        # Cache the result
        if data is not None:
            import json
            try:
                serialized_data = json.dumps(data, default=str)
                await cls.redis_set(cache_key, serialized_data, expire_seconds)
            except (TypeError, ValueError):
                # If data can't be serialized, store as string
                await cls.redis_set(cache_key, str(data), expire_seconds)
        
        return data
    
    @classmethod
    async def invalidate_cache_pattern(cls, pattern: str):
        """Invalidate all cache keys matching pattern."""
        
        client = cls.get_redis_client()
        
        # Find all keys matching pattern
        keys = []
        cursor = 0
        
        while True:
            cursor, batch = await client.scan(cursor, match=pattern, count=100)
            keys.extend(batch)
            if cursor == 0:
                break
        
        # Delete all matching keys
        if keys:
            await client.delete(*keys)
            logger.info("Cache invalidated", pattern=pattern, keys_deleted=len(keys))
    
    # =============================================================================
    # HEALTH DATA SPECIFIC METHODS
    # =============================================================================
    
    @classmethod
    async def get_user_health_summary(cls, user_id: str) -> Dict[str, Any]:
        """Get comprehensive health summary for user."""
        
        cache_key = f"health_summary:{user_id}"
        
        async def fetch_health_summary():
            # Get health metrics from MongoDB
            health_metrics = await cls.mongodb_find_many(
                "health_metrics",
                {"user_id": user_id},
                sort=[("measured_at", -1)],
                limit=50
            )
            
            # Get predictions from MongoDB
            predictions = await cls.mongodb_find_many(
                "predictions",
                {"user_id": user_id},
                sort=[("created_at", -1)]
            )
            
            # Get family health graph
            family_graph = await cls.mongodb_find_one(
                "family_graph",
                {"user_id": user_id}
            )
            
            return {
                "user_id": user_id,
                "health_metrics": health_metrics,
                "predictions": predictions,
                "family_graph": family_graph,
                "last_updated": "2025-12-17T10:30:00Z"
            }
        
        return await cls.get_cached_or_fetch(
            cache_key,
            fetch_health_summary,
            expire_seconds=settings.cache_ttl_health_data
        )
    
    @classmethod
    async def store_health_event(
        cls,
        user_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        ai_analysis: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store health event in MongoDB with optional AI analysis."""
        
        from datetime import datetime
        
        event_document = {
            "user_id": user_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow(),
            "data": event_data,
            "source": "api",
            "metadata": {
                "api_version": settings.api_version,
                "created_via": "backend_service"
            }
        }
        
        if ai_analysis:
            event_document["ai_analysis"] = ai_analysis
        
        event_id = await cls.mongodb_insert_one("health_events", event_document)
        
        # Invalidate related caches
        await cls.invalidate_cache_pattern(f"health_summary:{user_id}")
        await cls.invalidate_cache_pattern(f"health_events:{user_id}:*")
        
        logger.info(
            "Health event stored",
            user_id=user_id,
            event_type=event_type,
            event_id=event_id
        )
        
        return event_id


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

# Create global instances for easy access
mongodb_client = None
redis_client = None


async def get_mongodb_client():
    """Get global MongoDB client."""
    return DatabaseService._mongodb_client


async def get_redis_client():
    """Get global Redis client."""
    return DatabaseService._redis_client


# =============================================================================
# DATABASE UTILITIES
# =============================================================================

async def execute_with_retry(
    operation,
    max_retries: int = 3,
    delay_seconds: float = 1.0
) -> Any:
    """Execute database operation with retry logic."""
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            if asyncio.iscoroutinefunction(operation):
                return await operation()
            else:
                return operation()
                
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                logger.warning(
                    "Database operation failed, retrying",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e)
                )
                await asyncio.sleep(delay_seconds * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(
                    "Database operation failed after all retries",
                    attempts=max_retries,
                    error=str(e)
                )
    
    raise last_exception


async def batch_insert_postgres(
    table_name: str,
    columns: List[str],
    data: List[List[Any]],
    batch_size: int = 1000
) -> int:
    """Batch insert data into PostgreSQL table."""
    
    if not data:
        return 0
    
    total_inserted = 0
    placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
    query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
    
    async with DatabaseService.postgres_transaction() as conn:
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            for row in batch:
                await conn.execute(query, *row)
                total_inserted += 1
    
    logger.info(
        "Batch insert completed",
        table=table_name,
        rows_inserted=total_inserted
    )
    
    return total_inserted