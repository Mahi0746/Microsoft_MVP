# HealthSync AI - MongoDB Atlas Service
import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class MongoDBAtlasService:
    """MongoDB Atlas service for HealthSync AI"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.connection_string = os.getenv("MONGODB_URL")
        self.database_name = "healthsync_db"
        
    async def connect(self):
        """Connect to MongoDB Atlas"""
        try:
            if not self.connection_string or "temp_" in self.connection_string:
                logger.warning("Using demo mode - MongoDB Atlas not configured")
                return False
                
            self.client = AsyncIOMotorClient(self.connection_string)
            self.database = self.client[self.database_name]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("✅ Connected to MongoDB Atlas successfully!")
            
            # Initialize collections
            await self.initialize_collections()
            return True
            
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB Atlas: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ MongoDB Atlas connection error: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MongoDB Atlas"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB Atlas")
    
    async def initialize_collections(self):
        """Initialize required collections with indexes"""
        try:
            collections = [
                "users",
                "voice_sessions", 
                "ar_scans",
                "therapy_sessions",
                "doctor_profiles",
                "appointments",
                "future_simulations",
                "health_records",
                "family_graphs",
                "marketplace_listings"
            ]
            
            for collection_name in collections:
                collection = self.database[collection_name]
                
                # Create indexes based on collection type
                if collection_name == "users":
                    await collection.create_index("email", unique=True)
                    await collection.create_index("user_id", unique=True)
                elif collection_name == "voice_sessions":
                    await collection.create_index("user_id")
                    await collection.create_index("session_id", unique=True)
                elif collection_name == "ar_scans":
                    await collection.create_index("user_id")
                    await collection.create_index("scan_id", unique=True)
                elif collection_name == "appointments":
                    await collection.create_index("patient_id")
                    await collection.create_index("doctor_id")
                    await collection.create_index("appointment_date")
                
                logger.info(f"✅ Initialized collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Error initializing collections: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MongoDB Atlas connection health"""
        try:
            if not self.client:
                return {"status": "disconnected", "error": "No client connection"}
            
            # Ping the database
            result = await self.client.admin.command('ping')
            
            # Get database stats
            stats = await self.database.command("dbStats")
            
            return {
                "status": "healthy",
                "ping": result,
                "database": self.database_name,
                "collections": stats.get("collections", 0),
                "data_size": stats.get("dataSize", 0),
                "storage_size": stats.get("storageSize", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    # User Management
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user"""
        try:
            user_data["created_at"] = datetime.now()
            user_data["updated_at"] = datetime.now()
            
            result = await self.database.users.insert_one(user_data)
            
            return {
                "success": True,
                "user_id": str(result.inserted_id),
                "message": "User created successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            user = await self.database.users.find_one({"user_id": user_id})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            user = await self.database.users.find_one({"email": email})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"Error getting user by email: {e}")
            return None
    
    # Voice Sessions
    async def create_voice_session(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a voice AI session"""
        try:
            session_data["created_at"] = datetime.now()
            session_data["status"] = "active"
            
            result = await self.database.voice_sessions.insert_one(session_data)
            
            return {
                "success": True,
                "session_id": session_data["session_id"],
                "mongo_id": str(result.inserted_id)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_voice_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get voice session by ID"""
        try:
            session = await self.database.voice_sessions.find_one({"session_id": session_id})
            if session:
                session["_id"] = str(session["_id"])
            return session
        except Exception as e:
            logger.error(f"Error getting voice session: {e}")
            return None
    
    async def update_voice_session(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """Update voice session"""
        try:
            update_data["updated_at"] = datetime.now()
            result = await self.database.voice_sessions.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating voice session: {e}")
            return False
    
    # AR Scans
    async def create_ar_scan(self, scan_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create AR scan record"""
        try:
            scan_data["created_at"] = datetime.now()
            result = await self.database.ar_scans.insert_one(scan_data)
            
            return {
                "success": True,
                "scan_id": scan_data["scan_id"],
                "mongo_id": str(result.inserted_id)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_ar_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get AR scan by ID"""
        try:
            scan = await self.database.ar_scans.find_one({"scan_id": scan_id})
            if scan:
                scan["_id"] = str(scan["_id"])
            return scan
        except Exception as e:
            logger.error(f"Error getting AR scan: {e}")
            return None
    
    async def get_user_ar_scans(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's AR scans"""
        try:
            cursor = self.database.ar_scans.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
            scans = []
            async for scan in cursor:
                scan["_id"] = str(scan["_id"])
                scans.append(scan)
            return scans
        except Exception as e:
            logger.error(f"Error getting user AR scans: {e}")
            return []
    
    # Future Simulations
    async def create_future_simulation(self, simulation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create future simulation record"""
        try:
            simulation_data["created_at"] = datetime.now()
            result = await self.database.future_simulations.insert_one(simulation_data)
            
            return {
                "success": True,
                "simulation_id": simulation_data["simulation_id"],
                "mongo_id": str(result.inserted_id)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_future_simulation(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get future simulation by ID"""
        try:
            simulation = await self.database.future_simulations.find_one({"simulation_id": simulation_id})
            if simulation:
                simulation["_id"] = str(simulation["_id"])
            return simulation
        except Exception as e:
            logger.error(f"Error getting future simulation: {e}")
            return None
    
    # Appointments
    async def create_appointment(self, appointment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create appointment"""
        try:
            appointment_data["created_at"] = datetime.now()
            appointment_data["status"] = "scheduled"
            
            result = await self.database.appointments.insert_one(appointment_data)
            
            return {
                "success": True,
                "appointment_id": str(result.inserted_id)
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_user_appointments(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get user's appointments"""
        try:
            cursor = self.database.appointments.find(
                {"$or": [{"patient_id": user_id}, {"doctor_id": user_id}]}
            ).sort("appointment_date", -1).limit(limit)
            
            appointments = []
            async for appointment in cursor:
                appointment["_id"] = str(appointment["_id"])
                appointments.append(appointment)
            return appointments
        except Exception as e:
            logger.error(f"Error getting user appointments: {e}")
            return []
    
    # Analytics
    async def get_platform_stats(self) -> Dict[str, Any]:
        """Get platform statistics"""
        try:
            stats = {}
            
            # Count documents in each collection
            stats["total_users"] = await self.database.users.count_documents({})
            stats["total_voice_sessions"] = await self.database.voice_sessions.count_documents({})
            stats["total_ar_scans"] = await self.database.ar_scans.count_documents({})
            stats["total_appointments"] = await self.database.appointments.count_documents({})
            stats["total_simulations"] = await self.database.future_simulations.count_documents({})
            
            # Recent activity (last 24 hours)
            from datetime import timedelta
            yesterday = datetime.now() - timedelta(days=1)
            
            stats["recent_users"] = await self.database.users.count_documents({"created_at": {"$gte": yesterday}})
            stats["recent_sessions"] = await self.database.voice_sessions.count_documents({"created_at": {"$gte": yesterday}})
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting platform stats: {e}")
            return {}

# Global MongoDB Atlas service instance
mongodb_service = MongoDBAtlasService()

async def get_mongodb_service() -> MongoDBAtlasService:
    """Get MongoDB Atlas service instance"""
    if not mongodb_service.client:
        await mongodb_service.connect()
    return mongodb_service

async def close_mongodb_connection():
    """Close MongoDB Atlas connection"""
    await mongodb_service.disconnect()