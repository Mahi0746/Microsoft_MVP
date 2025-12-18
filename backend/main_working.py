# HealthSync AI - Working Backend (Handles Missing API Keys)
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from datetime import datetime, timedelta
import uvicorn
import logging
from contextlib import asynccontextmanager
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if MongoDB Atlas is configured
def check_mongodb_atlas():
    mongodb_url = os.getenv("MONGODB_URL", "")
    if mongodb_url and "mongodb+srv://" in mongodb_url and "cluster" in mongodb_url:
        return True
    return False

# Initialize MongoDB Atlas if available
mongodb_service = None
if check_mongodb_atlas():
    try:
        from services.mongodb_atlas_service import get_mongodb_service, close_mongodb_connection
        logger.info("✅ MongoDB Atlas configuration detected")
    except ImportError:
        logger.warning("⚠️ MongoDB Atlas service not available, using demo mode")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting HealthSync AI...")
    
    global mongodb_service
    if check_mongodb_atlas():
        try:
            mongodb_service = await get_mongodb_service()
            if mongodb_service and mongodb_service.client:
                logger.info("✅ MongoDB Atlas connected successfully!")
            else:
                logger.warning("⚠️ MongoDB Atlas connection failed, using demo mode")
                mongodb_service = None
        except Exception as e:
            logger.warning(f"⚠️ MongoDB Atlas error: {e}, using demo mode")
            mongodb_service = None
    else:
        logger.info("ℹ️ MongoDB Atlas not configured, using demo mode")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down HealthSync AI...")
    if mongodb_service:
        try:
            await close_mongodb_connection()
        except:
            pass
    logger.info("✅ Shutdown complete")

# FastAPI app
app = FastAPI(
    title="HealthSync AI - Working Backend",
    description="Healthcare AI platform that works with or without API keys",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:19006", "exp://192.168.*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check with system status"""
    
    # Check environment variables
    env_status = {
        "mongodb_atlas": check_mongodb_atlas(),
        "groq_api": bool(os.getenv("GROQ_API_KEY", "").startswith("gsk_")),
        "replicate_api": bool(os.getenv("REPLICATE_API_TOKEN", "").startswith("r8_")),
        "huggingface_api": bool(os.getenv("HUGGINGFACE_API_KEY", "").startswith("hf_")),
        "supabase": bool(os.getenv("SUPABASE_URL", "").startswith("https://")),
    }
    
    # Database health
    db_health = {"status": "demo_mode", "type": "in_memory"}
    if mongodb_service:
        try:
            db_health = await mongodb_service.health_check()
        except:
            db_health = {"status": "error", "type": "mongodb_atlas"}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "HealthSync AI Backend is running!",
        "version": "1.0.0",
        "environment": env_status,
        "database": db_health,
        "mode": "production" if any(env_status.values()) else "demo"
    }

# In-memory storage for demo mode
demo_storage = {
    "users": {},
    "voice_sessions": {},
    "ar_scans": {},
    "appointments": {},
    "simulations": {}
}

# Helper function to get or create demo user
def get_or_create_demo_user(email: str, role: str = "patient"):
    if email not in demo_storage["users"]:
        user_id = str(uuid.uuid4())
        demo_storage["users"][email] = {
            "user_id": user_id,
            "email": email,
            "firstName": "Demo",
            "lastName": "User",
            "role": role,
            "created_at": datetime.now().isoformat()
        }
    return demo_storage["users"][email]

# Authentication endpoints
@app.post("/api/auth/register")
async def register(user_data: dict):
    """Register new user"""
    try:
        email = user_data.get("email", "")
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Check if using MongoDB Atlas
        if mongodb_service:
            try:
                # Check if user already exists
                existing_user = await mongodb_service.get_user_by_email(email)
                if existing_user:
                    raise HTTPException(status_code=400, detail="User already exists")
                
                # Create new user in MongoDB
                user_id = str(uuid.uuid4())
                new_user = {
                    "user_id": user_id,
                    "email": email,
                    "firstName": user_data.get("firstName", ""),
                    "lastName": user_data.get("lastName", ""),
                    "role": user_data.get("role", "patient"),
                    "password_hash": "hashed_password_placeholder",
                    "phone": user_data.get("phone", ""),
                    "date_of_birth": user_data.get("dateOfBirth"),
                    "gender": user_data.get("gender"),
                }
                
                result = await mongodb_service.create_user(new_user)
                
                if result["success"]:
                    return {
                        "success": True,
                        "message": "Registration successful (MongoDB Atlas)",
                        "user": {
                            "id": user_id,
                            "email": new_user["email"],
                            "firstName": new_user["firstName"],
                            "lastName": new_user["lastName"],
                            "role": new_user["role"]
                        },
                        "token": f"jwt_token_{user_id}"
                    }
                else:
                    raise HTTPException(status_code=500, detail=result["error"])
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"MongoDB registration error: {e}")
                # Fall back to demo mode
                pass
        
        # Demo mode registration
        if email in demo_storage["users"]:
            raise HTTPException(status_code=400, detail="User already exists (demo mode)")
        
        user = get_or_create_demo_user(email, user_data.get("role", "patient"))
        user.update({
            "firstName": user_data.get("firstName", "Demo"),
            "lastName": user_data.get("lastName", "User"),
        })
        
        return {
            "success": True,
            "message": "Registration successful (demo mode)",
            "user": {
                "id": user["user_id"],
                "email": user["email"],
                "firstName": user["firstName"],
                "lastName": user["lastName"],
                "role": user["role"]
            },
            "token": f"demo_jwt_token_{user['user_id']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login")
async def login(credentials: dict):
    """User login"""
    try:
        email = credentials.get("email", "")
        password = credentials.get("password", "")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email is required")
        
        # Check MongoDB Atlas first
        if mongodb_service:
            try:
                user = await mongodb_service.get_user_by_email(email)
                if user:
                    return {
                        "success": True,
                        "message": "Login successful (MongoDB Atlas)",
                        "user": {
                            "id": user["user_id"],
                            "email": user["email"],
                            "firstName": user["firstName"],
                            "lastName": user["lastName"],
                            "role": user["role"]
                        },
                        "token": f"jwt_token_{user['user_id']}"
                    }
            except Exception as e:
                logger.error(f"MongoDB login error: {e}")
        
        # Demo mode login - create user if doesn't exist
        role = "doctor" if "doctor" in email.lower() else "patient"
        user = get_or_create_demo_user(email, role)
        
        return {
            "success": True,
            "message": "Login successful (demo mode)",
            "user": {
                "id": user["user_id"],
                "email": user["email"],
                "firstName": user["firstName"],
                "lastName": user["lastName"],
                "role": user["role"]
            },
            "token": f"demo_jwt_token_{user['user_id']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# Voice AI endpoints
@app.post("/api/voice/start-session")
async def start_voice_session(session_data: dict):
    """Start voice AI session"""
    try:
        session_id = str(uuid.uuid4())
        user_id = session_data.get("user_id", "demo_user")
        
        voice_session = {
            "session_id": session_id,
            "user_id": user_id,
            "status": "active",
            "conversation": [],
            "symptoms": session_data.get("symptoms", []),
            "created_at": datetime.now().isoformat()
        }
        
        # Try MongoDB Atlas first
        if mongodb_service:
            try:
                result = await mongodb_service.create_voice_session(voice_session)
                if result["success"]:
                    return {
                        "success": True,
                        "session_id": session_id,
                        "message": "Voice AI session started (MongoDB Atlas)",
                        "ai_response": "Hello! I'm your AI doctor. How can I help you today?",
                        "storage": "mongodb_atlas"
                    }
            except Exception as e:
                logger.error(f"MongoDB voice session error: {e}")
        
        # Demo mode storage
        demo_storage["voice_sessions"][session_id] = voice_session
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Voice AI session started (demo mode)",
            "ai_response": "Hello! I'm your AI doctor. I can help analyze your symptoms. What brings you here today? (Demo mode - connect real AI for advanced features)",
            "storage": "demo_mode"
        }
        
    except Exception as e:
        logger.error(f"Voice session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start voice session")

@app.get("/api/voice/session/{session_id}")
async def get_voice_session(session_id: str):
    """Get voice session details"""
    try:
        # Try MongoDB Atlas first
        if mongodb_service:
            try:
                session = await mongodb_service.get_voice_session(session_id)
                if session:
                    return {"success": True, "session": session, "storage": "mongodb_atlas"}
            except Exception as e:
                logger.error(f"MongoDB get session error: {e}")
        
        # Demo mode
        if session_id in demo_storage["voice_sessions"]:
            return {
                "success": True,
                "session": demo_storage["voice_sessions"][session_id],
                "storage": "demo_mode"
            }
        
        raise HTTPException(status_code=404, detail="Session not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get voice session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")

# AR Scanner endpoints
@app.post("/api/ar-scanner/scan-document")
async def scan_document(scan_data: dict):
    """Scan medical document with AR"""
    try:
        scan_id = str(uuid.uuid4())
        user_id = scan_data.get("user_id", "demo_user")
        
        # Simulate OCR results
        extracted_text = "Patient: John Doe\nMedication: Ibuprofen 400mg\nDosage: Take twice daily with food\nDuration: 7 days\nDoctor: Dr. Smith"
        
        analysis = {
            "document_type": "prescription",
            "medications": [{"name": "Ibuprofen", "dosage": "400mg", "frequency": "twice daily"}],
            "warnings": ["Take with food"],
            "confidence": 0.92
        }
        
        ar_scan = {
            "scan_id": scan_id,
            "user_id": user_id,
            "extracted_text": extracted_text,
            "analysis": analysis,
            "created_at": datetime.now().isoformat()
        }
        
        # Try MongoDB Atlas first
        if mongodb_service:
            try:
                result = await mongodb_service.create_ar_scan(ar_scan)
                if result["success"]:
                    return {
                        "success": True,
                        "scan_id": scan_id,
                        "extracted_text": extracted_text,
                        "analysis": analysis,
                        "storage": "mongodb_atlas"
                    }
            except Exception as e:
                logger.error(f"MongoDB AR scan error: {e}")
        
        # Demo mode storage
        demo_storage["ar_scans"][scan_id] = ar_scan
        
        return {
            "success": True,
            "scan_id": scan_id,
            "extracted_text": extracted_text,
            "analysis": analysis,
            "storage": "demo_mode",
            "note": "Demo OCR - connect real AI services for advanced document analysis"
        }
        
    except Exception as e:
        logger.error(f"AR scan error: {e}")
        raise HTTPException(status_code=500, detail="Failed to scan document")

# Doctor Marketplace endpoints
@app.get("/api/marketplace/doctors")
async def get_doctors(specialty: str = None):
    """Get available doctors"""
    doctors = [
        {
            "id": "doc_001",
            "name": "Dr. Sarah Johnson",
            "specialty": "Cardiology",
            "rating": 4.8,
            "reviews": 156,
            "availability": "Available Today",
            "price": 150,
            "location": "New York, NY",
            "experience": "15 years",
            "next_available": "2:00 PM Today"
        },
        {
            "id": "doc_002",
            "name": "Dr. Michael Chen",
            "specialty": "Dermatology",
            "rating": 4.9,
            "reviews": 203,
            "availability": "Available Tomorrow",
            "price": 120,
            "location": "Los Angeles, CA",
            "experience": "12 years",
            "next_available": "10:00 AM Tomorrow"
        }
    ]
    
    if specialty:
        doctors = [d for d in doctors if specialty.lower() in d["specialty"].lower()]
    
    return {"success": True, "doctors": doctors, "total": len(doctors)}

# Dashboard endpoints
@app.get("/api/dashboard/stats")
async def get_dashboard_stats(user_id: str = "demo_user"):
    """Get dashboard statistics"""
    try:
        # Try to get real stats from MongoDB
        if mongodb_service:
            try:
                platform_stats = await mongodb_service.get_platform_stats()
                return {
                    "success": True,
                    "stats": {
                        "totalPatients": platform_stats.get("total_users", 156),
                        "todayAppointments": 12,
                        "monthlyRevenue": 12450,
                        "completedConsultations": platform_stats.get("total_voice_sessions", 89),
                        "averageRating": 4.8,
                        "totalScans": platform_stats.get("total_ar_scans", 45),
                        "totalSimulations": platform_stats.get("total_simulations", 23)
                    },
                    "storage": "mongodb_atlas",
                    "platform_stats": platform_stats
                }
            except Exception as e:
                logger.error(f"MongoDB stats error: {e}")
        
        # Demo mode stats
        return {
            "success": True,
            "stats": {
                "totalPatients": len(demo_storage["users"]),
                "todayAppointments": 12,
                "monthlyRevenue": 12450,
                "completedConsultations": len(demo_storage["voice_sessions"]),
                "averageRating": 4.8,
                "totalScans": len(demo_storage["ar_scans"]),
                "totalSimulations": len(demo_storage["simulations"])
            },
            "storage": "demo_mode"
        }
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return {
            "success": True,
            "stats": {
                "totalPatients": 156,
                "todayAppointments": 12,
                "monthlyRevenue": 12450,
                "completedConsultations": 89,
                "averageRating": 4.8
            },
            "storage": "fallback"
        }

# Future Simulator endpoint
@app.post("/api/future-simulator/create-simulation")
async def create_simulation(simulation_data: dict):
    """Create future health simulation"""
    try:
        simulation_id = str(uuid.uuid4())
        
        simulation = {
            "simulation_id": simulation_id,
            "user_id": simulation_data.get("user_id", "demo_user"),
            "aged_image_url": f"https://via.placeholder.com/400x400/4F46E5/FFFFFF?text=Future+You+{simulation_id[:8]}",
            "health_predictions": {
                "cardiovascular_risk": 25,
                "diabetes_risk": 15,
                "life_expectancy": 78,
                "health_score": 75
            },
            "ai_narrative": "Based on your current health profile, you're on track for healthy aging! (Demo analysis)",
            "created_at": datetime.now().isoformat()
        }
        
        # Try MongoDB first
        if mongodb_service:
            try:
                result = await mongodb_service.create_future_simulation(simulation)
                if result["success"]:
                    return {**simulation, "success": True, "storage": "mongodb_atlas"}
            except Exception as e:
                logger.error(f"MongoDB simulation error: {e}")
        
        # Demo mode
        demo_storage["simulations"][simulation_id] = simulation
        return {**simulation, "success": True, "storage": "demo_mode"}
        
    except Exception as e:
        logger.error(f"Future simulation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create simulation")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to HealthSync AI!",
        "status": "running",
        "database": "MongoDB Atlas" if mongodb_service else "Demo Mode",
        "docs": "/docs",
        "health": "/health",
        "note": "This backend works with or without API keys!"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)