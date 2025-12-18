# HealthSync AI - Backend with MongoDB Atlas
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from datetime import datetime, timedelta
import uvicorn
import logging
from contextlib import asynccontextmanager
from services.mongodb_atlas_service import get_mongodb_service, close_mongodb_connection, MongoDBAtlasService
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting HealthSync AI with MongoDB Atlas...")
    mongodb_service = await get_mongodb_service()
    if mongodb_service.client:
        logger.info("✅ MongoDB Atlas connected successfully!")
    else:
        logger.warning("⚠️ Running in demo mode - MongoDB Atlas not configured")
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down HealthSync AI...")
    await close_mongodb_connection()
    logger.info("✅ Shutdown complete")

# FastAPI app with MongoDB Atlas
app = FastAPI(
    title="HealthSync AI - MongoDB Atlas Backend",
    description="Healthcare AI platform with MongoDB Atlas database",
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
    """Health check with MongoDB Atlas status"""
    mongodb_service = await get_mongodb_service()
    db_health = await mongodb_service.health_check()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "HealthSync AI Backend is running!",
        "version": "1.0.0",
        "database": db_health
    }

# Authentication endpoints
@app.post("/api/auth/register")
async def register(user_data: dict, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Register new user"""
    try:
        # Check if user already exists
        existing_user = await mongodb.get_user_by_email(user_data.get("email"))
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Create new user
        user_id = str(uuid.uuid4())
        new_user = {
            "user_id": user_id,
            "email": user_data.get("email"),
            "firstName": user_data.get("firstName", ""),
            "lastName": user_data.get("lastName", ""),
            "role": user_data.get("role", "patient"),
            "password_hash": "hashed_password_placeholder",  # In real app, hash the password
            "phone": user_data.get("phone", ""),
            "date_of_birth": user_data.get("dateOfBirth"),
            "gender": user_data.get("gender"),
            "medical_history": [],
            "emergency_contact": user_data.get("emergencyContact", {}),
            "preferences": {
                "notifications": True,
                "language": "en",
                "timezone": "UTC"
            }
        }
        
        result = await mongodb.create_user(new_user)
        
        if result["success"]:
            return {
                "success": True,
                "message": "Registration successful",
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
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login")
async def login(credentials: dict, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """User login"""
    try:
        email = credentials.get("email")
        password = credentials.get("password")
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        # Get user by email
        user = await mongodb.get_user_by_email(email)
        
        if not user:
            # For demo purposes, create a demo user
            user_id = str(uuid.uuid4())
            demo_user = {
                "user_id": user_id,
                "email": email,
                "firstName": "Demo",
                "lastName": "User",
                "role": "doctor" if "doctor" in email else "patient",
                "password_hash": "demo_hash"
            }
            await mongodb.create_user(demo_user)
            user = demo_user
        
        # In real app, verify password hash here
        
        return {
            "success": True,
            "message": "Login successful",
            "user": {
                "id": user["user_id"],
                "email": user["email"],
                "firstName": user["firstName"],
                "lastName": user["lastName"],
                "role": user["role"]
            },
            "token": f"jwt_token_{user['user_id']}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

# Voice AI endpoints
@app.post("/api/voice/start-session")
async def start_voice_session(session_data: dict, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
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
            "diagnosis": None,
            "recommendations": [],
            "duration": 0
        }
        
        result = await mongodb.create_voice_session(voice_session)
        
        return {
            "success": True,
            "session_id": session_id,
            "message": "Voice AI session started",
            "ai_response": "Hello! I'm your AI doctor. I can help analyze your symptoms and provide medical guidance. What brings you here today?",
            "mongodb_result": result
        }
        
    except Exception as e:
        logger.error(f"Voice session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start voice session")

@app.get("/api/voice/session/{session_id}")
async def get_voice_session(session_id: str, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Get voice session details"""
    try:
        session = await mongodb.get_voice_session(session_id)
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "success": True,
            "session": session
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get voice session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")

@app.post("/api/voice/send-audio")
async def process_voice_audio(audio_data: dict, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Process voice audio input"""
    try:
        session_id = audio_data.get("session_id")
        user_message = audio_data.get("message", "I have a headache and feel tired")
        
        # Simulate AI processing
        ai_responses = [
            "I understand you're experiencing a headache and fatigue. Can you tell me how long you've been having these symptoms?",
            "Based on your symptoms, this could be related to stress, dehydration, or lack of sleep. Have you been drinking enough water?",
            "I recommend getting adequate rest, staying hydrated, and monitoring your symptoms. If they persist for more than 2 days, please consult a healthcare provider.",
        ]
        
        import random
        ai_response = random.choice(ai_responses)
        
        # Update session in MongoDB
        if session_id:
            session = await mongodb.get_voice_session(session_id)
            if session:
                conversation = session.get("conversation", [])
                conversation.extend([
                    {"role": "user", "message": user_message, "timestamp": datetime.now().isoformat()},
                    {"role": "ai", "message": ai_response, "timestamp": datetime.now().isoformat()}
                ])
                
                await mongodb.update_voice_session(session_id, {"conversation": conversation})
        
        return {
            "success": True,
            "ai_response": ai_response,
            "session_id": session_id,
            "analysis": {
                "confidence": 0.85,
                "urgency": "low",
                "recommendations": ["Rest", "Hydration", "Monitor symptoms"]
            }
        }
        
    except Exception as e:
        logger.error(f"Voice audio processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio")

# AR Scanner endpoints
@app.post("/api/ar-scanner/scan-document")
async def scan_document(scan_data: dict, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Scan medical document with AR"""
    try:
        scan_id = str(uuid.uuid4())
        user_id = scan_data.get("user_id", "demo_user")
        
        # Simulate OCR and analysis
        extracted_text = "Patient: John Doe\nMedication: Ibuprofen 400mg\nDosage: Take twice daily with food\nDuration: 7 days\nDoctor: Dr. Smith"
        
        analysis = {
            "document_type": "prescription",
            "medications": [
                {
                    "name": "Ibuprofen",
                    "dosage": "400mg",
                    "frequency": "twice daily",
                    "instructions": "with food"
                }
            ],
            "warnings": ["Take with food to avoid stomach irritation"],
            "interactions": []
        }
        
        ar_scan = {
            "scan_id": scan_id,
            "user_id": user_id,
            "document_type": "prescription",
            "extracted_text": extracted_text,
            "analysis": analysis,
            "confidence_score": 0.92,
            "image_url": f"https://example.com/scans/{scan_id}.jpg"
        }
        
        result = await mongodb.create_ar_scan(ar_scan)
        
        return {
            "success": True,
            "scan_id": scan_id,
            "extracted_text": extracted_text,
            "analysis": analysis,
            "confidence_score": 0.92,
            "mongodb_result": result
        }
        
    except Exception as e:
        logger.error(f"AR scan error: {e}")
        raise HTTPException(status_code=500, detail="Failed to scan document")

@app.get("/api/ar-scanner/user-scans")
async def get_user_scans(user_id: str = "demo_user", mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Get user's AR scans"""
    try:
        scans = await mongodb.get_user_ar_scans(user_id, limit=20)
        
        return {
            "success": True,
            "scans": scans,
            "total": len(scans)
        }
        
    except Exception as e:
        logger.error(f"Get user scans error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scans")

# Future Simulator endpoints
@app.post("/api/future-simulator/create-simulation")
async def create_simulation(simulation_data: dict, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Create future health simulation"""
    try:
        simulation_id = str(uuid.uuid4())
        user_id = simulation_data.get("user_id", "demo_user")
        
        # Simulate AI processing
        health_predictions = {
            "cardiovascular_risk": 25,
            "diabetes_risk": 15,
            "life_expectancy": 78,
            "health_score": 75,
            "bmi_projection": 24.5,
            "fitness_level": "good"
        }
        
        lifestyle_scenarios = {
            "improved": {
                "cardiovascular_risk": 15,
                "diabetes_risk": 8,
                "life_expectancy": 82,
                "health_score": 88
            },
            "current": health_predictions,
            "declined": {
                "cardiovascular_risk": 40,
                "diabetes_risk": 28,
                "life_expectancy": 72,
                "health_score": 58
            }
        }
        
        future_simulation = {
            "simulation_id": simulation_id,
            "user_id": user_id,
            "current_age": simulation_data.get("current_age", 35),
            "target_age": simulation_data.get("target_age", 65),
            "health_predictions": health_predictions,
            "lifestyle_scenarios": lifestyle_scenarios,
            "aged_image_url": f"https://via.placeholder.com/400x400/4F46E5/FFFFFF?text=Aged+Photo+{simulation_id[:8]}",
            "ai_narrative": "Based on your current health profile, you're on a positive trajectory for healthy aging. Regular exercise and a balanced diet will significantly improve your long-term health outcomes.",
            "recommendations": [
                "Maintain regular cardiovascular exercise",
                "Follow a Mediterranean-style diet",
                "Get 7-8 hours of quality sleep",
                "Manage stress through mindfulness",
                "Regular health screenings"
            ]
        }
        
        result = await mongodb.create_future_simulation(future_simulation)
        
        return {
            "success": True,
            "simulation_id": simulation_id,
            "aged_image_url": future_simulation["aged_image_url"],
            "health_predictions": health_predictions,
            "lifestyle_scenarios": lifestyle_scenarios,
            "ai_narrative": future_simulation["ai_narrative"],
            "mongodb_result": result
        }
        
    except Exception as e:
        logger.error(f"Future simulation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create simulation")

# Doctor Marketplace endpoints
@app.get("/api/marketplace/doctors")
async def get_doctors(specialty: str = None, location: str = None):
    """Get available doctors"""
    try:
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
                "image": "https://via.placeholder.com/150x150/4F46E5/FFFFFF?text=SJ",
                "experience": "15 years",
                "languages": ["English", "Spanish"],
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
                "image": "https://via.placeholder.com/150x150/10B981/FFFFFF?text=MC",
                "experience": "12 years",
                "languages": ["English", "Mandarin"],
                "next_available": "10:00 AM Tomorrow"
            },
            {
                "id": "doc_003",
                "name": "Dr. Emily Rodriguez",
                "specialty": "Pediatrics",
                "rating": 4.7,
                "reviews": 89,
                "availability": "Available",
                "price": 100,
                "location": "Chicago, IL",
                "image": "https://via.placeholder.com/150x150/F59E0B/FFFFFF?text=ER",
                "experience": "8 years",
                "languages": ["English", "Spanish"],
                "next_available": "4:30 PM Today"
            }
        ]
        
        # Filter by specialty if provided
        if specialty:
            doctors = [d for d in doctors if specialty.lower() in d["specialty"].lower()]
        
        return {
            "success": True,
            "doctors": doctors,
            "total": len(doctors)
        }
        
    except Exception as e:
        logger.error(f"Get doctors error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get doctors")

@app.post("/api/marketplace/book-appointment")
async def book_appointment(appointment_data: dict, mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Book appointment with doctor"""
    try:
        appointment = {
            "appointment_id": str(uuid.uuid4()),
            "patient_id": appointment_data.get("patient_id", "demo_user"),
            "doctor_id": appointment_data.get("doctor_id"),
            "appointment_date": appointment_data.get("appointment_date"),
            "appointment_time": appointment_data.get("appointment_time"),
            "type": appointment_data.get("type", "consultation"),
            "symptoms": appointment_data.get("symptoms", []),
            "notes": appointment_data.get("notes", ""),
            "price": appointment_data.get("price", 150),
            "payment_status": "pending"
        }
        
        result = await mongodb.create_appointment(appointment)
        
        return {
            "success": True,
            "appointment": appointment,
            "message": "Appointment booked successfully",
            "mongodb_result": result
        }
        
    except Exception as e:
        logger.error(f"Book appointment error: {e}")
        raise HTTPException(status_code=500, detail="Failed to book appointment")

# Dashboard endpoints
@app.get("/api/dashboard/stats")
async def get_dashboard_stats(user_id: str = "demo_user", mongodb: MongoDBAtlasService = Depends(get_mongodb_service)):
    """Get dashboard statistics"""
    try:
        # Get platform stats from MongoDB
        platform_stats = await mongodb.get_platform_stats()
        
        # Get user-specific stats
        user_appointments = await mongodb.get_user_appointments(user_id, limit=100)
        user_scans = await mongodb.get_user_ar_scans(user_id, limit=100)
        
        stats = {
            "totalPatients": platform_stats.get("total_users", 156),
            "todayAppointments": len([a for a in user_appointments if a.get("appointment_date") == datetime.now().strftime("%Y-%m-%d")]),
            "monthlyRevenue": 12450,
            "completedConsultations": len([a for a in user_appointments if a.get("status") == "completed"]),
            "averageRating": 4.8,
            "totalScans": len(user_scans),
            "activeVoiceSessions": platform_stats.get("recent_sessions", 5),
            "totalSimulations": platform_stats.get("total_simulations", 23)
        }
        
        return {
            "success": True,
            "stats": stats,
            "platform_stats": platform_stats
        }
        
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        # Return demo data if MongoDB fails
        return {
            "success": True,
            "stats": {
                "totalPatients": 156,
                "todayAppointments": 12,
                "monthlyRevenue": 12450,
                "completedConsultations": 89,
                "averageRating": 4.8,
                "totalScans": 45,
                "activeVoiceSessions": 8,
                "totalSimulations": 23
            },
            "note": "Demo data - MongoDB connection issue"
        }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to HealthSync AI with MongoDB Atlas!",
        "status": "running",
        "database": "MongoDB Atlas",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)