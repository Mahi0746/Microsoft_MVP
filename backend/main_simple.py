# HealthSync AI - Simplified Backend for Quick Start
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import uvicorn

# Simple FastAPI app without complex dependencies
app = FastAPI(
    title="HealthSync AI - Simple Backend",
    description="Simplified backend for quick testing",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:19006"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "HealthSync AI Backend is running!",
        "version": "1.0.0"
    }

# Simple authentication endpoints
@app.post("/api/auth/login")
async def login(credentials: dict):
    return {
        "success": True,
        "message": "Login successful (demo mode)",
        "user": {
            "id": "demo_user_123",
            "email": credentials.get("email", "demo@healthsync.ai"),
            "firstName": "Demo",
            "lastName": "User",
            "role": "doctor"
        },
        "token": "demo_jwt_token_12345"
    }

@app.post("/api/auth/register")
async def register(user_data: dict):
    return {
        "success": True,
        "message": "Registration successful (demo mode)",
        "user": {
            "id": "demo_user_456",
            "email": user_data.get("email", "new@healthsync.ai"),
            "firstName": user_data.get("firstName", "New"),
            "lastName": user_data.get("lastName", "User"),
            "role": "patient"
        }
    }

# Simple voice AI endpoint
@app.post("/api/voice/start-session")
async def start_voice_session():
    return {
        "success": True,
        "session_id": "demo_voice_session_789",
        "message": "Voice AI session started (demo mode)",
        "ai_response": "Hello! I'm your AI doctor. How can I help you today? (This is demo mode - connect real AI services for full functionality)"
    }

# Simple AR scanner endpoint
@app.post("/api/ar-scanner/scan-document")
async def scan_document():
    return {
        "success": True,
        "scan_id": "demo_scan_101",
        "message": "Document scanned successfully (demo mode)",
        "extracted_text": "Sample medical document text extracted via OCR (demo)",
        "analysis": "This appears to be a medical prescription (demo analysis)"
    }

# Simple marketplace endpoint
@app.get("/api/marketplace/doctors")
async def get_doctors():
    return {
        "success": True,
        "doctors": [
            {
                "id": "doc_001",
                "name": "Dr. Sarah Johnson",
                "specialty": "Cardiology",
                "rating": 4.8,
                "availability": "Available",
                "price": "$150/consultation"
            },
            {
                "id": "doc_002", 
                "name": "Dr. Michael Chen",
                "specialty": "Dermatology",
                "rating": 4.9,
                "availability": "Busy",
                "price": "$120/consultation"
            }
        ]
    }

# Simple future simulator endpoint
@app.post("/api/future-simulator/create-simulation")
async def create_simulation():
    return {
        "success": True,
        "simulation_id": "sim_demo_202",
        "message": "Future health simulation created (demo mode)",
        "aged_image_url": "https://via.placeholder.com/400x400/4F46E5/FFFFFF?text=Aged+Photo+Demo",
        "health_predictions": {
            "cardiovascular_risk": 25,
            "diabetes_risk": 15,
            "life_expectancy": 78,
            "health_score": 75
        },
        "ai_narrative": "Based on your current lifestyle, you're on a good path for healthy aging! (Demo analysis)"
    }

# Simple therapy game endpoint
@app.post("/api/therapy-game/start-session")
async def start_therapy_session():
    return {
        "success": True,
        "session_id": "therapy_demo_303",
        "message": "Therapy game session started (demo mode)",
        "game_type": "shoulder_rehabilitation",
        "instructions": "Follow the on-screen movements for shoulder exercises (demo mode)"
    }

# Dashboard stats endpoint
@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    return {
        "success": True,
        "stats": {
            "totalPatients": 156,
            "todayAppointments": 12,
            "monthlyRevenue": 12450,
            "completedConsultations": 89,
            "averageRating": 4.8
        }
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to HealthSync AI Backend!",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)