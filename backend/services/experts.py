# experts.py - Domain-specific tools for the voice assistant

from langchain.tools import tool
from typing import List, Dict, Any
import datetime
from services.mongodb_atlas_service import get_mongodb_service

# Expert 1: Consultant Navigation
@tool
def open_voice_consultant(query: str = "") -> dict:
    """
    Opens the voice consultant feature for patient consultation.
    Use this ONLY when the user explicitly asks to 'speak to a doctor', 'start a consultation', 'talk to a human', or 'connect with a professional'.
    Do NOT use this for general health questions or symptom checking.
    """
    return {
        "action": "navigate",
        "target": "consultant",
        "message": "Opening voice consultant for you."
    }

# Expert 2: Medication Manager
@tool
def get_current_medication(patient_id: str = "current_user") -> dict:
    """
    Retrieves current medication list for the patient.
    Use this when the user asks about their medicines, prescriptions, or pills.
    """
    # In a real system, we would query the database here using patient_id
    # For MVP, we'll return mock data or fetch from a service if available
    meds = [
        {"name": "Metformin", "dosage": "500mg", "frequency": "2x daily", "reason": "Type 2 Diabetes"},
        {"name": "Lisinopril", "dosage": "10mg", "frequency": "1x daily", "reason": "Hypertension"},
        {"name": "Vitamin D3", "dosage": "1000 IU", "frequency": "1x daily", "reason": "Supplement"}
    ]
    return {
        "action": "display",
        "target": "medication",
        "data": meds,
        "message": f"I found {len(meds)} active medications in your profile."
    }

# Expert 3: Appointment Manager
@tool
def get_appointments(timeframe: str = "upcoming") -> dict:
    """
    Gets patient appointments.
    Use this when the user asks about their schedule, doctor visits, or upcoming appointments.
    """
    # Mock data for demonstration
    today = datetime.date.today()
    appointments = [
        {
            "date": (today + datetime.timedelta(days=2)).isoformat(),
            "time": "10:00 AM", 
            "doctor": "Dr. Sarah Shah", 
            "specialty": "Cardiologist",
            "type": "Checkup"
        },
        {
            "date": (today + datetime.timedelta(days=14)).isoformat(),
            "time": "2:30 PM", 
            "doctor": "Dr. James Wilson", 
            "specialty": "General Practitioner",
            "type": "Follow-up"
        }
    ]
    return {
        "action": "display",
        "target": "appointments",
        "data": appointments,
        "message": f"You have {len(appointments)} upcoming appointments scheduled."
    }

# Expert 4: Reports Manager
@tool
def get_medical_reports(report_type: str = "all") -> dict:
    """
    Retrieves medical reports, lab results, and scan documents.
    Use this when the user asks to see their lab results, blood tests, X-rays, MRI, CT scans, or medical documents.
    """
    return {
        "action": "navigate",
        "target": "reports",
        "message": "Opening your medical reports and scans section."
    }

# Expert 5: General Health Q&A
@tool
def answer_health_question(question: str) -> str:
    """
    Answers general health questions or symptom checks (e.g. 'I have a headache', 'is coffee bad').
    Do NOT use this for questions about the user's SPECIFIC daily plan, exercises, or diet -- use get_daily_health_plan for those.
    """
    # Return a prompt to the Agent to generate the answer itself.
    return f"User asked: '{question}'. Please provide a helpful, empathetic medical answer now based on your knowledge. Do not suggest opening the consultant unless asked."

# Expert 6: Dashboard Navigation
@tool
def navigate_dashboard(section: str) -> dict:
    """
    Navigates to a specific section of the dashboard.
    Valid sections: 'home', 'profile', 'settings', 'community', 'marketplace'.
    Use this when the user explicitly asks to go to a specific page.
    """
    valid_sections = ['home', 'profile', 'settings', 'community', 'marketplace', 'voice-doctor']
    target = section.lower().strip()
    
    # Simple mapping/correction
    if 'doctor' in target: target = 'marketplace'
    if 'consult' in target: target = 'voice-doctor'
    if 'voice' in target: target = 'voice-doctor'
    
    if target not in valid_sections:
        target = 'home'
        
    return {
        "action": "navigate",
        "target": target,
        "message": f"Navigating to {target}."
    }

# Expert 7: Daily Health Plan
@tool
async def get_daily_health_plan(user_id: str = "current_user") -> dict:
    """
    Retrieves the user's SPECIFIC daily health plan, prescribed exercises, diet, and medication reminders from the AI Coach.
    Use this for: "details of my plan", "what exercises do I have", "my workout", "diet recommendations", "what should I eat".
    ALWAYS use this if the user asks about "my plan" or "prescribed".
    """
    try:
        # For MVP, try to find a plan for today.
        mongodb = await get_mongodb_service()
        if not mongodb or not mongodb.client:
            return {"error": "Database unavailable"}

        # Try to find a plan for today for ANY user if specific one not provided, or specific one.
        today = datetime.datetime.utcnow().date().isoformat()
        
        query = {"date": today}
        if user_id and user_id not in ["current_user", "patient", "user"]:
             query["user_id"] = user_id
        elif user_id == "current_user":
             # Try to find the user in context? For now we might just query the latest one or a demo user.
             # Ideally the agent passes the ID.
             pass
             
        # Find one
        plan_doc = await mongodb.database.daily_agent_memory.find_one(query)
        
        if not plan_doc:
            # Fallback: check most recent plan regardless of date if specific date fails?
            # Or check for 'demo_user'
            plan_doc = await mongodb.database.daily_agent_memory.find_one({"user_id": "demo_user", "date": today})
            
        if not plan_doc:
             # Last resort: just get the latest one
             plan_doc = await mongodb.database.daily_agent_memory.find_one({}, sort=[("date", -1)])
            
        if not plan_doc:
            return {
                "action": "answer",
                "message": "I couldn't find a health plan generated for today. Please go to the AI Coach page to generate one."
            }
            
        daily_plan = plan_doc.get("daily_plan", {})
        
        return {
            "action": "display",
            "message": "Here is your daily health plan.",
            "data": daily_plan
        }

    except Exception as e:
        return {"error": f"Failed to fetch health plan: {str(e)}"}
