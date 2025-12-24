"""
Observer Agent - Medical Data Aggregator (Refactored for Daily Plans)

Collects medical data from MongoDB instead of fitness data.
No LLM usage - pure Python data aggregation.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from services.mongodb_atlas_service import get_mongodb_service


class ObserverAgent:
    """
    Observer Agent for Daily Medical Plans
    
    Collects:
    - Medical conditions
    - Current prescriptions
    - Consultation notes
    - Voice/text chat logs (symptoms)
    - Therapy session adherence
    - Previous daily plans
    - Today's plan status (duplicate prevention)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    async def observe(self, user_id: str) -> Dict[str, Any]:
        """
        Main observation method - collects all medical data
        
        Returns:
        {
            "user_id": str,
            "date": str,
            "today_generated": bool,
            "medical_conditions": List[str],
            "current_prescriptions": List[Dict],
            "recent_consultations": List[Dict],
            "chat_symptoms": List[str],
            "therapy_adherence": Dict,
            "previous_reflections": List[Dict]
        }
        """
        self.logger.info(f"Observer Agent collecting medical data for user {user_id}")
        
        today = datetime.utcnow().date()
        
        # Check if today's plan already exists (duplicate prevention)
        today_generated = await self._check_daily_plan_exists(user_id, today)
        
        # Collect medical data
        medical_conditions = await self._get_medical_conditions(user_id)
        prescriptions = await self._get_current_prescriptions(user_id)
        consultations = await self._get_recent_consultations(user_id, days=7)
        chat_symptoms = await self._extract_chat_symptoms(user_id, days=7)
        therapy_adherence = await self._get_therapy_adherence(user_id, days=7)
        previous_reflections = await self._get_previous_reflections(user_id, days=3)
        
        observed_data = {
            "user_id": user_id,
            "date": today.isoformat(),
            "today_generated": today_generated,
            "medical_conditions": medical_conditions,
            "current_prescriptions": prescriptions,
            "recent_consultations": consultations,
            "chat_symptoms": chat_symptoms,
            "therapy_adherence": therapy_adherence,
            "previous_reflections": previous_reflections,
            "observed_at": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"Observer Agent completed data collection for user {user_id}")
        return observed_data
    
    async def _check_daily_plan_exists(self, user_id: str, date) -> bool:
        """Check if daily plan already generated for today"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return False
            
            existing_plan = await mongodb.database.daily_agent_memory.find_one({
                "user_id": user_id,
                "date": date.isoformat()
            })
            
            if existing_plan:
                self.logger.info(f"Daily plan already exists for {user_id} on {date}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to check daily plan: {e}")
            return False
    
    async def _get_medical_conditions(self, user_id: str) -> List[str]:
        """Get user's medical conditions from profile"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return []
            
            user = await mongodb.database.users.find_one({"user_id": user_id})
            
            if user and "medical_conditions" in user:
                conditions = user["medical_conditions"]
                self.logger.debug(f"Found {len(conditions)} medical conditions")
                return conditions
            
            # Fallback: demo data if no conditions stored
            return ["general health monitoring"]
            
        except Exception as e:
            self.logger.error(f"Failed to get medical conditions: {e}")
            return []
    
    async def _get_current_prescriptions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get active prescriptions"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return []
            
            user = await mongodb.database.users.find_one({"user_id": user_id})
            
            if user and "current_prescriptions" in user:
                prescriptions = user["current_prescriptions"]
                self.logger.debug(f"Found {len(prescriptions)} prescriptions")
                return prescriptions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Failed to get prescriptions: {e}")
            return []
    
    async def _get_recent_consultations(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get recent doctor consultations"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return []
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Query consultations collection (if it exists)
            consultations = await mongodb.database.consultations.find({
                "user_id": user_id,
                "consultation_date": {"$gte": cutoff_date}
            }).sort("consultation_date", -1).to_list(length=10)
            
            result = []
            for consult in consultations:
                result.append({
                    "date": consult.get("consultation_date", "").isoformat() if consult.get("consultation_date") else "",
                    "doctor": consult.get("doctor_name", "Unknown"),
                    "notes": consult.get("notes", ""),
                    "restrictions": consult.get("exercise_restrictions", [])
                })
            
            self.logger.debug(f"Found {len(result)} recent consultations")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get consultations: {e}")
            return []
    
    async def _extract_chat_symptoms(self, user_id: str, days: int = 7) -> List[str]:
        """Extract symptoms from voice/text chat logs"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return []
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Get voice sessions
            voice_sessions = await mongodb.database.voice_sessions.find({
                "user_id": user_id,
                "created_at": {"$gte": cutoff_date}
            }).to_list(length=50)
            
            symptoms = []
            
            for session in voice_sessions:
                transcript = session.get("transcript", "")
                
                # Simple keyword extraction (in production, use NLP)
                symptom_keywords = ["pain", "ache", "fever", "tired", "fatigue", 
                                   "dizzy", "nausea", "headache", "sore", "swelling"]
                
                for keyword in symptom_keywords:
                    if keyword in transcript.lower():
                        # Extract context around keyword
                        words = transcript.split()
                        for i, word in enumerate(words):
                            if keyword in word.lower():
                                context_start = max(0, i - 3)
                                context_end = min(len(words), i + 4)
                                symptom_context = " ".join(words[context_start:context_end])
                                symptoms.append(symptom_context)
                                break
            
            self.logger.debug(f"Extracted {len(symptoms)} symptoms from chat")
            return symptoms[:10]  # Limit to 10 most recent
            
        except Exception as e:
            self.logger.error(f"Failed to extract chat symptoms: {e}")
            return []
    
    async def _get_therapy_adherence(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get therapy session adherence data"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return {"total_sessions": 0, "completion_rate": 0}
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            sessions = await mongodb.database.therapy_sessions.find({
                "user_id": user_id,
                "created_at": {"$gte": cutoff_date}
            }).to_list(length=100)
            
            total = len(sessions)
            completed = sum(1 for s in sessions if s.get("completed", False))
            
            # Get pain/fatigue reports
            pain_reports = [s.get("pain_level", 0) for s in sessions if "pain_level" in s]
            avg_pain = sum(pain_reports) / len(pain_reports) if pain_reports else 0
            
            return {
                "total_sessions": total,
                "completed_sessions": completed,
                "completion_rate": (completed / total * 100) if total > 0 else 0,
                "avg_pain_level": round(avg_pain, 1),
                "last_session_date": sessions[0].get("created_at").isoformat() if sessions else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get therapy adherence: {e}")
            return {"total_sessions": 0, "completion_rate": 0}
    
    async def _get_previous_reflections(self, user_id: str, days: int = 3) -> List[Dict[str, Any]]:
        """Get previous daily reflections for learning"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return []
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            memories = await mongodb.database.daily_agent_memory.find({
                "user_id": user_id,
                "created_at": {"$gte": cutoff_date},
                "reflection": {"$exists": True}
            }).sort("date", -1).to_list(length=days)
            
            reflections = []
            for mem in memories:
                reflection = mem.get("reflection", {})
                if reflection:
                    reflections.append({
                        "date": mem.get("date"),
                        "adherence_score": reflection.get("adherence_score", 0),
                        "lessons_learned": reflection.get("lessons_learned", []),
                        "next_day_suggestions": reflection.get("next_day_suggestions", [])
                    })
            
            self.logger.debug(f"Found {len(reflections)} previous reflections")
            return reflections
            
        except Exception as e:
            self.logger.error(f"Failed to get previous reflections: {e}")
            return []


# Singleton instance
_observer_agent = None

def get_observer_agent() -> ObserverAgent:
    """Get or create Observer Agent singleton"""
    global _observer_agent
    if _observer_agent is None:
        _observer_agent = ObserverAgent()
    return _observer_agent
