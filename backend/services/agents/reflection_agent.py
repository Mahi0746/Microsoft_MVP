"""
Reflection Agent - Daily Adherence Analyzer (Refactored)

Analyzes daily plan adherence and learns for next day.
Uses LLM (Gemini) or rule-based analysis.
"""

import logging
import json
from typing import Dict, Any
from datetime import datetime, timedelta
from services.mongodb_atlas_service import get_mongodb_service

try:
    import google.generativeai as genai
    from config_flexible import settings
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


class ReflectionAgent:
    """
    Reflection Agent for Daily Medical Plans
    
    Analyzes:
    - Medication compliance
    - Exercise completion
    - Overall adherence
    - Learns from patterns
    - Suggests improvements for tomorrow
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance._initialize_llm()
        return cls._instance
    
    def _initialize_llm(self):
        """Initialize Gemini LLM"""
        self.use_llm = False
        
        if GEMINI_AVAILABLE:
            api_key = getattr(settings, 'gemini_api_key', None)
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.model = genai.GenerativeModel('gemini-1.5-flash')
                    self.use_llm = True
                    self.logger.info("Reflection Agent using Gemini LLM")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini: {e}")
        
        if not self.use_llm:
            self.logger.info("Reflection Agent using rule-based analysis")
    
    async def reflect_on_day(
        self,
        user_id: str,
        date: str
    ) -> Dict[str, Any]:
        """
        Analyze today's plan adherence
        
        Returns:
        {
            "date": "YYYY-MM-DD",
            "adherence_score": 0-100,
            "medication_compliance": 0-100,
            "exercise_completion": 0-100,
            "lessons_learned": [...],
            "next_day_suggestions": [...]
        }
        """
        self.logger.info(f"Reflection Agent analyzing day for user {user_id}")
        
        # Get today's plan and adherence data
        plan_data = await self._get_plan_data(user_id, date)
        adherence_data = await self._get_adherence_data(user_id, date)
        
        if not plan_data:
            self.logger.warning(f"No plan found for {user_id} on {date}")
            return {"date": date, "adherence_score": 0}
        
        if self.use_llm:
            reflection = await self._reflect_with_llm(plan_data, adherence_data)
        else:
            reflection = self._reflect_with_rules(plan_data, adherence_data)
        
        reflection["date"] = date
        
        # Update daily_agent_memory with reflection
        await self._store_reflection(user_id, date, reflection)
        
        self.logger.info(f"Reflection completed: adherence score {reflection.get('adherence_score', 0)}")
        return reflection
    
    async def _get_plan_data(self, user_id: str, date: str) -> Dict[str, Any]:
        """Get today's plan from daily_agent_memory"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return {}
            
            plan = await mongodb.database.daily_agent_memory.find_one({
                "user_id": user_id,
                "date": date
            })
            
            return plan if plan else {}
            
        except Exception as e:
            self.logger.error(f"Failed to get plan data: {e}")
            return {}
    
    async def _get_adherence_data(self, user_id: str, date: str) -> Dict[str, Any]:
        """Get adherence data (completed tasks/reminders)"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return {}
            
            # Get medication reminders
            med_reminders = await mongodb.database.user_reminders.find({
                "user_id": user_id,
                "date": date,
                "type": "medication"
            }).to_list(length=100)
            
            # Get exercise tasks
            exercise_tasks = await mongodb.database.user_tasks.find({
                "user_id": user_id,
                "date": date,
                "type": "exercise"
            }).to_list(length=100)
            
            return {
                "medication_reminders": med_reminders,
                "exercise_tasks": exercise_tasks
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get adherence data: {e}")
            return {}
    
    async def _reflect_with_llm(self, plan_data: Dict[str, Any], adherence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect using Gemini LLM"""
        prompt = f"""
SYSTEM ROLE: Daily Health Reflection AI

TASK: Analyze today's plan adherence and suggest improvements for tomorrow.

TODAY'S PLAN:
{json.dumps(plan_data.get('daily_plan', {}), indent=2)}

ADHERENCE DATA:
- Medication Reminders: {len(adherence_data.get('medication_reminders', []))} total
  - Completed: {sum(1 for r in adherence_data.get('medication_reminders', []) if r.get('completed'))}
- Exercise Tasks: {len(adherence_data.get('exercise_tasks', []))} total
  - Completed: {sum(1 for t in adherence_data.get('exercise_tasks', []) if t.get('completed'))}

OUTPUT ONLY valid JSON:
{{
  "adherence_score": 0-100,
  "medication_compliance": 0-100,
  "exercise_completion": 0-100,
  "lessons_learned": ["Specific observations about what worked/didn't work"],
  "next_day_suggestions": ["Actionable recommendations for tomorrow"]
}}

ANALYZE:
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text
            
            result = json.loads(json_str)
            self.logger.info("LLM reflection completed")
            return result
            
        except Exception as e:
            self.logger.error(f"LLM reflection failed: {e}, using fallback")
            return self._reflect_with_rules(plan_data, adherence_data)
    
    def _reflect_with_rules(self, plan_data: Dict[str, Any], adherence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based reflection"""
        daily_plan = plan_data.get('daily_plan', {})
        
        # Calculate medication compliance
        med_reminders = adherence_data.get('medication_reminders', [])
        med_total = len(med_reminders)
        med_completed = sum(1 for r in med_reminders if r.get('completed'))
        med_compliance = (med_completed / med_total * 100) if med_total > 0 else 0
        
        # Calculate exercise completion
        exercise_tasks = adherence_data.get('exercise_tasks', [])
        ex_total = len(exercise_tasks)
        ex_completed = sum(1 for t in exercise_tasks if t.get('completed'))
        ex_completion = (ex_completed / ex_total * 100) if ex_total > 0 else 0
        
        # Overall adherence score (weighted)
        adherence_score = int((med_compliance * 0.6) + (ex_completion * 0.4))
        
        # Generate lessons
        lessons_learned = []
        if med_compliance < 100:
            lessons_learned.append(f"Missed {med_total - med_completed} medication(s) - set more reminders")
        if ex_completion < 50:
            lessons_learned.append("Low exercise completion - may need to reduce difficulty")
        if adherence_score >= 80:
            lessons_learned.append("Good adherence overall - maintain current plan")
        
        # Generate suggestions
        next_day_suggestions = []
        if med_compliance < 80:
            next_day_suggestions.append("Add extra medication reminders 30 min before scheduled time")
        if ex_completion < 50:
            next_day_suggestions.append("Reduce exercise intensity or duration")
        if adherence_score >= 90:
            next_day_suggestions.append("Consider slightly increasing exercise challenge")
        
        return {
            "adherence_score": adherence_score,
            "medication_compliance": int(med_compliance),
            "exercise_completion": int(ex_completion),
            "lessons_learned": lessons_learned,
            "next_day_suggestions": next_day_suggestions
        }
    
    async def _store_reflection(self, user_id: str, date: str, reflection: Dict[str, Any]):
        """Store reflection in daily_agent_memory"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return
            
            await mongodb.database.daily_agent_memory.update_one(
                {"user_id": user_id, "date": date},
                {"$set": {"reflection": reflection}}
            )
            
            self.logger.info(f"Stored reflection for {user_id} on {date}")
            
        except Exception as e:
            self.logger.error(f"Failed to store reflection: {e}")


# Singleton
_reflection_agent = None

def get_reflection_agent() -> ReflectionAgent:
    """Get or create Reflection Agent singleton"""
    global _reflection_agent
    if _reflection_agent is None:
        _reflection_agent = ReflectionAgent()
    return _reflection_agent
