"""
Planner Agent - Daily Health Plan Generator (Refactored)

Generates personalized daily health plans using LLM (Gemini).
Considers medical conditions, prescriptions, consultations.
Only generates if today_generated is False.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import google.generativeai as genai
    from config_flexible import settings
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


class PlannerAgent:
    """
    Planner Agent for Daily Medical Plans
    
    Generates:
    - Daily exercises (safe for conditions)
    - Diet recommendations
    - Medication reminders
    - Health tips
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
                    self.logger.info("Planner Agent using Gemini LLM")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini: {e}")
        
        if not self.use_llm:
            self.logger.info("Planner Agent using rule-based fallback")
    
    async def generate_daily_plan(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate daily health plan
        
        Returns:
        {
            "plan_date": "YYYY-MM-DD",
            "generated": true|false,
            "skip_reason": "...",
            "exercises": [...],
            "diet_recommendations": [...],
            "medication_reminders": [...],
            "health_tips": [...],
            "notes": "..."
        }
        """
        self.logger.info(f"Planner Agent generating daily plan for user {observed_data.get('user_id')}")
        
        # Check if today's plan already generated
        if observed_data.get("today_generated", False):
            self.logger.info("Plan already generated for today - skipping")
            return {
                "plan_date": observed_data.get("date"),
                "generated": False,
                "skip_reason": "Plan already exists for today"
            }
        
        # Generate plan using LLM or fallback
        if self.use_llm:
            plan = await self._generate_with_llm(observed_data)
        else:
            plan = self._generate_with_rules(observed_data)
        
        plan["plan_date"] = observed_data.get("date")
        plan["generated"] = True
        
        self.logger.info(f"Planner Agent completed plan generation")
        return plan
    
    async def _generate_with_llm(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate plan using Gemini LLM"""
        prompt = self._build_prompt(observed_data)
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            else:
                json_str = response_text
            
            plan = json.loads(json_str)
            self.logger.info("LLM generated plan successfully")
            return plan
            
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}, using fallback")
            return self._generate_with_rules(observed_data)
    
    def _build_prompt(self, observed_data: Dict[str, Any]) -> str:
        """Build prompt for LLM"""
        return f"""
SYSTEM ROLE: Daily Health Planner AI

TASK: Generate a personalized daily health plan for {observed_data.get('date')}.

USER PROFILE:
- Medical Conditions: {', '.join(observed_data.get('medical_conditions', ['none']))}
- Current Prescriptions: {json.dumps(observed_data.get('current_prescriptions', []), indent=2)}
- Recent Consultations: {json.dumps(observed_data.get('recent_consultations', []), indent=2)}
- Recent Symptoms (from chats): {', '.join(observed_data.get('chat_symptoms', ['none reported']))}
- Therapy Adherence: {observed_data.get('therapy_adherence', {})}
- Previous Reflections: {json.dumps(observed_data.get('previous_reflections', []), indent=2)}

RULES:
1. Exercises MUST be safe for user's medical conditions
2. Include medication reminders with EXACT times from prescriptions
3. Diet must align with conditions (e.g., low sugar for diabetes, low sodium for hypertension)
4. Consider recent symptoms and adjust intensity
5. Learn from previous reflections
6. Output ONLY valid JSON, no extra text

OUTPUT SCHEMA:
{{
  "exercises": [
    {{"name": "shoulder stretch", "reps": "10", "duration": "5 min", "time": "morning", "intensity": "low"}}
  ],
  "diet_recommendations": [
    "Low sugar breakfast",
    "High fiber lunch",
    "Light dinner before 8 PM"
  ],
  "medication_reminders": [
    {{"drug": "DrugName", "dose": "500mg", "time": "8:00 AM"}},
    {{"drug": "DrugName", "dose": "500mg", "time": "8:00 PM"}}
  ],
  "health_tips": [
    "Drink 8 glasses of water",
    "Take a 10-minute walk after meals"
  ],
  "notes": "Focus on gentle exercises due to recent symptoms. Monitor condition closely."
}}

GENERATE PLAN:
"""
    
    def _generate_with_rules(self, observed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based plan generation"""
        conditions = observed_data.get('medical_conditions', [])
        prescriptions = observed_data.get('current_prescriptions', [])
        therapy_adherence = observed_data.get('therapy_adherence', {})
        
        # Safe baseline exercises
        exercises = [
            {"name": "Gentle stretching", "reps": "10", "duration": "5 min", "time": "morning", "intensity": "low"},
            {"name": "Light walking", "reps": "1", "duration": "15 min", "time": "evening", "intensity": "low"}
        ]
        
        # Adjust based on conditions
        if "shoulder pain" in [c.lower() for c in conditions]:
            exercises.append({"name": "Shoulder mobility", "reps": "10", "duration": "5 min", "time": "afternoon", "intensity": "very low"})
        
        # Diet recommendations
        diet_recommendations = ["Balanced meals", "Hydrate well"]
        
        if "diabetes" in [c.lower() for c in conditions]:
            diet_recommendations.append("Low sugar, high fiber breakfast")
        
        if "hypertension" in [c.lower() for c in conditions]:
            diet_recommendations.append("Low sodium diet")
        
        # Medication reminders from prescriptions
        medication_reminders = []
        for rx in prescriptions:
            medication_reminders.append({
                "drug": rx.get("drug", "Medication"),
                "dose": rx.get("dose", "As prescribed"),
                "time": rx.get("time", "morning")
            })
        
        # Health tips
        health_tips = [
            "Drink 8 glasses of water",
            "Get 7-8 hours of sleep",
            "Monitor symptoms and report changes"
        ]
        
        return {
            "exercises": exercises,
            "diet_recommendations": diet_recommendations,
            "medication_reminders": medication_reminders,
            "health_tips": health_tips,
            "notes": "Rule-based plan generated. Stay consistent with therapy sessions."
        }


# Singleton
_planner_agent = None

def get_planner_agent() -> PlannerAgent:
    """Get or create Planner Agent singleton"""
    global _planner_agent
    if _planner_agent is None:
        _planner_agent = PlannerAgent()
    return _planner_agent
