"""
Safety Agent - Medical Safety Validator (Refactored)

Reviews daily plans for medical safety conflicts.
Uses LLM (Gemini) or rule-based validation.
"""

import logging
import json
from typing import Dict, Any

try:
    import google.generativeai as genai
    from config_flexible import settings
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


class SafetyAgent:
    """
    Safety Agent for Daily Medical Plans
    
    Validates:
    - Exercise conflicts with medical conditions
    - Diet conflicts with prescriptions
    - Missing medication reminders
    - Unsafe intensity based on recent pain/symptoms
    """
    
    _instance = None
    
    # Medical safety rules
    CONDITION_RESTRICTIONS = {
        "diabetes": {
            "diet_avoid": ["high sugar", "refined carbs", "sugary drinks"],
            "exercise_caution": ["monitor glucose before/after exercise"]
        },
        "hypertension": {
            "diet_avoid": ["high sodium", "salty foods", "processed foods"],
            "exercise_avoid": ["heavy lifting", "intense strength training"]
        },
        "shoulder pain": {
            "exercise_avoid": ["overhead press", "heavy shoulder work", "pull-ups"]
        },
        "heart disease": {
            "exercise_avoid": ["high-intensity interval training", "heavy weights"],
            "exercise_caution": ["monitor heart rate", "start slow"]
        }
    }
    
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
                    self.logger.info("Safety Agent using Gemini LLM")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Gemini: {e}")
        
        if not self.use_llm:
            self.logger.info("Safety Agent using rule-based validation")
    
    async def validate_plan(self, observed_data: Dict[str, Any], proposed_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate daily plan for safety
        
        Returns:
        {
            "approved": true|false,
            "risk_reason": "...",
            "suggestions": [...]
        }
        """
        self.logger.info("Safety Agent validating daily plan")
        
        if self.use_llm:
            result = await self._validate_with_llm(observed_data, proposed_plan)
        else:
            result = self._validate_with_rules(observed_data, proposed_plan)
        
        self.logger.info(f"Safety validation: {'APPROVED' if result['approved'] else 'REJECTED'}")
        return result
    
    async def _validate_with_llm(self, observed_data: Dict[str, Any], proposed_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Validate using Gemini LLM"""
        prompt = f"""
SYSTEM ROLE: Medical Safety Validator

TASK: Review the proposed daily health plan for safety conflicts.

USER PROFILE:
- Medical Conditions: {', '.join(observed_data.get('medical_conditions', []))}
- Current Prescriptions: {json.dumps(observed_data.get('current_prescriptions', []))}
- Recent Symptoms: {', '.join(observed_data.get('chat_symptoms', []))}
- Therapy Adherence: {observed_data.get('therapy_adherence', {})}

PROPOSED PLAN:
{json.dumps(proposed_plan, indent=2)}

CRITICAL SAFETY CHECKS:
1. Do any exercises conflict with medical conditions?
2. Do diet recommendations conflict with medications?
3. Are medication reminders included for all prescriptions?
4. Is exercise intensity safe given recent symptoms/pain?

OUTPUT ONLY valid JSON:
{{
  "approved": true|false,
  "risk_reason": "Specific reason if rejected, null otherwise",
  "suggestions": ["Alternative recommendations if rejected"]
}}

VALIDATE:
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
            self.logger.info("LLM validation completed")
            return result
            
        except Exception as e:
            self.logger.error(f"LLM validation failed: {e}, using fallback")
            return self._validate_with_rules(observed_data, proposed_plan)
    
    def _validate_with_rules(self, observed_data: Dict[str, Any], proposed_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based validation"""
        conditions = [c.lower() for c in observed_data.get('medical_conditions', [])]
        exercises = proposed_plan.get('exercises', [])
        diet_recs = proposed_plan.get('diet_recommendations', [])
        med_reminders = proposed_plan.get('medication_reminders', [])
        prescriptions = observed_data.get('current_prescriptions', [])
        
        # Check 1: Exercise conflicts
        for condition in conditions:
            if condition in self.CONDITION_RESTRICTIONS:
                restrictions = self.CONDITION_RESTRICTIONS[condition]
                avoid_exercises = restrictions.get('exercise_avoid', [])
                
                for exercise in exercises:
                    exercise_name = exercise.get('name', '').lower()
                    for avoid in avoid_exercises:
                        if avoid.lower() in exercise_name:
                            return {
                                "approved": False,
                                "risk_reason": f"Exercise '{exercise['name']}' conflicts with {condition}. Avoid: {', '.join(avoid_exercises)}",
                                "suggestions": ["Replace with gentle stretching", "Low-impact walking"]
                            }
        
        # Check 2: Diet conflicts
        for condition in conditions:
            if condition in self.CONDITION_RESTRICTIONS:
                restrictions = self.CONDITION_RESTRICTIONS[condition]
                diet_avoid = restrictions.get('diet_avoid', [])
                
                for diet_rec in diet_recs:
                    diet_lower = diet_rec.lower()
                    for avoid in diet_avoid:
                        if avoid in diet_lower:
                            return {
                                "approved": False,
                                "risk_reason": f"Diet recommendation conflicts with {condition}. Avoid: {', '.join(diet_avoid)}",
                                "suggestions": [f"Low {condition}-friendly diet"]
                            }
        
        # Check 3: Missing medication reminders
        if prescriptions and len(med_reminders) < len(prescriptions):
            return {
                "approved": False,
                "risk_reason": f"Missing medication reminders. Expected {len(prescriptions)}, got {len(med_reminders)}",
                "suggestions": ["Add all prescribed medication reminders"]
            }
        
        # Check 4: High pain + high intensity
        therapy = observed_data.get('therapy_adherence', {})
        avg_pain = therapy.get('avg_pain_level', 0)
        
        if avg_pain > 6:  # High pain
            for exercise in exercises:
                intensity = exercise.get('intensity', 'low')
                if intensity in ['high', 'very high']:
                    return {
                        "approved": False,
                        "risk_reason": f"High-intensity exercise unsafe with recent pain level of {avg_pain}/10",
                        "suggestions": ["Reduce to low-intensity exercises", "Focus on recovery"]
                    }
        
        # All checks passed
        return {
            "approved": True,
            "risk_reason": None,
            "suggestions": []
        }


# Singleton
_safety_agent = None

def get_safety_agent() -> SafetyAgent:
    """Get or create Safety Agent singleton"""
    global _safety_agent
    if _safety_agent is None:
        _safety_agent = SafetyAgent()
    return _safety_agent
