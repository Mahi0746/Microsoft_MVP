"""
Orchestrator Service - Multi-Agent Coordinator (Refactored for Daily Plans)

Coordinates the full agent cycle:
Observer → Planner → Safety → Action → Reflection → Memory
"""

import logging
from typing import Dict, Any
from datetime import datetime

from .observer_agent import get_observer_agent
from .planner_agent import get_planner_agent
from .safety_agent import get_safety_agent
from .action_agent import get_action_agent
from .reflection_agent import get_reflection_agent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Coordinates all 5 agents in DAILY health planning cycle.
    
    Flow:
    1. Observer collects medical data
    2. Check if today's plan exists (skip if yes)
    3. Planner generates daily plan
    4. Safety reviews plan (retry if rejected)
    5. Action executes approved plan
    6. Reflection analyzes adherence (end of day)
    7. Store to memory
    """
    
    MAX_RETRY_ATTEMPTS = 2
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def run_daily_cycle(self, user_id: str) -> Dict[str, Any]:
        """
        Execute full DAILY agent cycle for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Cycle summary with all agent outputs
        """
        self.logger.info(f"=== Starting DAILY Agent Cycle for User {user_id} ===")
        
        today = datetime.utcnow().date().isoformat()
        
        cycle_result = {
            "user_id": user_id,
            "date": today,
            "cycle_started_at": datetime.utcnow().isoformat(),
            "success": False,
            "stages": {}
        }
        
        try:
            # Stage 1: Observe (includes checking today_generated flag)
            observer = get_observer_agent()
            observed_data = await observer.observe(user_id)
            cycle_result["stages"]["observe"] = {
                "success": True,
                "data": observed_data
            }
            self.logger.info("✓ Observer stage complete")
            
            # Check if today's plan already generated
            if observed_data.get("today_generated", False):
                self.logger.info("Plan already generated for today - skipping cycle")
                cycle_result["success"] = False
                cycle_result["skip_reason"] = "Plan already exists for today"
                cycle_result["cycle_completed_at"] = datetime.utcnow().isoformat()
                return cycle_result
            
            # Stage 2: Plan (with retry logic)
            planner = get_planner_agent()
            safety = get_safety_agent()
            
            proposed_plan = None
            safety_result = None
            approved = False
            retry_count = 0
            
            while not approved and retry_count < self.MAX_RETRY_ATTEMPTS:
                # Generate plan
                proposed_plan = await planner.generate_daily_plan(observed_data)
                
                # Check if planner skipped
                if not proposed_plan.get("generated", True):
                    cycle_result["skip_reason"] = proposed_plan.get("skip_reason")
                    cycle_result["success"] = False
                    return cycle_result
                
                # Safety review
                safety_result = await safety.validate_plan(observed_data, proposed_plan)
                approved = safety_result.get("approved", False)
                
                if not approved:
                    risk_reason = safety_result.get("risk_reason", "Unknown risk")
                    self.logger.warning(f"Plan rejected (attempt {retry_count + 1}): {risk_reason}")
                    retry_count += 1
                    
                    # If rejected, try again
                    # For final attempt, use safe fallback
                    if retry_count >= self.MAX_RETRY_ATTEMPTS:
                        self.logger.error("Max retries reached - using safe fallback")
                        proposed_plan = self._get_safe_fallback_plan()
                        approved = True  # Accept fallback
                        safety_result = {"approved": True, "fallback": True}
                else:
                    self.logger.info("✓ Safety approved plan")
            
            cycle_result["stages"]["plan"] = {
                "success": approved,
                "proposed_plan": proposed_plan,
                "safety_result": safety_result,
                "retry_count": retry_count
            }
            
            # Stage 3: Execute (only if approved)
            if approved:
                action = get_action_agent()
                execution_result = await action.execute_plan(
                    user_id,
                    observed_data,
                    proposed_plan,
                    safety_result
                )
                cycle_result["stages"]["action"] = execution_result
                self.logger.info(f"✓ Action stage complete: {len(execution_result.get('executed_actions', []))} actions")
            else:
                cycle_result["stages"]["action"] = {
                    "success": False,
                    "error": "Plan not executed - safety rejection"
                }
            
            # Stage 4: Reflection (skip for now, runs end-of-day)
            cycle_result["stages"]["reflection"] = {
                "skipped": True,
                "reason": "Reflection runs end-of-day after user completes tasks"
            }
            
            cycle_result["success"] = True
            cycle_result["cycle_completed_at"] = datetime.utcnow().isoformat()
            
            self.logger.info(f"=== DAILY Agent Cycle Complete for User {user_id} ===")
            return cycle_result
            
        except Exception as e:
            self.logger.error(f"Agent cycle failed: {e}")
            cycle_result["success"] = False
            cycle_result["error"] = str(e)
            return cycle_result
    
    async def run_end_of_day_reflection(self, user_id: str) -> Dict[str, Any]:
        """
        Run reflection at end of day to analyze adherence
        
        Args:
            user_id: User identifier
            
        Returns:
            Reflection result
        """
        self.logger.info(f"Running end-of-day reflection for user {user_id}")
        
        today = datetime.utcnow().date().isoformat()
        
        try:
            reflection = get_reflection_agent()
            reflection_result = await reflection.reflect_on_day(user_id, today)
            
            self.logger.info(f"✓ Reflection complete: adherence score {reflection_result.get('adherence_score', 0)}")
            return {
                "success": True,
                "reflection": reflection_result
            }
            
        except Exception as e:
            self.logger.error(f"Reflection failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_safe_fallback_plan(self) -> Dict[str, Any]:
        """Return ultra-safe fallback plan if all retries fail"""
        return {
            "plan_date": datetime.utcnow().date().isoformat(),
            "exercises": [
                {"name": "Gentle stretching", "reps": "10", "duration": "5 min", "time": "morning", "intensity": "very low"}
            ],
            "diet_recommendations": [
                "Balanced meals",
                "Stay hydrated"
            ],
            "medication_reminders": [],
            "health_tips": [
                "Rest and recover",
                "Monitor symptoms"
            ],
            "notes": "Safe fallback plan - gentle activities only"
        }


# Singleton
_orchestrator_instance = None


def get_orchestrator() -> AgentOrchestrator:
    """Get or create Orchestrator singleton"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AgentOrchestrator()
    return _orchestrator_instance
