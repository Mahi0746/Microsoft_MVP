"""
Action Agent - Daily Plan Executor (Refactored)

Executes approved daily plans.
Stores plan in database, creates reminders.
Pure Python - no LLM required.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime
from services.mongodb_atlas_service import get_mongodb_service


class ActionAgent:
    """
    Action Agent for Daily Medical Plans
    
    Executes:
    - Store daily plan in database
    - Create medication reminders
    - Add exercises to dashboard
    - Send notifications
    - Update today_generated flag
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    async def execute_plan(
        self,
        user_id: str,
        observed_data: Dict[str, Any],
        daily_plan: Dict[str, Any],
        safety_approval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute approved daily plan
        
        Returns:
        {
            "executed_actions": [...],
            "plan_stored": true|false,
            "reminders_created": int,
            "executed_at": "datetime"
        }
        """
        self.logger.info(f"Action Agent executing plan for user {user_id}")
        
        executed_actions = []
        
        # 1. Store daily plan in database
        plan_stored = await self._store_daily_plan(
            user_id,
            observed_data,
            daily_plan,
            safety_approval
        )
        
        if plan_stored:
            executed_actions.append("plan_stored_in_db")
        
        # 2. Create medication reminders
        reminders_count = await self._create_medication_reminders(
            user_id,
            daily_plan.get('medication_reminders', [])
        )
        executed_actions.append(f"medication_reminders_created_{reminders_count}")
        
        # 3. Add exercises to dashboard (as tasks)
        exercises_added = await self._add_exercises_to_dashboard(
            user_id,
            daily_plan.get('exercises', [])
        )
        executed_actions.append(f"exercises_added_{exercises_added}")
        
        # 4. Send notification
        notification_sent = await self._send_daily_notification(
            user_id,
            daily_plan
        )
        if notification_sent:
            executed_actions.append("notification_sent")
        
        result = {
            "executed_actions": executed_actions,
            "plan_stored": plan_stored,
            "reminders_created": reminders_count,
            "exercises_added": exercises_added,
            "executed_at": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"Action Agent completed execution: {len(executed_actions)} actions")
        return result
    
    async def _store_daily_plan(
        self,
        user_id: str,
        observed_data: Dict[str, Any],
        daily_plan: Dict[str, Any],
        safety_approval: Dict[str, Any]
    ) -> bool:
        """Store complete daily plan in daily_agent_memory collection"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                self.logger.error("MongoDB not available")
                return False
            
            plan_date = daily_plan.get('plan_date')
            
            document = {
                "user_id": user_id,
                "date": plan_date,
                "observed_data": observed_data,
                "daily_plan": daily_plan,
                "safety_approval": safety_approval,
                "executed_actions": [],  # Will be updated by Reflection Agent
                "reflection": {},  # Will be added by Reflection Agent
                "created_at": datetime.utcnow()
            }
            
            # Upsert to prevent duplicates
            result = await mongodb.database.daily_agent_memory.update_one(
                {"user_id": user_id, "date": plan_date},
                {"$set": document},
                upsert=True
            )
            
            self.logger.info(f"Stored daily plan for {user_id} on {plan_date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store daily plan: {e}")
            return False
    
    async def _create_medication_reminders(
        self,
        user_id: str,
        medication_reminders: List[Dict[str, Any]]
    ) -> int:
        """Create medication reminder documents"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return 0
            
            count = 0
            for reminder in medication_reminders:
                # Store in user_reminders collection (create if doesn't exist)
                reminder_doc = {
                    "user_id": user_id,
                    "type": "medication",
                    "drug": reminder.get('drug'),
                    "dose": reminder.get('dose'),
                    "time": reminder.get('time'),
                    "date": datetime.utcnow().date().isoformat(),
                    "completed": False,
                    "created_at": datetime.utcnow()
                }
                
                await mongodb.database.user_reminders.insert_one(reminder_doc)
                count += 1
            
            self.logger.info(f"Created {count} medication reminders")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to create medication reminders: {e}")
            return 0
    
    async def _add_exercises_to_dashboard(
        self,
        user_id: str,
        exercises: List[Dict[str, Any]]
    ) -> int:
        """Add exercises as dashboard tasks"""
        try:
            mongodb = await get_mongodb_service()
            if not mongodb or not mongodb.client:
                return 0
            
            count = 0
            for exercise in exercises:
                task_doc = {
                    "user_id": user_id,
                    "type": "exercise",
                    "name": exercise.get('name'),
                    "reps": exercise.get('reps'),
                    "duration": exercise.get('duration'),
                    "time": exercise.get('time'),
                    "intensity": exercise.get('intensity'),
                    "date": datetime.utcnow().date().isoformat(),
                    "completed": False,
                    "created_at": datetime.utcnow()
                }
                
                await mongodb.database.user_tasks.insert_one(task_doc)
                count += 1
            
            self.logger.info(f"Added {count} exercises to dashboard")
            return count
            
        except Exception as e:
            self.logger.error(f"Failed to add exercises: {e}")
            return 0
    
    async def _send_daily_notification(
        self,
        user_id: str,
        daily_plan: Dict[str, Any]
    ) -> bool:
        """Send daily plan notification (mock for now)"""
        try:
            # In production: integrate with notification service
            exercise_count = len(daily_plan.get('exercises', []))
            med_count = len(daily_plan.get('medication_reminders', []))
            
            notification_message = f"Your daily plan is ready! {exercise_count} exercises, {med_count} medications"
            
            self.logger.info(f"Notification (mock): {notification_message}")
            
            # TODO: Integrate with actual notification service
            # await send_push_notification(user_id, notification_message)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {e}")
            return False


# Singleton
_action_agent = None

def get_action_agent() -> ActionAgent:
    """Get or create Action Agent singleton"""
    global _action_agent
    if _action_agent is None:
        _action_agent = ActionAgent()
    return _action_agent
