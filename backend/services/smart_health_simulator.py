"""
Smart Health Simulator - Generates realistic health data from lifestyle questions
Replaces Google Fit integration for instant, reliable data generation
"""

import random
from typing import Dict, Any
from datetime import datetime


class SmartHealthSimulator:
    """
    Generates realistic health data based on user lifestyle profile
    Perfect for demos and users without connected devices
    """
    
    PROFILES = {
        "sedentary": {
            "steps_range": (2000, 5000),
            "exercise_range": (0, 15),
            "sleep_range": (5.5, 7.0),
            "rhr_range": (75, 90),
            "fitness_level": 0.4
        },
        "average": {
            "steps_range": (5000, 9000),
            "exercise_range": (15, 35),
            "sleep_range": (6.5, 8.0),
            "rhr_range": (65, 78),
            "fitness_level": 0.6
        },
        "active": {
            "steps_range": (9000, 15000),
            "exercise_range": (35, 60),
            "sleep_range": (7.0, 8.5),
            "rhr_range": (55, 68),
            "fitness_level": 0.85
        },
        "athlete": {
            "steps_range": (12000, 20000),
            "exercise_range": (60, 120),
            "sleep_range": (7.5, 9.0),
            "rhr_range": (45, 60),
            "fitness_level": 0.95
        }
    }
    
    @staticmethod
    def determine_profile_from_answers(answers: Dict[str, int]) -> str:
        """Determine lifestyle profile from question answers"""
        score = sum(answers.values())
        
        if score <= 7:
            return "sedentary"
        elif score <= 12:
            return "average"
        elif score <= 16:
            return "active"
        else:
            return "athlete"
    
    @staticmethod
    def generate_health_data(profile: str, answers: Dict[str, int]) -> Dict[str, Any]:
        """
        Generate realistic 7-day health data based on lifestyle profile
        
        Args:
            profile: Lifestyle profile (sedentary/average/active/athlete)
            answers: Raw answers from 5 questions
            
        Returns:
            Comprehensive health metrics
        """
        profile_data = SmartHealthSimulator.PROFILES[profile]
        
        # Generate 7 days of realistic data
        daily_steps = []
        daily_sleep = []
        daily_heart_rate = []
        
        for day in range(7):
            is_weekend = day >= 5
            
            # Steps (more on weekends for active, less for sedentary)
            base_steps = random.randint(*profile_data['steps_range'])
            if is_weekend:
                weekend_factor = 1.2 if profile_data['fitness_level'] > 0.6 else 0.8
                steps = int(base_steps * weekend_factor)
            else:
                steps = base_steps
            daily_steps.append(steps)
            
            # Sleep (slightly more on weekends)
            base_sleep = random.uniform(*profile_data['sleep_range'])
            if is_weekend:
                sleep = min(9.5, base_sleep + random.uniform(0, 1))
            else:
                sleep = base_sleep
            daily_sleep.append(round(sleep, 1))
            
            # Heart rate
            base_rhr = random.randint(*profile_data['rhr_range'])
            daily_heart_rate.append(base_rhr)
        
        # Calculate averages
        avg_steps = sum(daily_steps) // 7
        avg_sleep = round(sum(daily_sleep) / 7, 1)
        avg_exercise = random.randint(*profile_data['exercise_range'])
        avg_rhr = sum(daily_heart_rate) // 7
        
        # Calculate derived metrics
        hrv = int(80 - (avg_rhr - 50) * 0.8)
        hrv = max(20, min(80, hrv))
        
        stress_from_sleep = max(0, (7.5 - avg_sleep) * 2)
        stress_from_activity = max(0, (8000 - avg_steps) / 1000)
        stress_level = min(10, stress_from_sleep + stress_from_activity + (10 - answers.get('stress', 2)))
        stress_level = round(stress_level, 1)
        
        recovery = int(
            40 +
            (avg_sleep / 8) * 20 +
            (hrv / 60) * 20 +
            (avg_steps / 10000) * 15 +
            (10 - stress_level) * 0.5
        )
        recovery = min(100, max(0, recovery))
        
        # Determine steps trend
        if avg_steps > 8000:
            steps_trend = "increasing"
        elif avg_steps < 5000:
            steps_trend = "decreasing"
        else:
            steps_trend = "stable"
        
        return {
            "avg_steps_per_day": avg_steps,
            "steps_trend": steps_trend,
            "resting_hr": avg_rhr,
            "avg_sleep_hours": avg_sleep,
            "exercise_minutes": avg_exercise,
            "hrv": hrv,
            "stress_level": stress_level,
            "recovery_score": recovery,
            "calories_burned": 1800 + (avg_steps // 10) + (avg_exercise * 8),
            "generated_at": datetime.utcnow().isoformat(),
            "data_source": "smart_simulator",
            "profile": profile
        }
