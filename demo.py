#!/usr/bin/env python3
"""
BIOLOGICAL AGE CALCULATOR - SMART REALISTIC SIMULATION
Uses intelligent patterns to simulate YOUR realistic health data
Perfect for Imagine Cup demo - shows automation concept
"""

import random
import time
from datetime import datetime, timedelta
import json

class SmartHealthSimulator:
    """
    Simulates realistic health data based on user profile
    In production: Replace with real API calls
    """
    
    def __init__(self, age, height, lifestyle_profile="average"):
        self.age = age
        self.height = height
        self.lifestyle_profile = lifestyle_profile
        
        # Define lifestyle profiles
        self.profiles = {
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
    
    def ask_lifestyle_questions(self):
        """Quick 5-question assessment to determine profile"""
        print("\n" + "="*70)
        print("🎯 QUICK LIFESTYLE ASSESSMENT (5 questions)")
        print("="*70)
        print("\nThis helps us simulate realistic data for you\n")
        
        score = 0
        
        # Q1: Activity level
        print("1️⃣ How active are you?")
        print("   1) Mostly sitting/desk work")
        print("   2) Light activity (some walking)")
        print("   3) Moderately active (regular exercise)")
        print("   4) Very active (daily intense exercise)")
        activity = int(input("   Your answer (1-4): ") or 2)
        score += activity
        
        # Q2: Exercise frequency
        print("\n2️⃣ How often do you exercise?")
        print("   1) Rarely/Never")
        print("   2) 1-2 times per week")
        print("   3) 3-4 times per week")
        print("   4) 5+ times per week")
        exercise = int(input("   Your answer (1-4): ") or 2)
        score += exercise
        
        # Q3: Sleep quality
        print("\n3️⃣ How would you rate your sleep?")
        print("   1) Poor (often tired)")
        print("   2) Fair (sometimes tired)")
        print("   3) Good (usually rested)")
        print("   4) Excellent (always energized)")
        sleep = int(input("   Your answer (1-4): ") or 3)
        score += sleep
        
        # Q4: Stress level
        print("\n4️⃣ Your stress level?")
        print("   1) Very high stress")
        print("   2) Moderate stress")
        print("   3) Low stress")
        print("   4) Very relaxed")
        stress = int(input("   Your answer (1-4): ") or 2)
        score += stress
        
        # Q5: Diet quality
        print("\n5️⃣ How's your diet?")
        print("   1) Mostly fast food/processed")
        print("   2) Mixed (some healthy)")
        print("   3) Mostly healthy")
        print("   4) Very clean/nutritious")
        diet = int(input("   Your answer (1-4): ") or 2)
        score += diet
        
        # Determine profile
        if score <= 7:
            profile = "sedentary"
        elif score <= 12:
            profile = "average"
        elif score <= 16:
            profile = "active"
        else:
            profile = "athlete"
        
        print(f"\n✅ Profile identified: {profile.upper()}")
        return profile, {
            'activity': activity,
            'exercise': exercise,
            'sleep': sleep,
            'stress': stress,
            'diet': diet
        }
    
    def generate_realistic_data(self, profile_answers):
        """Generate 7 days of realistic health data"""
        print("\n" + "="*70)
        print("📊 GENERATING YOUR PERSONALIZED HEALTH DATA (7-day pattern)")
        print("="*70)
        time.sleep(1)
        
        profile = self.profiles[self.lifestyle_profile]
        
        # Generate 7 days of data with realistic variation
        weekly_data = {
            'daily_steps': [],
            'daily_sleep': [],
            'daily_exercise': [],
            'daily_heart_rate': []
        }
        
        for day in range(7):
            # Weekend vs weekday variation
            is_weekend = day >= 5
            
            # Steps (more on weekends for active people, less for sedentary)
            base_steps = random.randint(*profile['steps_range'])
            if is_weekend:
                weekend_factor = 1.2 if profile['fitness_level'] > 0.6 else 0.8
                steps = int(base_steps * weekend_factor)
            else:
                steps = base_steps
            weekly_data['daily_steps'].append(steps)
            
            # Sleep (slightly more on weekends)
            base_sleep = random.uniform(*profile['sleep_range'])
            if is_weekend:
                sleep = min(9.5, base_sleep + random.uniform(0, 1))
            else:
                sleep = base_sleep
            weekly_data['daily_sleep'].append(round(sleep, 1))
            
            # Exercise
            base_exercise = random.randint(*profile['exercise_range'])
            if is_weekend and profile['fitness_level'] > 0.6:
                exercise = int(base_exercise * 1.3)
            else:
                exercise = base_exercise
            weekly_data['daily_exercise'].append(exercise)
            
            # Resting heart rate (improves slightly as week progresses)
            base_rhr = random.randint(*profile['rhr_range'])
            weekly_data['daily_heart_rate'].append(base_rhr)
        
        # Calculate averages
        avg_steps = sum(weekly_data['daily_steps']) // 7
        avg_sleep = round(sum(weekly_data['daily_sleep']) / 7, 1)
        avg_exercise = sum(weekly_data['daily_exercise']) // 7
        avg_rhr = sum(weekly_data['daily_heart_rate']) // 7
        
        print("\n📱 Data Collection Complete:")
        print(f"  ✅ Average steps: {avg_steps:,}/day")
        print(f"  ✅ Average sleep: {avg_sleep} hours/night")
        print(f"  ✅ Average exercise: {avg_exercise} min/day")
        print(f"  ✅ Resting heart rate: {avg_rhr} bpm")
        
        # Calculate derived metrics using AI-like inference
        print("\n🤖 AI inferring additional health metrics...")
        time.sleep(0.8)
        
        # HRV correlates inversely with RHR
        hrv = int(80 - (avg_rhr - 50) * 0.8)
        hrv = max(20, min(80, hrv))
        
        # Stress level based on multiple factors
        stress_from_sleep = max(0, (7.5 - avg_sleep) * 2)
        stress_from_activity = max(0, (8000 - avg_steps) / 1000)
        stress_level = min(10, stress_from_sleep + stress_from_activity + (10 - profile_answers['stress']))
        stress_level = round(stress_level, 1)
        
        # Recovery score
        recovery = int(
            40 +
            (avg_sleep / 8) * 20 +
            (hrv / 60) * 20 +
            (avg_steps / 10000) * 15 +
            (10 - stress_level) * 0.5
        )
        recovery = min(100, max(0, recovery))
        
        # Diet quality (from questionnaire + activity correlation)
        diet_quality = profile_answers['diet'] + (avg_steps / 3000)
        diet_quality = min(10, diet_quality)
        
        # Water intake (active people drink more)
        water_intake = 1.5 + (avg_steps / 5000) + (avg_exercise / 50)
        water_intake = round(min(4, water_intake), 1)
        
        # Blood pressure estimation
        bp_systolic = 110 + (avg_rhr - 60) // 2 + int(stress_level * 1.5)
        bp_diastolic = 70 + (avg_rhr - 60) // 3
        
        # Sleep quality score
        sleep_quality = min(10, int(avg_sleep) + random.randint(-1, 1))
        
        print(f"  ✅ Heart Rate Variability: {hrv}")
        print(f"  ✅ Stress level: {stress_level}/10")
        print(f"  ✅ Recovery score: {recovery}/100")
        print(f"  ✅ Diet quality: {diet_quality:.1f}/10")
        print(f"  ✅ Water intake: {water_intake}L/day")
        print(f"  ✅ Blood pressure: {bp_systolic}/{bp_diastolic}")
        
        # Get weight
        weight = float(input(f"\n⚖️ Your weight (kg) [70]: ") or 70)
        
        return {
            'avg_steps_per_day': avg_steps,
            'avg_sleep_hours': avg_sleep,
            'exercise_minutes': avg_exercise,
            'resting_hr': avg_rhr,
            'hrv': hrv,
            'stress_level': stress_level,
            'recovery_score': recovery,
            'diet_quality': diet_quality,
            'water_intake': water_intake,
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'sleep_quality': sleep_quality,
            'weight': weight,
            'weekly_data': weekly_data
        }
    
    def calculate_biological_age(self, health_data):
        """Calculate biological age from health data"""
        print("\n" + "="*70)
        print("🧬 CALCULATING BIOLOGICAL AGE")
        print("="*70)
        time.sleep(1)
        
        # Calculate BMI
        bmi = health_data['weight'] / ((self.height / 100) ** 2)
        
        # Component Scores (0-100)
        
        # Sleep Score
        sleep_hours = health_data['avg_sleep_hours']
        if 7 <= sleep_hours <= 9:
            sleep_score = 100
        elif sleep_hours < 7:
            sleep_score = (sleep_hours / 7) * 100
        else:
            sleep_score = max(0, 100 - (sleep_hours - 9) * 10)
        sleep_score = (sleep_score * 0.7) + (health_data['sleep_quality'] * 10 * 0.3)
        
        # Exercise Score
        steps = health_data['avg_steps_per_day']
        exercise_min = health_data['exercise_minutes']
        exercise_score = min(100, (steps / 10000) * 50 + (exercise_min / 30) * 50)
        
        # Heart Health Score
        rhr = health_data['resting_hr']
        hrv = health_data['hrv']
        bp_sys = health_data['bp_systolic']
        bp_dia = health_data['bp_diastolic']
        
        rhr_score = max(0, 100 - abs(rhr - 65) * 2)
        hrv_score = min(100, (hrv / 50) * 100)
        bp_score = max(0, 100 - abs(bp_sys - 120) - abs(bp_dia - 80))
        heart_score = (rhr_score * 0.3) + (hrv_score * 0.4) + (bp_score * 0.3)
        
        # Stress Score (inverted)
        stress_score = max(0, 100 - health_data['stress_level'] * 10)
        
        # Nutrition Score
        nutrition_score = (health_data['diet_quality'] * 10 * 0.7) + \
                         min(100, (health_data['water_intake'] / 3) * 100 * 0.3)
        
        # Recovery Score
        recovery_score = health_data['recovery_score']
        
        # BMI Score
        if 18.5 <= bmi <= 24.9:
            bmi_score = 100
        elif bmi < 18.5:
            bmi_score = (bmi / 18.5) * 100
        else:
            bmi_score = max(0, 100 - (bmi - 24.9) * 10)
        
        # Overall Health Score (weighted average)
        health_score = (
            sleep_score * 0.16 +
            exercise_score * 0.16 +
            heart_score * 0.20 +
            stress_score * 0.15 +
            nutrition_score * 0.14 +
            recovery_score * 0.11 +
            bmi_score * 0.08
        )
        
        # Biological Age Calculation
        # Every 10 points below perfect = ~2.5 years aging
        aging_factor = ((100 - health_score) / 10) * 2.5
        bio_age = self.age + aging_factor
        
        # Organ-specific ages
        organ_ages = {
            'Cardiovascular': self.age + ((100 - heart_score) / 10) * 2.0,
            'Brain/Cognitive': self.age + ((100 - stress_score) / 10) * 1.8,
            'Metabolic': self.age + ((100 - nutrition_score) / 10) * 2.2,
            'Musculoskeletal': self.age + ((100 - exercise_score) / 10) * 1.5,
            'Immune System': self.age + ((100 - sleep_score) / 10) * 2.0,
        }
        
        # Generate recommendations
        scores = {
            'Sleep': sleep_score,
            'Exercise': exercise_score,
            'Heart Health': heart_score,
            'Stress Management': stress_score,
            'Nutrition': nutrition_score,
            'Recovery': recovery_score,
            'BMI': bmi_score
        }
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])
        
        recommendations = []
        tips = {
            'Sleep': f'🛌 Improve sleep (currently {sleep_hours}h). Target: 7-9 hours.',
            'Exercise': f'🏃 Increase activity (currently {steps:,} steps). Target: 10,000 steps.',
            'Heart Health': f'❤️ Improve heart health (RHR: {rhr}, HRV: {hrv}). Try cardio.',
            'Stress Management': f'🧘 Reduce stress (currently {health_data["stress_level"]}/10). Meditate daily.',
            'Nutrition': f'🥗 Better nutrition (currently {health_data["diet_quality"]:.1f}/10). More whole foods.',
            'Recovery': f'💪 Improve recovery (currently {recovery_score}/100). More rest days.',
            'BMI': f'⚖️ Optimize BMI (currently {bmi:.1f}). Target: 18.5-24.9.'
        }
        
        for area, score in sorted_scores[:3]:
            if score < 80:
                impact = round((100 - score) / 40, 1)
                recommendations.append({
                    'area': area,
                    'score': round(score, 1),
                    'priority': 'HIGH' if score < 50 else 'MEDIUM',
                    'tip': tips[area],
                    'impact': f'Could reduce bio age by ~{impact} years'
                })
        
        return {
            'chronological_age': self.age,
            'biological_age': round(bio_age, 1),
            'age_difference': round(bio_age - self.age, 1),
            'health_score': round(health_score, 1),
            'bmi': round(bmi, 1),
            'component_scores': {k: round(v, 1) for k, v in scores.items()},
            'organ_ages': {k: round(v, 1) for k, v in organ_ages.items()},
            'recommendations': recommendations
        }
    
    def print_results(self, results):
        """Beautiful output"""
        print("\n" + "="*70)
        print("🎉 YOUR BIOLOGICAL AGE ANALYSIS")
        print("="*70)
        
        print(f"\n📅 Chronological Age: {results['chronological_age']} years")
        print(f"🧬 Biological Age: {results['biological_age']} years")
        
        diff = results['age_difference']
        if diff < -3:
            print(f"🌟 AMAZING! You are {abs(diff):.1f} years YOUNGER! 🎉")
        elif diff < 0:
            print(f"✅ GREAT! You are {abs(diff):.1f} years younger biologically!")
        elif diff < 2:
            print(f"😐 Your biological age matches your actual age")
        elif diff < 5:
            print(f"⚠️ You are {diff:.1f} years OLDER - time to improve")
        else:
            print(f"🚨 You are {diff:.1f} years OLDER - urgent action needed!")
        
        print(f"\n💯 Overall Health Score: {results['health_score']}/100")
        print(f"📊 BMI: {results['bmi']}")
        
        print("\n" + "-"*70)
        print("📈 HEALTH COMPONENT SCORES")
        print("-"*70)
        
        for name, score in results['component_scores'].items():
            bar = "█" * int(score/5) + "░" * (20 - int(score/5))
            status = "✅" if score >= 75 else "⚠️" if score >= 50 else "❌"
            print(f"{status} {name:20} [{bar}] {score:5.1f}/100")
        
        print("\n" + "-"*70)
        print("🏥 ORGAN SYSTEM BIOLOGICAL AGES")
        print("-"*70)
        
        for organ, age in results['organ_ages'].items():
            diff_org = age - results['chronological_age']
            if diff_org < 0:
                status = "✅"
                label = f"({abs(diff_org):.1f} years younger)"
            elif diff_org < 3:
                status = "😐"
                label = f"(+{diff_org:.1f} years)"
            else:
                status = "⚠️"
                label = f"(+{diff_org:.1f} years)"
            
            print(f"{status} {organ:22} {age:5.1f} years {label}")
        
        if results['recommendations']:
            print("\n" + "-"*70)
            print("💡 TOP IMPROVEMENT RECOMMENDATIONS")
            print("-"*70)
            
            for i, rec in enumerate(results['recommendations'], 1):
                emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡"
                print(f"\n{i}. {emoji} {rec['area'].upper()} - Current: {rec['score']}/100")
                print(f"   {rec['tip']}")
                print(f"   💪 {rec['impact']}")
        
        print("\n" + "="*70)


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("\n🏥 BIOLOGICAL AGE CALCULATOR - INTELLIGENT SIMULATION")
    print("="*70)
    print("\n✨ This demo shows 90% automation concept")
    print("📱 In production: connects to Google Fit, Fitbit, Apple Health\n")
    
    # Basic input
    age = int(input("Your age: ") or 20)
    height = float(input("Height (cm): ") or 170)
    
    # Initialize
    simulator = SmartHealthSimulator(age, height)
    
    # Quick assessment
    profile, answers = simulator.ask_lifestyle_questions()
    simulator.lifestyle_profile = profile
    
    # Generate realistic data
    health_data = simulator.generate_realistic_data(answers)
    
    # Calculate biological age
    results = simulator.calculate_biological_age(health_data)
    
    # Show results
    simulator.print_results(results)
    
    print("\n💡 FOR IMAGINE CUP DEMO:")
    print("   • This shows the automation concept")
    print("   • Replace simulation with real APIs for production")
    print("   • Add your test email to Google Console for real data")
    print("   • Or use Fitbit API (easier auth)")
    print("\n🚀 Backend ready for UI integration!\n")