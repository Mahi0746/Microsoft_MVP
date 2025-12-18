#!/usr/bin/env python3
"""
Simple test script for therapy game functionality.
Run this to verify the therapy game routes are working.
"""

import asyncio
import json
from datetime import datetime
from services.therapy_game_service import TherapyGameService, ExerciseType, DifficultyLevel


async def test_therapy_game_service():
    """Test the therapy game service functionality."""
    
    print("🎮 Testing HealthSync AI Therapy Game Service")
    print("=" * 50)
    
    try:
        # Test 1: Initialize MediaPipe
        print("1. Initializing MediaPipe models...")
        await TherapyGameService.initialize_mediapipe()
        print("✅ MediaPipe models initialized successfully")
        
        # Test 2: Start a therapy session
        print("\n2. Starting therapy session...")
        session_data = await TherapyGameService.start_therapy_session(
            user_id="test_user_123",
            exercise_type=ExerciseType.NECK_ROTATION.value,
            difficulty=DifficultyLevel.BEGINNER.value,
            duration_minutes=5,
            pain_level_before=6
        )
        
        print(f"✅ Session started: {session_data['session_id']}")
        print(f"   Exercise: {session_data['exercise_config']['name']}")
        print(f"   Target reps: {session_data['target_repetitions']}")
        print(f"   Estimated calories: {session_data['estimated_calories']}")
        
        # Test 3: Exercise configurations
        print("\n3. Testing exercise configurations...")
        for exercise_type in ExerciseType:
            config = TherapyGameService.EXERCISE_CONFIGS[exercise_type]
            print(f"   {exercise_type.value}: {config['name']}")
        
        print(f"✅ Found {len(ExerciseType)} exercise types")
        
        # Test 4: Achievement system
        print("\n4. Testing achievement system...")
        achievements = TherapyGameService.ACHIEVEMENT_SYSTEM
        print(f"✅ Found {len(achievements)} achievements:")
        for achievement_id, achievement in achievements.items():
            print(f"   {achievement_id}: {achievement['name']} ({achievement['points']} pts)")
        
        # Test 5: Difficulty levels
        print("\n5. Testing difficulty levels...")
        for difficulty in DifficultyLevel:
            print(f"   {difficulty.value}")
        
        print(f"✅ Found {len(DifficultyLevel)} difficulty levels")
        
        print("\n🎉 All therapy game tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_exercise_types():
    """Test exercise type validation."""
    
    print("\n🏋️ Testing Exercise Types")
    print("-" * 30)
    
    # Valid exercise types
    valid_types = [
        "neck_rotation",
        "shoulder_rolls", 
        "arm_raises",
        "spine_twist",
        "leg_lifts",
        "balance_training",
        "breathing_exercise",
        "finger_exercises"
    ]
    
    for exercise_type in valid_types:
        try:
            enum_value = ExerciseType(exercise_type)
            config = TherapyGameService.EXERCISE_CONFIGS[enum_value]
            print(f"✅ {exercise_type}: {config['name']}")
        except Exception as e:
            print(f"❌ {exercise_type}: {str(e)}")


def test_difficulty_levels():
    """Test difficulty level validation."""
    
    print("\n📊 Testing Difficulty Levels")
    print("-" * 30)
    
    valid_difficulties = [
        "beginner",
        "intermediate", 
        "advanced",
        "rehabilitation"
    ]
    
    for difficulty in valid_difficulties:
        try:
            enum_value = DifficultyLevel(difficulty)
            print(f"✅ {difficulty}")
        except Exception as e:
            print(f"❌ {difficulty}: {str(e)}")


def test_calorie_estimation():
    """Test calorie estimation functionality."""
    
    print("\n🔥 Testing Calorie Estimation")
    print("-" * 30)
    
    test_cases = [
        (ExerciseType.NECK_ROTATION, 5, DifficultyLevel.BEGINNER),
        (ExerciseType.ARM_RAISES, 10, DifficultyLevel.INTERMEDIATE),
        (ExerciseType.LEG_LIFTS, 15, DifficultyLevel.ADVANCED),
        (ExerciseType.BREATHING_EXERCISE, 20, DifficultyLevel.REHABILITATION)
    ]
    
    for exercise_type, duration, difficulty in test_cases:
        calories = TherapyGameService._estimate_calories(exercise_type, duration, difficulty)
        print(f"✅ {exercise_type.value} ({difficulty.value}, {duration}min): {calories} calories")


def test_points_calculation():
    """Test points calculation functionality."""
    
    print("\n🏆 Testing Points Calculation")
    print("-" * 30)
    
    test_cases = [
        (10, DifficultyLevel.BEGINNER),
        (15, DifficultyLevel.INTERMEDIATE),
        (20, DifficultyLevel.ADVANCED),
        (8, DifficultyLevel.REHABILITATION)
    ]
    
    for target_reps, difficulty in test_cases:
        points = TherapyGameService._calculate_potential_points(target_reps, difficulty)
        print(f"✅ {target_reps} reps ({difficulty.value}): {points} potential points")


async def main():
    """Run all tests."""
    
    print("🚀 HealthSync AI - Therapy Game Test Suite")
    print("=" * 60)
    
    # Run synchronous tests first
    test_exercise_types()
    test_difficulty_levels()
    test_calorie_estimation()
    test_points_calculation()
    
    # Run async tests
    await test_therapy_game_service()
    
    print("\n" + "=" * 60)
    print("✨ Test suite completed!")


if __name__ == "__main__":
    asyncio.run(main())