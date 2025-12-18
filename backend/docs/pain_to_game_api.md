# Pain-to-Game Therapy API Documentation

The HealthSync AI Pain-to-Game Therapy system transforms rehabilitation exercises into an engaging, gamified experience with real-time pain detection, adaptive difficulty, and comprehensive progress tracking. This API provides advanced motion tracking, facial pain analysis, and social gamification features.

## Features

- **🎯 Gamified Exercise System**: Points, levels, achievements, and streaks
- **😣 Real-time Pain Detection**: Facial Action Unit analysis (AU4, AU6, AU7, AU9, AU10)
- **🎚️ Adaptive Difficulty**: Auto-adjust based on pain levels and performance
- **🏆 Leaderboards**: Weekly, monthly, and all-time rankings
- **🎮 Daily Challenges**: Personalized challenges based on user level
- **🔥 Streak Tracking**: Consistency rewards and milestone achievements
- **📊 Advanced Analytics**: Performance trends and pain tracking
- **💪 Social Features**: Anonymous competition and motivation

## Base URL

```
/api/v1/therapy-game
```

## Authentication

All endpoints require JWT authentication via the `Authorization: Bearer <token>` header.

## Core Exercise System

### Start Enhanced Therapy Session

```http
POST /session/start
```

**Request Body:**
```json
{
  "exercise_type": "arm_raises",
  "difficulty": "intermediate",
  "duration_minutes": 10,
  "pain_level_before": 6
}
```

**Enhanced Response:**
```json
{
  "session_id": "therapy_user123_20251217_103000",
  "exercise_config": {
    "name": "Therapeutic Arm Raises",
    "description": "Controlled arm raises for shoulder rehabilitation",
    "target_muscles": ["deltoids", "rotator_cuff", "serratus_anterior"],
    "pain_conditions": ["shoulder_impingement", "rotator_cuff_injury"]
  },
  "target_repetitions": 15,
  "instructions": {
    "setup": ["Position yourself in front of camera"],
    "exercise_steps": ["Stand with feet shoulder-width apart", "Raise both arms to shoulder height"],
    "safety_tips": ["Move slowly and gently", "Stop if pain increases"],
    "pain_adaptations": ["Exercise will auto-adapt based on facial pain detection"]
  },
  "tracking_points": ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder"],
  "estimated_calories": 12.5,
  "potential_points": 180,
  "gamification": {
    "current_level": 3,
    "level_name": "Apprentice",
    "level_multiplier": 1.2,
    "points_to_next_level": 250,
    "current_streak": 5,
    "daily_challenge_progress": "2/3 completed"
  }
}
```

### Enhanced Motion Processing with Pain Detection

```http
POST /session/motion
```

**Request Body:**
```json
{
  "session_id": "therapy_user123_20251217_103000",
  "frame_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "timestamp": 1702810200.123
}
```

**Enhanced Response with Pain Analysis:**
```json
{
  "success": true,
  "movement_analysis": {
    "movement_detected": true,
    "repetition_completed": true,
    "accuracy_score": 0.85,
    "quality_score": 0.82,
    "form_feedback": ["Good range of motion", "Keep shoulders level"]
  },
  "pain_analysis": {
    "pain_detected": true,
    "pain_level": 0.3,
    "pain_category": "mild",
    "confidence": 0.78,
    "action_units": {
      "AU4": {"intensity": 0.2, "confidence": 0.7, "description": "Brow lowering"},
      "AU6": {"intensity": 0.4, "confidence": 0.6, "description": "Cheek raising"},
      "AU7": {"intensity": 0.3, "confidence": 0.65, "description": "Eye tightening"}
    },
    "recommendations": [
      "Continue with caution",
      "Gentle movements recommended",
      "Monitor for pain increase"
    ]
  },
  "adaptation": {
    "difficulty_changed": false,
    "recommendations": ["Take a 30-second rest break", "Focus on form over speed"],
    "auto_adjustments": ["Rest break suggested"]
  },
  "feedback": {
    "message": "😐 Mild discomfort - proceed with caution",
    "type": "info",
    "encouragement": "You're doing great, stay mindful",
    "points_earned": 12,
    "total_points": 1250,
    "achievement": null,
    "milestone": null,
    "motivation": "👍 Good job, keep it up!",
    "level_progress": "🎮 50 points to Expert!",
    "visual_cues": ["Reduce intensity", "Take breaks as needed"],
    "audio_cue": "gentle_chime"
  },
  "progress": {
    "completed_reps": 8,
    "target_reps": 15,
    "accuracy": 0.83,
    "quality_score": 0.81,
    "session_points": 96,
    "streak_bonus": 5
  }
}
```

## Gamification System

### Get Leaderboard

```http
GET /leaderboard?timeframe=weekly&exercise_type=arm_raises&limit=50
```

**Response:**
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "user_id": null,
      "display_name": "User7892",
      "total_points": 2850,
      "level": {
        "level": 5,
        "name": "Expert",
        "color": "#2196F3"
      },
      "stats": {
        "total_sessions": 12,
        "total_repetitions": 180,
        "average_accuracy": 92.5,
        "total_calories": 45.2,
        "pain_reduction_total": 18,
        "achievements_count": 8
      },
      "last_active": "2025-12-17T09:30:00Z",
      "is_current_user": false
    },
    {
      "rank": 2,
      "user_id": "user_12345",
      "display_name": "User2345",
      "total_points": 2650,
      "level": {
        "level": 4,
        "name": "Practitioner",
        "color": "#00BCD4"
      },
      "stats": {
        "total_sessions": 10,
        "total_repetitions": 150,
        "average_accuracy": 88.2,
        "total_calories": 38.5,
        "pain_reduction_total": 15,
        "achievements_count": 6
      },
      "last_active": "2025-12-17T10:15:00Z",
      "is_current_user": true
    }
  ],
  "timeframe": "weekly",
  "exercise_type": "arm_raises",
  "total_entries": 25,
  "user_rank": 2,
  "last_updated": "2025-12-17T10:30:00Z"
}
```

### Get Personal Ranking

```http
GET /leaderboard/personal?timeframe=weekly
```

**Response:**
```json
{
  "user_rank": 2,
  "total_competitors": 25,
  "user_entry": {
    "rank": 2,
    "display_name": "User2345",
    "total_points": 2650,
    "level": {"level": 4, "name": "Practitioner", "color": "#00BCD4"}
  },
  "nearby_competitors": [
    {
      "rank": 1,
      "display_name": "User7892",
      "total_points": 2850,
      "points_ahead": 200
    },
    {
      "rank": 3,
      "display_name": "User5671",
      "total_points": 2400,
      "points_behind": 250
    }
  ],
  "percentile": 92.0
}
```

### Get Level System

```http
GET /levels
```

**Response:**
```json
{
  "current_level": {
    "level": 4,
    "name": "Practitioner",
    "color": "#00BCD4",
    "multiplier": 1.3,
    "points_required": 2500
  },
  "next_level": {
    "level": 5,
    "name": "Expert",
    "color": "#2196F3",
    "multiplier": 1.4,
    "points_required": 5000
  },
  "user_stats": {
    "total_points": 2650,
    "points_needed_for_next": 2350,
    "progress_percentage": 6.0
  },
  "all_levels": [
    {
      "level": 1,
      "name": "Beginner",
      "points_required": 0,
      "multiplier": 1.0,
      "color": "#8BC34A",
      "unlocked": true
    },
    {
      "level": 10,
      "name": "Immortal",
      "points_required": 150000,
      "multiplier": 2.0,
      "color": "#FFD700",
      "unlocked": false
    }
  ]
}
```

### Get User Streaks

```http
GET /streaks
```

**Response:**
```json
{
  "current_streak": 7,
  "longest_streak": 15,
  "weekly_consistency": 85.7,
  "monthly_consistency": 73.3,
  "total_active_days": 45,
  "streak_history": [
    {
      "date": "2025-12-17",
      "has_session": true,
      "day_name": "Tue"
    },
    {
      "date": "2025-12-16",
      "has_session": true,
      "day_name": "Mon"
    }
  ],
  "streak_milestones": {
    "next_milestone": {
      "days": 14,
      "name": "Fortnight Fighter",
      "days_remaining": 7
    },
    "milestone_progress": 50.0
  }
}
```

## Challenge System

### Get Daily Challenges

```http
GET /challenges
```

**Response:**
```json
{
  "daily_challenges": [
    {
      "id": "daily_session_20251217",
      "name": "Daily Exercise",
      "description": "Complete one therapy session today",
      "type": "session_count",
      "target": 1,
      "reward_points": 50,
      "difficulty": "easy",
      "icon": "🎯",
      "completed": true
    },
    {
      "id": "accuracy_challenge_20251217",
      "name": "Precision Practice",
      "description": "Achieve 80%+ accuracy in a session",
      "type": "accuracy",
      "target": 0.8,
      "reward_points": 75,
      "difficulty": "medium",
      "icon": "🎯",
      "completed": false
    },
    {
      "id": "pain_reduction_20251217",
      "name": "Pain Warrior",
      "description": "Reduce pain level by 3+ points in a session",
      "type": "pain_reduction",
      "target": 3,
      "reward_points": 150,
      "difficulty": "hard",
      "icon": "⚔️",
      "completed": false
    }
  ],
  "weekly_challenges": [
    {
      "id": "weekly_consistency_2025-W50",
      "name": "Weekly Warrior",
      "description": "Complete exercises 5 days this week",
      "type": "weekly_sessions",
      "target": 5,
      "reward_points": 300,
      "difficulty": "medium",
      "icon": "📅",
      "expires": "Sunday 23:59 UTC"
    }
  ],
  "challenge_date": "2025-12-17",
  "user_level": 4,
  "refresh_time": "00:00 UTC"
}
```

### Complete Challenge

```http
POST /challenges/{challenge_id}/complete
```

**Response:**
```json
{
  "success": true,
  "message": "Challenge completed!",
  "points_awarded": 75,
  "challenge_id": "accuracy_challenge_20251217",
  "new_total_points": 2725,
  "level_up": false,
  "achievements_unlocked": []
}
```

## Pain Detection System

### Get Pain Detection Configuration

```http
GET /pain-detection/config
```

**Response:**
```json
{
  "facial_action_units": {
    "AU4": {
      "name": "Brow Lowerer",
      "pain_indicator": true,
      "weight": 0.3
    },
    "AU6": {
      "name": "Cheek Raiser",
      "pain_indicator": true,
      "weight": 0.2
    },
    "AU7": {
      "name": "Lid Tightener",
      "pain_indicator": true,
      "weight": 0.25
    },
    "AU9": {
      "name": "Nose Wrinkler",
      "pain_indicator": true,
      "weight": 0.15
    },
    "AU10": {
      "name": "Upper Lip Raiser",
      "pain_indicator": true,
      "weight": 0.1
    }
  },
  "pain_thresholds": {
    "none": 0.0,
    "mild": 0.2,
    "moderate": 0.4,
    "severe": 0.6,
    "extreme": 0.8
  },
  "adaptation_rules": {
    "pain_increase": {
      "difficulty_reduction": 0.2,
      "rest_suggestion": true
    },
    "pain_stable": {
      "continue_current": true
    },
    "pain_decrease": {
      "difficulty_increase": 0.1,
      "encouragement": true
    }
  },
  "enabled": true
}
```

### Get Game Mechanics

```http
GET /game-mechanics
```

**Response:**
```json
{
  "game_mechanics": {
    "base_points_per_rep": 10,
    "accuracy_bonus_multiplier": 0.5,
    "streak_bonus_multiplier": 0.1,
    "pain_reduction_bonus": 20,
    "perfect_session_bonus": 100,
    "consistency_bonus": 50,
    "level_bonus_multiplier": 0.2
  },
  "achievement_system": {
    "first_session": {
      "name": "Getting Started",
      "points": 50,
      "description": "Complete your first therapy session",
      "icon": "🎯"
    },
    "pain_reducer": {
      "name": "Pain Reducer",
      "points": 100,
      "description": "Reduce pain level by 3+ points in a session",
      "icon": "💊"
    },
    "streak_master": {
      "name": "Streak Master",
      "points": 500,
      "description": "Maintain 30-day exercise streak",
      "icon": "🔥"
    }
  },
  "level_system": {
    "1": {"name": "Beginner", "points_required": 0, "multiplier": 1.0, "color": "#8BC34A"},
    "5": {"name": "Expert", "points_required": 5000, "multiplier": 1.4, "color": "#2196F3"},
    "10": {"name": "Immortal", "points_required": 150000, "multiplier": 2.0, "color": "#FFD700"}
  },
  "scoring_explanation": {
    "base_points": "10 points per repetition",
    "accuracy_bonus": "Up to 50% bonus for perfect form",
    "streak_bonus": "10% bonus for consecutive days",
    "pain_reduction_bonus": "20 points for reducing pain",
    "perfect_session_bonus": "100 points for 100% accuracy",
    "level_multiplier": "20% bonus per level"
  }
}
```

## Pain Detection Features

### Facial Action Units (AU) Analysis

The system analyzes facial expressions to detect pain indicators:

| Action Unit | Description | Pain Indicator | Weight |
|-------------|-------------|----------------|---------|
| AU4 | Brow Lowerer | ✅ High | 0.30 |
| AU6 | Cheek Raiser | ✅ Medium | 0.20 |
| AU7 | Lid Tightener | ✅ Medium | 0.25 |
| AU9 | Nose Wrinkler | ✅ Low | 0.15 |
| AU10 | Upper Lip Raiser | ✅ Low | 0.10 |

### Pain Categories and Responses

| Category | Pain Level | System Response |
|----------|------------|-----------------|
| **None** | 0.0 - 0.2 | Normal exercise progression |
| **Mild** | 0.2 - 0.4 | Gentle reminders, continue with caution |
| **Moderate** | 0.4 - 0.6 | Suggest rest breaks, reduce intensity |
| **Severe** | 0.6 - 0.8 | Auto-reduce difficulty, frequent breaks |
| **Extreme** | 0.8 - 1.0 | Stop exercise immediately, seek help |

### Adaptive Difficulty Rules

```json
{
  "pain_increase": {
    "action": "Reduce difficulty by 20%",
    "suggestion": "Take rest breaks",
    "monitoring": "Increased pain level monitoring"
  },
  "pain_stable": {
    "action": "Continue current difficulty",
    "suggestion": "Maintain current pace",
    "monitoring": "Regular pain level checks"
  },
  "pain_decrease": {
    "action": "Consider increasing difficulty by 10%",
    "suggestion": "Positive reinforcement",
    "monitoring": "Celebrate progress"
  }
}
```

## Gamification Elements

### Achievement System

| Achievement | Points | Trigger | Icon |
|-------------|--------|---------|------|
| Getting Started | 50 | First session | 🎯 |
| Pain Reducer | 100 | Reduce pain by 3+ points | 💊 |
| Pain-Free Hero | 1000 | Report pain level 0 | 🏆 |
| Weekly Warrior | 200 | 5 sessions in a week | 📅 |
| Streak Master | 500 | 30-day streak | 🔥 |
| Precision Pro | 150 | 90%+ accuracy in 5 sessions | 🎯 |
| Perfect Form | 250 | 100% accuracy in session | ✨ |
| Pain Warrior | 400 | Exercise despite high pain | ⚔️ |

### Level System

| Level | Name | Points Required | Multiplier | Color |
|-------|------|----------------|------------|-------|
| 1 | Beginner | 0 | 1.0x | Green |
| 2 | Novice | 500 | 1.1x | Light Green |
| 3 | Apprentice | 1,200 | 1.2x | Teal |
| 4 | Practitioner | 2,500 | 1.3x | Cyan |
| 5 | Expert | 5,000 | 1.4x | Blue |
| 6 | Master | 10,000 | 1.5x | Indigo |
| 7 | Grandmaster | 20,000 | 1.6x | Purple |
| 8 | Legend | 40,000 | 1.7x | Pink |
| 9 | Champion | 75,000 | 1.8x | Orange |
| 10 | Immortal | 150,000 | 2.0x | Gold |

### Points Calculation

```javascript
// Base points per repetition
basePoints = 10

// Accuracy bonus (up to 50% extra)
accuracyBonus = basePoints * accuracy * 0.5

// Streak bonus (10% per consecutive day)
streakBonus = basePoints * streakDays * 0.1

// Level multiplier (20% per level)
levelMultiplier = 1 + (userLevel * 0.2)

// Total points
totalPoints = (basePoints + accuracyBonus + streakBonus) * levelMultiplier

// Special bonuses
if (accuracy === 1.0) totalPoints += 100  // Perfect form
if (painReduction >= 3) totalPoints += 20  // Pain reduction
```

## Real-time Feedback System

### Feedback Types

| Type | Description | Visual Indicator | Audio Cue |
|------|-------------|------------------|-----------|
| **Success** | Repetition completed | Green checkmark | Success chime |
| **Warning** | Form issues or mild pain | Yellow caution | Gentle tone |
| **Error** | Severe pain detected | Red stop sign | Warning tone |
| **Info** | General guidance | Blue info icon | Soft beep |

### Motivational Messages

Based on performance and pain levels:

- **No Pain + High Accuracy**: "🔥 You're on fire! Amazing form!"
- **No Pain + Good Accuracy**: "💪 Excellent technique!"
- **No Pain + Average**: "👍 Good job, keep it up!"
- **Mild Pain**: "😐 Mild discomfort - proceed with caution"
- **Moderate Pain**: "😣 Discomfort detected - take it easy"
- **Severe Pain**: "⚠️ Pain detected - please stop and rest"

## WebSocket Integration

For real-time pain detection and feedback:

```javascript
const ws = new WebSocket('wss://api.healthsync.ai/ws/therapy-game');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'pain_detected') {
    // Handle pain detection
    showPainWarning(data.pain_analysis);
    adaptExercise(data.adaptation);
  }
  
  if (data.type === 'achievement_unlocked') {
    // Show achievement animation
    showAchievement(data.achievement);
  }
};
```

## Error Handling

### Pain Detection Errors

```json
{
  "error": true,
  "message": "Pain detection failed - camera not accessible",
  "code": "PAIN_DETECTION_ERROR",
  "fallback": "Continue with manual pain reporting",
  "retry_suggestion": "Check camera permissions"
}
```

### Gamification Errors

```json
{
  "error": true,
  "message": "Leaderboard temporarily unavailable",
  "code": "LEADERBOARD_ERROR",
  "fallback": "Show personal progress only",
  "retry_after": 300
}
```

## Privacy and Safety

### Data Protection
- Facial analysis is processed in real-time, no images stored
- Leaderboards use anonymized display names
- Personal health data is encrypted and secured
- Users can opt-out of pain detection features

### Safety Features
- Automatic exercise termination on severe pain detection
- Medical disclaimers on all pain-related feedback
- Encouragement to consult healthcare providers
- Gradual difficulty progression to prevent injury

## SDK Examples

### React Native Integration

```javascript
import { PainToGameTherapy } from '@healthsync/therapy-sdk';

const therapy = new PainToGameTherapy({
  apiKey: 'your-jwt-token',
  enablePainDetection: true,
  enableGamification: true
});

// Start session with pain detection
const session = await therapy.startSession({
  exerciseType: 'arm_raises',
  difficulty: 'intermediate',
  painLevelBefore: 6
});

// Process frame with pain analysis
const result = await therapy.processFrame(session.sessionId, frameData);

if (result.pain_analysis.pain_detected) {
  // Handle pain detection
  showPainFeedback(result.pain_analysis);
}

// Show gamification elements
updatePoints(result.feedback.points_earned);
showAchievements(result.feedback.achievement);
```

### Python Integration

```python
from healthsync import PainToGameClient

client = PainToGameClient(
    api_key='your-jwt-token',
    enable_pain_detection=True
)

# Start enhanced session
session = await client.start_session(
    exercise_type='arm_raises',
    difficulty='intermediate',
    pain_level_before=6
)

# Process with pain detection
result = await client.process_frame_with_pain_detection(
    session.session_id,
    frame_data
)

# Handle adaptive difficulty
if result.adaptation.difficulty_changed:
    print(f"Difficulty adapted: {result.adaptation.new_difficulty}")
```

## Best Practices

### Pain Detection
- Ensure good lighting for facial analysis
- Position camera at eye level
- Avoid shadows on face
- Calibrate pain thresholds per user
- Provide manual override options

### Gamification
- Balance challenge with achievability
- Provide multiple progression paths
- Celebrate small wins frequently
- Maintain fair competition
- Respect privacy in leaderboards

### User Experience
- Gradual introduction of features
- Clear explanation of pain detection
- Optional gamification elements
- Accessible design for all abilities
- Regular feedback and encouragement