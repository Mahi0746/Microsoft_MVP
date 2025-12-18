# Therapy Game API Documentation

The HealthSync AI Therapy Game provides gamified rehabilitation exercises with real-time motion tracking using MediaPipe. This API enables users to perform therapeutic exercises while receiving AI-powered feedback and progress tracking.

## Features

- **8 Exercise Types**: Neck rotation, shoulder rolls, arm raises, spine twists, leg lifts, balance training, breathing exercises, and finger exercises
- **4 Difficulty Levels**: Beginner, Intermediate, Advanced, and Rehabilitation
- **Real-time Motion Tracking**: Using MediaPipe for pose, hand, and face detection
- **Gamification**: Points, achievements, and progress tracking
- **Pain Management**: Before/after pain level tracking
- **Analytics**: Performance trends and insights

## Base URL

```
/api/v1/therapy-game
```

## Authentication

All endpoints require JWT authentication via the `Authorization: Bearer <token>` header.

## Endpoints

### Session Management

#### Start Therapy Session
```http
POST /session/start
```

**Request Body:**
```json
{
  "exercise_type": "neck_rotation",
  "difficulty": "beginner", 
  "duration_minutes": 5,
  "pain_level_before": 6
}
```

**Response:**
```json
{
  "session_id": "therapy_user123_20251217_103000",
  "exercise_config": {
    "name": "Neck Rotation Therapy",
    "description": "Gentle neck rotations to improve mobility",
    "target_muscles": ["neck", "upper_trapezius"],
    "pain_conditions": ["neck_pain", "tension_headaches"]
  },
  "target_repetitions": 8,
  "instructions": {
    "setup": ["Position yourself in front of camera"],
    "exercise_steps": ["Sit with spine straight", "Turn head right"],
    "safety_tips": ["Move slowly and gently"]
  },
  "tracking_points": ["nose", "left_ear", "right_ear"],
  "estimated_calories": 7.5,
  "potential_points": 80
}
```

#### Process Motion Frame
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

**Response:**
```json
{
  "success": true,
  "movement_analysis": {
    "movement_detected": true,
    "repetition_completed": true,
    "accuracy_score": 0.85,
    "quality_score": 0.82,
    "form_feedback": ["Good range of motion"],
    "movement_range": 25.5
  },
  "feedback": {
    "message": "Great! Repetition completed.",
    "type": "success",
    "visual_cues": [],
    "audio_cue": "success_chime",
    "encouragement": "Keep it up!"
  },
  "progress": {
    "completed_reps": 3,
    "target_reps": 8,
    "accuracy": 0.83,
    "quality_score": 0.81
  }
}
```

#### Complete Session
```http
POST /session/complete
```

**Request Body:**
```json
{
  "session_id": "therapy_user123_20251217_103000",
  "pain_level_after": 3,
  "user_feedback": "Felt much better after the exercises"
}
```

**Response:**
```json
{
  "session_result": {
    "session_id": "therapy_user123_20251217_103000",
    "completed_repetitions": 8,
    "accuracy_score": 0.85,
    "pain_level_before": 6,
    "pain_level_after": 3,
    "calories_burned": 6.8,
    "points_earned": 95,
    "achievements_unlocked": ["pain_reducer"]
  },
  "summary": {
    "completion_rate": "100.0%",
    "accuracy_score": "85.0%",
    "pain_reduction": 3,
    "calories_burned": 6.8,
    "points_earned": 95,
    "achievements_unlocked": 1
  },
  "achievements": ["pain_reducer"],
  "recommendations": [
    "Great job! You reduced your pain level by 3 points.",
    "Excellent completion rate! Consider increasing difficulty next time."
  ]
}
```

### Progress & Analytics

#### Get User Progress
```http
GET /progress
```

**Response:**
```json
{
  "total_sessions": 15,
  "total_points": 1250,
  "total_calories": 125.5,
  "total_exercise_time": 75,
  "achievements": ["first_session", "pain_reducer", "accuracy_master"],
  "exercise_stats": {
    "neck_rotation": {
      "sessions": 8,
      "total_reps": 64,
      "avg_accuracy": 0.82,
      "best_accuracy": 0.95
    }
  },
  "pain_trend": "improving",
  "consistency_score": 0.85
}
```

#### Get Session History
```http
GET /sessions/history?limit=10&offset=0&exercise_type=neck_rotation
```

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "therapy_user123_20251217_103000",
      "exercise_type": "neck_rotation",
      "difficulty": "beginner",
      "start_time": "2025-12-17T10:30:00Z",
      "duration_minutes": 5,
      "completed_repetitions": 8,
      "target_repetitions": 8,
      "accuracy_score": 0.85,
      "pain_level_before": 6,
      "pain_level_after": 3,
      "calories_burned": 6.8,
      "points_earned": 95
    }
  ],
  "total_count": 15,
  "has_more": true
}
```

### Exercise Information

#### Get Available Exercises
```http
GET /exercises
```

**Response:**
```json
{
  "exercises": [
    {
      "type": "neck_rotation",
      "name": "Neck Rotation Therapy",
      "description": "Gentle neck rotations to improve mobility",
      "target_muscles": ["neck", "upper_trapezius"],
      "pain_conditions": ["neck_pain", "tension_headaches"],
      "duration_range": [2, 10],
      "repetition_range": [5, 20],
      "difficulty_levels": ["beginner", "intermediate", "advanced", "rehabilitation"]
    }
  ],
  "total_count": 8
}
```

#### Get Exercise Details
```http
GET /exercises/neck_rotation
```

**Response:**
```json
{
  "exercise": {
    "type": "neck_rotation",
    "name": "Neck Rotation Therapy",
    "description": "Gentle neck rotations to improve mobility",
    "target_muscles": ["neck", "upper_trapezius"],
    "pain_conditions": ["neck_pain", "tension_headaches"],
    "tracking_points": ["nose", "left_ear", "right_ear"],
    "movement_threshold": 15.0,
    "accuracy_threshold": 0.7
  },
  "user_stats": {
    "sessions": 8,
    "total_reps": 64,
    "avg_accuracy": 0.82,
    "best_accuracy": 0.95
  },
  "difficulty_levels": [
    {
      "level": "beginner",
      "description": "Perfect for those new to exercise or with limited mobility"
    }
  ]
}
```

### Achievements

#### Get Achievements
```http
GET /achievements
```

**Response:**
```json
{
  "achievements": [
    {
      "id": "first_session",
      "name": "Getting Started",
      "description": "Complete your first therapy session",
      "points": 50,
      "unlocked": true,
      "unlock_date": null
    }
  ],
  "stats": {
    "total_achievements": 7,
    "unlocked_achievements": 3,
    "completion_percentage": 42.9,
    "total_achievement_points": 350
  }
}
```

### Analytics

#### Get Pain Trends
```http
GET /analytics/pain-trends?days=30
```

**Response:**
```json
{
  "trends": [
    {
      "date": "2025-12-17T10:30:00Z",
      "pain_before": 6,
      "pain_after": 3,
      "reduction": 3,
      "exercise_type": "neck_rotation"
    }
  ],
  "stats": {
    "average_reduction": 2.1,
    "best_reduction": 4,
    "worst_reduction": -1,
    "total_sessions": 15,
    "improvement_rate": 86.7
  },
  "period_days": 30,
  "data_points": 15
}
```

#### Get Exercise Performance
```http
GET /analytics/exercise-performance?exercise_type=neck_rotation
```

**Response:**
```json
{
  "performance": {
    "neck_rotation": {
      "total_sessions": 8,
      "total_repetitions": 64,
      "average_accuracy": 82.0,
      "best_accuracy": 95.0,
      "average_reps_per_session": 8.0
    }
  },
  "total_exercise_types": 1,
  "filter_applied": true
}
```

## Exercise Types

| Type | Name | Target Areas |
|------|------|--------------|
| `neck_rotation` | Neck Rotation Therapy | Neck, upper trapezius |
| `shoulder_rolls` | Shoulder Roll Therapy | Deltoids, trapezius, rhomboids |
| `arm_raises` | Therapeutic Arm Raises | Deltoids, rotator cuff |
| `spine_twist` | Spinal Rotation Therapy | Obliques, erector spinae |
| `leg_lifts` | Therapeutic Leg Lifts | Hip flexors, quadriceps, core |
| `balance_training` | Balance and Stability Training | Core, stabilizers |
| `breathing_exercise` | Therapeutic Breathing | Diaphragm, intercostals |
| `finger_exercises` | Hand and Finger Therapy | Hand muscles, finger flexors |

## Difficulty Levels

| Level | Description |
|-------|-------------|
| `beginner` | Perfect for those new to exercise or with limited mobility |
| `intermediate` | Suitable for regular exercisers with good mobility |
| `advanced` | Challenging exercises for experienced users |
| `rehabilitation` | Gentle exercises designed for injury recovery |

## Achievement System

| Achievement | Points | Description |
|-------------|--------|-------------|
| Getting Started | 50 | Complete your first therapy session |
| Pain Reducer | 100 | Reduce pain level by 3+ points in a session |
| Weekly Warrior | 200 | Complete exercises 5 days in a week |
| Precision Pro | 150 | Achieve 90%+ accuracy in 5 sessions |
| Endurance Champion | 300 | Complete 30-minute session |
| Streak Master | 500 | Maintain 30-day exercise streak |
| Pain-Free Hero | 1000 | Report pain level 0 after session |

## Error Handling

All endpoints return structured error responses:

```json
{
  "error": true,
  "message": "Invalid exercise type: invalid_exercise",
  "status_code": 400,
  "timestamp": "2025-12-17T10:30:00Z",
  "path": "/api/v1/therapy-game/session/start",
  "request_id": "req_123456"
}
```

## Rate Limits

- General endpoints: 100 requests/minute
- Motion processing: 5 requests/minute (due to image processing)
- All other endpoints: 100 requests/minute

## WebSocket Support

For real-time motion tracking, consider using the WebSocket endpoint at `/ws/therapy-game` for lower latency frame processing.

## SDK Examples

### JavaScript/TypeScript
```javascript
const therapyGame = new HealthSyncTherapyGame({
  apiKey: 'your-jwt-token',
  baseUrl: 'https://api.healthsync.ai/api/v1/therapy-game'
});

// Start session
const session = await therapyGame.startSession({
  exerciseType: 'neck_rotation',
  difficulty: 'beginner',
  durationMinutes: 5,
  painLevelBefore: 6
});

// Process video frame
const result = await therapyGame.processFrame(session.sessionId, frameData);
```

### Python
```python
from healthsync import TherapyGameClient

client = TherapyGameClient(api_key='your-jwt-token')

# Start session
session = await client.start_session(
    exercise_type='neck_rotation',
    difficulty='beginner',
    duration_minutes=5,
    pain_level_before=6
)

# Process frame
result = await client.process_frame(session.session_id, frame_data)
```