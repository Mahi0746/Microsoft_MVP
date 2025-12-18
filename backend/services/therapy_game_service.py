# HealthSync AI - Pain-to-Game Therapy Service
import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import structlog
import cv2
import mediapipe as mp
from dataclasses import dataclass
from enum import Enum

from config import settings
from services.db_service import DatabaseService
from services.ai_service import AIService


logger = structlog.get_logger(__name__)


class ExerciseType(Enum):
    """Types of therapeutic exercises."""
    NECK_ROTATION = "neck_rotation"
    SHOULDER_ROLLS = "shoulder_rolls"
    ARM_RAISES = "arm_raises"
    SPINE_TWIST = "spine_twist"
    LEG_LIFTS = "leg_lifts"
    BALANCE_TRAINING = "balance_training"
    BREATHING_EXERCISE = "breathing_exercise"
    FINGER_EXERCISES = "finger_exercises"


class DifficultyLevel(Enum):
    """Exercise difficulty levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    REHABILITATION = "rehabilitation"


@dataclass
class ExerciseSession:
    """Exercise session data structure."""
    session_id: str
    user_id: str
    exercise_type: ExerciseType
    difficulty: DifficultyLevel
    duration_minutes: int
    target_repetitions: int
    completed_repetitions: int
    accuracy_score: float
    pain_level_before: int
    pain_level_after: int
    calories_burned: float
    points_earned: int
    achievements_unlocked: List[str]
    timestamp: datetime


class TherapyGameService:
    """Advanced therapy game service using MediaPipe for motion tracking."""
    
    # MediaPipe models
    _pose_model = None
    _hands_model = None
    _face_model = None
    
    # Exercise configurations
    EXERCISE_CONFIGS = {
        ExerciseType.NECK_ROTATION: {
            "name": "Neck Rotation Therapy",
            "description": "Gentle neck rotations to improve mobility and reduce stiffness",
            "target_muscles": ["neck", "upper_trapezius"],
            "pain_conditions": ["neck_pain", "tension_headaches", "cervical_strain"],
            "duration_range": (2, 10),
            "repetition_range": (5, 20),
            "tracking_points": ["nose", "left_ear", "right_ear"],
            "movement_threshold": 15.0,
            "accuracy_threshold": 0.7
        },
        ExerciseType.SHOULDER_ROLLS: {
            "name": "Shoulder Roll Therapy",
            "description": "Shoulder rolls to release tension and improve range of motion",
            "target_muscles": ["deltoids", "trapezius", "rhomboids"],
            "pain_conditions": ["shoulder_pain", "upper_back_tension", "frozen_shoulder"],
            "duration_range": (3, 15),
            "repetition_range": (8, 25),
            "tracking_points": ["left_shoulder", "right_shoulder"],
            "movement_threshold": 20.0,
            "accuracy_threshold": 0.75
        },
        ExerciseType.ARM_RAISES: {
            "name": "Therapeutic Arm Raises",
            "description": "Controlled arm raises for shoulder rehabilitation",
            "target_muscles": ["deltoids", "rotator_cuff", "serratus_anterior"],
            "pain_conditions": ["shoulder_impingement", "rotator_cuff_injury", "post_surgery_rehab"],
            "duration_range": (5, 20),
            "repetition_range": (10, 30),
            "tracking_points": ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder"],
            "movement_threshold": 25.0,
            "accuracy_threshold": 0.8
        },
        ExerciseType.SPINE_TWIST: {
            "name": "Spinal Rotation Therapy",
            "description": "Gentle spinal twists for back mobility and pain relief",
            "target_muscles": ["obliques", "erector_spinae", "multifidus"],
            "pain_conditions": ["lower_back_pain", "spinal_stiffness", "disc_issues"],
            "duration_range": (4, 12),
            "repetition_range": (6, 18),
            "tracking_points": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
            "movement_threshold": 18.0,
            "accuracy_threshold": 0.72
        },
        ExerciseType.LEG_LIFTS: {
            "name": "Therapeutic Leg Lifts",
            "description": "Controlled leg lifts for hip and core strengthening",
            "target_muscles": ["hip_flexors", "quadriceps", "core_muscles"],
            "pain_conditions": ["hip_pain", "lower_back_pain", "post_injury_rehab"],
            "duration_range": (5, 18),
            "repetition_range": (8, 24),
            "tracking_points": ["left_knee", "right_knee", "left_ankle", "right_ankle"],
            "movement_threshold": 22.0,
            "accuracy_threshold": 0.78
        },
        ExerciseType.BALANCE_TRAINING: {
            "name": "Balance and Stability Training",
            "description": "Balance exercises for proprioception and fall prevention",
            "target_muscles": ["core", "stabilizers", "proprioceptors"],
            "pain_conditions": ["balance_issues", "post_injury_rehab", "elderly_care"],
            "duration_range": (3, 15),
            "repetition_range": (5, 15),
            "tracking_points": ["left_ankle", "right_ankle", "nose"],
            "movement_threshold": 10.0,
            "accuracy_threshold": 0.85
        },
        ExerciseType.BREATHING_EXERCISE: {
            "name": "Therapeutic Breathing",
            "description": "Guided breathing exercises for pain management and relaxation",
            "target_muscles": ["diaphragm", "intercostals"],
            "pain_conditions": ["chronic_pain", "anxiety", "stress_related_pain"],
            "duration_range": (5, 30),
            "repetition_range": (10, 50),
            "tracking_points": ["nose", "chest"],
            "movement_threshold": 5.0,
            "accuracy_threshold": 0.9
        },
        ExerciseType.FINGER_EXERCISES: {
            "name": "Hand and Finger Therapy",
            "description": "Fine motor exercises for hand rehabilitation",
            "target_muscles": ["finger_flexors", "finger_extensors", "intrinsic_hand_muscles"],
            "pain_conditions": ["arthritis", "carpal_tunnel", "post_surgery_rehab"],
            "duration_range": (3, 12),
            "repetition_range": (15, 40),
            "tracking_points": ["thumb_tip", "index_finger_tip", "middle_finger_tip"],
            "movement_threshold": 8.0,
            "accuracy_threshold": 0.82
        }
    }
    
    # Enhanced Gamification System
    ACHIEVEMENT_SYSTEM = {
        # Basic Achievements
        "first_session": {"name": "Getting Started", "points": 50, "description": "Complete your first therapy session", "icon": "🎯"},
        "pain_reducer": {"name": "Pain Reducer", "points": 100, "description": "Reduce pain level by 3+ points in a session", "icon": "💊"},
        "pain_free": {"name": "Pain-Free Hero", "points": 1000, "description": "Report pain level 0 after session", "icon": "🏆"},
        
        # Consistency Achievements
        "consistency_week": {"name": "Weekly Warrior", "points": 200, "description": "Complete exercises 5 days in a week", "icon": "📅"},
        "streak_master": {"name": "Streak Master", "points": 500, "description": "Maintain 30-day exercise streak", "icon": "🔥"},
        "daily_champion": {"name": "Daily Champion", "points": 75, "description": "Complete daily exercise goal", "icon": "⭐"},
        
        # Performance Achievements
        "accuracy_master": {"name": "Precision Pro", "points": 150, "description": "Achieve 90%+ accuracy in 5 sessions", "icon": "🎯"},
        "endurance_champion": {"name": "Endurance Champion", "points": 300, "description": "Complete 30-minute session", "icon": "💪"},
        "perfect_form": {"name": "Perfect Form", "points": 250, "description": "Achieve 100% accuracy in a session", "icon": "✨"},
        
        # Progress Achievements
        "level_up": {"name": "Level Up", "points": 100, "description": "Advance to next difficulty level", "icon": "📈"},
        "milestone_100": {"name": "Century Club", "points": 300, "description": "Complete 100 repetitions total", "icon": "💯"},
        "milestone_500": {"name": "Elite Performer", "points": 500, "description": "Complete 500 repetitions total", "icon": "🌟"},
        
        # Special Achievements
        "pain_warrior": {"name": "Pain Warrior", "points": 400, "description": "Exercise despite high pain (7+) and reduce it", "icon": "⚔️"},
        "comeback_king": {"name": "Comeback King", "points": 200, "description": "Return to exercise after 7+ day break", "icon": "👑"},
        "night_owl": {"name": "Night Owl", "points": 50, "description": "Complete session after 9 PM", "icon": "🦉"},
        "early_bird": {"name": "Early Bird", "points": 50, "description": "Complete session before 7 AM", "icon": "🐦"}
    }
    
    # Level System
    LEVEL_SYSTEM = {
        1: {"name": "Beginner", "points_required": 0, "multiplier": 1.0, "color": "#8BC34A"},
        2: {"name": "Novice", "points_required": 500, "multiplier": 1.1, "color": "#4CAF50"},
        3: {"name": "Apprentice", "points_required": 1200, "multiplier": 1.2, "color": "#009688"},
        4: {"name": "Practitioner", "points_required": 2500, "multiplier": 1.3, "color": "#00BCD4"},
        5: {"name": "Expert", "points_required": 5000, "multiplier": 1.4, "color": "#2196F3"},
        6: {"name": "Master", "points_required": 10000, "multiplier": 1.5, "color": "#3F51B5"},
        7: {"name": "Grandmaster", "points_required": 20000, "multiplier": 1.6, "color": "#9C27B0"},
        8: {"name": "Legend", "points_required": 40000, "multiplier": 1.7, "color": "#E91E63"},
        9: {"name": "Champion", "points_required": 75000, "multiplier": 1.8, "color": "#FF5722"},
        10: {"name": "Immortal", "points_required": 150000, "multiplier": 2.0, "color": "#FFD700"}
    }
    
    # Pain Detection Configuration
    PAIN_DETECTION_CONFIG = {
        "facial_action_units": {
            "AU4": {"name": "Brow Lowerer", "pain_indicator": True, "weight": 0.3},
            "AU6": {"name": "Cheek Raiser", "pain_indicator": True, "weight": 0.2},
            "AU7": {"name": "Lid Tightener", "pain_indicator": True, "weight": 0.25},
            "AU9": {"name": "Nose Wrinkler", "pain_indicator": True, "weight": 0.15},
            "AU10": {"name": "Upper Lip Raiser", "pain_indicator": True, "weight": 0.1}
        },
        "pain_thresholds": {
            "none": 0.0,
            "mild": 0.2,
            "moderate": 0.4,
            "severe": 0.6,
            "extreme": 0.8
        },
        "adaptation_rules": {
            "pain_increase": {"difficulty_reduction": 0.2, "rest_suggestion": True},
            "pain_stable": {"continue_current": True},
            "pain_decrease": {"difficulty_increase": 0.1, "encouragement": True}
        }
    }
    
    # Game Mechanics
    GAME_MECHANICS = {
        "base_points_per_rep": 10,
        "accuracy_bonus_multiplier": 0.5,
        "streak_bonus_multiplier": 0.1,
        "pain_reduction_bonus": 20,
        "perfect_session_bonus": 100,
        "consistency_bonus": 50,
        "level_bonus_multiplier": 0.2
    }
    
    @classmethod
    async def initialize_mediapipe(cls):
        """Initialize MediaPipe models."""
        
        try:
            logger.info("Initializing MediaPipe models")
            
            # Initialize pose detection
            cls._pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize hand tracking
            cls._hands_model = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize face mesh
            cls._face_model = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logger.info("MediaPipe models initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize MediaPipe", error=str(e))
            raise
    
    @classmethod
    async def start_therapy_session(
        cls,
        user_id: str,
        exercise_type: str,
        difficulty: str,
        duration_minutes: int,
        pain_level_before: int
    ) -> Dict[str, Any]:
        """Start a new therapy game session."""
        
        try:
            # Validate inputs
            exercise_enum = ExerciseType(exercise_type)
            difficulty_enum = DifficultyLevel(difficulty)
            
            if not (1 <= pain_level_before <= 10):
                raise ValueError("Pain level must be between 1 and 10")
            
            # Get exercise configuration
            config = cls.EXERCISE_CONFIGS[exercise_enum]
            
            # Calculate target repetitions based on difficulty and duration
            base_reps = config["repetition_range"][0]
            max_reps = config["repetition_range"][1]
            
            difficulty_multiplier = {
                DifficultyLevel.BEGINNER: 0.6,
                DifficultyLevel.INTERMEDIATE: 0.8,
                DifficultyLevel.ADVANCED: 1.0,
                DifficultyLevel.REHABILITATION: 0.4
            }
            
            target_reps = int(base_reps + (max_reps - base_reps) * difficulty_multiplier[difficulty_enum])
            target_reps = min(target_reps, int(duration_minutes * 2))  # Adjust for duration
            
            # Create session
            session_id = f"therapy_{user_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "exercise_type": exercise_type,
                "difficulty": difficulty,
                "duration_minutes": duration_minutes,
                "target_repetitions": target_reps,
                "pain_level_before": pain_level_before,
                "config": config,
                "start_time": datetime.utcnow(),
                "status": "active",
                "tracking_data": [],
                "real_time_feedback": [],
                "performance_metrics": {
                    "completed_repetitions": 0,
                    "accuracy_scores": [],
                    "movement_quality": [],
                    "timing_consistency": []
                }
            }
            
            # Store session in MongoDB
            await DatabaseService.mongodb_insert_one("therapy_sessions", session_data)
            
            # Get personalized instructions
            instructions = await cls._generate_personalized_instructions(
                user_id, exercise_enum, difficulty_enum, pain_level_before
            )
            
            logger.info(
                "Therapy session started",
                user_id=user_id,
                session_id=session_id,
                exercise_type=exercise_type,
                target_reps=target_reps
            )
            
            return {
                "session_id": session_id,
                "exercise_config": config,
                "target_repetitions": target_reps,
                "instructions": instructions,
                "tracking_points": config["tracking_points"],
                "movement_threshold": config["movement_threshold"],
                "accuracy_threshold": config["accuracy_threshold"],
                "estimated_calories": cls._estimate_calories(exercise_enum, duration_minutes, difficulty_enum),
                "potential_points": cls._calculate_potential_points(target_reps, difficulty_enum)
            }
            
        except Exception as e:
            logger.error("Failed to start therapy session", user_id=user_id, error=str(e))
            raise
    
    @classmethod
    async def process_motion_frame(
        cls,
        session_id: str,
        frame_data: str,
        timestamp: float
    ) -> Dict[str, Any]:
        """Process motion tracking frame using MediaPipe."""
        
        try:
            # Get session data
            session = await DatabaseService.mongodb_find_one(
                "therapy_sessions",
                {"session_id": session_id, "status": "active"}
            )
            
            if not session:
                raise ValueError("Active session not found")
            
            # Decode frame
            import base64
            import io
            from PIL import Image
            
            frame_bytes = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
            pil_image = Image.open(io.BytesIO(frame_bytes))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Process frame based on exercise type
            exercise_type = ExerciseType(session["exercise_type"])
            
            if exercise_type in [ExerciseType.FINGER_EXERCISES]:
                landmarks = cls._process_hand_tracking(frame)
            elif exercise_type in [ExerciseType.NECK_ROTATION, ExerciseType.BREATHING_EXERCISE]:
                landmarks = cls._process_face_tracking(frame)
            else:
                landmarks = cls._process_pose_tracking(frame)
            
            if not landmarks:
                return {
                    "success": False,
                    "message": "No landmarks detected",
                    "feedback": "Please ensure you're visible in the camera"
                }
            
            # Analyze movement
            movement_analysis = cls._analyze_movement(
                landmarks,
                exercise_type,
                session["config"],
                session["performance_metrics"]
            )
            
            # Detect pain from facial expressions (if face tracking is available)
            pain_analysis = {}
            if landmarks and landmarks.get("type") == "face":
                pain_analysis = cls._detect_pain_from_face(landmarks)
                
                # Auto-adapt exercise difficulty based on pain detection
                adaptation = await cls._adapt_exercise_difficulty(
                    session_id,
                    pain_analysis,
                    session["performance_metrics"]
                )
                
                # Add adaptation info to movement analysis
                movement_analysis["pain_analysis"] = pain_analysis
                movement_analysis["adaptation"] = adaptation
            
            # Update session data
            session["tracking_data"].append({
                "timestamp": timestamp,
                "landmarks": landmarks,
                "movement_analysis": movement_analysis,
                "pain_analysis": pain_analysis
            })
            
            # Generate enhanced real-time feedback with pain awareness
            feedback = cls._generate_enhanced_real_time_feedback(
                movement_analysis,
                exercise_type,
                session["performance_metrics"],
                pain_analysis
            )
            
            session["real_time_feedback"].append({
                "timestamp": timestamp,
                "feedback": feedback
            })
            
            # Update performance metrics
            cls._update_performance_metrics(session["performance_metrics"], movement_analysis)
            
            # Save updated session
            await DatabaseService.mongodb_update_one(
                "therapy_sessions",
                {"session_id": session_id},
                {"$set": {
                    "tracking_data": session["tracking_data"][-100:],  # Keep last 100 frames
                    "real_time_feedback": session["real_time_feedback"][-50:],  # Keep last 50 feedback
                    "performance_metrics": session["performance_metrics"]
                }}
            )
            
            return {
                "success": True,
                "movement_analysis": movement_analysis,
                "feedback": feedback,
                "progress": {
                    "completed_reps": session["performance_metrics"]["completed_repetitions"],
                    "target_reps": session["target_repetitions"],
                    "accuracy": np.mean(session["performance_metrics"]["accuracy_scores"]) if session["performance_metrics"]["accuracy_scores"] else 0,
                    "quality_score": np.mean(session["performance_metrics"]["movement_quality"]) if session["performance_metrics"]["movement_quality"] else 0
                }
            }
            
        except Exception as e:
            logger.error("Motion frame processing failed", session_id=session_id, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "feedback": "Processing error occurred"
            }
    
    @classmethod
    def _process_pose_tracking(cls, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process pose tracking using MediaPipe."""
        
        try:
            if cls._pose_model is None:
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = cls._pose_model.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            # Extract landmarks
            landmarks = {}
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mp.solutions.pose.PoseLandmark(i).name.lower()
                landmarks[landmark_name] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                }
            
            return {
                "type": "pose",
                "landmarks": landmarks,
                "confidence": np.mean([lm["visibility"] for lm in landmarks.values()])
            }
            
        except Exception as e:
            logger.error("Pose tracking failed", error=str(e))
            return None
    
    @classmethod
    def _process_hand_tracking(cls, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process hand tracking using MediaPipe."""
        
        try:
            if cls._hands_model is None:
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = cls._hands_model.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
            
            # Extract hand landmarks
            hands_data = []
            
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_data = {
                    "handedness": handedness.classification[0].label.lower(),
                    "landmarks": {}
                }
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    landmark_name = f"landmark_{i}"
                    hand_data["landmarks"][landmark_name] = {
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z
                    }
                
                hands_data.append(hand_data)
            
            return {
                "type": "hands",
                "hands": hands_data,
                "confidence": 0.8  # MediaPipe doesn't provide hand confidence
            }
            
        except Exception as e:
            logger.error("Hand tracking failed", error=str(e))
            return None
    
    @classmethod
    def _process_face_tracking(cls, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process face tracking using MediaPipe."""
        
        try:
            if cls._face_model is None:
                return None
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = cls._face_model.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return None
            
            # Extract face landmarks (simplified)
            face_landmarks = results.multi_face_landmarks[0]
            
            # Key facial points for neck/breathing exercises
            key_points = {
                "nose_tip": face_landmarks.landmark[1],
                "left_ear": face_landmarks.landmark[234],
                "right_ear": face_landmarks.landmark[454],
                "chin": face_landmarks.landmark[175]
            }
            
            landmarks = {}
            for name, landmark in key_points.items():
                landmarks[name] = {
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                }
            
            return {
                "type": "face",
                "landmarks": landmarks,
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error("Face tracking failed", error=str(e))
            return None
    
    @classmethod
    def _detect_pain_from_face(cls, face_landmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Detect pain indicators from facial landmarks using Action Units."""
        
        try:
            pain_analysis = {
                "pain_detected": False,
                "pain_level": 0.0,
                "pain_category": "none",
                "action_units": {},
                "confidence": 0.0,
                "recommendations": []
            }
            
            if not face_landmarks or face_landmarks["type"] != "face":
                return pain_analysis
            
            landmarks = face_landmarks["landmarks"]
            
            # Analyze facial action units for pain detection
            au_scores = cls._analyze_action_units(landmarks)
            pain_analysis["action_units"] = au_scores
            
            # Calculate overall pain score
            total_pain_score = 0.0
            total_weight = 0.0
            
            for au_name, au_config in cls.PAIN_DETECTION_CONFIG["facial_action_units"].items():
                if au_name in au_scores:
                    au_score = au_scores[au_name]["intensity"]
                    weight = au_config["weight"]
                    total_pain_score += au_score * weight
                    total_weight += weight
            
            if total_weight > 0:
                pain_analysis["pain_level"] = total_pain_score / total_weight
                pain_analysis["confidence"] = min(face_landmarks.get("confidence", 0.0), 0.85)
            
            # Categorize pain level
            pain_thresholds = cls.PAIN_DETECTION_CONFIG["pain_thresholds"]
            
            if pain_analysis["pain_level"] >= pain_thresholds["extreme"]:
                pain_analysis["pain_category"] = "extreme"
                pain_analysis["pain_detected"] = True
            elif pain_analysis["pain_level"] >= pain_thresholds["severe"]:
                pain_analysis["pain_category"] = "severe"
                pain_analysis["pain_detected"] = True
            elif pain_analysis["pain_level"] >= pain_thresholds["moderate"]:
                pain_analysis["pain_category"] = "moderate"
                pain_analysis["pain_detected"] = True
            elif pain_analysis["pain_level"] >= pain_thresholds["mild"]:
                pain_analysis["pain_category"] = "mild"
                pain_analysis["pain_detected"] = True
            
            # Generate recommendations based on pain level
            if pain_analysis["pain_detected"]:
                pain_analysis["recommendations"] = cls._generate_pain_recommendations(
                    pain_analysis["pain_category"],
                    pain_analysis["pain_level"]
                )
            
            return pain_analysis
            
        except Exception as e:
            logger.error("Pain detection failed", error=str(e))
            return {
                "pain_detected": False,
                "pain_level": 0.0,
                "pain_category": "none",
                "error": str(e)
            }
    
    @classmethod
    def _analyze_action_units(cls, landmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze facial action units from landmarks."""
        
        try:
            au_scores = {}
            
            # Get key facial points
            nose_tip = landmarks.get("nose_tip", {})
            left_ear = landmarks.get("left_ear", {})
            right_ear = landmarks.get("right_ear", {})
            chin = landmarks.get("chin", {})
            
            if not all([nose_tip, left_ear, right_ear, chin]):
                return au_scores
            
            # AU4 - Brow Lowerer (estimated from face geometry)
            # Simplified calculation based on relative positions
            brow_tension = cls._calculate_brow_tension(nose_tip, left_ear, right_ear)
            au_scores["AU4"] = {
                "intensity": brow_tension,
                "confidence": 0.7,
                "description": "Brow lowering indicating concentration or discomfort"
            }
            
            # AU6 - Cheek Raiser (estimated from eye area)
            cheek_raise = cls._calculate_cheek_raise(nose_tip, chin)
            au_scores["AU6"] = {
                "intensity": cheek_raise,
                "confidence": 0.6,
                "description": "Cheek raising often associated with pain"
            }
            
            # AU7 - Lid Tightener (estimated from eye region)
            lid_tension = cls._calculate_lid_tension(left_ear, right_ear, nose_tip)
            au_scores["AU7"] = {
                "intensity": lid_tension,
                "confidence": 0.65,
                "description": "Eye tightening indicating discomfort"
            }
            
            # AU9 - Nose Wrinkler (estimated from nose area)
            nose_wrinkle = cls._calculate_nose_wrinkle(nose_tip, chin)
            au_scores["AU9"] = {
                "intensity": nose_wrinkle,
                "confidence": 0.5,
                "description": "Nose wrinkling associated with pain or disgust"
            }
            
            # AU10 - Upper Lip Raiser (estimated from mouth area)
            lip_raise = cls._calculate_lip_raise(nose_tip, chin)
            au_scores["AU10"] = {
                "intensity": lip_raise,
                "confidence": 0.55,
                "description": "Upper lip raising indicating discomfort"
            }
            
            return au_scores
            
        except Exception as e:
            logger.error("Action unit analysis failed", error=str(e))
            return {}
    
    @classmethod
    def _calculate_brow_tension(cls, nose_tip: Dict, left_ear: Dict, right_ear: Dict) -> float:
        """Calculate brow tension indicator."""
        
        try:
            # Simplified calculation based on facial geometry
            # In a real implementation, this would use more sophisticated facial landmark analysis
            
            ear_distance = abs(right_ear["x"] - left_ear["x"])
            nose_ear_ratio = abs(nose_tip["y"] - (left_ear["y"] + right_ear["y"]) / 2)
            
            # Normalize and calculate tension indicator
            tension = min(nose_ear_ratio / ear_distance * 2, 1.0) if ear_distance > 0 else 0.0
            
            return max(0.0, min(1.0, tension))
            
        except Exception:
            return 0.0
    
    @classmethod
    def _calculate_cheek_raise(cls, nose_tip: Dict, chin: Dict) -> float:
        """Calculate cheek raise indicator."""
        
        try:
            # Simplified calculation
            nose_chin_distance = abs(chin["y"] - nose_tip["y"])
            
            # Normalize (smaller distance might indicate cheek raising)
            raise_indicator = max(0.0, (0.15 - nose_chin_distance) / 0.15) if nose_chin_distance < 0.15 else 0.0
            
            return max(0.0, min(1.0, raise_indicator))
            
        except Exception:
            return 0.0
    
    @classmethod
    def _calculate_lid_tension(cls, left_ear: Dict, right_ear: Dict, nose_tip: Dict) -> float:
        """Calculate eyelid tension indicator."""
        
        try:
            # Simplified calculation based on eye region geometry
            ear_midpoint_y = (left_ear["y"] + right_ear["y"]) / 2
            eye_tension = abs(nose_tip["y"] - ear_midpoint_y)
            
            # Normalize tension indicator
            tension = min(eye_tension * 3, 1.0)
            
            return max(0.0, min(1.0, tension))
            
        except Exception:
            return 0.0
    
    @classmethod
    def _calculate_nose_wrinkle(cls, nose_tip: Dict, chin: Dict) -> float:
        """Calculate nose wrinkle indicator."""
        
        try:
            # Simplified calculation
            nose_position = nose_tip["y"]
            chin_position = chin["y"]
            
            # Calculate relative position (wrinkled nose might be higher)
            relative_position = (chin_position - nose_position) / abs(chin_position - nose_position) if chin_position != nose_position else 0
            
            wrinkle_indicator = max(0.0, (relative_position - 0.6) / 0.4) if relative_position > 0.6 else 0.0
            
            return max(0.0, min(1.0, wrinkle_indicator))
            
        except Exception:
            return 0.0
    
    @classmethod
    def _calculate_lip_raise(cls, nose_tip: Dict, chin: Dict) -> float:
        """Calculate upper lip raise indicator."""
        
        try:
            # Simplified calculation
            nose_chin_ratio = abs(nose_tip["y"] - chin["y"])
            
            # Normalize lip raise indicator
            lip_raise = max(0.0, (0.12 - nose_chin_ratio) / 0.12) if nose_chin_ratio < 0.12 else 0.0
            
            return max(0.0, min(1.0, lip_raise))
            
        except Exception:
            return 0.0
    
    @classmethod
    def _generate_pain_recommendations(cls, pain_category: str, pain_level: float) -> List[str]:
        """Generate recommendations based on detected pain level."""
        
        recommendations = []
        
        if pain_category == "extreme":
            recommendations.extend([
                "Stop exercise immediately - extreme pain detected",
                "Consider seeking medical attention",
                "Rest and apply ice if appropriate",
                "Do not continue until pain subsides"
            ])
        elif pain_category == "severe":
            recommendations.extend([
                "Reduce exercise intensity significantly",
                "Take frequent breaks",
                "Consider stopping if pain persists",
                "Monitor pain levels closely"
            ])
        elif pain_category == "moderate":
            recommendations.extend([
                "Reduce exercise intensity",
                "Focus on gentle movements",
                "Take breaks as needed",
                "Listen to your body"
            ])
        elif pain_category == "mild":
            recommendations.extend([
                "Continue with caution",
                "Gentle movements recommended",
                "Monitor for pain increase"
            ])
        
        return recommendations
    
    @classmethod
    async def _adapt_exercise_difficulty(
        cls,
        session_id: str,
        pain_analysis: Dict[str, Any],
        current_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Automatically adapt exercise difficulty based on pain detection."""
        
        try:
            adaptation = {
                "difficulty_changed": False,
                "new_difficulty": None,
                "recommendations": [],
                "auto_adjustments": []
            }
            
            # Get current session
            session = await DatabaseService.mongodb_find_one(
                "therapy_sessions",
                {"session_id": session_id, "status": "active"}
            )
            
            if not session:
                return adaptation
            
            pain_detected = pain_analysis.get("pain_detected", False)
            pain_category = pain_analysis.get("pain_category", "none")
            
            adaptation_rules = cls.PAIN_DETECTION_CONFIG["adaptation_rules"]
            
            if pain_detected and pain_category in ["severe", "extreme"]:
                # Reduce difficulty significantly
                adaptation["difficulty_changed"] = True
                adaptation["new_difficulty"] = "rehabilitation"
                adaptation["recommendations"].extend([
                    "Exercise difficulty automatically reduced due to pain detection",
                    "Focus on gentle, pain-free movements",
                    "Stop if pain increases"
                ])
                adaptation["auto_adjustments"].append("Difficulty reduced to rehabilitation level")
                
                # Update session difficulty
                await DatabaseService.mongodb_update_one(
                    "therapy_sessions",
                    {"session_id": session_id},
                    {"$set": {"difficulty": "rehabilitation", "auto_adapted": True}}
                )
                
            elif pain_detected and pain_category == "moderate":
                # Suggest rest breaks
                adaptation["recommendations"].extend([
                    "Take a 30-second rest break",
                    "Reduce movement speed",
                    "Focus on form over speed"
                ])
                adaptation["auto_adjustments"].append("Rest break suggested")
                
            elif not pain_detected and current_performance.get("accuracy", 0) > 0.9:
                # Consider increasing difficulty if performing well with no pain
                adaptation["recommendations"].append(
                    "Excellent performance with no pain detected - consider increasing difficulty next session"
                )
            
            return adaptation
            
        except Exception as e:
            logger.error("Exercise adaptation failed", session_id=session_id, error=str(e))
            return {"difficulty_changed": False, "error": str(e)}
    
    @classmethod
    def _analyze_movement(
        cls,
        landmarks: Dict[str, Any],
        exercise_type: ExerciseType,
        config: Dict[str, Any],
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze movement quality and detect exercise completion."""
        
        try:
            analysis = {
                "movement_detected": False,
                "repetition_completed": False,
                "accuracy_score": 0.0,
                "quality_score": 0.0,
                "form_feedback": [],
                "movement_range": 0.0,
                "timing_score": 0.0
            }
            
            if exercise_type == ExerciseType.NECK_ROTATION:
                analysis = cls._analyze_neck_rotation(landmarks, config, analysis)
                
            elif exercise_type == ExerciseType.SHOULDER_ROLLS:
                analysis = cls._analyze_shoulder_rolls(landmarks, config, analysis)
                
            elif exercise_type == ExerciseType.ARM_RAISES:
                analysis = cls._analyze_arm_raises(landmarks, config, analysis)
                
            elif exercise_type == ExerciseType.SPINE_TWIST:
                analysis = cls._analyze_spine_twist(landmarks, config, analysis)
                
            elif exercise_type == ExerciseType.LEG_LIFTS:
                analysis = cls._analyze_leg_lifts(landmarks, config, analysis)
                
            elif exercise_type == ExerciseType.BALANCE_TRAINING:
                analysis = cls._analyze_balance_training(landmarks, config, analysis)
                
            elif exercise_type == ExerciseType.BREATHING_EXERCISE:
                analysis = cls._analyze_breathing_exercise(landmarks, config, analysis)
                
            elif exercise_type == ExerciseType.FINGER_EXERCISES:
                analysis = cls._analyze_finger_exercises(landmarks, config, analysis)
            
            return analysis
            
        except Exception as e:
            logger.error("Movement analysis failed", error=str(e))
            return {
                "movement_detected": False,
                "error": str(e)
            }
    
    @classmethod
    def _analyze_neck_rotation(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze neck rotation exercise."""
        
        try:
            if landmarks["type"] != "face":
                return analysis
            
            face_landmarks = landmarks["landmarks"]
            
            # Calculate head rotation angle
            nose = face_landmarks.get("nose_tip", {})
            left_ear = face_landmarks.get("left_ear", {})
            right_ear = face_landmarks.get("right_ear", {})
            
            if not all([nose, left_ear, right_ear]):
                return analysis
            
            # Calculate rotation angle
            ear_diff_x = right_ear["x"] - left_ear["x"]
            ear_diff_y = right_ear["y"] - left_ear["y"]
            rotation_angle = np.degrees(np.arctan2(ear_diff_y, ear_diff_x))
            
            # Detect movement
            movement_threshold = config["movement_threshold"]
            if abs(rotation_angle) > movement_threshold:
                analysis["movement_detected"] = True
                analysis["movement_range"] = abs(rotation_angle)
                
                # Check for full rotation (simplified)
                if abs(rotation_angle) > movement_threshold * 1.5:
                    analysis["repetition_completed"] = True
                    analysis["accuracy_score"] = min(1.0, abs(rotation_angle) / (movement_threshold * 2))
            
            # Form feedback
            if abs(rotation_angle) > 45:
                analysis["form_feedback"].append("Reduce rotation range to avoid strain")
            elif abs(rotation_angle) < movement_threshold * 0.5:
                analysis["form_feedback"].append("Increase rotation range for better effect")
            
            analysis["quality_score"] = analysis["accuracy_score"] * landmarks["confidence"]
            
            return analysis
            
        except Exception as e:
            logger.error("Neck rotation analysis failed", error=str(e))
            return analysis
    
    @classmethod
    def _analyze_shoulder_rolls(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze shoulder roll exercise."""
        
        try:
            if landmarks["type"] != "pose":
                return analysis
            
            pose_landmarks = landmarks["landmarks"]
            
            # Get shoulder positions
            left_shoulder = pose_landmarks.get("left_shoulder", {})
            right_shoulder = pose_landmarks.get("right_shoulder", {})
            
            if not all([left_shoulder, right_shoulder]):
                return analysis
            
            # Calculate shoulder movement
            shoulder_center_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
            shoulder_width = abs(right_shoulder["x"] - left_shoulder["x"])
            
            # Detect vertical movement (shoulder rolls)
            movement_threshold = config["movement_threshold"] / 100  # Normalize to image coordinates
            
            # This is simplified - in practice, you'd track movement over time
            if shoulder_width > 0.15:  # Shoulders visible and apart
                analysis["movement_detected"] = True
                analysis["movement_range"] = shoulder_width * 100
                
                # Simplified repetition detection
                if shoulder_center_y < 0.3:  # Shoulders raised
                    analysis["repetition_completed"] = True
                    analysis["accuracy_score"] = 0.8
            
            analysis["quality_score"] = analysis["accuracy_score"] * landmarks["confidence"]
            
            return analysis
            
        except Exception as e:
            logger.error("Shoulder roll analysis failed", error=str(e))
            return analysis
    
    @classmethod
    def _analyze_arm_raises(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze arm raise exercise."""
        
        try:
            if landmarks["type"] != "pose":
                return analysis
            
            pose_landmarks = landmarks["landmarks"]
            
            # Get arm positions
            left_wrist = pose_landmarks.get("left_wrist", {})
            right_wrist = pose_landmarks.get("right_wrist", {})
            left_shoulder = pose_landmarks.get("left_shoulder", {})
            right_shoulder = pose_landmarks.get("right_shoulder", {})
            
            if not all([left_wrist, right_wrist, left_shoulder, right_shoulder]):
                return analysis
            
            # Calculate arm elevation angles
            left_arm_angle = cls._calculate_arm_angle(left_shoulder, left_wrist)
            right_arm_angle = cls._calculate_arm_angle(right_shoulder, right_wrist)
            
            avg_angle = (left_arm_angle + right_arm_angle) / 2
            
            # Detect arm raise
            if avg_angle > 45:  # Arms raised above horizontal
                analysis["movement_detected"] = True
                analysis["movement_range"] = avg_angle
                
                if avg_angle > 90:  # Full arm raise
                    analysis["repetition_completed"] = True
                    analysis["accuracy_score"] = min(1.0, avg_angle / 180)
            
            # Form feedback
            angle_diff = abs(left_arm_angle - right_arm_angle)
            if angle_diff > 20:
                analysis["form_feedback"].append("Keep both arms at similar height")
            
            analysis["quality_score"] = analysis["accuracy_score"] * landmarks["confidence"]
            
            return analysis
            
        except Exception as e:
            logger.error("Arm raise analysis failed", error=str(e))
            return analysis
    
    @classmethod
    def _calculate_arm_angle(cls, shoulder: Dict[str, float], wrist: Dict[str, float]) -> float:
        """Calculate arm elevation angle."""
        
        try:
            dx = wrist["x"] - shoulder["x"]
            dy = wrist["y"] - shoulder["y"]
            
            # Calculate angle from horizontal
            angle = np.degrees(np.arctan2(-dy, dx))  # Negative dy because y increases downward
            
            # Normalize to 0-180 range
            if angle < 0:
                angle += 360
            
            return min(angle, 180)
            
        except Exception:
            return 0.0
    
    @classmethod
    def _analyze_spine_twist(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spinal twist exercise."""
        
        # Simplified implementation
        analysis["movement_detected"] = True
        analysis["accuracy_score"] = 0.75
        analysis["quality_score"] = 0.75
        return analysis
    
    @classmethod
    def _analyze_leg_lifts(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze leg lift exercise."""
        
        # Simplified implementation
        analysis["movement_detected"] = True
        analysis["accuracy_score"] = 0.8
        analysis["quality_score"] = 0.8
        return analysis
    
    @classmethod
    def _analyze_balance_training(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze balance training exercise."""
        
        # Simplified implementation
        analysis["movement_detected"] = True
        analysis["accuracy_score"] = 0.85
        analysis["quality_score"] = 0.85
        return analysis
    
    @classmethod
    def _analyze_breathing_exercise(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze breathing exercise."""
        
        # Simplified implementation
        analysis["movement_detected"] = True
        analysis["accuracy_score"] = 0.9
        analysis["quality_score"] = 0.9
        return analysis
    
    @classmethod
    def _analyze_finger_exercises(cls, landmarks: Dict[str, Any], config: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze finger exercise."""
        
        # Simplified implementation
        analysis["movement_detected"] = True
        analysis["accuracy_score"] = 0.82
        analysis["quality_score"] = 0.82
        return analysis
    
    @classmethod
    def _generate_real_time_feedback(
        cls,
        movement_analysis: Dict[str, Any],
        exercise_type: ExerciseType,
        performance_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate real-time feedback for the user."""
        
        feedback = {
            "message": "",
            "type": "info",  # info, success, warning, error
            "visual_cues": [],
            "audio_cue": None,
            "encouragement": ""
        }
        
        if not movement_analysis.get("movement_detected", False):
            feedback["message"] = "No movement detected. Please ensure you're visible in the camera."
            feedback["type"] = "warning"
            return feedback
        
        if movement_analysis.get("repetition_completed", False):
            feedback["message"] = "Great! Repetition completed."
            feedback["type"] = "success"
            feedback["audio_cue"] = "success_chime"
            feedback["encouragement"] = "Keep it up!"
            
        elif movement_analysis.get("accuracy_score", 0) < 0.5:
            feedback["message"] = "Try to improve your form for better results."
            feedback["type"] = "warning"
            
        else:
            feedback["message"] = "Good movement! Continue the exercise."
            feedback["type"] = "info"
        
        # Add form feedback
        form_feedback = movement_analysis.get("form_feedback", [])
        if form_feedback:
            feedback["visual_cues"] = form_feedback
        
        # Add exercise-specific encouragement
        completed_reps = performance_metrics.get("completed_repetitions", 0)
        if completed_reps > 0 and completed_reps % 5 == 0:
            feedback["encouragement"] = f"Excellent! {completed_reps} repetitions completed!"
        
        return feedback
    
    @classmethod
    def _generate_enhanced_real_time_feedback(
        cls,
        movement_analysis: Dict[str, Any],
        exercise_type: ExerciseType,
        performance_metrics: Dict[str, Any],
        pain_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate enhanced real-time feedback with pain awareness and gamification."""
        
        # Start with basic feedback
        feedback = cls._generate_real_time_feedback(movement_analysis, exercise_type, performance_metrics)
        
        # Enhance with pain-aware feedback
        if pain_analysis and pain_analysis.get("pain_detected", False):
            pain_category = pain_analysis.get("pain_category", "none")
            pain_level = pain_analysis.get("pain_level", 0.0)
            
            # Override feedback based on pain detection
            if pain_category in ["severe", "extreme"]:
                feedback["message"] = "⚠️ Pain detected - please stop and rest"
                feedback["type"] = "error"
                feedback["audio_cue"] = "warning_tone"
                feedback["encouragement"] = "Your safety is most important"
                feedback["visual_cues"] = pain_analysis.get("recommendations", [])
                
            elif pain_category == "moderate":
                feedback["message"] = "😣 Discomfort detected - take it easy"
                feedback["type"] = "warning"
                feedback["encouragement"] = "Listen to your body"
                feedback["visual_cues"].extend([
                    "Reduce intensity",
                    "Take breaks as needed"
                ])
                
            elif pain_category == "mild":
                feedback["message"] = "😐 Mild discomfort - proceed with caution"
                feedback["type"] = "info"
                feedback["encouragement"] = "You're doing great, stay mindful"
        
        # Add gamification elements
        completed_reps = performance_metrics.get("completed_repetitions", 0)
        accuracy_scores = performance_metrics.get("accuracy_scores", [])
        
        # Points calculation
        if movement_analysis.get("repetition_completed", False):
            base_points = cls.GAME_MECHANICS["base_points_per_rep"]
            accuracy_bonus = 0
            
            if accuracy_scores:
                current_accuracy = accuracy_scores[-1]
                accuracy_bonus = int(base_points * current_accuracy * cls.GAME_MECHANICS["accuracy_bonus_multiplier"])
            
            total_points = base_points + accuracy_bonus
            
            feedback["points_earned"] = total_points
            feedback["total_points"] = sum([base_points] * completed_reps) + sum([int(base_points * acc * 0.5) for acc in accuracy_scores])
            
            # Special achievements
            if current_accuracy >= 1.0:
                feedback["achievement"] = "Perfect Form! 🌟"
                feedback["bonus_points"] = cls.GAME_MECHANICS["perfect_session_bonus"]
            
            # Streak tracking
            if completed_reps > 0 and completed_reps % 10 == 0:
                feedback["milestone"] = f"🎯 {completed_reps} reps milestone reached!"
                feedback["streak_bonus"] = int(completed_reps * cls.GAME_MECHANICS["streak_bonus_multiplier"])
        
        # Level progression hints
        current_level = cls._calculate_user_level(feedback.get("total_points", 0))
        next_level = current_level + 1
        
        if next_level <= 10:
            next_level_info = cls.LEVEL_SYSTEM.get(next_level, {})
            points_needed = next_level_info.get("points_required", 0) - feedback.get("total_points", 0)
            
            if points_needed > 0 and points_needed <= 100:
                feedback["level_progress"] = f"🎮 {points_needed} points to {next_level_info.get('name', 'next level')}!"
        
        # Motivational messages based on performance
        if not pain_analysis.get("pain_detected", False):
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0
            
            if avg_accuracy >= 0.9:
                feedback["motivation"] = "🔥 You're on fire! Amazing form!"
            elif avg_accuracy >= 0.8:
                feedback["motivation"] = "💪 Excellent technique!"
            elif avg_accuracy >= 0.7:
                feedback["motivation"] = "👍 Good job, keep it up!"
            elif completed_reps >= 5:
                feedback["motivation"] = "🌟 Great persistence!"
        
        return feedback
    
    @classmethod
    def _calculate_user_level(cls, total_points: int) -> int:
        """Calculate user level based on total points."""
        
        for level in range(10, 0, -1):
            level_info = cls.LEVEL_SYSTEM.get(level, {})
            if total_points >= level_info.get("points_required", 0):
                return level
        
        return 1
    
    @classmethod
    def _update_performance_metrics(cls, metrics: Dict[str, Any], movement_analysis: Dict[str, Any]):
        """Update performance metrics based on movement analysis."""
        
        if movement_analysis.get("repetition_completed", False):
            metrics["completed_repetitions"] += 1
            
        if movement_analysis.get("accuracy_score", 0) > 0:
            metrics["accuracy_scores"].append(movement_analysis["accuracy_score"])
            
        if movement_analysis.get("quality_score", 0) > 0:
            metrics["movement_quality"].append(movement_analysis["quality_score"])
    
    @classmethod
    async def complete_therapy_session(
        cls,
        session_id: str,
        pain_level_after: int,
        user_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete therapy session and calculate results."""
        
        try:
            # Get session data
            session = await DatabaseService.mongodb_find_one(
                "therapy_sessions",
                {"session_id": session_id}
            )
            
            if not session:
                raise ValueError("Session not found")
            
            # Calculate session results
            performance = session["performance_metrics"]
            completed_reps = performance["completed_repetitions"]
            target_reps = session["target_repetitions"]
            
            # Calculate scores
            completion_rate = completed_reps / target_reps if target_reps > 0 else 0
            avg_accuracy = np.mean(performance["accuracy_scores"]) if performance["accuracy_scores"] else 0
            avg_quality = np.mean(performance["movement_quality"]) if performance["movement_quality"] else 0
            
            # Calculate calories burned
            exercise_type = ExerciseType(session["exercise_type"])
            difficulty = DifficultyLevel(session["difficulty"])
            actual_duration = (datetime.utcnow() - session["start_time"]).total_seconds() / 60
            calories_burned = cls._calculate_calories_burned(exercise_type, actual_duration, difficulty, completion_rate)
            
            # Calculate points earned
            points_earned = cls._calculate_points_earned(
                completed_reps, avg_accuracy, completion_rate, difficulty
            )
            
            # Check for achievements
            achievements = await cls._check_achievements(
                session["user_id"], session, pain_level_after, completion_rate, avg_accuracy
            )
            
            # Create session result
            session_result = ExerciseSession(
                session_id=session_id,
                user_id=session["user_id"],
                exercise_type=exercise_type,
                difficulty=difficulty,
                duration_minutes=int(actual_duration),
                target_repetitions=target_reps,
                completed_repetitions=completed_reps,
                accuracy_score=avg_accuracy,
                pain_level_before=session["pain_level_before"],
                pain_level_after=pain_level_after,
                calories_burned=calories_burned,
                points_earned=points_earned,
                achievements_unlocked=achievements,
                timestamp=datetime.utcnow()
            )
            
            # Update session status
            await DatabaseService.mongodb_update_one(
                "therapy_sessions",
                {"session_id": session_id},
                {"$set": {
                    "status": "completed",
                    "end_time": datetime.utcnow(),
                    "session_result": session_result.__dict__,
                    "user_feedback": user_feedback
                }}
            )
            
            # Store session result in PostgreSQL
            await cls._store_session_result(session_result)
            
            # Update user progress
            await cls._update_user_progress(session["user_id"], session_result)
            
            # Store health event
            await DatabaseService.store_health_event(
                session["user_id"],
                "therapy_session_completed",
                {
                    "exercise_type": session["exercise_type"],
                    "completion_rate": completion_rate,
                    "pain_reduction": session["pain_level_before"] - pain_level_after,
                    "points_earned": points_earned,
                    "achievements": achievements
                }
            )
            
            logger.info(
                "Therapy session completed",
                user_id=session["user_id"],
                session_id=session_id,
                completion_rate=completion_rate,
                pain_reduction=session["pain_level_before"] - pain_level_after
            )
            
            return {
                "session_result": session_result.__dict__,
                "summary": {
                    "completion_rate": f"{completion_rate:.1%}",
                    "accuracy_score": f"{avg_accuracy:.1%}",
                    "quality_score": f"{avg_quality:.1%}",
                    "pain_reduction": session["pain_level_before"] - pain_level_after,
                    "calories_burned": calories_burned,
                    "points_earned": points_earned,
                    "achievements_unlocked": len(achievements)
                },
                "achievements": achievements,
                "recommendations": cls._generate_session_recommendations(session_result)
            }
            
        except Exception as e:
            logger.error("Failed to complete therapy session", session_id=session_id, error=str(e))
            raise
    
    @classmethod
    async def _generate_personalized_instructions(
        cls,
        user_id: str,
        exercise_type: ExerciseType,
        difficulty: DifficultyLevel,
        pain_level: int
    ) -> Dict[str, Any]:
        """Generate personalized exercise instructions."""
        
        config = cls.EXERCISE_CONFIGS[exercise_type]
        
        instructions = {
            "setup": [
                "Position yourself in front of the camera",
                "Ensure good lighting and clear visibility",
                "Wear comfortable clothing that allows movement"
            ],
            "exercise_steps": [],
            "safety_tips": [],
            "modifications": []
        }
        
        # Exercise-specific instructions
        if exercise_type == ExerciseType.NECK_ROTATION:
            instructions["exercise_steps"] = [
                "Sit or stand with your spine straight",
                "Slowly turn your head to the right",
                "Hold for 2-3 seconds",
                "Return to center",
                "Repeat to the left side"
            ]
            instructions["safety_tips"] = [
                "Move slowly and gently",
                "Stop if you feel dizzy or pain",
                "Don't force the movement"
            ]
            
        elif exercise_type == ExerciseType.ARM_RAISES:
            instructions["exercise_steps"] = [
                "Stand with feet shoulder-width apart",
                "Raise both arms to shoulder height",
                "Hold for 2 seconds",
                "Lower arms slowly",
                "Repeat the movement"
            ]
            
        # Add difficulty modifications
        if difficulty == DifficultyLevel.BEGINNER:
            instructions["modifications"].append("Start with smaller range of motion")
            instructions["modifications"].append("Take breaks between repetitions if needed")
        elif difficulty == DifficultyLevel.REHABILITATION:
            instructions["modifications"].append("Focus on gentle, pain-free movement")
            instructions["modifications"].append("Stop immediately if pain increases")
        
        # Add pain-level specific advice
        if pain_level >= 7:
            instructions["safety_tips"].append("Exercise very gently due to high pain level")
            instructions["modifications"].append("Reduce range of motion significantly")
        
        return instructions
    
    @classmethod
    def _estimate_calories(cls, exercise_type: ExerciseType, duration_minutes: int, difficulty: DifficultyLevel) -> float:
        """Estimate calories burned during exercise."""
        
        # Base calories per minute for different exercises
        base_calories = {
            ExerciseType.NECK_ROTATION: 1.5,
            ExerciseType.SHOULDER_ROLLS: 2.0,
            ExerciseType.ARM_RAISES: 3.0,
            ExerciseType.SPINE_TWIST: 2.5,
            ExerciseType.LEG_LIFTS: 4.0,
            ExerciseType.BALANCE_TRAINING: 2.8,
            ExerciseType.BREATHING_EXERCISE: 1.0,
            ExerciseType.FINGER_EXERCISES: 0.8
        }
        
        # Difficulty multipliers
        difficulty_multipliers = {
            DifficultyLevel.BEGINNER: 0.8,
            DifficultyLevel.INTERMEDIATE: 1.0,
            DifficultyLevel.ADVANCED: 1.3,
            DifficultyLevel.REHABILITATION: 0.6
        }
        
        base = base_calories.get(exercise_type, 2.0)
        multiplier = difficulty_multipliers.get(difficulty, 1.0)
        
        return round(base * duration_minutes * multiplier, 1)
    
    @classmethod
    def _calculate_potential_points(cls, target_reps: int, difficulty: DifficultyLevel) -> int:
        """Calculate potential points for session."""
        
        base_points = target_reps * 10
        
        difficulty_bonus = {
            DifficultyLevel.BEGINNER: 1.0,
            DifficultyLevel.INTERMEDIATE: 1.2,
            DifficultyLevel.ADVANCED: 1.5,
            DifficultyLevel.REHABILITATION: 0.8
        }
        
        return int(base_points * difficulty_bonus.get(difficulty, 1.0))
    
    @classmethod
    def _calculate_calories_burned(
        cls,
        exercise_type: ExerciseType,
        duration_minutes: float,
        difficulty: DifficultyLevel,
        completion_rate: float
    ) -> float:
        """Calculate actual calories burned."""
        
        estimated = cls._estimate_calories(exercise_type, int(duration_minutes), difficulty)
        return round(estimated * completion_rate, 1)
    
    @classmethod
    def _calculate_points_earned(
        cls,
        completed_reps: int,
        avg_accuracy: float,
        completion_rate: float,
        difficulty: DifficultyLevel
    ) -> int:
        """Calculate points earned in session."""
        
        base_points = completed_reps * 10
        accuracy_bonus = int(base_points * avg_accuracy * 0.5)
        completion_bonus = int(base_points * completion_rate * 0.3)
        
        difficulty_multiplier = {
            DifficultyLevel.BEGINNER: 1.0,
            DifficultyLevel.INTERMEDIATE: 1.2,
            DifficultyLevel.ADVANCED: 1.5,
            DifficultyLevel.REHABILITATION: 0.8
        }
        
        total_points = (base_points + accuracy_bonus + completion_bonus) * difficulty_multiplier.get(difficulty, 1.0)
        
        return int(total_points)
    
    @classmethod
    async def _check_achievements(
        cls,
        user_id: str,
        session: Dict[str, Any],
        pain_level_after: int,
        completion_rate: float,
        avg_accuracy: float
    ) -> List[str]:
        """Check for unlocked achievements."""
        
        achievements = []
        
        # Get user's session history
        user_sessions = await DatabaseService.mongodb_find_many(
            "therapy_sessions",
            {"user_id": user_id, "status": "completed"},
            limit=100
        )
        
        # First session achievement
        if len(user_sessions) == 0:
            achievements.append("first_session")
        
        # Pain reduction achievement
        pain_reduction = session["pain_level_before"] - pain_level_after
        if pain_reduction >= 3:
            achievements.append("pain_reducer")
        
        # Pain-free achievement
        if pain_level_after == 0:
            achievements.append("pain_free")
        
        # Accuracy achievement
        if avg_accuracy >= 0.9:
            # Check if user has achieved 90%+ accuracy in 5 sessions
            high_accuracy_sessions = [s for s in user_sessions if np.mean(s.get("performance_metrics", {}).get("accuracy_scores", [0])) >= 0.9]
            if len(high_accuracy_sessions) >= 4:  # 4 previous + current = 5
                achievements.append("accuracy_master")
        
        # Duration achievement
        duration = (datetime.utcnow() - session["start_time"]).total_seconds() / 60
        if duration >= 30:
            achievements.append("endurance_champion")
        
        return achievements
    
    @classmethod
    async def _store_session_result(cls, session_result: ExerciseSession):
        """Store session result in PostgreSQL."""
        
        try:
            await DatabaseService.execute_query(
                """
                INSERT INTO therapy_sessions (
                    session_id, user_id, exercise_type, difficulty, duration_minutes,
                    target_repetitions, completed_repetitions, accuracy_score,
                    pain_level_before, pain_level_after, calories_burned,
                    points_earned, achievements_unlocked, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """,
                session_result.session_id,
                session_result.user_id,
                session_result.exercise_type.value,
                session_result.difficulty.value,
                session_result.duration_minutes,
                session_result.target_repetitions,
                session_result.completed_repetitions,
                session_result.accuracy_score,
                session_result.pain_level_before,
                session_result.pain_level_after,
                session_result.calories_burned,
                session_result.points_earned,
                session_result.achievements_unlocked,
                session_result.timestamp
            )
            
        except Exception as e:
            logger.error("Failed to store session result", error=str(e))
    
    @classmethod
    async def _update_user_progress(cls, user_id: str, session_result: ExerciseSession):
        """Update user's overall progress and statistics."""
        
        try:
            # Get or create user progress
            progress = await DatabaseService.mongodb_find_one(
                "user_therapy_progress",
                {"user_id": user_id}
            )
            
            if not progress:
                progress = {
                    "user_id": user_id,
                    "total_sessions": 0,
                    "total_points": 0,
                    "total_calories": 0.0,
                    "total_exercise_time": 0,
                    "achievements": [],
                    "exercise_stats": {},
                    "pain_tracking": [],
                    "created_at": datetime.utcnow()
                }
            
            # Update statistics
            progress["total_sessions"] += 1
            progress["total_points"] += session_result.points_earned
            progress["total_calories"] += session_result.calories_burned
            progress["total_exercise_time"] += session_result.duration_minutes
            
            # Add new achievements
            for achievement in session_result.achievements_unlocked:
                if achievement not in progress["achievements"]:
                    progress["achievements"].append(achievement)
            
            # Update exercise-specific stats
            exercise_type = session_result.exercise_type.value
            if exercise_type not in progress["exercise_stats"]:
                progress["exercise_stats"][exercise_type] = {
                    "sessions": 0,
                    "total_reps": 0,
                    "avg_accuracy": 0.0,
                    "best_accuracy": 0.0
                }
            
            exercise_stats = progress["exercise_stats"][exercise_type]
            exercise_stats["sessions"] += 1
            exercise_stats["total_reps"] += session_result.completed_repetitions
            exercise_stats["avg_accuracy"] = (exercise_stats["avg_accuracy"] * (exercise_stats["sessions"] - 1) + session_result.accuracy_score) / exercise_stats["sessions"]
            exercise_stats["best_accuracy"] = max(exercise_stats["best_accuracy"], session_result.accuracy_score)
            
            # Track pain levels
            progress["pain_tracking"].append({
                "date": session_result.timestamp,
                "before": session_result.pain_level_before,
                "after": session_result.pain_level_after,
                "exercise_type": exercise_type
            })
            
            # Keep only last 100 pain tracking entries
            progress["pain_tracking"] = progress["pain_tracking"][-100:]
            
            progress["updated_at"] = datetime.utcnow()
            
            # Save updated progress
            await DatabaseService.mongodb_update_one(
                "user_therapy_progress",
                {"user_id": user_id},
                {"$set": progress},
                upsert=True
            )
            
        except Exception as e:
            logger.error("Failed to update user progress", user_id=user_id, error=str(e))
    
    @classmethod
    def _generate_session_recommendations(cls, session_result: ExerciseSession) -> List[str]:
        """Generate recommendations based on session performance."""
        
        recommendations = []
        
        completion_rate = session_result.completed_repetitions / session_result.target_repetitions
        pain_reduction = session_result.pain_level_before - session_result.pain_level_after
        
        # Completion rate recommendations
        if completion_rate >= 0.9:
            recommendations.append("Excellent completion rate! Consider increasing difficulty next time.")
        elif completion_rate < 0.5:
            recommendations.append("Try to complete more repetitions. Consider reducing difficulty if needed.")
        
        # Pain level recommendations
        if pain_reduction > 0:
            recommendations.append(f"Great job! You reduced your pain level by {pain_reduction} points.")
        elif pain_reduction < 0:
            recommendations.append("Your pain increased during exercise. Consider gentler movements or consult a healthcare provider.")
        else:
            recommendations.append("Pain level remained stable. Continue regular exercise for long-term benefits.")
        
        # Accuracy recommendations
        if session_result.accuracy_score >= 0.8:
            recommendations.append("Excellent form! Your movement accuracy was very good.")
        elif session_result.accuracy_score < 0.6:
            recommendations.append("Focus on movement quality over speed. Consider reviewing exercise instructions.")
        
        # Exercise-specific recommendations
        if session_result.exercise_type == ExerciseType.NECK_ROTATION:
            recommendations.append("Continue neck exercises daily to maintain mobility.")
        elif session_result.exercise_type == ExerciseType.BALANCE_TRAINING:
            recommendations.append("Balance training is excellent for fall prevention. Keep it up!")
        
        return recommendations
    
    @classmethod
    async def get_user_progress(cls, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user therapy progress."""
        
        try:
            progress = await DatabaseService.mongodb_find_one(
                "user_therapy_progress",
                {"user_id": user_id}
            )
            
            if not progress:
                return {"message": "No therapy progress found"}
            
            # Calculate additional metrics
            recent_sessions = await DatabaseService.mongodb_find_many(
                "therapy_sessions",
                {"user_id": user_id, "status": "completed"},
                limit=10,
                sort=[("start_time", -1)]
            )
            
            # Calculate trends
            pain_trend = cls._calculate_pain_trend(progress.get("pain_tracking", []))
            consistency_score = cls._calculate_consistency_score(recent_sessions)
            
            return {
                "user_progress": progress,
                "trends": {
                    "pain_trend": pain_trend,
                    "consistency_score": consistency_score
                },
                "recent_sessions": len(recent_sessions),
                "next_recommendations": cls._generate_progress_recommendations(progress, recent_sessions)
            }
            
        except Exception as e:
            logger.error("Failed to get user progress", user_id=user_id, error=str(e))
            return {"error": str(e)}
    
    @classmethod
    def _calculate_pain_trend(cls, pain_tracking: List[Dict[str, Any]]) -> str:
        """Calculate pain level trend."""
        
        if len(pain_tracking) < 3:
            return "insufficient_data"
        
        recent_reductions = []
        for entry in pain_tracking[-10:]:
            reduction = entry["before"] - entry["after"]
            recent_reductions.append(reduction)
        
        avg_reduction = np.mean(recent_reductions)
        
        if avg_reduction > 1.5:
            return "improving"
        elif avg_reduction > 0.5:
            return "stable_improvement"
        elif avg_reduction > -0.5:
            return "stable"
        else:
            return "worsening"
    
    @classmethod
    def _calculate_consistency_score(cls, recent_sessions: List[Dict[str, Any]]) -> float:
        """Calculate exercise consistency score."""
        
        if not recent_sessions:
            return 0.0
        
        # Calculate days between sessions
        session_dates = [session["start_time"] for session in recent_sessions]
        session_dates.sort()
        
        if len(session_dates) < 2:
            return 0.5
        
        # Calculate average days between sessions
        intervals = []
        for i in range(1, len(session_dates)):
            interval = (session_dates[i] - session_dates[i-1]).days
            intervals.append(interval)
        
        avg_interval = np.mean(intervals)
        
        # Score based on consistency (ideal is 1-2 days between sessions)
        if avg_interval <= 2:
            return 1.0
        elif avg_interval <= 4:
            return 0.8
        elif avg_interval <= 7:
            return 0.6
        else:
            return 0.3
    
    @classmethod
    def _generate_progress_recommendations(cls, progress: Dict[str, Any], recent_sessions: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on user progress."""
        
        recommendations = []
        
        total_sessions = progress.get("total_sessions", 0)
        
        if total_sessions < 5:
            recommendations.append("Keep building your exercise habit. Aim for 3-4 sessions per week.")
        elif total_sessions < 20:
            recommendations.append("Great progress! Consider trying different exercise types for variety.")
        else:
            recommendations.append("Excellent commitment! You're building strong healthy habits.")
        
        # Pain trend recommendations
        pain_tracking = progress.get("pain_tracking", [])
        if pain_tracking:
            pain_trend = cls._calculate_pain_trend(pain_tracking)
            
            if pain_trend == "improving":
                recommendations.append("Your pain levels are improving! Continue your current routine.")
            elif pain_trend == "worsening":
                recommendations.append("Consider consulting a healthcare provider about your pain levels.")
        
        # Exercise variety recommendations
        exercise_stats = progress.get("exercise_stats", {})
        if len(exercise_stats) == 1:
            recommendations.append("Try adding different exercise types to target various muscle groups.")
        
        return recommendations
    # =============================================================================
    # LEADERBOARD AND GAMIFICATION METHODS
    # =============================================================================
    
    @classmethod
    async def get_leaderboard(
        cls,
        timeframe: str = "weekly",
        exercise_type: Optional[str] = None,
        limit: int = 50,
        requesting_user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get therapy game leaderboard."""
        
        try:
            # Calculate date range based on timeframe
            now = datetime.utcnow()
            
            if timeframe == "weekly":
                start_date = now - timedelta(days=7)
            elif timeframe == "monthly":
                start_date = now - timedelta(days=30)
            else:  # all_time
                start_date = datetime(2020, 1, 1)  # Far back date
            
            # Build aggregation pipeline
            match_stage = {
                "status": "completed",
                "start_time": {"$gte": start_date}
            }
            
            if exercise_type:
                match_stage["exercise_type"] = exercise_type
            
            pipeline = [
                {"$match": match_stage},
                {
                    "$group": {
                        "_id": "$user_id",
                        "total_points": {"$sum": "$session_result.points_earned"},
                        "total_sessions": {"$sum": 1},
                        "total_reps": {"$sum": "$session_result.completed_repetitions"},
                        "avg_accuracy": {"$avg": "$session_result.accuracy_score"},
                        "total_calories": {"$sum": "$session_result.calories_burned"},
                        "pain_reduction_total": {
                            "$sum": {
                                "$subtract": [
                                    "$session_result.pain_level_before",
                                    "$session_result.pain_level_after"
                                ]
                            }
                        },
                        "achievements_count": {"$sum": {"$size": "$session_result.achievements_unlocked"}},
                        "last_session": {"$max": "$start_time"}
                    }
                },
                {"$sort": {"total_points": -1}},
                {"$limit": limit}
            ]
            
            # Execute aggregation
            leaderboard_data = await DatabaseService.mongodb_aggregate("therapy_sessions", pipeline)
            
            # Get user details and format leaderboard
            leaderboard = []
            user_rank = None
            
            for i, entry in enumerate(leaderboard_data):
                user_id = entry["_id"]
                
                # Get user info (anonymized for privacy)
                user_info = await DatabaseService.get_user_by_id(user_id)
                display_name = f"User{user_id[-4:]}" if user_info else "Anonymous"
                
                # Calculate level
                total_points = entry["total_points"]
                level = cls._calculate_user_level(total_points)
                level_info = cls.LEVEL_SYSTEM.get(level, {})
                
                leaderboard_entry = {
                    "rank": i + 1,
                    "user_id": user_id if user_id == requesting_user_id else None,  # Only show own ID
                    "display_name": display_name,
                    "total_points": total_points,
                    "level": {
                        "level": level,
                        "name": level_info.get("name", "Unknown"),
                        "color": level_info.get("color", "#8BC34A")
                    },
                    "stats": {
                        "total_sessions": entry["total_sessions"],
                        "total_repetitions": entry["total_reps"],
                        "average_accuracy": round(entry["avg_accuracy"] * 100, 1),
                        "total_calories": round(entry["total_calories"], 1),
                        "pain_reduction_total": entry["pain_reduction_total"],
                        "achievements_count": entry["achievements_count"]
                    },
                    "last_active": entry["last_session"],
                    "is_current_user": user_id == requesting_user_id
                }
                
                leaderboard.append(leaderboard_entry)
                
                # Track current user's rank
                if user_id == requesting_user_id:
                    user_rank = i + 1
            
            return {
                "leaderboard": leaderboard,
                "timeframe": timeframe,
                "exercise_type": exercise_type,
                "total_entries": len(leaderboard),
                "user_rank": user_rank,
                "last_updated": now.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get leaderboard", error=str(e))
            return {"error": str(e)}
    
    @classmethod
    async def get_personal_ranking(
        cls,
        user_id: str,
        timeframe: str = "weekly"
    ) -> Dict[str, Any]:
        """Get user's personal ranking and nearby competitors."""
        
        try:
            # Get full leaderboard
            full_leaderboard = await cls.get_leaderboard(
                timeframe=timeframe,
                limit=1000,  # Get more entries to find user's position
                requesting_user_id=user_id
            )
            
            if "error" in full_leaderboard:
                return full_leaderboard
            
            leaderboard = full_leaderboard["leaderboard"]
            user_rank = None
            user_entry = None
            
            # Find user's position
            for entry in leaderboard:
                if entry["is_current_user"]:
                    user_rank = entry["rank"]
                    user_entry = entry
                    break
            
            if not user_entry:
                return {
                    "message": "User not found in leaderboard",
                    "user_rank": None,
                    "nearby_competitors": []
                }
            
            # Get nearby competitors (5 above and 5 below)
            nearby_start = max(0, user_rank - 6)
            nearby_end = min(len(leaderboard), user_rank + 5)
            nearby_competitors = leaderboard[nearby_start:nearby_end]
            
            return {
                "user_rank": user_rank,
                "total_competitors": len(leaderboard),
                "user_entry": user_entry,
                "nearby_competitors": nearby_competitors,
                "percentile": round((1 - (user_rank - 1) / len(leaderboard)) * 100, 1) if len(leaderboard) > 1 else 100.0
            }
            
        except Exception as e:
            logger.error("Failed to get personal ranking", user_id=user_id, error=str(e))
            return {"error": str(e)}
    
    @classmethod
    async def get_user_streaks(cls, user_id: str) -> Dict[str, Any]:
        """Get user's exercise streaks and consistency data."""
        
        try:
            # Get user's completed sessions
            sessions = await DatabaseService.mongodb_find_many(
                "therapy_sessions",
                {"user_id": user_id, "status": "completed"},
                sort=[("start_time", -1)],
                limit=100
            )
            
            if not sessions:
                return {
                    "current_streak": 0,
                    "longest_streak": 0,
                    "weekly_consistency": 0.0,
                    "monthly_consistency": 0.0,
                    "streak_history": []
                }
            
            # Calculate streaks
            session_dates = []
            for session in sessions:
                session_date = session["start_time"].date()
                if session_date not in session_dates:
                    session_dates.append(session_date)
            
            session_dates.sort(reverse=True)
            
            # Calculate current streak
            current_streak = 0
            today = datetime.utcnow().date()
            
            for i, session_date in enumerate(session_dates):
                expected_date = today - timedelta(days=i)
                if session_date == expected_date:
                    current_streak += 1
                else:
                    break
            
            # Calculate longest streak
            longest_streak = 0
            temp_streak = 1
            
            for i in range(1, len(session_dates)):
                if session_dates[i-1] - session_dates[i] == timedelta(days=1):
                    temp_streak += 1
                else:
                    longest_streak = max(longest_streak, temp_streak)
                    temp_streak = 1
            
            longest_streak = max(longest_streak, temp_streak)
            
            # Calculate consistency percentages
            now = datetime.utcnow()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            weekly_sessions = [s for s in sessions if s["start_time"] >= week_ago]
            monthly_sessions = [s for s in sessions if s["start_time"] >= month_ago]
            
            weekly_consistency = min(100.0, (len(weekly_sessions) / 7) * 100)
            monthly_consistency = min(100.0, (len(monthly_sessions) / 30) * 100)
            
            # Create streak history (last 30 days)
            streak_history = []
            for i in range(30):
                date = today - timedelta(days=i)
                has_session = date in session_dates
                streak_history.append({
                    "date": date.isoformat(),
                    "has_session": has_session,
                    "day_name": date.strftime("%a")
                })
            
            streak_history.reverse()  # Show chronologically
            
            return {
                "current_streak": current_streak,
                "longest_streak": longest_streak,
                "weekly_consistency": round(weekly_consistency, 1),
                "monthly_consistency": round(monthly_consistency, 1),
                "total_active_days": len(session_dates),
                "streak_history": streak_history,
                "streak_milestones": {
                    "next_milestone": cls._get_next_streak_milestone(current_streak),
                    "milestone_progress": cls._get_streak_milestone_progress(current_streak)
                }
            }
            
        except Exception as e:
            logger.error("Failed to get user streaks", user_id=user_id, error=str(e))
            return {"error": str(e)}
    
    @classmethod
    def _get_next_streak_milestone(cls, current_streak: int) -> Dict[str, Any]:
        """Get the next streak milestone."""
        
        milestones = [7, 14, 30, 60, 100, 365]
        
        for milestone in milestones:
            if current_streak < milestone:
                return {
                    "days": milestone,
                    "name": cls._get_milestone_name(milestone),
                    "days_remaining": milestone - current_streak
                }
        
        return {
            "days": 365,
            "name": "Streak Legend",
            "days_remaining": 0
        }
    
    @classmethod
    def _get_milestone_name(cls, days: int) -> str:
        """Get milestone name for streak days."""
        
        milestone_names = {
            7: "Week Warrior",
            14: "Fortnight Fighter",
            30: "Monthly Master",
            60: "Consistency Champion",
            100: "Century Streaker",
            365: "Year-Long Legend"
        }
        
        return milestone_names.get(days, f"{days}-Day Streak")
    
    @classmethod
    def _get_streak_milestone_progress(cls, current_streak: int) -> float:
        """Get progress towards next milestone as percentage."""
        
        next_milestone = cls._get_next_streak_milestone(current_streak)
        milestone_days = next_milestone["days"]
        
        if milestone_days == 0:
            return 100.0
        
        # Find previous milestone
        milestones = [0, 7, 14, 30, 60, 100, 365]
        prev_milestone = 0
        
        for milestone in milestones:
            if milestone < milestone_days:
                prev_milestone = milestone
            else:
                break
        
        progress_range = milestone_days - prev_milestone
        current_progress = current_streak - prev_milestone
        
        return (current_progress / progress_range) * 100 if progress_range > 0 else 0.0
    
    @classmethod
    async def get_daily_challenges(cls, user_id: str) -> Dict[str, Any]:
        """Get daily and weekly challenges for the user."""
        
        try:
            today = datetime.utcnow().date()
            
            # Get user's progress to determine appropriate challenges
            progress_data = await cls.get_user_progress(user_id)
            
            if "error" in progress_data:
                user_level = 1
                total_sessions = 0
            else:
                user_progress = progress_data.get("user_progress", {})
                total_points = user_progress.get("total_points", 0)
                user_level = cls._calculate_user_level(total_points)
                total_sessions = user_progress.get("total_sessions", 0)
            
            # Generate daily challenges
            daily_challenges = cls._generate_daily_challenges(user_level, total_sessions)
            
            # Generate weekly challenges
            weekly_challenges = cls._generate_weekly_challenges(user_level, total_sessions)
            
            # Check completion status
            completed_today = await cls._get_completed_challenges_today(user_id, today)
            
            for challenge in daily_challenges:
                challenge["completed"] = challenge["id"] in completed_today
            
            return {
                "daily_challenges": daily_challenges,
                "weekly_challenges": weekly_challenges,
                "challenge_date": today.isoformat(),
                "user_level": user_level,
                "refresh_time": "00:00 UTC"
            }
            
        except Exception as e:
            logger.error("Failed to get daily challenges", user_id=user_id, error=str(e))
            return {"error": str(e)}
    
    @classmethod
    def _generate_daily_challenges(cls, user_level: int, total_sessions: int) -> List[Dict[str, Any]]:
        """Generate daily challenges based on user level."""
        
        challenges = []
        
        # Basic challenges for all levels
        challenges.append({
            "id": f"daily_session_{datetime.utcnow().strftime('%Y%m%d')}",
            "name": "Daily Exercise",
            "description": "Complete one therapy session today",
            "type": "session_count",
            "target": 1,
            "reward_points": 50,
            "difficulty": "easy",
            "icon": "🎯"
        })
        
        # Level-appropriate challenges
        if user_level >= 2:
            challenges.append({
                "id": f"accuracy_challenge_{datetime.utcnow().strftime('%Y%m%d')}",
                "name": "Precision Practice",
                "description": "Achieve 80%+ accuracy in a session",
                "type": "accuracy",
                "target": 0.8,
                "reward_points": 75,
                "difficulty": "medium",
                "icon": "🎯"
            })
        
        if user_level >= 3:
            challenges.append({
                "id": f"reps_challenge_{datetime.utcnow().strftime('%Y%m%d')}",
                "name": "Rep Master",
                "description": "Complete 20+ repetitions in a single session",
                "type": "repetitions",
                "target": 20,
                "reward_points": 100,
                "difficulty": "medium",
                "icon": "💪"
            })
        
        if user_level >= 5:
            challenges.append({
                "id": f"pain_reduction_{datetime.utcnow().strftime('%Y%m%d')}",
                "name": "Pain Warrior",
                "description": "Reduce pain level by 3+ points in a session",
                "type": "pain_reduction",
                "target": 3,
                "reward_points": 150,
                "difficulty": "hard",
                "icon": "⚔️"
            })
        
        return challenges
    
    @classmethod
    def _generate_weekly_challenges(cls, user_level: int, total_sessions: int) -> List[Dict[str, Any]]:
        """Generate weekly challenges based on user level."""
        
        challenges = []
        week_start = datetime.utcnow().strftime("%Y-W%U")
        
        challenges.append({
            "id": f"weekly_consistency_{week_start}",
            "name": "Weekly Warrior",
            "description": "Complete exercises 5 days this week",
            "type": "weekly_sessions",
            "target": 5,
            "reward_points": 300,
            "difficulty": "medium",
            "icon": "📅",
            "expires": "Sunday 23:59 UTC"
        })
        
        if user_level >= 3:
            challenges.append({
                "id": f"weekly_points_{week_start}",
                "name": "Point Collector",
                "description": "Earn 1000+ points this week",
                "type": "weekly_points",
                "target": 1000,
                "reward_points": 500,
                "difficulty": "hard",
                "icon": "🌟",
                "expires": "Sunday 23:59 UTC"
            })
        
        return challenges
    
    @classmethod
    async def _get_completed_challenges_today(cls, user_id: str, date: datetime.date) -> List[str]:
        """Get list of challenge IDs completed today."""
        
        try:
            # Get completed challenges from database
            completed = await DatabaseService.mongodb_find_one(
                "user_challenges",
                {
                    "user_id": user_id,
                    "date": date.isoformat()
                }
            )
            
            return completed.get("completed_challenges", []) if completed else []
            
        except Exception as e:
            logger.error("Failed to get completed challenges", user_id=user_id, error=str(e))
            return []
    
    @classmethod
    async def complete_challenge(cls, user_id: str, challenge_id: str) -> Dict[str, Any]:
        """Mark a challenge as completed and award points."""
        
        try:
            today = datetime.utcnow().date()
            
            # Check if challenge is already completed
            completed_today = await cls._get_completed_challenges_today(user_id, today)
            
            if challenge_id in completed_today:
                return {
                    "success": False,
                    "message": "Challenge already completed today"
                }
            
            # Add to completed challenges
            await DatabaseService.mongodb_update_one(
                "user_challenges",
                {"user_id": user_id, "date": today.isoformat()},
                {
                    "$addToSet": {"completed_challenges": challenge_id},
                    "$setOnInsert": {"user_id": user_id, "date": today.isoformat()}
                },
                upsert=True
            )
            
            # Award points (this would be more sophisticated in production)
            challenge_points = 100  # Default points
            
            # Update user progress
            await DatabaseService.mongodb_update_one(
                "user_therapy_progress",
                {"user_id": user_id},
                {
                    "$inc": {"total_points": challenge_points},
                    "$push": {"challenge_completions": {
                        "challenge_id": challenge_id,
                        "completed_at": datetime.utcnow(),
                        "points_awarded": challenge_points
                    }}
                }
            )
            
            return {
                "success": True,
                "message": "Challenge completed!",
                "points_awarded": challenge_points,
                "challenge_id": challenge_id
            }
            
        except Exception as e:
            logger.error("Failed to complete challenge", user_id=user_id, challenge_id=challenge_id, error=str(e))
            return {"success": False, "error": str(e)}