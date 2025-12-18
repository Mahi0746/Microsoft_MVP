# HealthSync AI - WebSocket Routes
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState
import structlog

from config import settings
from api.middleware.auth import verify_token
from services.ai_service import AIService
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# CONNECTION MANAGER
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time features."""
    
    def __init__(self):
        # Active connections by user_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Voice streaming sessions
        self.voice_sessions: Dict[str, Dict[str, Any]] = {}
        # Therapy motion tracking sessions
        self.therapy_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, connection_type: str = "general"):
        """Accept WebSocket connection and add to manager."""
        
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        
        self.active_connections[user_id].append(websocket)
        
        logger.info(
            "WebSocket connected",
            user_id=user_id,
            connection_type=connection_type,
            total_connections=len(self.active_connections[user_id])
        )
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove WebSocket connection from manager."""
        
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            
            # Clean up empty connection lists
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        # Clean up any active sessions for this connection
        self._cleanup_sessions(user_id, websocket)
        
        logger.info("WebSocket disconnected", user_id=user_id)
    
    async def send_personal_message(self, message: Dict[str, Any], user_id: str):
        """Send message to specific user's connections."""
        
        if user_id in self.active_connections:
            disconnected = []
            
            for connection in self.active_connections[user_id]:
                try:
                    if connection.client_state == WebSocketState.CONNECTED:
                        await connection.send_text(json.dumps(message))
                    else:
                        disconnected.append(connection)
                except Exception as e:
                    logger.warning("Failed to send message", user_id=user_id, error=str(e))
                    disconnected.append(connection)
            
            # Remove disconnected connections
            for conn in disconnected:
                self.disconnect(conn, user_id)
    
    async def broadcast_to_doctors(self, message: Dict[str, Any], specialty: Optional[str] = None):
        """Broadcast message to all connected doctors."""
        
        # This would require tracking doctor specialties
        # For now, broadcast to all connections
        for user_id, connections in self.active_connections.items():
            await self.send_personal_message(message, user_id)
    
    def _cleanup_sessions(self, user_id: str, websocket: WebSocket):
        """Clean up sessions associated with disconnected WebSocket."""
        
        # Clean up voice sessions
        sessions_to_remove = []
        for session_id, session_data in self.voice_sessions.items():
            if session_data.get("user_id") == user_id and session_data.get("websocket") == websocket:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.voice_sessions[session_id]
        
        # Clean up therapy sessions
        sessions_to_remove = []
        for session_id, session_data in self.therapy_sessions.items():
            if session_data.get("user_id") == user_id and session_data.get("websocket") == websocket:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.therapy_sessions[session_id]


# Global connection manager
manager = ConnectionManager()


# =============================================================================
# WEBSOCKET AUTHENTICATION
# =============================================================================

async def get_websocket_user(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """Authenticate WebSocket connection using token."""
    
    try:
        # Get token from query parameters or headers
        token = None
        
        # Try query parameter first
        if "token" in websocket.query_params:
            token = websocket.query_params["token"]
        
        # Try authorization header
        elif "authorization" in websocket.headers:
            auth_header = websocket.headers["authorization"]
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
        
        if not token:
            return None
        
        # Verify token
        payload = verify_token(token)
        
        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("role")
        }
        
    except Exception as e:
        logger.warning("WebSocket authentication failed", error=str(e))
        return None


# =============================================================================
# VOICE STREAMING WEBSOCKET
# =============================================================================

@router.websocket("/voice/stream")
async def voice_stream_websocket(websocket: WebSocket):
    """Real-time voice analysis WebSocket endpoint."""
    
    # Authenticate user
    user = await get_websocket_user(websocket)
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    user_id = user["user_id"]
    
    # Connect to manager
    await manager.connect(websocket, user_id, "voice_stream")
    
    # Initialize voice session
    session_id = f"voice_stream_{int(time.time())}_{user_id[:8]}"
    manager.voice_sessions[session_id] = {
        "user_id": user_id,
        "websocket": websocket,
        "start_time": time.time(),
        "audio_chunks": [],
        "processing": False
    }
    
    try:
        # Send session started message
        await websocket.send_text(json.dumps({
            "type": "session_started",
            "session_id": session_id,
            "message": "Voice streaming session started"
        }))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "audio_chunk":
                await _handle_voice_chunk(websocket, session_id, message)
            
            elif message_type == "start_analysis":
                await _start_voice_analysis(websocket, session_id)
            
            elif message_type == "stop_session":
                await _stop_voice_session(websocket, session_id)
                break
            
            elif message_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
    
    except WebSocketDisconnect:
        logger.info("Voice streaming WebSocket disconnected", user_id=user_id, session_id=session_id)
    
    except Exception as e:
        logger.error("Voice streaming error", user_id=user_id, session_id=session_id, error=str(e))
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Voice streaming error occurred"
        }))
    
    finally:
        # Clean up session
        if session_id in manager.voice_sessions:
            del manager.voice_sessions[session_id]
        
        manager.disconnect(websocket, user_id)


async def _handle_voice_chunk(websocket: WebSocket, session_id: str, message: Dict[str, Any]):
    """Handle incoming voice audio chunk."""
    
    try:
        session = manager.voice_sessions.get(session_id)
        if not session:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Session not found"
            }))
            return
        
        # Get audio data
        audio_data = message.get("audio_data")  # Base64 encoded
        chunk_index = message.get("chunk_index", 0)
        
        if not audio_data:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "No audio data provided"
            }))
            return
        
        # Store audio chunk
        session["audio_chunks"].append({
            "index": chunk_index,
            "data": audio_data,
            "timestamp": time.time()
        })
        
        # Send acknowledgment
        await websocket.send_text(json.dumps({
            "type": "chunk_received",
            "chunk_index": chunk_index,
            "total_chunks": len(session["audio_chunks"])
        }))
        
        # Auto-analyze if we have enough chunks (e.g., every 5 seconds)
        if len(session["audio_chunks"]) >= 5 and not session["processing"]:
            await _process_voice_chunks(websocket, session_id)
    
    except Exception as e:
        logger.error("Voice chunk handling failed", session_id=session_id, error=str(e))
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Failed to process audio chunk"
        }))


async def _process_voice_chunks(websocket: WebSocket, session_id: str):
    """Process accumulated voice chunks for analysis."""
    
    try:
        session = manager.voice_sessions.get(session_id)
        if not session or session["processing"]:
            return
        
        session["processing"] = True
        
        # Combine audio chunks
        import base64
        combined_audio = b""
        
        for chunk in session["audio_chunks"]:
            chunk_bytes = base64.b64decode(chunk["data"])
            combined_audio += chunk_bytes
        
        # Send processing status
        await websocket.send_text(json.dumps({
            "type": "processing_started",
            "message": "Analyzing voice data..."
        }))
        
        # Analyze voice using AI service
        analysis_result = await AIService.analyze_voice_audio(
            combined_audio,
            session["user_id"]
        )
        
        # Send analysis result
        await websocket.send_text(json.dumps({
            "type": "analysis_result",
            "session_id": analysis_result["session_id"],
            "transcript": analysis_result["transcript"],
            "voice_analysis": analysis_result["voice_analysis"],
            "assessment": analysis_result["assessment"],
            "processing_time_ms": analysis_result["processing_time_ms"]
        }))
        
        # Clear processed chunks
        session["audio_chunks"] = []
        session["processing"] = False
    
    except Exception as e:
        logger.error("Voice processing failed", session_id=session_id, error=str(e))
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Voice analysis failed"
        }))
        
        if session_id in manager.voice_sessions:
            manager.voice_sessions[session_id]["processing"] = False


async def _start_voice_analysis(websocket: WebSocket, session_id: str):
    """Start analysis of current voice session."""
    
    session = manager.voice_sessions.get(session_id)
    if not session:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Session not found"
        }))
        return
    
    if session["audio_chunks"]:
        await _process_voice_chunks(websocket, session_id)
    else:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "No audio data to analyze"
        }))


async def _stop_voice_session(websocket: WebSocket, session_id: str):
    """Stop voice streaming session."""
    
    session = manager.voice_sessions.get(session_id)
    if not session:
        return
    
    # Process any remaining chunks
    if session["audio_chunks"] and not session["processing"]:
        await _process_voice_chunks(websocket, session_id)
    
    # Send session ended message
    await websocket.send_text(json.dumps({
        "type": "session_ended",
        "session_id": session_id,
        "duration_seconds": time.time() - session["start_time"],
        "total_chunks_processed": len(session["audio_chunks"])
    }))


# =============================================================================
# THERAPY MOTION TRACKING WEBSOCKET
# =============================================================================

@router.websocket("/therapy/motion")
async def therapy_motion_websocket(websocket: WebSocket):
    """Real-time therapy motion tracking WebSocket endpoint."""
    
    # Authenticate user
    user = await get_websocket_user(websocket)
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    user_id = user["user_id"]
    
    # Connect to manager
    await manager.connect(websocket, user_id, "therapy_motion")
    
    # Initialize therapy session
    session_id = f"therapy_{int(time.time())}_{user_id[:8]}"
    manager.therapy_sessions[session_id] = {
        "user_id": user_id,
        "websocket": websocket,
        "start_time": time.time(),
        "exercise_type": None,
        "pose_data": [],
        "repetitions": 0,
        "target_reps": 10,
        "form_scores": [],
        "pain_detected": False
    }
    
    try:
        # Send session started message
        await websocket.send_text(json.dumps({
            "type": "session_started",
            "session_id": session_id,
            "message": "Therapy motion tracking started"
        }))
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "start_exercise":
                await _start_therapy_exercise(websocket, session_id, message)
            
            elif message_type == "pose_data":
                await _handle_pose_data(websocket, session_id, message)
            
            elif message_type == "stop_exercise":
                await _stop_therapy_exercise(websocket, session_id)
                break
            
            elif message_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }))
    
    except WebSocketDisconnect:
        logger.info("Therapy motion WebSocket disconnected", user_id=user_id, session_id=session_id)
    
    except Exception as e:
        logger.error("Therapy motion error", user_id=user_id, session_id=session_id, error=str(e))
    
    finally:
        # Save session data
        await _save_therapy_session(session_id)
        
        # Clean up session
        if session_id in manager.therapy_sessions:
            del manager.therapy_sessions[session_id]
        
        manager.disconnect(websocket, user_id)


async def _start_therapy_exercise(websocket: WebSocket, session_id: str, message: Dict[str, Any]):
    """Start a therapy exercise session."""
    
    session = manager.therapy_sessions.get(session_id)
    if not session:
        return
    
    exercise_type = message.get("exercise_type", "arm_raises")
    target_reps = message.get("target_reps", 10)
    
    session["exercise_type"] = exercise_type
    session["target_reps"] = target_reps
    session["repetitions"] = 0
    session["pose_data"] = []
    session["form_scores"] = []
    
    await websocket.send_text(json.dumps({
        "type": "exercise_started",
        "exercise_type": exercise_type,
        "target_reps": target_reps,
        "message": f"Starting {exercise_type} exercise"
    }))


async def _handle_pose_data(websocket: WebSocket, session_id: str, message: Dict[str, Any]):
    """Handle incoming pose detection data."""
    
    try:
        session = manager.therapy_sessions.get(session_id)
        if not session:
            return
        
        pose_data = message.get("pose_data", {})
        timestamp = time.time()
        
        # Store pose data
        session["pose_data"].append({
            "timestamp": timestamp,
            "keypoints": pose_data.get("keypoints", []),
            "confidence": pose_data.get("confidence", 0.0)
        })
        
        # Analyze form (simplified)
        form_score = _analyze_exercise_form(
            session["exercise_type"],
            pose_data.get("keypoints", [])
        )
        
        session["form_scores"].append(form_score)
        
        # Check for repetition completion
        if _is_repetition_complete(session["exercise_type"], session["pose_data"]):
            session["repetitions"] += 1
            
            # Calculate points
            points = _calculate_exercise_points(form_score, session["repetitions"])
            
            await websocket.send_text(json.dumps({
                "type": "repetition_completed",
                "repetition": session["repetitions"],
                "target_reps": session["target_reps"],
                "form_score": form_score,
                "points_earned": points,
                "progress_percentage": (session["repetitions"] / session["target_reps"]) * 100
            }))
            
            # Check if exercise is complete
            if session["repetitions"] >= session["target_reps"]:
                await _complete_therapy_exercise(websocket, session_id)
        
        else:
            # Send real-time feedback
            await websocket.send_text(json.dumps({
                "type": "form_feedback",
                "form_score": form_score,
                "feedback": _get_form_feedback(session["exercise_type"], form_score)
            }))
    
    except Exception as e:
        logger.error("Pose data handling failed", session_id=session_id, error=str(e))


async def _complete_therapy_exercise(websocket: WebSocket, session_id: str):
    """Complete therapy exercise and calculate final scores."""
    
    session = manager.therapy_sessions.get(session_id)
    if not session:
        return
    
    # Calculate final statistics
    avg_form_score = sum(session["form_scores"]) / len(session["form_scores"]) if session["form_scores"] else 0
    total_points = sum(_calculate_exercise_points(score, i+1) for i, score in enumerate(session["form_scores"]))
    duration = time.time() - session["start_time"]
    
    await websocket.send_text(json.dumps({
        "type": "exercise_completed",
        "repetitions_completed": session["repetitions"],
        "target_reps": session["target_reps"],
        "average_form_score": round(avg_form_score, 2),
        "total_points": total_points,
        "duration_seconds": round(duration, 1),
        "pain_detected": session["pain_detected"]
    }))


async def _save_therapy_session(session_id: str):
    """Save therapy session to database."""
    
    try:
        session = manager.therapy_sessions.get(session_id)
        if not session or not session.get("exercise_type"):
            return
        
        # Calculate session statistics
        duration = time.time() - session["start_time"]
        avg_form_score = sum(session["form_scores"]) / len(session["form_scores"]) if session["form_scores"] else 0
        total_points = sum(_calculate_exercise_points(score, i+1) for i, score in enumerate(session["form_scores"]))
        
        # Save to PostgreSQL
        await DatabaseService.execute_query(
            """
            INSERT INTO therapy_sessions (user_id, exercise_type, duration_seconds, 
                                        repetitions_completed, repetitions_target, 
                                        form_accuracy, points_earned, pose_data)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            session["user_id"],
            session["exercise_type"],
            int(duration),
            session["repetitions"],
            session["target_reps"],
            avg_form_score,
            total_points,
            {"pose_data": session["pose_data"][:100]}  # Limit stored data
        )
        
        logger.info(
            "Therapy session saved",
            user_id=session["user_id"],
            exercise_type=session["exercise_type"],
            repetitions=session["repetitions"],
            points=total_points
        )
    
    except Exception as e:
        logger.error("Failed to save therapy session", session_id=session_id, error=str(e))


# =============================================================================
# APPOINTMENT NOTIFICATIONS WEBSOCKET
# =============================================================================

@router.websocket("/appointments/updates")
async def appointment_updates_websocket(websocket: WebSocket):
    """Real-time appointment and bidding updates."""
    
    # Authenticate user
    user = await get_websocket_user(websocket)
    if not user:
        await websocket.close(code=4001, reason="Authentication required")
        return
    
    user_id = user["user_id"]
    
    # Connect to manager
    await manager.connect(websocket, user_id, "appointments")
    
    try:
        # Send connection confirmation
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Connected to appointment updates"
        }))
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    
    except WebSocketDisconnect:
        logger.info("Appointment updates WebSocket disconnected", user_id=user_id)
    
    finally:
        manager.disconnect(websocket, user_id)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _analyze_exercise_form(exercise_type: str, keypoints: List[Dict[str, Any]]) -> float:
    """Analyze exercise form based on pose keypoints."""
    
    if not keypoints:
        return 0.0
    
    # Simplified form analysis (would be more sophisticated in production)
    if exercise_type == "arm_raises":
        # Check if arms are raised properly
        # This would analyze shoulder, elbow, and wrist positions
        return 0.85  # Mock score
    
    elif exercise_type == "knee_bends":
        # Check knee bend angle and balance
        return 0.90  # Mock score
    
    elif exercise_type == "neck_rotations":
        # Check neck movement range and smoothness
        return 0.75  # Mock score
    
    return 0.5  # Default score


def _is_repetition_complete(exercise_type: str, pose_data: List[Dict[str, Any]]) -> bool:
    """Check if a repetition is complete based on pose sequence."""
    
    if len(pose_data) < 10:  # Need minimum data points
        return False
    
    # Simplified repetition detection
    # In production, this would analyze the movement pattern
    return len(pose_data) % 20 == 0  # Mock: every 20 pose frames = 1 rep


def _calculate_exercise_points(form_score: float, repetition: int) -> int:
    """Calculate points earned for exercise repetition."""
    
    base_points = 10
    form_bonus = int(form_score * 20)  # Up to 20 bonus points for perfect form
    
    return base_points + form_bonus


def _get_form_feedback(exercise_type: str, form_score: float) -> str:
    """Get real-time form feedback message."""
    
    if form_score >= 0.9:
        return "Excellent form! Keep it up!"
    elif form_score >= 0.7:
        return "Good form, minor adjustments needed"
    elif form_score >= 0.5:
        return "Focus on proper posture and movement"
    else:
        return "Please check your form and slow down"


# =============================================================================
# NOTIFICATION HELPERS
# =============================================================================

async def notify_new_appointment_bid(patient_id: str, appointment_id: str, doctor_name: str, bid_amount: float):
    """Notify patient of new appointment bid."""
    
    message = {
        "type": "new_bid",
        "appointment_id": appointment_id,
        "doctor_name": doctor_name,
        "bid_amount": bid_amount,
        "timestamp": time.time()
    }
    
    await manager.send_personal_message(message, patient_id)


async def notify_appointment_accepted(doctor_id: str, appointment_id: str, patient_name: str):
    """Notify doctor that their bid was accepted."""
    
    message = {
        "type": "bid_accepted",
        "appointment_id": appointment_id,
        "patient_name": patient_name,
        "timestamp": time.time()
    }
    
    await manager.send_personal_message(message, doctor_id)


async def notify_health_alert(user_id: str, alert_type: str, message: str, severity: str = "medium"):
    """Send health alert notification to user."""
    
    notification = {
        "type": "health_alert",
        "alert_type": alert_type,
        "message": message,
        "severity": severity,
        "timestamp": time.time()
    }
    
    await manager.send_personal_message(notification, user_id)