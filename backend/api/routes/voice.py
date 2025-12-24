# HealthSync AI - Voice AI Doctor Routes with Real Audio Processing
import asyncio
import base64
import uuid
import time
import tempfile
import os
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import structlog

from config_flexible import settings
from services.voice_service import VoiceProcessingService
from services.voice_agent_service import VoiceAgentService
from services.mongodb_atlas_service import get_mongodb_service

logger = structlog.get_logger(__name__)
router = APIRouter()


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class VoiceSessionRequest(BaseModel):
    user_id: str
    symptoms: Optional[list] = []

class VoiceMessageRequest(BaseModel):
    session_id: str
    message: Optional[str] = None
    transcript: Optional[str] = None

class VoiceSessionResponse(BaseModel):
    success: bool
    session_id: str
    ai_response: str
    analysis: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class VoiceMessageResponse(BaseModel):
    success: bool
    ai_response: str
    session_id: str
    analysis: Optional[Dict[str, Any]] = None
    conversation: Optional[list] = None

class VoiceCommandRequest(BaseModel):
    text: str
    user_id: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


# =============================================================================
# VOICE AI DOCTOR ROUTES
# =============================================================================

@router.post("/start-session")
async def start_voice_session(session_data: VoiceSessionRequest):
    """Start a new voice AI consultation session"""
    try:
        session_id = str(uuid.uuid4())
        
        # Create session data
        session = {
            "session_id": session_id,
            "user_id": session_data.user_id,
            "status": "active",
            "conversation": [],
            "symptoms": session_data.symptoms,
            "analysis": {
                "stress_level": "normal",
                "urgency": "low",
                "confidence": 0.85
            },
            "recommendations": []
        }
        
        # Store session in MongoDB
        try:
            mongodb = await get_mongodb_service()
            if mongodb.client:
                await mongodb.create_voice_session(session)
                logger.info(f"Voice session created in MongoDB: {session_id}")
        except Exception as e:
            logger.warning(f"Could not save session to MongoDB: {e}")
        
        # Generate initial AI response using Groq
        ai_response = await _get_groq_response(
            "You are a helpful AI medical assistant. Provide supportive, informative responses while always recommending professional medical consultation for serious concerns.",
            f"A patient has started a consultation session with symptoms: {session_data.symptoms or 'general health inquiry'}. Greet them warmly and ask how you can help."
        )
        
        return {
            "success": True,
            "session_id": session_id,
            "ai_response": ai_response,
            "analysis": session["analysis"],
            "message": "Voice AI session started successfully"
        }
        
    except Exception as e:
        logger.error(f"Voice session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start voice session")


@router.post("/send-audio")
async def process_voice_audio(
    audio_file: UploadFile = File(...),
    session_id: str = Form(default=""),
    user_id: str = Form(default="demo_user")
):
    """
    Process uploaded voice audio and return AI response.
    
    - **audio_file**: Audio file (webm, wav, mp3, mp4)
    - **session_id**: Active voice session ID
    - **user_id**: User identifier
    """
    temp_audio_path = None
    
    try:
        logger.info(f"=== Voice Audio Upload Request ===")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"Audio file: {audio_file.filename if audio_file else 'None'}")
        logger.info(f"Content type: {audio_file.content_type if audio_file else 'None'}")
        
        # Validate inputs
        if not audio_file:
            logger.error("No audio file provided")
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        if not session_id or session_id.strip() == "":
            logger.error("Session ID is missing or empty")
            raise HTTPException(status_code=400, detail="Session ID is required")
        
        if not audio_file.filename:
            logger.error("Audio file has no filename")
            raise HTTPException(status_code=400, detail="Invalid audio file")
        
        # Read audio file
        audio_content = await audio_file.read()
        logger.info(f"Audio content size: {len(audio_content)} bytes")
        
        if len(audio_content) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # Determine file extension based on content type
        file_extension = '.webm'
        if audio_file.content_type:
            if 'wav' in audio_file.content_type:
                file_extension = '.wav'
            elif 'mp4' in audio_file.content_type:
                file_extension = '.mp4'
            elif 'ogg' in audio_file.content_type:
                file_extension = '.ogg'
        
        # Save audio to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(audio_content)
            temp_audio_path = temp_file.name
        
        logger.info(f"Saved audio to temporary file: {temp_audio_path}")
        
        # Convert audio to text using Groq Whisper
        transcript = await _transcribe_audio_with_groq(temp_audio_path)
        
        if not transcript or transcript.strip() == "":
            transcript = "I couldn't understand the audio clearly. Could you please try again or type your message?"
        
        logger.info(f"Transcription: {transcript}")
        
        # Fetch conversation history
        history = []
        try:
            mongodb = await get_mongodb_service()
            if mongodb.client:
                session_data = await mongodb.get_voice_session(session_id)
                if session_data and "conversation" in session_data:
                    # Get last 10 messages for context (prevent token limit issues)
                    history = session_data["conversation"][-10:]
        except Exception as e:
            logger.warning(f"Could not fetch history: {e}")

        # Get AI response using Groq
        ai_response = await _get_groq_response(
            "You are an AI medical assistant. Analyze the patient's message and provide helpful, supportive guidance while recommending professional consultation when appropriate.",
            transcript,
            history
        )
        
        logger.info(f"AI response generated: {ai_response[:100]}...")
        
        # Create conversation entry with string timestamps
        conversation_entry = [
            {
                "role": "user", 
                "message": transcript, 
                "timestamp": str(time.time())
            },
            {
                "role": "ai", 
                "message": ai_response, 
                "timestamp": str(time.time())
            }
        ]
        
        # Simple voice analysis (you can enhance this)
        analysis = {
            "confidence": 0.85,
            "urgency": "high" if any(word in transcript.lower() for word in ["emergency", "severe", "urgent", "help"]) else "low",
            "sentiment": "neutral",
            "recommendations": ["Stay hydrated", "Monitor symptoms", "Consult healthcare provider if symptoms persist"]
        }
        
        # Save to MongoDB
        try:
            mongodb = await get_mongodb_service()
            if mongodb.client:
                # Add analysis to the last message (AI response)
                if conversation_entry and len(conversation_entry) > 0:
                    conversation_entry[-1]["analysis"] = analysis
                
                await mongodb.add_message_to_voice_session(session_id, conversation_entry)
                logger.info(f"Saved {len(conversation_entry)} messages to session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save conversation to MongoDB: {e}")
        
        # Return JSON response (no binary data)
        return {
            "success": True,
            "ai_response": ai_response,
            "session_id": session_id,
            "analysis": analysis,
            "conversation": conversation_entry
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice audio processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process audio: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                logger.info(f"Cleaned up temporary file: {temp_audio_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")


@router.post("/send-message")
async def send_text_message(message_data: VoiceMessageRequest):
    """Send text message and get AI response (fallback for when voice fails)"""
    try:
        user_message = message_data.message or message_data.transcript or ""
        
        if not user_message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Fetch conversation history
        history = []
        try:
            mongodb = await get_mongodb_service()
            if mongodb.client:
                session_data = await mongodb.get_voice_session(message_data.session_id)
                if session_data and "conversation" in session_data:
                    # Get last 10 messages for context
                    history = session_data["conversation"][-10:]
        except Exception as e:
            logger.warning(f"Could not fetch history: {e}")

        # Get AI response using Groq
        ai_response = await _get_groq_response(
            "You are an AI medical assistant. Analyze the patient's message and provide helpful, supportive guidance while recommending professional consultation when appropriate.",
            user_message,
            history
        )
        
        # Create conversation entry with string timestamps
        conversation_entry = [
            {
                "role": "user", 
                "message": user_message, 
                "timestamp": str(time.time())
            },
            {
                "role": "ai", 
                "message": ai_response, 
                "timestamp": str(time.time())
            }
        ]
        
        # Simple analysis
        analysis = {
            "confidence": 0.85,
            "urgency": "high" if any(word in user_message.lower() for word in ["emergency", "severe", "urgent", "help"]) else "low",
            "sentiment": "neutral",
            "recommendations": ["Stay hydrated", "Monitor symptoms", "Consult healthcare provider if symptoms persist"]
        }
        
        # Save to MongoDB
        try:
            mongodb = await get_mongodb_service()
            if mongodb.client:
                # Add analysis to the last message (AI response)
                if conversation_entry and len(conversation_entry) > 0:
                    conversation_entry[-1]["analysis"] = analysis
                
                await mongodb.add_message_to_voice_session(message_data.session_id, conversation_entry)
        except Exception as e:
            logger.error(f"Failed to save conversation to MongoDB: {e}")
        
        return {
            "success": True,
            "ai_response": ai_response,
            "session_id": message_data.session_id,
            "analysis": analysis,
            "conversation": conversation_entry
        }
        
    except Exception as e:
        logger.error(f"Text message processing error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process message")


@router.post("/command")
async def process_voice_command(request: VoiceCommandRequest):
    """
    Process a text command from the voice interface using the LangChain Agent.
    This enables the 'Experts' pattern (navigation, medication lookup, etc).
    """
    try:
        logger.info(f"Processing voice command: {request.text}")
        
        # Get History/Session Context
        history = []
        session_id = request.session_id
        mongodb = await get_mongodb_service()
        
        if mongodb.client:
            # If no session_id provided, try to find the most recent active one for this user
            if not session_id:
                recent_sessions = await mongodb.get_user_voice_sessions(request.user_id, limit=1)
                if recent_sessions:
                    session_id = recent_sessions[0]["session_id"]
                    logger.info(f"Auto-detected recent session: {session_id}")
            
            # Fetch history if we have a session_id
            if session_id:
                session_data = await mongodb.get_voice_session(session_id)
                if session_data and "conversation" in session_data:
                    history = session_data["conversation"][-10:]
        
        # Use the new VoiceAgentService with history
        user_context = request.context or {"patient_id": request.user_id, "role": "patient"}
        result = await VoiceAgentService.process_voice_command(request.text, user_context, history=history)
        
        # Save interaction to DB
        try:
            if mongodb.client:
                # If we still don't have a session_id, create one now to start tracking history
                if not session_id:
                    session_id = str(uuid.uuid4())
                    new_session = {
                        "session_id": session_id,
                        "user_id": request.user_id,
                        "status": "active",
                        "conversation": [],
                        "created_at": time.time()
                    }
                    await mongodb.create_voice_session(new_session)
                    logger.info(f"Created new voice session for command: {session_id}")

                # Construct conversation turn
                # The result is a dict, we need to extract the text message for the log
                ai_text = result.get("message", "")
                if isinstance(ai_text, dict):
                    ai_text = json.dumps(ai_text)
                
                conversation_entry = [
                    {
                        "role": "user", 
                        "message": request.text, 
                        "timestamp": str(time.time())
                    },
                    {
                        "role": "ai", 
                        "message": ai_text, 
                        "timestamp": str(time.time())
                    }
                ]
                
                await mongodb.add_message_to_voice_session(session_id, conversation_entry)
                
        except Exception as e:
            logger.warning(f"Could not log command to DB: {e}")
            
        return {
            "success": True,
            "result": result,
            "session_id": session_id 
        }
        
    except Exception as e:
        logger.error(f"Voice command processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process command: {str(e)}")





# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def _transcribe_audio_with_groq(audio_file_path: str) -> str:
    """Transcribe audio using Groq Whisper API"""
    try:
        if not settings.has_real_groq_key():
            # Demo mode - return placeholder transcript based on file size
            file_size = os.path.getsize(audio_file_path) if os.path.exists(audio_file_path) else 0
            if file_size > 1000:
                return "This is a demo transcription of your audio recording. Please connect your Groq API key for real speech-to-text functionality. I heard you speaking about your health concerns."
            else:
                return "Demo mode: Audio file seems too small. Please record for at least 2-3 seconds and try again."
        
        from groq import Groq
        
        client = Groq(api_key=settings.groq_api_key)
        
        # Check file size
        if not os.path.exists(audio_file_path):
            raise Exception("Audio file not found")
        
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            raise Exception("Audio file is empty")
        
        if file_size < 100:
            raise Exception("Audio file is too small (less than 100 bytes)")
        
        logger.info(f"Transcribing audio file: {audio_file_path} ({file_size} bytes)")
        
        with open(audio_file_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), file, "audio/webm"),
                model="whisper-large-v3",
                response_format="text",
                language="en",
                temperature=0.0
            )
        
        result = transcription.strip() if transcription else ""
        logger.info(f"Transcription result: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Groq transcription error: {str(e)}")
        return f"I couldn't transcribe the audio clearly. Error: {str(e)}. Please try speaking more clearly or check your microphone."


async def _get_groq_response(system_prompt: str, user_message: str, history: list = None) -> str:
    """Get AI response using Groq API with conversation history, falling back to OpenRouter on rate limit"""
    try:
        if not settings.has_real_groq_key():
            # Demo responses
            demo_responses = [
                f"Thank you for sharing: '{user_message}'. Based on this information, I recommend staying hydrated and getting adequate rest. If symptoms persist, please consult a healthcare provider.",
                f"I've analyzed your input: '{user_message}'. This could be related to various factors. I suggest monitoring your symptoms and seeking medical attention if they worsen.",
                f"Regarding '{user_message}', it's important to consider lifestyle factors like stress, sleep, and diet. Please consult with a doctor for a proper evaluation."
            ]
            import random
            return random.choice(demo_responses)
        
        from groq import Groq, RateLimitError
        
        client = Groq(api_key=settings.groq_api_key)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if provided
        if history:
            for msg in history:
                role = "assistant" if msg.get("role") == "ai" else "user"
                content = msg.get("message", "")
                if content:
                    messages.append({"role": role, "content": content})
                    
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = client.chat.completions.create(
                model=settings.groq_model,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except RateLimitError as e:
            logger.warning(f"Groq Rate Limit exceeded: {e}. Switching to OpenRouter fallback...")
            return await _get_openrouter_response(messages)
            
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        # Try fallback for other errors too if appropriate, but primarily for rate limits
        if "429" in str(e) or "Rate limit" in str(e):
             return await _get_openrouter_response(messages)
             
        return f"I understand your concern about '{user_message}'. Please consult with a healthcare professional for proper medical advice."

async def _get_openrouter_response(messages: list) -> str:
    """Get AI response using OpenRouter (Fallback) via direct HTTP request"""
    try:
        import httpx
        
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://healthsync.ai",
            "X-Title": "HealthSync AI",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": settings.openrouter_model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            logger.info("Successfully used OpenRouter fallback (via HTTP).")
            return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        logger.error(f"OpenRouter Fallback error: {e}")
        return "I apologize, but I am currently experiencing high traffic. Please try again in a moment."


@router.post("/test-upload")
async def test_audio_upload(
    audio_file: UploadFile = File(...),
    session_id: str = Form(default="test"),
    user_id: str = Form(default="test_user")
):
    """Test endpoint to debug file upload issues"""
    try:
        content = await audio_file.read()
        
        return {
            "success": True,
            "received": {
                "filename": audio_file.filename,
                "content_type": audio_file.content_type,
                "size": len(content),
                "session_id": session_id,
                "user_id": user_id
            }
        }
    except Exception as e:
        logger.error(f"Test upload error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@router.get("/session/{session_id}")
async def get_voice_session(session_id: str):
    """Get voice session details from MongoDB"""
    try:
        # Try to fetch from MongoDB
        try:
            mongodb = await get_mongodb_service()
            if mongodb.client:
                session = await mongodb.get_voice_session(session_id)
                if session:
                    return {
                        "success": True,
                        "session": session,
                        "storage": "mongodb"
                    }
        except Exception as e:
            logger.warning(f"Could not fetch from MongoDB: {e}")
        
        # Fallback to demo response
        return {
            "success": True,
            "session": {
                "session_id": session_id,
                "status": "active",
                "conversation": [],
                "created_at": time.time()
            },
            "storage": "demo_mode"
        }
        
    except Exception as e:
        logger.error(f"Get voice session error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session")


@router.get("/user-sessions/{user_id}")
async def get_user_voice_sessions(user_id: str, limit: int = 20):
    """Get all voice sessions for a user"""
    try:
        sessions = []
        
        # Try to fetch from MongoDB
        try:
            mongodb = await get_mongodb_service()
            if mongodb.client:
                sessions = await mongodb.get_user_voice_sessions(user_id, limit)
                logger.info(f"Found {len(sessions)} voice sessions for user {user_id}")
        except Exception as e:
            logger.warning(f"Could not fetch sessions from MongoDB: {e}")
        
        return {
            "success": True,
            "sessions": sessions,
            "count": len(sessions),
            "user_id": user_id
        }
        
    except Exception as e:
        logger.error(f"Get user voice sessions error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get user sessions")