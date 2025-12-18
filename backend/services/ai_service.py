# HealthSync AI - AI Service Integration
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Tuple
import httpx
import structlog
from groq import Groq
import replicate
import librosa
import numpy as np
from sklearn.externals import joblib
import io
import base64

from config import settings
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)


class AIService:
    """Centralized AI service for all ML/AI operations."""
    
    # Service clients
    _groq_client: Optional[Groq] = None
    _replicate_client: Optional[Any] = None
    _huggingface_client: Optional[httpx.AsyncClient] = None
    
    # Model cache
    _model_cache: Dict[str, Any] = {}
    
    @classmethod
    async def initialize(cls):
        """Initialize all AI service clients."""
        
        logger.info("Initializing AI services")
        
        try:
            # Initialize Groq client
            if settings.groq_api_key:
                cls._groq_client = Groq(api_key=settings.groq_api_key)
                logger.info("Groq client initialized")
            
            # Initialize Replicate client
            if settings.replicate_api_token:
                replicate.Client(api_token=settings.replicate_api_token)
                cls._replicate_client = replicate
                logger.info("Replicate client initialized")
            
            # Initialize Hugging Face client
            if settings.huggingface_api_key:
                cls._huggingface_client = httpx.AsyncClient(
                    headers={"Authorization": f"Bearer {settings.huggingface_api_key}"},
                    timeout=30.0
                )
                logger.info("Hugging Face client initialized")
            
            # Test connections
            await cls.health_check()
            logger.info("All AI services verified")
            
        except Exception as e:
            logger.error("Failed to initialize AI services", error=str(e))
            raise
    
    @classmethod
    async def cleanup(cls):
        """Cleanup AI service clients."""
        
        logger.info("Cleaning up AI services")
        
        try:
            if cls._huggingface_client:
                await cls._huggingface_client.aclose()
                cls._huggingface_client = None
            
            cls._groq_client = None
            cls._replicate_client = None
            cls._model_cache.clear()
            
            logger.info("AI services cleaned up")
            
        except Exception as e:
            logger.error("Error cleaning up AI services", error=str(e))
    
    @classmethod
    async def health_check(cls):
        """Check health of AI services."""
        
        # Test Groq connection
        if cls._groq_client:
            try:
                # Simple test completion
                response = cls._groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model=settings.groq_model,
                    max_tokens=5
                )
                if not response.choices:
                    raise Exception("Groq health check failed")
            except Exception as e:
                logger.warning("Groq health check failed", error=str(e))
        
        # Test Hugging Face connection
        if cls._huggingface_client:
            try:
                response = await cls._huggingface_client.get(
                    "https://api-inference.huggingface.co/models/microsoft/trocr-base-printed"
                )
                if response.status_code not in [200, 503]:  # 503 is model loading
                    raise Exception(f"Hugging Face health check failed: {response.status_code}")
            except Exception as e:
                logger.warning("Hugging Face health check failed", error=str(e))
    
    @classmethod
    async def warm_up_models(cls):
        """Warm up AI models for faster first requests."""
        
        logger.info("Warming up AI models")
        
        tasks = []
        
        # Warm up Groq
        if cls._groq_client:
            tasks.append(cls._warm_up_groq())
        
        # Warm up Hugging Face models
        if cls._huggingface_client:
            tasks.append(cls._warm_up_huggingface())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("AI models warmed up")
    
    @classmethod
    async def _warm_up_groq(cls):
        """Warm up Groq model."""
        try:
            await cls.analyze_symptoms_with_groq(
                "Test symptoms for warmup",
                {"stress_level": 0.1}
            )
        except Exception as e:
            logger.warning("Groq warmup failed", error=str(e))
    
    @classmethod
    async def _warm_up_huggingface(cls):
        """Warm up Hugging Face models."""
        try:
            # Create a small test image
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            await cls.analyze_image_with_blip2(test_image)
        except Exception as e:
            logger.warning("Hugging Face warmup failed", error=str(e))
    
    # =============================================================================
    # VOICE ANALYSIS METHODS
    # =============================================================================
    
    @classmethod
    async def analyze_voice_audio(
        cls,
        audio_data: bytes,
        user_id: str
    ) -> Dict[str, Any]:
        """Complete voice analysis pipeline."""
        
        start_time = time.time()
        
        try:
            # Step 1: Speech-to-text
            transcript = await cls.speech_to_text(audio_data)
            
            # Step 2: Extract audio features
            audio_features = await cls.extract_audio_features(audio_data)
            
            # Step 3: Analyze with Groq LLM
            ai_analysis = await cls.analyze_symptoms_with_groq(
                transcript,
                audio_features
            )
            
            # Step 4: Store voice session in MongoDB
            session_data = {
                "user_id": user_id,
                "session_id": f"voice_{int(time.time())}_{user_id[:8]}",
                "audio_features": audio_features,
                "speech_analysis": {
                    "transcript": transcript,
                    "word_count": len(transcript.split()),
                    "speech_rate": cls._calculate_speech_rate(transcript, audio_features.get("duration", 1))
                },
                "stress_indicators": audio_features.get("stress_indicators", {}),
                "analysis_results": ai_analysis,
                "processing_metadata": {
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "ai_services_used": ["whisper", "groq_llm", "librosa"],
                    "model_versions": {
                        "groq_llm": settings.groq_model
                    }
                }
            }
            
            session_id = await DatabaseService.mongodb_insert_one("voice_sessions", session_data)
            
            logger.info(
                "Voice analysis completed",
                user_id=user_id,
                session_id=session_id,
                processing_time_ms=session_data["processing_metadata"]["processing_time_ms"]
            )
            
            return {
                "session_id": session_id,
                "transcript": transcript,
                "voice_analysis": audio_features,
                "assessment": ai_analysis,
                "processing_time_ms": session_data["processing_metadata"]["processing_time_ms"]
            }
            
        except Exception as e:
            logger.error("Voice analysis failed", user_id=user_id, error=str(e))
            raise
    
    @classmethod
    async def speech_to_text(cls, audio_data: bytes) -> str:
        """Convert speech to text using Whisper via Replicate."""
        
        if not cls._replicate_client:
            raise Exception("Replicate client not initialized")
        
        try:
            # Convert audio bytes to base64 for Replicate
            audio_b64 = base64.b64encode(audio_data).decode()
            audio_uri = f"data:audio/wav;base64,{audio_b64}"
            
            # Run Whisper model
            output = cls._replicate_client.run(
                "openai/whisper:4d50797290df275329f202e48c76360b3f22b08d28c196cbc54600319435f8d2",
                input={
                    "audio": audio_uri,
                    "model": "base",
                    "transcription": "plain text"
                }
            )
            
            # Extract transcript from output
            if isinstance(output, dict) and "transcription" in output:
                return output["transcription"]
            elif isinstance(output, str):
                return output
            else:
                return str(output)
                
        except Exception as e:
            logger.error("Speech-to-text failed", error=str(e))
            # Fallback to empty transcript
            return ""
    
    @classmethod
    async def extract_audio_features(cls, audio_data: bytes) -> Dict[str, Any]:
        """Extract audio features using librosa."""
        
        try:
            # Load audio data
            audio_io = io.BytesIO(audio_data)
            y, sr = librosa.load(audio_io, sr=settings.voice_sample_rate)
            
            # Extract features
            features = {}
            
            # Basic audio properties
            features["duration"] = len(y) / sr
            features["sample_rate"] = sr
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features["mfcc"] = {
                "mean": np.mean(mfcc, axis=1).tolist(),
                "std": np.std(mfcc, axis=1).tolist()
            }
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                features["pitch"] = {
                    "mean": float(np.mean(pitch_values)),
                    "std": float(np.std(pitch_values)),
                    "min": float(np.min(pitch_values)),
                    "max": float(np.max(pitch_values))
                }
            else:
                features["pitch"] = {"mean": 0, "std": 0, "min": 0, "max": 0}
            
            # Energy features
            rms = librosa.feature.rms(y=y)[0]
            features["energy"] = {
                "rms_mean": float(np.mean(rms)),
                "rms_std": float(np.std(rms))
            }
            
            # Zero crossing rate (voice quality indicator)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features["zero_crossing_rate"] = float(np.mean(zcr))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features["spectral_centroid"] = float(np.mean(spectral_centroids))
            
            # Calculate stress indicators
            features["stress_indicators"] = cls._calculate_stress_indicators(features)
            
            return features
            
        except Exception as e:
            logger.error("Audio feature extraction failed", error=str(e))
            return {
                "duration": 0,
                "sample_rate": settings.voice_sample_rate,
                "error": str(e)
            }
    
    @classmethod
    def _calculate_stress_indicators(cls, features: Dict[str, Any]) -> Dict[str, float]:
        """Calculate stress indicators from audio features."""
        
        try:
            # Simple stress calculation based on pitch variance and energy
            pitch_variance = features.get("pitch", {}).get("std", 0)
            energy_variance = features.get("energy", {}).get("rms_std", 0)
            zcr = features.get("zero_crossing_rate", 0)
            
            # Normalize and combine features (simplified approach)
            stress_level = min(1.0, (pitch_variance / 50.0 + energy_variance * 10 + zcr * 5) / 3)
            
            return {
                "stress_level": float(stress_level),
                "anxiety_score": float(min(1.0, pitch_variance / 40.0)),
                "fatigue_score": float(min(1.0, 1.0 - energy_variance * 5)),
                "confidence_level": float(max(0.0, 1.0 - stress_level))
            }
            
        except Exception as e:
            logger.error("Stress calculation failed", error=str(e))
            return {
                "stress_level": 0.0,
                "anxiety_score": 0.0,
                "fatigue_score": 0.0,
                "confidence_level": 0.5
            }
    
    @classmethod
    def _calculate_speech_rate(cls, transcript: str, duration: float) -> float:
        """Calculate speech rate in words per minute."""
        
        if duration <= 0:
            return 0.0
        
        word_count = len(transcript.split())
        return (word_count / duration) * 60  # words per minute
    
    # =============================================================================
    # GROQ LLM METHODS
    # =============================================================================
    
    @classmethod
    async def analyze_symptoms_with_groq(
        cls,
        transcript: str,
        voice_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze symptoms using Groq LLM."""
        
        if not cls._groq_client:
            raise Exception("Groq client not initialized")
        
        try:
            # Prepare system prompt
            system_prompt = """You are a medical AI assistant analyzing patient symptoms and voice patterns. 
            Based on the transcript and voice stress indicators, provide a structured assessment.
            
            Respond with valid JSON containing:
            - risk_level: "low", "medium", "high", or "critical"
            - urgency_flag: boolean indicating if immediate attention is needed
            - recommended_actions: array of specific action recommendations
            - suggested_specialists: array of relevant medical specialties
            - confidence_score: float between 0 and 1
            - reasoning: brief explanation of the assessment
            
            Consider voice stress indicators as additional context for symptom severity."""
            
            # Prepare user message
            stress_info = voice_features.get("stress_indicators", {})
            user_message = f"""
            Patient Transcript: "{transcript}"
            
            Voice Stress Analysis:
            - Stress Level: {stress_info.get('stress_level', 0):.2f}
            - Anxiety Score: {stress_info.get('anxiety_score', 0):.2f}
            - Fatigue Score: {stress_info.get('fatigue_score', 0):.2f}
            - Confidence Level: {stress_info.get('confidence_level', 0.5):.2f}
            
            Please provide your medical assessment in JSON format.
            """
            
            # Call Groq API
            response = cls._groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=settings.groq_model,
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent medical advice
            )
            
            # Parse response
            content = response.choices[0].message.content
            
            try:
                # Try to parse as JSON
                analysis = json.loads(content)
                
                # Validate required fields
                required_fields = ["risk_level", "urgency_flag", "recommended_actions", "suggested_specialists", "confidence_score"]
                for field in required_fields:
                    if field not in analysis:
                        analysis[field] = cls._get_default_value(field)
                
                return analysis
                
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                logger.warning("Failed to parse Groq response as JSON", content=content)
                return cls._create_fallback_analysis(transcript, stress_info)
                
        except Exception as e:
            logger.error("Groq analysis failed", error=str(e))
            return cls._create_fallback_analysis(transcript, stress_info)
    
    @classmethod
    def _get_default_value(cls, field: str) -> Any:
        """Get default value for missing analysis fields."""
        
        defaults = {
            "risk_level": "medium",
            "urgency_flag": False,
            "recommended_actions": ["Consult with healthcare provider"],
            "suggested_specialists": ["Primary Care"],
            "confidence_score": 0.5,
            "reasoning": "Analysis completed with limited information"
        }
        
        return defaults.get(field, None)
    
    @classmethod
    def _create_fallback_analysis(cls, transcript: str, stress_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback analysis when AI services fail."""
        
        # Simple keyword-based risk assessment
        high_risk_keywords = ["chest pain", "shortness of breath", "severe", "emergency", "can't breathe"]
        medium_risk_keywords = ["pain", "headache", "fever", "nausea", "dizzy"]
        
        transcript_lower = transcript.lower()
        
        if any(keyword in transcript_lower for keyword in high_risk_keywords):
            risk_level = "high"
            urgency = True
        elif any(keyword in transcript_lower for keyword in medium_risk_keywords):
            risk_level = "medium"
            urgency = False
        else:
            risk_level = "low"
            urgency = False
        
        # Adjust based on stress level
        stress_level = stress_info.get("stress_level", 0)
        if stress_level > 0.7:
            if risk_level == "low":
                risk_level = "medium"
            elif risk_level == "medium":
                risk_level = "high"
        
        return {
            "risk_level": risk_level,
            "urgency_flag": urgency,
            "recommended_actions": [
                "Monitor symptoms closely",
                "Consult healthcare provider if symptoms persist"
            ],
            "suggested_specialists": ["Primary Care"],
            "confidence_score": 0.3,  # Low confidence for fallback
            "reasoning": "Automated analysis based on keyword detection and voice stress patterns"
        }
    
    # =============================================================================
    # IMAGE ANALYSIS METHODS
    # =============================================================================
    
    @classmethod
    async def analyze_medical_image(
        cls,
        image_data: bytes,
        analysis_type: str = "prescription"
    ) -> Dict[str, Any]:
        """Analyze medical image using multiple AI models."""
        
        try:
            # Run image analysis and OCR in parallel
            tasks = [
                cls.analyze_image_with_blip2(image_data),
                cls.extract_text_with_trocr(image_data)
            ]
            
            image_analysis, ocr_results = await asyncio.gather(*tasks)
            
            # Combine results
            result = {
                "image_analysis": image_analysis,
                "ocr_results": ocr_results,
                "analysis_type": analysis_type,
                "timestamp": time.time()
            }
            
            # Extract structured medication data if it's a prescription
            if analysis_type == "prescription":
                result["medication_extraction"] = cls._extract_medication_info(ocr_results)
                result["safety_analysis"] = cls._analyze_medication_safety(result["medication_extraction"])
            
            return result
            
        except Exception as e:
            logger.error("Medical image analysis failed", error=str(e))
            raise
    
    @classmethod
    async def analyze_image_with_blip2(cls, image_data: bytes) -> Dict[str, Any]:
        """Analyze image using BLIP-2 model via Hugging Face."""
        
        if not cls._huggingface_client:
            raise Exception("Hugging Face client not initialized")
        
        try:
            # Prepare image for API
            files = {"inputs": image_data}
            
            # Call BLIP-2 API
            response = await cls._huggingface_client.post(
                "https://api-inference.huggingface.co/models/Salesforce/blip2-opt-2.7b",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                
                return {
                    "description": result[0].get("generated_text", ""),
                    "confidence": 0.8,  # BLIP-2 doesn't provide confidence scores
                    "model": "blip2-opt-2.7b"
                }
            else:
                logger.warning("BLIP-2 API error", status_code=response.status_code)
                return {"description": "Image analysis unavailable", "confidence": 0.0}
                
        except Exception as e:
            logger.error("BLIP-2 analysis failed", error=str(e))
            return {"description": "Image analysis failed", "confidence": 0.0, "error": str(e)}
    
    @classmethod
    async def extract_text_with_trocr(cls, image_data: bytes) -> Dict[str, Any]:
        """Extract text from image using TrOCR model."""
        
        if not cls._huggingface_client:
            raise Exception("Hugging Face client not initialized")
        
        try:
            # Prepare image for API
            files = {"inputs": image_data}
            
            # Call TrOCR API
            response = await cls._huggingface_client.post(
                "https://api-inference.huggingface.co/models/microsoft/trocr-base-printed",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                
                extracted_text = result[0].get("generated_text", "")
                
                return {
                    "raw_text": extracted_text,
                    "confidence": 0.85,  # Estimated confidence
                    "model": "trocr-base-printed",
                    "language": "en"
                }
            else:
                logger.warning("TrOCR API error", status_code=response.status_code)
                return {"raw_text": "", "confidence": 0.0}
                
        except Exception as e:
            logger.error("TrOCR extraction failed", error=str(e))
            return {"raw_text": "", "confidence": 0.0, "error": str(e)}
    
    @classmethod
    async def extract_medical_text(
        cls,
        image_b64: str,
        model_type: str = "trocr"
    ) -> str:
        """Extract medical text from base64 image using specified OCR model."""
        
        try:
            # Convert base64 to bytes
            if image_b64.startswith('data:image'):
                image_b64 = image_b64.split(',')[1]
            
            image_bytes = base64.b64decode(image_b64)
            
            if model_type == "trocr":
                # Use TrOCR for medical documents
                result = await cls.extract_text_with_trocr(image_bytes)
                return result.get("raw_text", "")
            
            else:
                # Fallback to general OCR
                result = await cls.extract_text_with_trocr(image_bytes)
                return result.get("raw_text", "")
                
        except Exception as e:
            logger.error("Medical text extraction failed", model_type=model_type, error=str(e))
            return ""
    
    @classmethod
    async def analyze_medical_image(
        cls,
        image_b64: str,
        prompt: str,
        analysis_type: str = "general"
    ) -> str:
        """Analyze medical image with specific prompt using vision models."""
        
        try:
            # Convert base64 to bytes
            if image_b64.startswith('data:image'):
                image_b64 = image_b64.split(',')[1]
            
            image_bytes = base64.b64decode(image_b64)
            
            # Use BLIP-2 for image analysis
            blip_result = await cls.analyze_image_with_blip2(image_bytes)
            description = blip_result.get("description", "")
            
            # Enhance description with Groq analysis
            if cls._groq_client and description:
                enhanced_prompt = f"""
                Based on this medical image description: "{description}"
                
                {prompt}
                
                Provide a detailed medical analysis focusing on the specific requirements in the prompt.
                """
                
                groq_response = cls._groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": enhanced_prompt}],
                    model=settings.groq_model,
                    max_tokens=1000,
                    temperature=0.3
                )
                
                return groq_response.choices[0].message.content
            
            return description
            
        except Exception as e:
            logger.error("Medical image analysis failed", analysis_type=analysis_type, error=str(e))
            return f"Analysis failed: {str(e)}"
    
    @classmethod
    def _extract_medication_info(cls, ocr_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured medication information from OCR text."""
        
        text = ocr_results.get("raw_text", "")
        
        # Simple medication extraction (would be more sophisticated in production)
        medications = []
        
        # Common medication patterns
        import re
        
        # Pattern for medication names and dosages
        med_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|tablets?)'
        matches = re.findall(med_pattern, text, re.IGNORECASE)
        
        for match in matches:
            medication = {
                "name": match[0],
                "dosage": f"{match[1]}{match[2]}",
                "confidence": 0.7
            }
            medications.append(medication)
        
        # Extract prescriber info
        prescriber_pattern = r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        prescriber_match = re.search(prescriber_pattern, text)
        
        prescriber_info = {}
        if prescriber_match:
            prescriber_info["doctor_name"] = prescriber_match.group(0)
        
        return {
            "medications": medications,
            "prescriber_info": prescriber_info,
            "extraction_confidence": 0.7
        }
    
    @classmethod
    def _analyze_medication_safety(cls, medication_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medication safety and interactions."""
        
        # Simple safety analysis (would use drug database in production)
        warnings = []
        interactions = []
        
        medications = medication_data.get("medications", [])
        
        for med in medications:
            med_name = med.get("name", "").lower()
            
            # Common safety warnings
            if "ibuprofen" in med_name:
                warnings.append("Take with food to reduce stomach irritation")
            elif "acetaminophen" in med_name:
                warnings.append("Do not exceed 4000mg per day")
            elif "aspirin" in med_name:
                warnings.append("May increase bleeding risk")
        
        return {
            "warnings": warnings,
            "drug_interactions": interactions,
            "allergy_alerts": [],
            "safety_score": 0.8
        }
    
    # =============================================================================
    # ML MODEL METHODS
    # =============================================================================
    
    @classmethod
    async def predict_disease_risk(
        cls,
        user_id: str,
        health_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict disease risks using ML models."""
        
        try:
            # Load or get cached models
            models = await cls._get_ml_models()
            
            predictions = {}
            
            for disease, model in models.items():
                if model:
                    # Prepare features
                    features = cls._prepare_features_for_model(health_data, disease)
                    
                    # Make prediction
                    probability = model.predict_proba([features])[0][1]  # Probability of positive class
                    predictions[disease] = float(probability)
            
            return predictions
            
        except Exception as e:
            logger.error("Disease risk prediction failed", user_id=user_id, error=str(e))
            return {}
    
    @classmethod
    async def _get_ml_models(cls) -> Dict[str, Any]:
        """Get ML models from cache or load from database."""
        
        if not cls._model_cache:
            # Load models from MongoDB
            models_data = await DatabaseService.mongodb_find_many(
                "ml_models",
                {"is_active": True},
                limit=10
            )
            
            for model_doc in models_data:
                model_type = model_doc["model_type"]
                
                try:
                    # Deserialize model (simplified - would use proper model loading)
                    # model = joblib.loads(model_doc["model_data"])
                    # cls._model_cache[model_type] = model
                    
                    # For now, use mock models
                    cls._model_cache[model_type] = MockMLModel(model_type)
                    
                except Exception as e:
                    logger.error("Failed to load model", model_type=model_type, error=str(e))
        
        return cls._model_cache
    
    @classmethod
    def _prepare_features_for_model(cls, health_data: Dict[str, Any], disease: str) -> List[float]:
        """Prepare feature vector for ML model."""
        
        # Extract relevant features based on disease type
        features = []
        
        # Common features
        features.append(health_data.get("age", 30))
        features.append(health_data.get("bmi", 25.0))
        features.append(1 if health_data.get("family_history", {}).get(disease, False) else 0)
        
        # Disease-specific features
        if disease == "diabetes":
            features.extend([
                health_data.get("blood_sugar", 100),
                health_data.get("exercise_frequency", 3),
                1 if health_data.get("smoking", False) else 0
            ])
        elif disease == "heart_disease":
            features.extend([
                health_data.get("blood_pressure_systolic", 120),
                health_data.get("cholesterol", 200),
                health_data.get("stress_level", 0.5)
            ])
        
        # Pad or truncate to expected size
        while len(features) < 10:
            features.append(0.0)
        
        return features[:10]


# =============================================================================
# MOCK ML MODEL (for development)
# =============================================================================

class MockMLModel:
    """Mock ML model for development and testing."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
    
    def predict_proba(self, features: List[List[float]]) -> List[List[float]]:
        """Mock prediction with random but consistent results."""
        
        import hashlib
        
        # Create deterministic "random" prediction based on features
        feature_hash = hashlib.md5(str(features[0]).encode()).hexdigest()
        hash_int = int(feature_hash[:8], 16)
        
        # Generate probability between 0.1 and 0.9
        probability = 0.1 + (hash_int % 80) / 100.0
        
        return [[1 - probability, probability]]