# HealthSync AI - Advanced Voice Processing Service
import asyncio
import io
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import skew, kurtosis
import structlog

from config import settings
from services.ai_service import AIService
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)


class VoiceProcessingService:
    """Advanced voice processing for medical analysis."""
    
    @classmethod
    async def process_voice_stream(
        cls,
        audio_chunks: List[bytes],
        user_id: str,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process streaming voice data for real-time analysis."""
        
        try:
            # Combine audio chunks
            combined_audio = cls._combine_audio_chunks(audio_chunks)
            
            # Extract comprehensive features
            audio_features = await cls._extract_comprehensive_features(combined_audio)
            
            # Analyze speech patterns
            speech_analysis = await cls._analyze_speech_patterns(combined_audio, audio_features)
            
            # Detect emotional state
            emotion_analysis = await cls._analyze_emotional_state(audio_features, speech_analysis)
            
            # Medical symptom detection
            symptom_indicators = await cls._detect_symptom_indicators(
                audio_features, 
                speech_analysis, 
                emotion_analysis
            )
            
            # Generate medical assessment
            medical_assessment = await cls._generate_medical_assessment(
                speech_analysis["transcript"],
                emotion_analysis,
                symptom_indicators,
                session_metadata or {}
            )
            
            # Store comprehensive analysis
            session_id = await cls._store_voice_analysis(
                user_id,
                {
                    "audio_features": audio_features,
                    "speech_analysis": speech_analysis,
                    "emotion_analysis": emotion_analysis,
                    "symptom_indicators": symptom_indicators,
                    "medical_assessment": medical_assessment,
                    "session_metadata": session_metadata or {}
                }
            )
            
            return {
                "session_id": session_id,
                "transcript": speech_analysis["transcript"],
                "emotion_analysis": emotion_analysis,
                "symptom_indicators": symptom_indicators,
                "medical_assessment": medical_assessment,
                "confidence_score": medical_assessment.get("confidence_score", 0.5),
                "processing_metadata": {
                    "audio_duration": audio_features.get("duration", 0),
                    "chunk_count": len(audio_chunks),
                    "features_extracted": len(audio_features),
                    "processing_time_ms": int(time.time() * 1000)
                }
            }
            
        except Exception as e:
            logger.error("Voice stream processing failed", user_id=user_id, error=str(e))
            raise
    
    @classmethod
    def _combine_audio_chunks(cls, audio_chunks: List[bytes]) -> bytes:
        """Combine multiple audio chunks into single audio stream."""
        
        if not audio_chunks:
            return b""
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        # Combine all chunks
        combined = b"".join(audio_chunks)
        
        return combined
    
    @classmethod
    async def _extract_comprehensive_features(cls, audio_data: bytes) -> Dict[str, Any]:
        """Extract comprehensive audio features for medical analysis."""
        
        try:
            # Load audio
            audio_io = io.BytesIO(audio_data)
            y, sr = librosa.load(audio_io, sr=settings.voice_sample_rate)
            
            if len(y) == 0:
                return {"error": "Empty audio data"}
            
            features = {
                "duration": len(y) / sr,
                "sample_rate": sr,
                "audio_length": len(y)
            }
            
            # Basic audio properties
            features.update(cls._extract_basic_features(y, sr))
            
            # Spectral features
            features.update(cls._extract_spectral_features(y, sr))
            
            # Prosodic features
            features.update(cls._extract_prosodic_features(y, sr))
            
            # Voice quality features
            features.update(cls._extract_voice_quality_features(y, sr))
            
            # Temporal features
            features.update(cls._extract_temporal_features(y, sr))
            
            # Emotional indicators
            features.update(cls._extract_emotional_features(y, sr))
            
            return features
            
        except Exception as e:
            logger.error("Feature extraction failed", error=str(e))
            return {"error": str(e)}
    
    @classmethod
    def _extract_basic_features(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract basic audio features."""
        
        features = {}
        
        # Energy features
        rms = librosa.feature.rms(y=y)[0]
        features["energy"] = {
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "rms_max": float(np.max(rms)),
            "rms_min": float(np.min(rms))
        }
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features["zero_crossing_rate"] = {
            "mean": float(np.mean(zcr)),
            "std": float(np.std(zcr)),
            "max": float(np.max(zcr)),
            "min": float(np.min(zcr))
        }
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features["mfcc"] = {
            "mean": np.mean(mfcc, axis=1).tolist(),
            "std": np.std(mfcc, axis=1).tolist(),
            "delta": np.mean(librosa.feature.delta(mfcc), axis=1).tolist(),
            "delta2": np.mean(librosa.feature.delta(mfcc, order=2), axis=1).tolist()
        }
        
        return features
    
    @classmethod
    def _extract_spectral_features(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features for voice analysis."""
        
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features["spectral_centroid"] = {
            "mean": float(np.mean(spectral_centroids)),
            "std": float(np.std(spectral_centroids)),
            "max": float(np.max(spectral_centroids)),
            "min": float(np.min(spectral_centroids))
        }
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features["spectral_rolloff"] = {
            "mean": float(np.mean(spectral_rolloff)),
            "std": float(np.std(spectral_rolloff))
        }
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features["spectral_bandwidth"] = {
            "mean": float(np.mean(spectral_bandwidth)),
            "std": float(np.std(spectral_bandwidth))
        }
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features["spectral_contrast"] = {
            "mean": np.mean(spectral_contrast, axis=1).tolist(),
            "std": np.std(spectral_contrast, axis=1).tolist()
        }
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma"] = {
            "mean": np.mean(chroma, axis=1).tolist(),
            "std": np.std(chroma, axis=1).tolist()
        }
        
        return features
    
    @classmethod
    def _extract_prosodic_features(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract prosodic features (pitch, rhythm, stress)."""
        
        features = {}
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        
        # Extract pitch values
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            pitch_array = np.array(pitch_values)
            features["pitch"] = {
                "mean": float(np.mean(pitch_array)),
                "std": float(np.std(pitch_array)),
                "min": float(np.min(pitch_array)),
                "max": float(np.max(pitch_array)),
                "range": float(np.max(pitch_array) - np.min(pitch_array)),
                "median": float(np.median(pitch_array)),
                "skewness": float(skew(pitch_array)),
                "kurtosis": float(kurtosis(pitch_array))
            }
            
            # Pitch contour analysis
            features["pitch_contour"] = cls._analyze_pitch_contour(pitch_array)
        else:
            features["pitch"] = {
                "mean": 0, "std": 0, "min": 0, "max": 0,
                "range": 0, "median": 0, "skewness": 0, "kurtosis": 0
            }
            features["pitch_contour"] = {"stability": 0, "variability": 0}
        
        # Rhythm analysis
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        features["rhythm"] = {
            "tempo": float(tempo),
            "beat_count": len(beats),
            "rhythm_regularity": cls._calculate_rhythm_regularity(beats)
        }
        
        return features
    
    @classmethod
    def _extract_voice_quality_features(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract voice quality indicators."""
        
        features = {}
        
        # Jitter (pitch perturbation)
        features["jitter"] = cls._calculate_jitter(y, sr)
        
        # Shimmer (amplitude perturbation)
        features["shimmer"] = cls._calculate_shimmer(y, sr)
        
        # Harmonics-to-noise ratio
        features["hnr"] = cls._calculate_hnr(y, sr)
        
        # Voice breaks and irregularities
        features["voice_breaks"] = cls._detect_voice_breaks(y, sr)
        
        return features
    
    @classmethod
    def _extract_temporal_features(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract temporal speech features."""
        
        features = {}
        
        # Speech/silence segmentation
        intervals = librosa.effects.split(y, top_db=20)
        
        if len(intervals) > 0:
            speech_durations = [(end - start) / sr for start, end in intervals]
            silence_durations = []
            
            for i in range(len(intervals) - 1):
                silence_start = intervals[i][1]
                silence_end = intervals[i + 1][0]
                silence_durations.append((silence_end - silence_start) / sr)
            
            features["speech_timing"] = {
                "speech_segments": len(intervals),
                "total_speech_time": sum(speech_durations),
                "total_silence_time": sum(silence_durations) if silence_durations else 0,
                "speech_rate": len(intervals) / (len(y) / sr),
                "avg_speech_duration": np.mean(speech_durations) if speech_durations else 0,
                "avg_silence_duration": np.mean(silence_durations) if silence_durations else 0
            }
        else:
            features["speech_timing"] = {
                "speech_segments": 0,
                "total_speech_time": 0,
                "total_silence_time": len(y) / sr,
                "speech_rate": 0,
                "avg_speech_duration": 0,
                "avg_silence_duration": 0
            }
        
        return features
    
    @classmethod
    def _extract_emotional_features(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract features indicative of emotional state."""
        
        features = {}
        
        # Formant analysis (simplified)
        features["formants"] = cls._estimate_formants(y, sr)
        
        # Intensity variations
        rms = librosa.feature.rms(y=y)[0]
        features["intensity_variation"] = {
            "coefficient_of_variation": float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0,
            "dynamic_range": float(np.max(rms) - np.min(rms))
        }
        
        # Spectral flux (measure of spectral change)
        stft = librosa.stft(y)
        spectral_flux = np.sum(np.diff(np.abs(stft), axis=1) ** 2, axis=0)
        features["spectral_flux"] = {
            "mean": float(np.mean(spectral_flux)),
            "std": float(np.std(spectral_flux))
        }
        
        return features
    
    @classmethod
    async def _analyze_speech_patterns(
        cls, 
        audio_data: bytes, 
        audio_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze speech patterns and extract transcript."""
        
        try:
            # Get transcript using AI service
            transcript = await AIService.speech_to_text(audio_data)
            
            # Analyze transcript
            words = transcript.split()
            
            # Calculate speech rate
            duration = audio_features.get("duration", 1)
            speech_rate = len(words) / duration * 60  # words per minute
            
            # Analyze linguistic patterns
            linguistic_analysis = cls._analyze_linguistic_patterns(transcript)
            
            # Detect disfluencies
            disfluencies = cls._detect_disfluencies(transcript)
            
            return {
                "transcript": transcript,
                "word_count": len(words),
                "speech_rate_wpm": speech_rate,
                "linguistic_analysis": linguistic_analysis,
                "disfluencies": disfluencies,
                "speech_clarity": cls._assess_speech_clarity(audio_features)
            }
            
        except Exception as e:
            logger.error("Speech pattern analysis failed", error=str(e))
            return {
                "transcript": "",
                "word_count": 0,
                "speech_rate_wpm": 0,
                "error": str(e)
            }
    
    @classmethod
    async def _analyze_emotional_state(
        cls,
        audio_features: Dict[str, Any],
        speech_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze emotional state from voice characteristics."""
        
        try:
            # Extract emotion indicators
            emotion_indicators = {}
            
            # Stress indicators
            pitch_features = audio_features.get("pitch", {})
            energy_features = audio_features.get("energy", {})
            
            # High pitch variation + high energy = stress/anxiety
            stress_score = cls._calculate_stress_score(pitch_features, energy_features)
            emotion_indicators["stress"] = stress_score
            
            # Fatigue indicators (low energy, slow speech)
            fatigue_score = cls._calculate_fatigue_score(
                energy_features, 
                speech_analysis.get("speech_rate_wpm", 0)
            )
            emotion_indicators["fatigue"] = fatigue_score
            
            # Anxiety indicators (high pitch, fast speech, disfluencies)
            anxiety_score = cls._calculate_anxiety_score(
                pitch_features,
                speech_analysis.get("speech_rate_wpm", 0),
                speech_analysis.get("disfluencies", {})
            )
            emotion_indicators["anxiety"] = anxiety_score
            
            # Depression indicators (low pitch, slow speech, monotone)
            depression_score = cls._calculate_depression_score(
                pitch_features,
                speech_analysis.get("speech_rate_wpm", 0),
                audio_features.get("pitch_contour", {})
            )
            emotion_indicators["depression"] = depression_score
            
            # Pain indicators (voice strain, irregular patterns)
            pain_score = cls._calculate_pain_score(
                audio_features.get("jitter", 0),
                audio_features.get("shimmer", 0),
                audio_features.get("voice_breaks", {})
            )
            emotion_indicators["pain"] = pain_score
            
            # Overall emotional state
            dominant_emotion = max(emotion_indicators.items(), key=lambda x: x[1])
            
            return {
                "emotion_indicators": emotion_indicators,
                "dominant_emotion": dominant_emotion[0],
                "dominant_emotion_confidence": dominant_emotion[1],
                "emotional_stability": cls._assess_emotional_stability(emotion_indicators),
                "arousal_level": cls._calculate_arousal_level(audio_features),
                "valence": cls._calculate_valence(speech_analysis, emotion_indicators)
            }
            
        except Exception as e:
            logger.error("Emotional analysis failed", error=str(e))
            return {"error": str(e)}
    
    @classmethod
    async def _detect_symptom_indicators(
        cls,
        audio_features: Dict[str, Any],
        speech_analysis: Dict[str, Any],
        emotion_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Detect potential medical symptoms from voice analysis."""
        
        symptom_indicators = {}
        
        # Respiratory symptoms
        symptom_indicators["respiratory"] = cls._detect_respiratory_symptoms(
            audio_features, speech_analysis
        )
        
        # Neurological symptoms
        symptom_indicators["neurological"] = cls._detect_neurological_symptoms(
            audio_features, speech_analysis
        )
        
        # Cardiovascular symptoms
        symptom_indicators["cardiovascular"] = cls._detect_cardiovascular_symptoms(
            audio_features, emotion_analysis
        )
        
        # Mental health symptoms
        symptom_indicators["mental_health"] = cls._detect_mental_health_symptoms(
            emotion_analysis, speech_analysis
        )
        
        # Pain symptoms
        symptom_indicators["pain"] = cls._detect_pain_symptoms(
            audio_features, emotion_analysis
        )
        
        # Overall symptom severity
        symptom_indicators["overall_severity"] = cls._calculate_overall_severity(
            symptom_indicators
        )
        
        return symptom_indicators
    
    @classmethod
    async def _generate_medical_assessment(
        cls,
        transcript: str,
        emotion_analysis: Dict[str, Any],
        symptom_indicators: Dict[str, Any],
        session_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive medical assessment using AI."""
        
        try:
            # Prepare comprehensive context for AI analysis
            analysis_context = {
                "transcript": transcript,
                "emotion_analysis": emotion_analysis,
                "symptom_indicators": symptom_indicators,
                "session_metadata": session_metadata
            }
            
            # Use Groq for medical reasoning
            assessment = await AIService.analyze_symptoms_with_groq(
                transcript,
                {
                    "emotion_analysis": emotion_analysis,
                    "symptom_indicators": symptom_indicators
                }
            )
            
            # Enhance assessment with voice-specific insights
            assessment["voice_insights"] = cls._generate_voice_insights(
                emotion_analysis, symptom_indicators
            )
            
            # Add follow-up recommendations
            assessment["follow_up_recommendations"] = cls._generate_follow_up_recommendations(
                assessment, symptom_indicators
            )
            
            return assessment
            
        except Exception as e:
            logger.error("Medical assessment generation failed", error=str(e))
            return {
                "risk_level": "medium",
                "urgency_flag": False,
                "recommended_actions": ["Consult healthcare provider"],
                "confidence_score": 0.3,
                "error": str(e)
            }
    
    @classmethod
    async def _store_voice_analysis(
        cls,
        user_id: str,
        analysis_data: Dict[str, Any]
    ) -> str:
        """Store comprehensive voice analysis in MongoDB."""
        
        try:
            session_document = {
                "user_id": user_id,
                "session_id": f"voice_{int(time.time())}_{user_id[:8]}",
                "timestamp": time.time(),
                **analysis_data
            }
            
            session_id = await DatabaseService.mongodb_insert_one(
                "voice_sessions",
                session_document
            )
            
            logger.info(
                "Voice analysis stored",
                user_id=user_id,
                session_id=session_id
            )
            
            return session_document["session_id"]
            
        except Exception as e:
            logger.error("Failed to store voice analysis", user_id=user_id, error=str(e))
            raise
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    @classmethod
    def _analyze_pitch_contour(cls, pitch_values: np.ndarray) -> Dict[str, float]:
        """Analyze pitch contour for stability and patterns."""
        
        if len(pitch_values) < 2:
            return {"stability": 0, "variability": 0}
        
        # Calculate pitch stability
        pitch_diff = np.diff(pitch_values)
        stability = 1.0 / (1.0 + np.std(pitch_diff))
        
        # Calculate pitch variability
        variability = np.std(pitch_values) / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0
        
        return {
            "stability": float(stability),
            "variability": float(variability)
        }
    
    @classmethod
    def _calculate_rhythm_regularity(cls, beats: np.ndarray) -> float:
        """Calculate rhythm regularity from beat tracking."""
        
        if len(beats) < 3:
            return 0.0
        
        # Calculate inter-beat intervals
        intervals = np.diff(beats)
        
        # Regularity is inverse of coefficient of variation
        if np.mean(intervals) > 0:
            cv = np.std(intervals) / np.mean(intervals)
            regularity = 1.0 / (1.0 + cv)
        else:
            regularity = 0.0
        
        return float(regularity)
    
    @classmethod
    def _calculate_jitter(cls, y: np.ndarray, sr: int) -> float:
        """Calculate jitter (pitch perturbation)."""
        
        # Simplified jitter calculation
        # In production, would use more sophisticated pitch tracking
        try:
            pitches, _ = librosa.piptrack(y=y, sr=sr)
            pitch_values = pitches[pitches > 0]
            
            if len(pitch_values) < 2:
                return 0.0
            
            # Calculate period-to-period variation
            periods = 1.0 / pitch_values
            period_diff = np.abs(np.diff(periods))
            jitter = np.mean(period_diff) / np.mean(periods) if np.mean(periods) > 0 else 0
            
            return float(jitter)
            
        except Exception:
            return 0.0
    
    @classmethod
    def _calculate_shimmer(cls, y: np.ndarray, sr: int) -> float:
        """Calculate shimmer (amplitude perturbation)."""
        
        try:
            # Calculate RMS energy in overlapping windows
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            rms = librosa.feature.rms(
                y=y, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            if len(rms) < 2:
                return 0.0
            
            # Calculate amplitude variation
            amp_diff = np.abs(np.diff(rms))
            shimmer = np.mean(amp_diff) / np.mean(rms) if np.mean(rms) > 0 else 0
            
            return float(shimmer)
            
        except Exception:
            return 0.0
    
    @classmethod
    def _calculate_hnr(cls, y: np.ndarray, sr: int) -> float:
        """Calculate harmonics-to-noise ratio."""
        
        try:
            # Simplified HNR calculation
            # Autocorrelation-based approach
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find fundamental period
            if len(autocorr) > 1:
                peak_idx = np.argmax(autocorr[1:]) + 1
                hnr = 10 * np.log10(autocorr[peak_idx] / (np.sum(autocorr) - autocorr[peak_idx]))
                return float(max(0, min(hnr, 30)))  # Clamp between 0-30 dB
            
            return 0.0
            
        except Exception:
            return 0.0
    
    @classmethod
    def _detect_voice_breaks(cls, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Detect voice breaks and irregularities."""
        
        try:
            # Detect silent segments
            intervals = librosa.effects.split(y, top_db=20)
            
            # Calculate voice break statistics
            total_duration = len(y) / sr
            speech_duration = sum((end - start) / sr for start, end in intervals)
            silence_duration = total_duration - speech_duration
            
            voice_breaks = {
                "break_count": len(intervals) - 1 if len(intervals) > 1 else 0,
                "total_break_duration": silence_duration,
                "break_percentage": (silence_duration / total_duration) * 100 if total_duration > 0 else 0,
                "average_break_duration": silence_duration / max(1, len(intervals) - 1) if len(intervals) > 1 else 0
            }
            
            return voice_breaks
            
        except Exception:
            return {"break_count": 0, "total_break_duration": 0, "break_percentage": 0}
    
    @classmethod
    def _estimate_formants(cls, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Estimate formant frequencies (simplified)."""
        
        try:
            # Use LPC to estimate formants
            from scipy.signal import lfilter
            
            # Pre-emphasis
            pre_emphasis = 0.97
            y_preemphasized = lfilter([1, -pre_emphasis], [1], y)
            
            # Window the signal
            windowed = y_preemphasized * np.hanning(len(y_preemphasized))
            
            # Simple formant estimation (would be more sophisticated in production)
            fft = np.fft.fft(windowed)
            magnitude = np.abs(fft[:len(fft)//2])
            
            # Find peaks (simplified formant detection)
            peaks = []
            for i in range(1, len(magnitude)-1):
                if magnitude[i] > magnitude[i-1] and magnitude[i] > magnitude[i+1]:
                    peaks.append((i * sr / len(fft), magnitude[i]))
            
            # Sort by magnitude and take top formants
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            formants = {"f1": 0, "f2": 0, "f3": 0}
            for i, (freq, _) in enumerate(peaks[:3]):
                formants[f"f{i+1}"] = freq
            
            return formants
            
        except Exception:
            return {"f1": 0, "f2": 0, "f3": 0}
    
    # Emotion calculation methods
    @classmethod
    def _calculate_stress_score(cls, pitch_features: Dict, energy_features: Dict) -> float:
        """Calculate stress score from voice features."""
        
        pitch_std = pitch_features.get("std", 0)
        energy_std = energy_features.get("rms_std", 0)
        
        # High pitch variation + high energy variation = stress
        stress_score = min(1.0, (pitch_std / 50.0 + energy_std * 10) / 2)
        return float(stress_score)
    
    @classmethod
    def _calculate_fatigue_score(cls, energy_features: Dict, speech_rate: float) -> float:
        """Calculate fatigue score from voice features."""
        
        energy_mean = energy_features.get("rms_mean", 0)
        
        # Low energy + slow speech = fatigue
        energy_factor = max(0, 1.0 - energy_mean * 20)
        speech_factor = max(0, 1.0 - speech_rate / 150.0)  # Normal speech ~150 WPM
        
        fatigue_score = (energy_factor + speech_factor) / 2
        return float(min(1.0, fatigue_score))
    
    @classmethod
    def _calculate_anxiety_score(cls, pitch_features: Dict, speech_rate: float, disfluencies: Dict) -> float:
        """Calculate anxiety score from voice features."""
        
        pitch_mean = pitch_features.get("mean", 0)
        pitch_std = pitch_features.get("std", 0)
        
        # High pitch + fast speech + disfluencies = anxiety
        pitch_factor = min(1.0, pitch_mean / 200.0)  # Normalize to typical range
        rate_factor = min(1.0, max(0, speech_rate - 150) / 100.0)  # Above normal rate
        disfluency_factor = min(1.0, disfluencies.get("total_count", 0) / 10.0)
        
        anxiety_score = (pitch_factor + rate_factor + disfluency_factor) / 3
        return float(anxiety_score)
    
    @classmethod
    def _calculate_depression_score(cls, pitch_features: Dict, speech_rate: float, pitch_contour: Dict) -> float:
        """Calculate depression score from voice features."""
        
        pitch_mean = pitch_features.get("mean", 0)
        pitch_variability = pitch_contour.get("variability", 0)
        
        # Low pitch + slow speech + monotone = depression
        pitch_factor = max(0, 1.0 - pitch_mean / 150.0)  # Lower pitch
        rate_factor = max(0, 1.0 - speech_rate / 120.0)   # Slower speech
        monotone_factor = max(0, 1.0 - pitch_variability * 5)  # Less variation
        
        depression_score = (pitch_factor + rate_factor + monotone_factor) / 3
        return float(depression_score)
    
    @classmethod
    def _calculate_pain_score(cls, jitter: float, shimmer: float, voice_breaks: Dict) -> float:
        """Calculate pain score from voice quality features."""
        
        # Voice strain indicators
        jitter_factor = min(1.0, jitter * 100)  # Normalize jitter
        shimmer_factor = min(1.0, shimmer * 50)  # Normalize shimmer
        breaks_factor = min(1.0, voice_breaks.get("break_percentage", 0) / 20.0)
        
        pain_score = (jitter_factor + shimmer_factor + breaks_factor) / 3
        return float(pain_score)
    
    # Additional utility methods for linguistic and symptom analysis would go here...
    
    @classmethod
    def _analyze_linguistic_patterns(cls, transcript: str) -> Dict[str, Any]:
        """Analyze linguistic patterns in transcript."""
        
        # Simplified linguistic analysis
        words = transcript.lower().split()
        
        # Medical keywords
        medical_keywords = [
            'pain', 'hurt', 'ache', 'sick', 'nausea', 'dizzy', 'tired', 'fatigue',
            'chest', 'head', 'stomach', 'back', 'breathe', 'breathing', 'cough'
        ]
        
        medical_word_count = sum(1 for word in words if any(keyword in word for keyword in medical_keywords))
        
        return {
            "medical_keyword_density": medical_word_count / len(words) if words else 0,
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "sentence_complexity": len(words) / max(1, transcript.count('.') + transcript.count('!') + transcript.count('?'))
        }
    
    @classmethod
    def _detect_disfluencies(cls, transcript: str) -> Dict[str, Any]:
        """Detect speech disfluencies in transcript."""
        
        # Common disfluency patterns
        filler_words = ['um', 'uh', 'er', 'ah', 'like', 'you know']
        repetitions = 0
        false_starts = 0
        
        words = transcript.lower().split()
        filler_count = sum(1 for word in words if word in filler_words)
        
        # Simple repetition detection
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repetitions += 1
        
        return {
            "filler_words": filler_count,
            "repetitions": repetitions,
            "false_starts": false_starts,
            "total_count": filler_count + repetitions + false_starts
        }
    
    @classmethod
    def _assess_speech_clarity(cls, audio_features: Dict[str, Any]) -> float:
        """Assess overall speech clarity."""
        
        # Combine multiple clarity indicators
        zcr_mean = audio_features.get("zero_crossing_rate", {}).get("mean", 0)
        spectral_centroid = audio_features.get("spectral_centroid", {}).get("mean", 0)
        
        # Higher ZCR and spectral centroid generally indicate clearer speech
        clarity_score = min(1.0, (zcr_mean * 10 + spectral_centroid / 2000) / 2)
        
        return float(clarity_score)
    
    @classmethod
    def _assess_emotional_stability(cls, emotion_indicators: Dict[str, float]) -> float:
        """Assess emotional stability from emotion indicators."""
        
        # Lower variation in emotion scores indicates more stability
        emotion_values = list(emotion_indicators.values())
        if not emotion_values:
            return 0.5
        
        stability = 1.0 - (max(emotion_values) - min(emotion_values))
        return float(max(0.0, min(1.0, stability)))
    
    @classmethod
    def _calculate_arousal_level(cls, audio_features: Dict[str, Any]) -> float:
        """Calculate arousal level from audio features."""
        
        energy_mean = audio_features.get("energy", {}).get("rms_mean", 0)
        pitch_mean = audio_features.get("pitch", {}).get("mean", 0)
        
        # High energy and pitch indicate high arousal
        arousal = min(1.0, (energy_mean * 20 + pitch_mean / 200) / 2)
        return float(arousal)
    
    @classmethod
    def _calculate_valence(cls, speech_analysis: Dict, emotion_indicators: Dict) -> float:
        """Calculate emotional valence (positive/negative)."""
        
        # Simplified valence calculation
        stress = emotion_indicators.get("stress", 0)
        anxiety = emotion_indicators.get("anxiety", 0)
        depression = emotion_indicators.get("depression", 0)
        
        # Lower stress/anxiety/depression = higher valence
        negative_emotions = (stress + anxiety + depression) / 3
        valence = 1.0 - negative_emotions
        
        return float(max(0.0, min(1.0, valence)))
    
    # Symptom detection methods
    @classmethod
    def _detect_respiratory_symptoms(cls, audio_features: Dict, speech_analysis: Dict) -> Dict[str, float]:
        """Detect respiratory symptoms from voice."""
        
        voice_breaks = audio_features.get("voice_breaks", {})
        speech_rate = speech_analysis.get("speech_rate_wpm", 0)
        
        # Frequent breaks + slow speech may indicate breathing difficulties
        breathing_difficulty = min(1.0, voice_breaks.get("break_percentage", 0) / 15.0)
        
        return {
            "breathing_difficulty": breathing_difficulty,
            "speech_interruption": min(1.0, voice_breaks.get("break_count", 0) / 10.0)
        }
    
    @classmethod
    def _detect_neurological_symptoms(cls, audio_features: Dict, speech_analysis: Dict) -> Dict[str, float]:
        """Detect neurological symptoms from voice."""
        
        jitter = audio_features.get("jitter", 0)
        shimmer = audio_features.get("shimmer", 0)
        disfluencies = speech_analysis.get("disfluencies", {})
        
        # Voice tremor and speech difficulties may indicate neurological issues
        voice_tremor = min(1.0, (jitter + shimmer) * 50)
        speech_difficulty = min(1.0, disfluencies.get("total_count", 0) / 15.0)
        
        return {
            "voice_tremor": voice_tremor,
            "speech_difficulty": speech_difficulty
        }
    
    @classmethod
    def _detect_cardiovascular_symptoms(cls, audio_features: Dict, emotion_analysis: Dict) -> Dict[str, float]:
        """Detect cardiovascular symptoms from voice."""
        
        stress = emotion_analysis.get("emotion_indicators", {}).get("stress", 0)
        arousal = emotion_analysis.get("arousal_level", 0)
        
        # High stress and arousal may indicate cardiovascular stress
        cardiovascular_stress = (stress + arousal) / 2
        
        return {
            "cardiovascular_stress": cardiovascular_stress
        }
    
    @classmethod
    def _detect_mental_health_symptoms(cls, emotion_analysis: Dict, speech_analysis: Dict) -> Dict[str, float]:
        """Detect mental health symptoms from voice."""
        
        emotion_indicators = emotion_analysis.get("emotion_indicators", {})
        
        return {
            "anxiety_level": emotion_indicators.get("anxiety", 0),
            "depression_level": emotion_indicators.get("depression", 0),
            "stress_level": emotion_indicators.get("stress", 0),
            "emotional_instability": 1.0 - emotion_analysis.get("emotional_stability", 0.5)
        }
    
    @classmethod
    def _detect_pain_symptoms(cls, audio_features: Dict, emotion_analysis: Dict) -> Dict[str, float]:
        """Detect pain symptoms from voice."""
        
        pain_score = emotion_analysis.get("emotion_indicators", {}).get("pain", 0)
        voice_strain = min(1.0, (audio_features.get("jitter", 0) + audio_features.get("shimmer", 0)) * 25)
        
        return {
            "pain_level": pain_score,
            "voice_strain": voice_strain
        }
    
    @classmethod
    def _calculate_overall_severity(cls, symptom_indicators: Dict[str, Any]) -> str:
        """Calculate overall symptom severity."""
        
        all_scores = []
        for category in symptom_indicators.values():
            if isinstance(category, dict):
                all_scores.extend(category.values())
        
        if not all_scores:
            return "low"
        
        max_score = max(all_scores)
        avg_score = sum(all_scores) / len(all_scores)
        
        if max_score > 0.8 or avg_score > 0.6:
            return "high"
        elif max_score > 0.5 or avg_score > 0.3:
            return "medium"
        else:
            return "low"
    
    @classmethod
    def _generate_voice_insights(cls, emotion_analysis: Dict, symptom_indicators: Dict) -> List[str]:
        """Generate voice-specific insights."""
        
        insights = []
        
        # Emotion insights
        dominant_emotion = emotion_analysis.get("dominant_emotion", "")
        if dominant_emotion and emotion_analysis.get("dominant_emotion_confidence", 0) > 0.6:
            insights.append(f"Voice analysis indicates elevated {dominant_emotion} levels")
        
        # Symptom insights
        overall_severity = symptom_indicators.get("overall_severity", "low")
        if overall_severity in ["medium", "high"]:
            insights.append(f"Voice patterns suggest {overall_severity} severity symptoms")
        
        return insights
    
    @classmethod
    def _generate_follow_up_recommendations(cls, assessment: Dict, symptom_indicators: Dict) -> List[str]:
        """Generate follow-up recommendations based on analysis."""
        
        recommendations = []
        
        risk_level = assessment.get("risk_level", "low")
        overall_severity = symptom_indicators.get("overall_severity", "low")
        
        if risk_level == "high" or overall_severity == "high":
            recommendations.extend([
                "Schedule immediate medical consultation",
                "Monitor symptoms closely",
                "Consider emergency care if symptoms worsen"
            ])
        elif risk_level == "medium" or overall_severity == "medium":
            recommendations.extend([
                "Schedule medical appointment within 24-48 hours",
                "Keep a symptom diary",
                "Follow up with voice analysis in 24 hours"
            ])
        else:
            recommendations.extend([
                "Continue monitoring symptoms",
                "Schedule routine check-up if symptoms persist",
                "Use voice analysis for ongoing health tracking"
            ])
        
        return recommendations