# HealthSync AI - Future-You Health Simulator Service
import asyncio
import json
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import structlog
import httpx
import replicate
from PIL import Image, ImageEnhance, ImageFilter
import io
import numpy as np

from config import settings
from services.db_service import DatabaseService
from services.ai_service import AIService
from services.ml_service import MLModelService
import os
from pathlib import Path


logger = structlog.get_logger(__name__)

# Create uploads directory for local file storage
UPLOADS_DIR = Path(__file__).parent.parent / "uploads" / "future_sim"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class FutureSimulatorService:
    """Advanced Future-You Health Simulator with AI-powered age progression and health projections."""
    
    # Age progression models and configurations
    AGE_PROGRESSION_MODELS = {
        "stable_diffusion": "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
        "face_aging": "cjwbw/roop:b9d68b78538b9e92c3c6e8b5b8b8b8b8b8b8b8b8"  # Alternative aging model
    }
    
    # Health condition visual effects mapping
    HEALTH_EFFECTS = {
        "diabetes": {
            "vision_effects": ["blurred_vision", "eye_damage"],
            "skin_effects": ["slow_healing", "infections"],
            "weight_effects": ["weight_gain", "fatigue_appearance"]
        },
        "heart_disease": {
            "facial_effects": ["pale_complexion", "fatigue_lines"],
            "physical_effects": ["reduced_vitality", "shortness_of_breath_appearance"],
            "posture_effects": ["slouched_posture", "reduced_energy"]
        },
        "smoking": {
            "skin_effects": ["wrinkles", "yellowed_teeth", "aged_skin"],
            "hair_effects": ["thinning_hair", "graying"],
            "facial_effects": ["smoker_lines", "dull_complexion"]
        },
        "obesity": {
            "facial_effects": ["fuller_face", "double_chin"],
            "skin_effects": ["stretch_marks", "skin_discoloration"],
            "posture_effects": ["altered_posture", "reduced_mobility_appearance"]
        },
        "hypertension": {
            "facial_effects": ["flushed_appearance", "stress_lines"],
            "eye_effects": ["bloodshot_eyes", "under_eye_bags"],
            "overall_effects": ["tension_appearance", "fatigue"]
        }
    }
    
    @classmethod
    async def initialize(cls):
        """Initialize the Future Simulator service."""
        
        logger.info("Initializing Future Simulator Service")
        
        try:
            # Initialize Replicate client
            if settings.replicate_api_token:
                replicate.Client(api_token=settings.replicate_api_token)
                logger.info("Replicate client initialized for age progression")
            
            # Test image processing capabilities
            await cls._test_image_processing()
            
            logger.info("Future Simulator Service initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize Future Simulator Service", error=str(e))
            raise
    
    @classmethod
    async def _test_image_processing(cls):
        """Test basic image processing capabilities."""
        
        try:
            # Create a test image
            test_image = Image.new('RGB', (256, 256), color='white')
            
            # Test basic operations
            enhanced = ImageEnhance.Contrast(test_image).enhance(1.2)
            blurred = test_image.filter(ImageFilter.GaussianBlur(radius=1))
            
            logger.info("Image processing capabilities verified")
            
        except Exception as e:
            logger.warning("Image processing test failed", error=str(e))
    
    @classmethod
    async def upload_and_validate_image(
        cls,
        image_data: bytes,
        user_id: str,
        content_type: str = "image/jpeg"
    ) -> Dict[str, Any]:
        """Upload and validate user image for age progression."""
        
        try:
            # Validate image size (max 10MB)
            if len(image_data) > settings.max_file_size_bytes:
                return {
                    "error": "Image too large",
                    "max_size_mb": settings.max_file_size_mb
                }
            
            # Validate content type
            if content_type not in settings.allowed_image_types:
                return {
                    "error": "Invalid image type",
                    "allowed_types": settings.allowed_image_types
                }
            
            # Load and validate image
            try:
                image = Image.open(io.BytesIO(image_data))
                
                # Check image dimensions
                width, height = image.size
                if width < 256 or height < 256:
                    return {"error": "Image too small (minimum 256x256 pixels)"}
                
                if width > 2048 or height > 2048:
                    # Resize if too large
                    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    
                    # Convert back to bytes
                    output = io.BytesIO()
                    image.save(output, format='JPEG', quality=settings.image_quality)
                    image_data = output.getvalue()
                
            except Exception as img_error:
                return {"error": f"Invalid image format: {str(img_error)}"}
            
            # Detect face in image (basic validation)
            face_detected = await cls._detect_face_in_image(image)
            if not face_detected:
                return {"error": "No face detected in image. Please upload a clear photo of your face."}
            
            # Generate unique filename
            image_hash = hashlib.md5(image_data).hexdigest()
            timestamp = int(datetime.now().timestamp())
            relative_path = f"{user_id}/{image_hash}_{timestamp}.jpg"
            
            # Save to local storage
            user_dir = UPLOADS_DIR / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = user_dir / f"{image_hash}_{timestamp}.jpg"
            with open(file_path, "wb") as f:
                f.write(image_data)
            
            # Generate local URL
            local_url = f"/uploads/future_sim/{relative_path}"
            
            # Store image metadata in MongoDB
            image_record = await DatabaseService.mongodb_insert_one("user_images", {
                "user_id": user_id,
                "file_path": relative_path,
                "original_filename": f"future_sim_{timestamp}.jpg",
                "file_size": len(image_data),
                "content_type": "image/jpeg",
                "purpose": "future_simulation",
                "created_at": datetime.now()
            })
            
            logger.info(
                "Image uploaded and validated successfully",
                user_id=user_id,
                image_id=str(image_record.inserted_id),
                file_size=len(image_data)
            )
            
            return {
                "success": True,
                "image_id": str(image_record.inserted_id),
                "file_path": relative_path,
                "signed_url": local_url,
                "file_size": len(image_data),
                "dimensions": f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error("Image upload and validation failed", user_id=user_id, error=str(e))
            return {"error": "Image processing failed"}
    
    @classmethod
    async def _detect_face_in_image(cls, image: Image.Image) -> bool:
        """Basic face detection using image analysis."""
        
        try:
            # Convert to numpy array for basic analysis
            img_array = np.array(image)
            
            # Simple heuristics for face detection
            # Check if image has reasonable aspect ratio for portrait
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            # Face photos typically have aspect ratio between 0.7 and 1.5
            if 0.7 <= aspect_ratio <= 1.5:
                # Check for skin-tone colors (basic heuristic)
                # Convert to HSV for better skin detection
                if len(img_array.shape) == 3:
                    # Simple skin tone detection
                    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
                    skin_mask = (r > 95) & (g > 40) & (b > 20) & \
                               (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b) > 15) & \
                               (np.abs(r - g) > 15) & (r > g) & (r > b)
                    
                    skin_percentage = np.sum(skin_mask) / (height * width)
                    
                    # If more than 10% of image contains skin tones, likely a face
                    return skin_percentage > 0.1
            
            # Fallback: assume face is present if basic checks pass
            return True
            
        except Exception as e:
            logger.warning("Face detection failed, assuming face present", error=str(e))
            return True
    
    @classmethod
    async def generate_age_progression(
        cls,
        image_path: str,
        target_age_years: int,
        user_id: str,
        current_age: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate age progression using AI models."""
        
        try:
            # Get image from local storage
            full_path = UPLOADS_DIR / image_path
            if not full_path.exists():
                return {"error": "Original image not found"}
            
            image_url = f"/uploads/future_sim/{image_path}"
            
            # Determine age progression prompt
            if current_age:
                age_difference = target_age_years
                if age_difference <= 0:
                    return {"error": "Target age must be greater than current age"}
            else:
                age_difference = target_age_years
            
            # Create age progression prompt
            age_prompt = cls._create_age_progression_prompt(age_difference)
            
            # Generate aged image using Replicate
            aged_image_result = await cls._generate_aged_image_replicate(
                image_url,
                age_prompt,
                user_id
            )
            
            if aged_image_result.get("error"):
                return aged_image_result
            
            # Store age progression record
            progression_record = await DatabaseService.execute_query(
                """
                INSERT INTO age_progressions (user_id, original_image_path, aged_image_path,
                                            target_age_years, generation_prompt, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING id
                """,
                user_id,
                image_path,
                aged_image_result["aged_image_path"],
                target_age_years,
                age_prompt,
                fetch_one=True
            )
            
            logger.info(
                "Age progression generated successfully",
                user_id=user_id,
                progression_id=progression_record["id"],
                target_age=target_age_years
            )
            
            return {
                "success": True,
                "progression_id": progression_record["id"],
                "original_image_url": image_url,
                "aged_image_url": aged_image_result["aged_image_url"],
                "target_age_years": target_age_years,
                "generation_prompt": age_prompt
            }
            
        except Exception as e:
            logger.error("Age progression generation failed", user_id=user_id, error=str(e))
            return {"error": "Age progression generation failed"}
    
    @classmethod
    def _create_age_progression_prompt(cls, age_years: int) -> str:
        """Create detailed prompt for age progression."""
        
        if age_years <= 5:
            return f"Age this person by {age_years} years, subtle aging, maintain facial features, realistic, high quality, professional photography"
        elif age_years <= 15:
            return f"Age this person by {age_years} years, show natural aging progression, some wrinkles, mature appearance, realistic, high quality"
        elif age_years <= 25:
            return f"Age this person by {age_years} years, significant aging, wrinkles, gray hair, mature features, realistic aging, high quality"
        else:
            return f"Age this person by {age_years} years, elderly appearance, deep wrinkles, gray/white hair, aged skin, realistic elderly person, high quality"
    
    @classmethod
    async def _generate_aged_image_replicate(
        cls,
        image_url: str,
        prompt: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Generate aged image using Replicate API."""
        
        try:
            # Use Stable Diffusion for age progression
            output = replicate.run(
                cls.AGE_PROGRESSION_MODELS["stable_diffusion"],
                input={
                    "image": image_url,
                    "prompt": prompt,
                    "negative_prompt": "blurry, low quality, distorted, cartoon, anime, painting",
                    "num_inference_steps": 20,
                    "guidance_scale": 7.5,
                    "strength": 0.7,  # How much to change the original image
                    "seed": None  # Random seed for variation
                }
            )
            
            if not output:
                return {"error": "No output from age progression model"}
            
            # Download the generated image
            aged_image_url = output[0] if isinstance(output, list) else output
            
            async with httpx.AsyncClient() as client:
                response = await client.get(aged_image_url)
                response.raise_for_status()
                aged_image_data = response.content
            
            # Save aged image to local storage
            timestamp = int(datetime.now().timestamp())
            relative_path = f"{user_id}/aged_{timestamp}.jpg"
            
            user_dir = UPLOADS_DIR / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            aged_file_path = user_dir / f"aged_{timestamp}.jpg"
            with open(aged_file_path, "wb") as f:
                f.write(aged_image_data)
            
            # Generate local URL for aged image
            aged_local_url = f"/uploads/future_sim/{relative_path}"
            
            return {
                "success": True,
                "aged_image_path": relative_path,
                "aged_image_url": aged_local_url
            }
            
        except Exception as e:
            logger.error("Replicate age progression failed", error=str(e))
            
            # Fallback: Create a simple aged version using PIL
            return await cls._create_fallback_aged_image(image_url, user_id)
    
    @classmethod
    async def _create_fallback_aged_image(
        cls,
        image_url: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Create a fallback aged image using basic image processing."""
        
        try:
            # Download original image
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                image_data = response.content
            
            # Load image
            image = Image.open(io.BytesIO(image_data))
            
            # Apply aging effects using PIL
            # Reduce brightness slightly
            enhancer = ImageEnhance.Brightness(image)
            aged_image = enhancer.enhance(0.9)
            
            # Reduce contrast slightly
            enhancer = ImageEnhance.Contrast(aged_image)
            aged_image = enhancer.enhance(0.95)
            
            # Add slight blur for softer appearance
            aged_image = aged_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Convert back to bytes
            output = io.BytesIO()
            aged_image.save(output, format='JPEG', quality=85)
            aged_image_data = output.getvalue()
            
            # Save fallback aged image to local storage
            timestamp = int(datetime.now().timestamp())
            relative_path = f"{user_id}/aged_fallback_{timestamp}.jpg"
            
            user_dir = UPLOADS_DIR / user_id
            user_dir.mkdir(parents=True, exist_ok=True)
            
            aged_file_path = user_dir / f"aged_fallback_{timestamp}.jpg"
            with open(aged_file_path, "wb") as f:
                f.write(aged_image_data)
            
            # Generate local URL
            aged_local_url = f"/uploads/future_sim/{relative_path}"
            
            logger.info("Fallback aged image created", user_id=user_id)
            
            return {
                "success": True,
                "aged_image_path": relative_path,
                "aged_image_url": aged_local_url,
                "fallback": True
            }
            
        except Exception as e:
            logger.error("Fallback aged image creation failed", error=str(e))
            return {"error": "Failed to create aged image"}
    
    @classmethod
    async def generate_health_projections(
        cls,
        user_id: str,
        target_age_years: int,
        lifestyle_scenario: str = "current"
    ) -> Dict[str, Any]:
        """Generate comprehensive health projections for future age."""
        
        try:
            # Get current health data and predictions
            current_health = await cls._get_current_health_data(user_id)
            
            if not current_health:
                return {"error": "No health data available for projections"}
            
            # Get ML predictions for future health risks
            future_predictions = await MLModelService.predict_future_health_risks(
                user_id,
                target_age_years,
                lifestyle_scenario
            )
            
            # Calculate life expectancy projection
            life_expectancy = await cls._calculate_life_expectancy(
                current_health,
                future_predictions,
                lifestyle_scenario
            )
            
            # Generate health condition projections
            condition_projections = await cls._generate_condition_projections(
                current_health,
                future_predictions,
                target_age_years
            )
            
            # Create lifestyle impact analysis
            lifestyle_impact = await cls._analyze_lifestyle_impact(
                current_health,
                future_predictions,
                lifestyle_scenario
            )
            
            # Generate AI narrative using Groq
            health_narrative = await cls._generate_health_narrative(
                current_health,
                future_predictions,
                condition_projections,
                lifestyle_scenario,
                target_age_years
            )
            
            # Store projection record
            projection_record = await DatabaseService.execute_query(
                """
                INSERT INTO health_projections (user_id, target_age_years, lifestyle_scenario,
                                              projections_data, life_expectancy, created_at)
                VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING id
                """,
                user_id,
                target_age_years,
                lifestyle_scenario,
                json.dumps({
                    "condition_projections": condition_projections,
                    "lifestyle_impact": lifestyle_impact
                }),
                life_expectancy,
                fetch_one=True
            )
            
            logger.info(
                "Health projections generated",
                user_id=user_id,
                projection_id=projection_record["id"],
                target_age=target_age_years,
                scenario=lifestyle_scenario
            )
            
            return {
                "success": True,
                "projection_id": projection_record["id"],
                "target_age_years": target_age_years,
                "lifestyle_scenario": lifestyle_scenario,
                "life_expectancy": life_expectancy,
                "condition_projections": condition_projections,
                "lifestyle_impact": lifestyle_impact,
                "health_narrative": health_narrative,
                "recommendations": lifestyle_impact.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error("Health projections generation failed", user_id=user_id, error=str(e))
            return {"error": "Health projections generation failed"}
    
    @classmethod
    async def _get_current_health_data(cls, user_id: str) -> Optional[Dict[str, Any]]:
        """Get current health data and risk factors."""
        
        try:
            # Get user demographics
            user_data = await DatabaseService.execute_query(
                """
                SELECT date_of_birth, gender, height, weight
                FROM users 
                WHERE id = $1
                """,
                user_id,
                fetch_one=True
            )
            
            if not user_data:
                return None
            
            # Calculate current age
            if user_data["date_of_birth"]:
                today = datetime.now().date()
                birth_date = user_data["date_of_birth"]
                current_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            else:
                current_age = 30  # Default age
            
            # Get latest health metrics
            health_metrics = await DatabaseService.execute_query(
                """
                SELECT metric_type, value, recorded_at
                FROM health_metrics 
                WHERE user_id = $1 
                AND recorded_at > NOW() - INTERVAL '6 months'
                ORDER BY recorded_at DESC
                """,
                user_id,
                fetch_all=True
            )
            
            # Get current ML predictions
            current_predictions = await DatabaseService.execute_query(
                """
                SELECT disease_type, probability, confidence_score
                FROM health_predictions 
                WHERE user_id = $1 
                AND created_at > NOW() - INTERVAL '30 days'
                ORDER BY created_at DESC
                """,
                user_id,
                fetch_all=True
            )
            
            # Get family history
            family_history = await DatabaseService.execute_query(
                """
                SELECT condition_name, relation, age_of_onset
                FROM family_health_history 
                WHERE user_id = $1
                """,
                user_id,
                fetch_all=True
            )
            
            # Process health metrics into structured format
            metrics_dict = {}
            for metric in health_metrics:
                metrics_dict[metric["metric_type"]] = {
                    "value": float(metric["value"]),
                    "recorded_at": metric["recorded_at"]
                }
            
            # Process predictions
            predictions_dict = {}
            for pred in current_predictions:
                predictions_dict[pred["disease_type"]] = {
                    "probability": float(pred["probability"]),
                    "confidence": float(pred["confidence_score"])
                }
            
            # Process family history
            family_conditions = {}
            for fh in family_history:
                condition = fh["condition_name"]
                if condition not in family_conditions:
                    family_conditions[condition] = []
                family_conditions[condition].append({
                    "relation": fh["relation"],
                    "age_of_onset": fh["age_of_onset"]
                })
            
            return {
                "current_age": current_age,
                "gender": user_data["gender"],
                "height": float(user_data["height"]) if user_data["height"] else None,
                "weight": float(user_data["weight"]) if user_data["weight"] else None,
                "health_metrics": metrics_dict,
                "current_predictions": predictions_dict,
                "family_history": family_conditions
            }
            
        except Exception as e:
            logger.error("Failed to get current health data", user_id=user_id, error=str(e))
            return None
    
    @classmethod
    async def _calculate_life_expectancy(
        cls,
        current_health: Dict[str, Any],
        future_predictions: Dict[str, Any],
        lifestyle_scenario: str
    ) -> float:
        """Calculate projected life expectancy based on health data."""
        
        try:
            # Base life expectancy by gender (US averages)
            base_expectancy = 78.8 if current_health["gender"] == "male" else 81.1
            
            # Adjust for current age
            current_age = current_health["current_age"]
            remaining_years = base_expectancy - current_age
            
            # Risk factor adjustments
            risk_adjustments = 0
            
            # Diabetes risk adjustment
            diabetes_risk = future_predictions.get("diabetes", {}).get("probability", 0)
            if diabetes_risk > 0.7:
                risk_adjustments -= 3.5
            elif diabetes_risk > 0.4:
                risk_adjustments -= 1.8
            
            # Heart disease risk adjustment
            heart_risk = future_predictions.get("heart_disease", {}).get("probability", 0)
            if heart_risk > 0.7:
                risk_adjustments -= 4.2
            elif heart_risk > 0.4:
                risk_adjustments -= 2.1
            
            # Cancer risk adjustment
            cancer_risk = future_predictions.get("cancer", {}).get("probability", 0)
            if cancer_risk > 0.6:
                risk_adjustments -= 2.8
            elif cancer_risk > 0.3:
                risk_adjustments -= 1.4
            
            # BMI adjustment
            if current_health.get("height") and current_health.get("weight"):
                height_m = current_health["height"] / 100
                bmi = current_health["weight"] / (height_m ** 2)
                
                if bmi > 35:  # Severely obese
                    risk_adjustments -= 2.5
                elif bmi > 30:  # Obese
                    risk_adjustments -= 1.5
                elif bmi < 18.5:  # Underweight
                    risk_adjustments -= 1.0
            
            # Lifestyle scenario adjustments
            lifestyle_adjustments = {
                "improved": 3.2,  # Healthy lifestyle improvements
                "current": 0,     # No change
                "declined": -2.8  # Unhealthy lifestyle decline
            }
            
            lifestyle_adj = lifestyle_adjustments.get(lifestyle_scenario, 0)
            
            # Calculate final life expectancy
            projected_expectancy = base_expectancy + risk_adjustments + lifestyle_adj
            
            # Ensure reasonable bounds
            projected_expectancy = max(current_age + 5, min(projected_expectancy, 95))
            
            return round(projected_expectancy, 1)
            
        except Exception as e:
            logger.error("Life expectancy calculation failed", error=str(e))
            return 78.0  # Default fallback
    
    @classmethod
    async def _generate_condition_projections(
        cls,
        current_health: Dict[str, Any],
        future_predictions: Dict[str, Any],
        target_age_years: int
    ) -> Dict[str, Any]:
        """Generate specific health condition projections."""
        
        try:
            projections = {}
            
            # Diabetes projection
            diabetes_prob = future_predictions.get("diabetes", {}).get("probability", 0)
            projections["diabetes"] = {
                "probability": diabetes_prob,
                "risk_level": "high" if diabetes_prob > 0.6 else "medium" if diabetes_prob > 0.3 else "low",
                "potential_complications": [
                    "Vision problems",
                    "Kidney disease", 
                    "Nerve damage",
                    "Slow wound healing"
                ] if diabetes_prob > 0.5 else [],
                "visual_effects": cls.HEALTH_EFFECTS["diabetes"] if diabetes_prob > 0.4 else {}
            }
            
            # Heart disease projection
            heart_prob = future_predictions.get("heart_disease", {}).get("probability", 0)
            projections["heart_disease"] = {
                "probability": heart_prob,
                "risk_level": "high" if heart_prob > 0.6 else "medium" if heart_prob > 0.3 else "low",
                "potential_complications": [
                    "Chest pain",
                    "Shortness of breath",
                    "Fatigue",
                    "Reduced exercise capacity"
                ] if heart_prob > 0.5 else [],
                "visual_effects": cls.HEALTH_EFFECTS["heart_disease"] if heart_prob > 0.4 else {}
            }
            
            # Cancer risk projection
            cancer_prob = future_predictions.get("cancer", {}).get("probability", 0)
            projections["cancer"] = {
                "probability": cancer_prob,
                "risk_level": "high" if cancer_prob > 0.5 else "medium" if cancer_prob > 0.25 else "low",
                "screening_recommendations": [
                    "Regular mammograms" if current_health["gender"] == "female" else "Prostate screening",
                    "Colonoscopy every 10 years",
                    "Annual skin checks",
                    "Lung CT if smoking history"
                ] if cancer_prob > 0.3 else []
            }
            
            # Mobility and aging effects
            age_effects_prob = min((target_age_years - current_health["current_age"]) / 30, 1.0)
            projections["aging_effects"] = {
                "probability": age_effects_prob,
                "potential_effects": [
                    "Reduced mobility",
                    "Joint stiffness",
                    "Decreased muscle mass",
                    "Balance issues",
                    "Slower reflexes"
                ] if age_effects_prob > 0.5 else [],
                "visual_effects": {
                    "posture_effects": ["slight_stoop", "reduced_height"],
                    "skin_effects": ["wrinkles", "age_spots"],
                    "hair_effects": ["graying", "thinning"]
                } if age_effects_prob > 0.3 else {}
            }
            
            return projections
            
        except Exception as e:
            logger.error("Condition projections generation failed", error=str(e))
            return {}
    
    @classmethod
    async def _analyze_lifestyle_impact(
        cls,
        current_health: Dict[str, Any],
        future_predictions: Dict[str, Any],
        lifestyle_scenario: str
    ) -> Dict[str, Any]:
        """Analyze the impact of different lifestyle scenarios."""
        
        try:
            # Calculate risk reductions/increases for different scenarios
            scenarios = {
                "improved": {
                    "description": "Healthy lifestyle improvements",
                    "changes": [
                        "Regular exercise (150 min/week)",
                        "Balanced diet with reduced processed foods",
                        "Adequate sleep (7-9 hours)",
                        "Stress management techniques",
                        "No smoking, limited alcohol"
                    ],
                    "risk_reductions": {
                        "diabetes": 0.3,
                        "heart_disease": 0.4,
                        "cancer": 0.2,
                        "obesity": 0.5
                    }
                },
                "current": {
                    "description": "Continue current lifestyle",
                    "changes": ["No significant lifestyle changes"],
                    "risk_reductions": {}
                },
                "declined": {
                    "description": "Unhealthy lifestyle decline",
                    "changes": [
                        "Sedentary lifestyle",
                        "Poor diet with high processed foods",
                        "Inadequate sleep",
                        "High stress levels",
                        "Increased smoking/alcohol"
                    ],
                    "risk_increases": {
                        "diabetes": 0.4,
                        "heart_disease": 0.5,
                        "cancer": 0.3,
                        "obesity": 0.6
                    }
                }
            }
            
            scenario_data = scenarios.get(lifestyle_scenario, scenarios["current"])
            
            # Calculate adjusted predictions
            adjusted_predictions = {}
            for condition, prediction in future_predictions.items():
                current_prob = prediction.get("probability", 0)
                
                if lifestyle_scenario == "improved":
                    reduction = scenario_data["risk_reductions"].get(condition, 0)
                    adjusted_prob = max(0, current_prob - reduction)
                elif lifestyle_scenario == "declined":
                    increase = scenario_data["risk_increases"].get(condition, 0)
                    adjusted_prob = min(1.0, current_prob + increase)
                else:
                    adjusted_prob = current_prob
                
                adjusted_predictions[condition] = {
                    "original_probability": current_prob,
                    "adjusted_probability": adjusted_prob,
                    "change": adjusted_prob - current_prob
                }
            
            # Generate recommendations
            recommendations = cls._generate_lifestyle_recommendations(
                current_health,
                future_predictions,
                lifestyle_scenario
            )
            
            return {
                "scenario": lifestyle_scenario,
                "description": scenario_data["description"],
                "lifestyle_changes": scenario_data["changes"],
                "adjusted_predictions": adjusted_predictions,
                "recommendations": recommendations,
                "potential_benefits": cls._calculate_potential_benefits(adjusted_predictions)
            }
            
        except Exception as e:
            logger.error("Lifestyle impact analysis failed", error=str(e))
            return {}
    
    @classmethod
    def _generate_lifestyle_recommendations(
        cls,
        current_health: Dict[str, Any],
        future_predictions: Dict[str, Any],
        lifestyle_scenario: str
    ) -> List[str]:
        """Generate personalized lifestyle recommendations."""
        
        recommendations = []
        
        # High diabetes risk recommendations
        if future_predictions.get("diabetes", {}).get("probability", 0) > 0.4:
            recommendations.extend([
                "Maintain a healthy weight through balanced diet and exercise",
                "Limit refined sugars and processed carbohydrates",
                "Monitor blood sugar levels regularly",
                "Include fiber-rich foods in your diet"
            ])
        
        # High heart disease risk recommendations
        if future_predictions.get("heart_disease", {}).get("probability", 0) > 0.4:
            recommendations.extend([
                "Engage in regular cardiovascular exercise",
                "Follow a heart-healthy diet (Mediterranean style)",
                "Manage stress through relaxation techniques",
                "Monitor blood pressure and cholesterol levels"
            ])
        
        # General aging recommendations
        recommendations.extend([
            "Stay physically active with age-appropriate exercises",
            "Maintain social connections and mental stimulation",
            "Get regular health screenings and check-ups",
            "Prioritize quality sleep and stress management"
        ])
        
        # BMI-based recommendations
        if current_health.get("height") and current_health.get("weight"):
            height_m = current_health["height"] / 100
            bmi = current_health["weight"] / (height_m ** 2)
            
            if bmi > 30:
                recommendations.append("Work with a healthcare provider on a safe weight loss plan")
            elif bmi < 18.5:
                recommendations.append("Consider working with a nutritionist to achieve healthy weight gain")
        
        return list(set(recommendations))  # Remove duplicates
    
    @classmethod
    def _calculate_potential_benefits(cls, adjusted_predictions: Dict[str, Any]) -> Dict[str, str]:
        """Calculate potential benefits of lifestyle changes."""
        
        benefits = {}
        
        for condition, data in adjusted_predictions.items():
            change = data.get("change", 0)
            
            if change < -0.1:  # Significant risk reduction
                benefits[condition] = f"Risk reduced by {abs(change)*100:.1f}%"
            elif change > 0.1:  # Significant risk increase
                benefits[condition] = f"Risk increased by {change*100:.1f}%"
            else:
                benefits[condition] = "No significant change"
        
        return benefits
    
    @classmethod
    async def _generate_health_narrative(
        cls,
        current_health: Dict[str, Any],
        future_predictions: Dict[str, Any],
        condition_projections: Dict[str, Any],
        lifestyle_scenario: str,
        target_age_years: int
    ) -> Dict[str, str]:
        """Generate AI-powered health narrative using Groq."""
        
        try:
            # Prepare context for AI narrative generation
            context = {
                "current_age": current_health["current_age"],
                "target_age": target_age_years,
                "gender": current_health["gender"],
                "lifestyle_scenario": lifestyle_scenario,
                "health_risks": {k: v.get("probability", 0) for k, v in future_predictions.items()},
                "condition_projections": condition_projections
            }
            
            # Create system prompt for health narrative
            system_prompt = """You are a compassionate health AI creating personalized future health narratives. 
            Based on the health data and projections, create encouraging yet realistic narratives for different scenarios.
            
            Provide two narratives:
            1. "current_path" - What might happen if current lifestyle continues
            2. "improved_path" - What could happen with healthy lifestyle changes
            
            Keep narratives:
            - Encouraging and motivational
            - Medically accurate but not alarming
            - Focused on actionable improvements
            - Personalized to the individual
            - 2-3 sentences each
            
            Respond with JSON format:
            {
                "current_path": "narrative text",
                "improved_path": "narrative text"
            }"""
            
            user_message = f"""
            Health Context:
            - Current age: {context['current_age']}, Target age: {context['target_age']}
            - Gender: {context['gender']}
            - Lifestyle scenario: {context['lifestyle_scenario']}
            
            Health Risk Probabilities:
            {json.dumps(context['health_risks'], indent=2)}
            
            Please generate personalized health narratives for this individual.
            """
            
            # Generate narrative using Groq
            narrative_response = await AIService.analyze_symptoms_with_groq(
                user_message,
                {"context": "health_narrative", "age": context['current_age']}
            )
            
            # Parse narrative response
            if isinstance(narrative_response, dict) and "reasoning" in narrative_response:
                try:
                    narrative_data = json.loads(narrative_response["reasoning"])
                    return narrative_data
                except json.JSONDecodeError:
                    pass
            
            # Fallback narrative generation
            return cls._generate_fallback_narrative(context)
            
        except Exception as e:
            logger.error("Health narrative generation failed", error=str(e))
            return cls._generate_fallback_narrative({
                "current_age": current_health.get("current_age", 30),
                "target_age": target_age_years,
                "lifestyle_scenario": lifestyle_scenario
            })
    
    @classmethod
    def _generate_fallback_narrative(cls, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate fallback health narrative when AI is unavailable."""
        
        age_diff = context.get("target_age", 50) - context.get("current_age", 30)
        
        if context.get("lifestyle_scenario") == "improved":
            current_path = f"By maintaining your current lifestyle, you may experience typical age-related changes over the next {age_diff} years. Regular health monitoring and preventive care will be important for early detection of any health issues."
            
            improved_path = f"With healthy lifestyle improvements, you could significantly reduce your risk of chronic diseases and maintain better physical and mental health as you age. Small changes today can lead to substantial benefits over the next {age_diff} years."
        else:
            current_path = f"Continuing your current health habits may lead to increased risk of age-related health conditions over the next {age_diff} years. Regular check-ups and preventive care remain important."
            
            improved_path = f"By adopting healthier habits like regular exercise, balanced nutrition, and stress management, you could dramatically improve your health trajectory and quality of life in the coming {age_diff} years."
        
        return {
            "current_path": current_path,
            "improved_path": improved_path
        }
    
    @classmethod
    async def get_user_simulations(
        cls,
        user_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get user's previous future simulations."""
        
        try:
            simulations = await DatabaseService.execute_query(
                """
                SELECT ap.id as progression_id, ap.target_age_years, ap.created_at as progression_date,
                       hp.id as health_projection_id, hp.lifestyle_scenario, hp.life_expectancy,
                       ui.file_path as original_image_path, ap.aged_image_path
                FROM age_progressions ap
                LEFT JOIN health_projections hp ON ap.user_id = hp.user_id 
                    AND DATE(ap.created_at) = DATE(hp.created_at)
                LEFT JOIN user_images ui ON ap.original_image_path = ui.file_path
                WHERE ap.user_id = $1
                ORDER BY ap.created_at DESC
                LIMIT $2
                """,
                user_id,
                limit,
                fetch_all=True
            )
            
            result = []
            for sim in simulations:
                # Generate local URLs for images
                original_url = None
                aged_url = None
                
                if sim["original_image_path"]:
                    original_url = f"/uploads/future_sim/{sim['original_image_path']}"
                
                if sim["aged_image_path"]:
                    aged_url = f"/uploads/future_sim/{sim['aged_image_path']}"
                
                result.append({
                    "progression_id": sim["progression_id"],
                    "health_projection_id": sim["health_projection_id"],
                    "target_age_years": sim["target_age_years"],
                    "lifestyle_scenario": sim["lifestyle_scenario"],
                    "life_expectancy": float(sim["life_expectancy"]) if sim["life_expectancy"] else None,
                    "original_image_url": original_url,
                    "aged_image_url": aged_url,
                    "created_at": sim["progression_date"]
                })
            
            return result
            
        except Exception as e:
            logger.error("Failed to get user simulations", user_id=user_id, error=str(e))
            return []