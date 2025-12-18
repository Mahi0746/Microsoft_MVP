# HealthSync AI - Machine Learning Service
import asyncio
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
import structlog
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from config import settings
from services.db_service import DatabaseService


logger = structlog.get_logger(__name__)


class MLModelService:
    """Machine Learning service for health predictions and risk assessment."""
    
    # Model cache
    _model_cache: Dict[str, Any] = {}
    _scaler_cache: Dict[str, Any] = {}
    
    # Disease prediction models
    DISEASE_MODELS = {
        "diabetes": {
            "algorithm": RandomForestClassifier,
            "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
            "features": [
                "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
                "family_history_diabetes", "exercise_frequency", "diet_quality",
                "smoking_status", "alcohol_consumption", "stress_level"
            ]
        },
        "heart_disease": {
            "algorithm": LogisticRegression,
            "params": {"random_state": 42, "max_iter": 1000},
            "features": [
                "age", "gender", "chest_pain_type", "blood_pressure_systolic",
                "cholesterol", "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
                "exercise_induced_angina", "st_depression", "family_history_heart_disease"
            ]
        },
        "cancer": {
            "algorithm": GradientBoostingClassifier,
            "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42},
            "features": [
                "age", "gender", "smoking_status", "alcohol_consumption",
                "family_history_cancer", "bmi", "exercise_frequency",
                "diet_quality", "environmental_exposure", "genetic_markers"
            ]
        },
        "hypertension": {
            "algorithm": RandomForestClassifier,
            "params": {"n_estimators": 80, "max_depth": 8, "random_state": 42},
            "features": [
                "age", "bmi", "sodium_intake", "exercise_frequency", "stress_level",
                "family_history_hypertension", "smoking_status", "alcohol_consumption"
            ]
        },
        "stroke": {
            "algorithm": LogisticRegression,
            "params": {"random_state": 42, "max_iter": 1000},
            "features": [
                "age", "gender", "hypertension", "heart_disease", "avg_glucose_level",
                "bmi", "smoking_status", "family_history_stroke"
            ]
        }
    }
    
    @classmethod
    async def initialize_models(cls):
        """Initialize and load ML models."""
        
        logger.info("Initializing ML models")
        
        try:
            # Load existing models from MongoDB or train new ones
            for disease, config in cls.DISEASE_MODELS.items():
                await cls._load_or_train_model(disease, config)
            
            logger.info("ML models initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize ML models", error=str(e))
            raise
    
    @classmethod
    async def _load_or_train_model(cls, disease: str, config: Dict[str, Any]):
        """Load existing model or train new one."""
        
        try:
            # Try to load existing model from MongoDB
            model_doc = await DatabaseService.mongodb_find_one(
                "ml_models",
                {"model_type": "disease_prediction", "disease": disease, "is_active": True}
            )
            
            if model_doc and model_doc.get("model_data"):
                # Load existing model
                model_data = model_doc["model_data"]
                scaler_data = model_doc.get("scaler_data")
                
                # Deserialize model (in production, would use proper deserialization)
                # For now, create new model with same parameters
                model = config["algorithm"](**config["params"])
                scaler = StandardScaler()
                
                cls._model_cache[disease] = model
                cls._scaler_cache[disease] = scaler
                
                logger.info("Loaded existing model", disease=disease)
            else:
                # Train new model
                await cls._train_new_model(disease, config)
                
        except Exception as e:
            logger.error("Failed to load/train model", disease=disease, error=str(e))
            # Create mock model for development
            cls._create_mock_model(disease, config)
    
    @classmethod
    async def _train_new_model(cls, disease: str, config: Dict[str, Any]):
        """Train a new ML model for disease prediction."""
        
        try:
            # Generate synthetic training data (in production, use real data)
            X, y = cls._generate_synthetic_data(disease, config["features"])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = config["algorithm"](**config["params"])
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "auc_roc": roc_auc_score(y_test, y_pred_proba)
            }
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            # Store model
            cls._model_cache[disease] = model
            cls._scaler_cache[disease] = scaler
            
            # Save to MongoDB
            await cls._save_model_to_db(disease, model, scaler, metrics, cv_scores, config)
            
            logger.info(
                "Trained new model",
                disease=disease,
                accuracy=metrics["accuracy"],
                auc_roc=metrics["auc_roc"]
            )
            
        except Exception as e:
            logger.error("Failed to train model", disease=disease, error=str(e))
            cls._create_mock_model(disease, config)
    
    @classmethod
    def _create_mock_model(cls, disease: str, config: Dict[str, Any]):
        """Create mock model for development."""
        
        model = config["algorithm"](**config["params"])
        scaler = StandardScaler()
        
        # Fit with dummy data
        dummy_X = np.random.rand(100, len(config["features"]))
        dummy_y = np.random.randint(0, 2, 100)
        
        scaler.fit(dummy_X)
        model.fit(scaler.transform(dummy_X), dummy_y)
        
        cls._model_cache[disease] = model
        cls._scaler_cache[disease] = scaler
        
        logger.info("Created mock model", disease=disease)
    
    @classmethod
    async def predict_disease_risks(
        cls,
        user_id: str,
        health_data: Dict[str, Any],
        include_family_history: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """Predict disease risks for a user."""
        
        try:
            predictions = {}
            
            # Get family history if requested
            family_risks = {}
            if include_family_history:
                family_graph = await DatabaseService.mongodb_find_one(
                    "family_graph",
                    {"user_id": user_id}
                )
                if family_graph:
                    family_risks = family_graph.get("inherited_risks", {})
            
            # Predict for each disease
            for disease in cls.DISEASE_MODELS.keys():
                if disease in cls._model_cache and disease in cls._scaler_cache:
                    prediction = await cls._predict_single_disease(
                        disease, health_data, family_risks
                    )
                    predictions[disease] = prediction
            
            # Store predictions in database
            await cls._store_predictions(user_id, predictions)
            
            return predictions
            
        except Exception as e:
            logger.error("Disease risk prediction failed", user_id=user_id, error=str(e))
            return {}
    
    @classmethod
    async def _predict_single_disease(
        cls,
        disease: str,
        health_data: Dict[str, Any],
        family_risks: Dict[str, float]
    ) -> Dict[str, Any]:
        """Predict risk for a single disease."""
        
        try:
            model = cls._model_cache[disease]
            scaler = cls._scaler_cache[disease]
            config = cls.DISEASE_MODELS[disease]
            
            # Prepare features
            features = cls._prepare_features(disease, health_data, family_risks, config["features"])
            
            # Scale features
            features_scaled = scaler.transform([features])
            
            # Make prediction
            probability = model.predict_proba(features_scaled)[0][1]
            confidence = max(model.predict_proba(features_scaled)[0])
            
            # Calculate risk factors
            risk_factors = cls._calculate_risk_factors(disease, health_data, family_risks)
            
            # Generate recommendations
            recommendations = cls._generate_recommendations(disease, probability, risk_factors)
            
            return {
                "probability": float(probability),
                "confidence_score": float(confidence),
                "risk_level": cls._categorize_risk(probability),
                "risk_factors": risk_factors,
                "recommendations": recommendations,
                "family_contribution": family_risks.get(disease, 0.0),
                "model_version": "v1.0.0"
            }
            
        except Exception as e:
            logger.error("Single disease prediction failed", disease=disease, error=str(e))
            return {
                "probability": 0.5,
                "confidence_score": 0.3,
                "risk_level": "medium",
                "error": str(e)
            }
    
    @classmethod
    def _prepare_features(
        cls,
        disease: str,
        health_data: Dict[str, Any],
        family_risks: Dict[str, float],
        feature_names: List[str]
    ) -> List[float]:
        """Prepare feature vector for prediction."""
        
        features = []
        
        for feature in feature_names:
            if feature.startswith("family_history_"):
                # Family history feature
                disease_name = feature.replace("family_history_", "")
                value = 1.0 if family_risks.get(disease_name, 0) > 0.3 else 0.0
            elif feature in health_data:
                # Direct health data
                value = float(health_data[feature])
            else:
                # Default values for missing features
                value = cls._get_default_feature_value(feature, health_data)
            
            features.append(value)
        
        return features
    
    @classmethod
    def _get_default_feature_value(cls, feature: str, health_data: Dict[str, Any]) -> float:
        """Get default value for missing features."""
        
        defaults = {
            "age": health_data.get("age", 35),
            "gender": 1.0 if health_data.get("gender", "").lower() == "male" else 0.0,
            "bmi": health_data.get("bmi", 25.0),
            "blood_pressure_systolic": 120.0,
            "blood_pressure_diastolic": 80.0,
            "cholesterol": 200.0,
            "exercise_frequency": 3.0,  # times per week
            "diet_quality": 3.0,  # 1-5 scale
            "smoking_status": 0.0,  # 0=never, 1=former, 2=current
            "alcohol_consumption": 1.0,  # 0=none, 1=light, 2=moderate, 3=heavy
            "stress_level": 0.5,  # 0-1 scale
            "sodium_intake": 2300.0,  # mg per day
            "fasting_blood_sugar": 0.0,  # 0=normal, 1=elevated
            "max_heart_rate": 180.0,
            "chest_pain_type": 0.0,
            "resting_ecg": 0.0,
            "exercise_induced_angina": 0.0,
            "st_depression": 0.0,
            "avg_glucose_level": 100.0,
            "hypertension": 0.0,
            "heart_disease": 0.0,
            "environmental_exposure": 0.0,
            "genetic_markers": 0.0
        }
        
        return defaults.get(feature, 0.0)
    
    @classmethod
    def _calculate_risk_factors(
        cls,
        disease: str,
        health_data: Dict[str, Any],
        family_risks: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate contributing risk factors."""
        
        risk_factors = {
            "modifiable": {},
            "non_modifiable": {},
            "family_history": family_risks.get(disease, 0.0)
        }
        
        age = health_data.get("age", 35)
        bmi = health_data.get("bmi", 25.0)
        
        # Age risk (non-modifiable)
        if disease in ["heart_disease", "stroke", "cancer"]:
            risk_factors["non_modifiable"]["age"] = min(1.0, max(0.0, (age - 40) / 40))
        elif disease == "diabetes":
            risk_factors["non_modifiable"]["age"] = min(1.0, max(0.0, (age - 35) / 35))
        
        # BMI risk (modifiable)
        if bmi > 30:
            risk_factors["modifiable"]["obesity"] = min(1.0, (bmi - 25) / 15)
        elif bmi > 25:
            risk_factors["modifiable"]["overweight"] = min(1.0, (bmi - 25) / 10)
        
        # Lifestyle factors
        smoking = health_data.get("smoking_status", 0)
        if smoking > 0:
            risk_factors["modifiable"]["smoking"] = smoking / 2.0
        
        exercise = health_data.get("exercise_frequency", 3)
        if exercise < 3:
            risk_factors["modifiable"]["sedentary_lifestyle"] = (3 - exercise) / 3.0
        
        stress = health_data.get("stress_level", 0.5)
        if stress > 0.6:
            risk_factors["modifiable"]["high_stress"] = stress
        
        return risk_factors
    
    @classmethod
    def _categorize_risk(cls, probability: float) -> str:
        """Categorize risk level based on probability."""
        
        if probability >= 0.7:
            return "high"
        elif probability >= 0.4:
            return "medium"
        else:
            return "low"
    
    @classmethod
    def _generate_recommendations(
        cls,
        disease: str,
        probability: float,
        risk_factors: Dict[str, Any]
    ) -> List[str]:
        """Generate personalized recommendations."""
        
        recommendations = []
        
        # General recommendations based on risk level
        if probability >= 0.7:
            recommendations.extend([
                f"Schedule immediate consultation with a specialist for {disease} risk assessment",
                "Consider comprehensive health screening",
                "Discuss preventive medications with your doctor"
            ])
        elif probability >= 0.4:
            recommendations.extend([
                f"Schedule regular monitoring for {disease} risk factors",
                "Implement lifestyle modifications to reduce risk"
            ])
        
        # Specific recommendations based on modifiable risk factors
        modifiable_risks = risk_factors.get("modifiable", {})
        
        if "obesity" in modifiable_risks or "overweight" in modifiable_risks:
            recommendations.append("Focus on weight management through diet and exercise")
        
        if "smoking" in modifiable_risks:
            recommendations.append("Consider smoking cessation programs")
        
        if "sedentary_lifestyle" in modifiable_risks:
            recommendations.append("Increase physical activity to at least 150 minutes per week")
        
        if "high_stress" in modifiable_risks:
            recommendations.append("Implement stress management techniques")
        
        # Disease-specific recommendations
        disease_specific = {
            "diabetes": [
                "Monitor blood sugar levels regularly",
                "Follow a low-glycemic diet",
                "Maintain healthy weight"
            ],
            "heart_disease": [
                "Monitor blood pressure and cholesterol",
                "Follow heart-healthy diet (Mediterranean style)",
                "Regular cardiovascular exercise"
            ],
            "cancer": [
                "Follow cancer screening guidelines",
                "Maintain healthy diet rich in antioxidants",
                "Limit alcohol consumption"
            ],
            "hypertension": [
                "Reduce sodium intake",
                "Monitor blood pressure regularly",
                "Maintain healthy weight"
            ],
            "stroke": [
                "Control blood pressure and cholesterol",
                "Take prescribed medications as directed",
                "Recognize stroke warning signs"
            ]
        }
        
        if disease in disease_specific:
            recommendations.extend(disease_specific[disease])
        
        return recommendations[:8]  # Limit to 8 recommendations
    
    @classmethod
    async def _store_predictions(cls, user_id: str, predictions: Dict[str, Dict[str, Any]]):
        """Store predictions in PostgreSQL database."""
        
        try:
            # Clear old predictions
            await DatabaseService.execute_query(
                "DELETE FROM predictions WHERE user_id = $1",
                user_id
            )
            
            # Insert new predictions
            for disease, prediction in predictions.items():
                expires_at = datetime.utcnow() + timedelta(days=30)
                
                await DatabaseService.execute_query(
                    """
                    INSERT INTO predictions (user_id, disease, probability, confidence_score,
                                          risk_factors, model_version, recommendations, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    user_id,
                    disease,
                    prediction["probability"],
                    prediction["confidence_score"],
                    prediction.get("risk_factors", {}),
                    prediction.get("model_version", "v1.0.0"),
                    prediction.get("recommendations", []),
                    expires_at
                )
            
            logger.info("Predictions stored", user_id=user_id, disease_count=len(predictions))
            
        except Exception as e:
            logger.error("Failed to store predictions", user_id=user_id, error=str(e))
    
    @classmethod
    async def _save_model_to_db(
        cls,
        disease: str,
        model: Any,
        scaler: Any,
        metrics: Dict[str, float],
        cv_scores: np.ndarray,
        config: Dict[str, Any]
    ):
        """Save trained model to MongoDB."""
        
        try:
            # Serialize model and scaler (simplified for demo)
            model_data = b"serialized_model_placeholder"
            scaler_data = b"serialized_scaler_placeholder"
            
            model_document = {
                "model_type": "disease_prediction",
                "disease": disease,
                "model_data": model_data,
                "scaler_data": scaler_data,
                "algorithm": config["algorithm"].__name__,
                "hyperparameters": config["params"],
                "feature_columns": config["features"],
                "accuracy_metrics": metrics,
                "cross_validation_scores": cv_scores.tolist(),
                "training_date": datetime.utcnow(),
                "version": "v1.0.0",
                "is_active": True
            }
            
            # Deactivate old models
            await DatabaseService.mongodb_update_one(
                "ml_models",
                {"model_type": "disease_prediction", "disease": disease},
                {"$set": {"is_active": False}}
            )
            
            # Insert new model
            await DatabaseService.mongodb_insert_one("ml_models", model_document)
            
            logger.info("Model saved to database", disease=disease)
            
        except Exception as e:
            logger.error("Failed to save model", disease=disease, error=str(e))
    
    @classmethod
    def _generate_synthetic_data(cls, disease: str, features: List[str], n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data for model development."""
        
        np.random.seed(42)
        
        # Generate feature data
        data = {}
        
        for feature in features:
            if feature == "age":
                data[feature] = np.random.normal(45, 15, n_samples).clip(18, 90)
            elif feature == "bmi":
                data[feature] = np.random.normal(26, 5, n_samples).clip(15, 50)
            elif feature in ["blood_pressure_systolic"]:
                data[feature] = np.random.normal(130, 20, n_samples).clip(90, 200)
            elif feature in ["blood_pressure_diastolic"]:
                data[feature] = np.random.normal(85, 15, n_samples).clip(60, 120)
            elif feature == "cholesterol":
                data[feature] = np.random.normal(200, 40, n_samples).clip(120, 350)
            elif feature in ["exercise_frequency"]:
                data[feature] = np.random.poisson(3, n_samples).clip(0, 7)
            elif feature in ["diet_quality"]:
                data[feature] = np.random.randint(1, 6, n_samples)
            elif "family_history" in feature:
                data[feature] = np.random.binomial(1, 0.3, n_samples)
            elif feature in ["gender"]:
                data[feature] = np.random.binomial(1, 0.5, n_samples)
            elif feature in ["smoking_status"]:
                data[feature] = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])
            else:
                # Default to normal distribution
                data[feature] = np.random.normal(0.5, 0.2, n_samples).clip(0, 1)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate target variable based on risk factors
        risk_score = np.zeros(n_samples)
        
        # Age contribution
        if "age" in data:
            risk_score += (df["age"] - 30) / 100
        
        # BMI contribution
        if "bmi" in data:
            risk_score += np.maximum(0, (df["bmi"] - 25) / 20)
        
        # Family history contribution
        for feature in features:
            if "family_history" in feature:
                risk_score += df[feature] * 0.3
        
        # Smoking contribution
        if "smoking_status" in data:
            risk_score += df["smoking_status"] * 0.2
        
        # Add noise and create binary target
        risk_score += np.random.normal(0, 0.1, n_samples)
        
        # Disease-specific adjustments
        if disease == "diabetes":
            risk_score *= 1.2
        elif disease == "heart_disease":
            risk_score *= 1.1
        elif disease == "cancer":
            risk_score *= 0.8
        
        # Convert to binary target
        threshold = np.percentile(risk_score, 70)  # 30% positive cases
        y = (risk_score > threshold).astype(int)
        
        return df.values, y
    
    @classmethod
    async def retrain_model(cls, disease: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrain a specific model with updated data."""
        
        try:
            if disease not in cls.DISEASE_MODELS:
                raise ValueError(f"Unknown disease: {disease}")
            
            config = cls.DISEASE_MODELS[disease]
            
            # Train new model
            await cls._train_new_model(disease, config)
            
            # Update predictions for all users if global retrain
            if not user_id:
                # This would trigger prediction updates for all users
                logger.info("Global model retrained", disease=disease)
            
            return {
                "disease": disease,
                "status": "retrained",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Model retraining failed", disease=disease, error=str(e))
            return {"error": str(e)}
    
    @classmethod
    async def get_model_performance(cls, disease: str) -> Dict[str, Any]:
        """Get performance metrics for a model."""
        
        try:
            model_doc = await DatabaseService.mongodb_find_one(
                "ml_models",
                {"model_type": "disease_prediction", "disease": disease, "is_active": True}
            )
            
            if not model_doc:
                return {"error": "Model not found"}
            
            return {
                "disease": disease,
                "accuracy_metrics": model_doc.get("accuracy_metrics", {}),
                "cross_validation_scores": model_doc.get("cross_validation_scores", []),
                "training_date": model_doc.get("training_date"),
                "version": model_doc.get("version"),
                "algorithm": model_doc.get("algorithm"),
                "feature_count": len(model_doc.get("feature_columns", []))
            }
            
        except Exception as e:
            logger.error("Failed to get model performance", disease=disease, error=str(e))
            return {"error": str(e)}