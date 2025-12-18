# HealthSync AI - Doctor Marketplace Service
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import structlog
from groq import Groq

from config import settings
from services.db_service import DatabaseService
from services.ai_service import AIService


logger = structlog.get_logger(__name__)


class MarketplaceService:
    """Comprehensive doctor marketplace service with AI-powered matching."""
    
    # Specialty mapping for symptom analysis
    SPECIALTY_KEYWORDS = {
        "cardiology": [
            "chest pain", "heart", "cardiac", "palpitations", "shortness of breath",
            "hypertension", "blood pressure", "arrhythmia", "angina", "heart attack"
        ],
        "dermatology": [
            "skin", "rash", "acne", "mole", "eczema", "psoriasis", "dermatitis",
            "itching", "hives", "skin cancer", "melanoma", "wart"
        ],
        "neurology": [
            "headache", "migraine", "seizure", "stroke", "memory loss", "dizziness",
            "numbness", "tingling", "tremor", "epilepsy", "parkinson", "alzheimer"
        ],
        "orthopedics": [
            "bone", "joint", "fracture", "arthritis", "back pain", "knee pain",
            "shoulder pain", "hip pain", "sports injury", "sprain", "strain"
        ],
        "gastroenterology": [
            "stomach", "abdominal pain", "nausea", "vomiting", "diarrhea", "constipation",
            "heartburn", "acid reflux", "ulcer", "ibs", "crohn", "colitis"
        ],
        "pulmonology": [
            "lung", "breathing", "cough", "asthma", "copd", "pneumonia", "bronchitis",
            "shortness of breath", "wheezing", "chest congestion"
        ],
        "endocrinology": [
            "diabetes", "thyroid", "hormone", "insulin", "blood sugar", "metabolism",
            "weight gain", "weight loss", "fatigue", "adrenal", "pituitary"
        ],
        "psychiatry": [
            "depression", "anxiety", "panic", "bipolar", "schizophrenia", "ptsd",
            "ocd", "adhd", "mental health", "mood", "stress", "insomnia"
        ],
        "gynecology": [
            "menstrual", "pregnancy", "pelvic pain", "ovarian", "uterine", "cervical",
            "breast", "contraception", "fertility", "menopause", "pms"
        ],
        "urology": [
            "kidney", "bladder", "urinary", "prostate", "erectile dysfunction",
            "kidney stone", "uti", "incontinence", "blood in urine"
        ]
    }
    
    @classmethod
    async def analyze_symptoms_for_specialists(
        cls,
        symptoms_summary: str,
        urgency_level: str = "medium",
        patient_age: Optional[int] = None,
        patient_gender: Optional[str] = None
    ) -> Dict[str, Any]:
        """Advanced symptom analysis with AI-powered specialist matching."""
        
        try:
            # Enhanced Groq analysis with specialist focus
            system_prompt = """You are an expert medical triage AI specializing in symptom analysis and specialist referrals.
            
            Analyze the patient's symptoms and provide:
            1. Primary specialty recommendations (ranked by relevance)
            2. Secondary specialties that might be relevant
            3. Urgency assessment with reasoning
            4. Red flags that require immediate attention
            5. Confidence score for each recommendation
            
            Consider patient demographics when available.
            
            Respond with valid JSON:
            {
                "primary_specialists": [
                    {"specialty": "cardiology", "confidence": 0.9, "reasoning": "chest pain with cardiac risk factors"},
                    {"specialty": "emergency_medicine", "confidence": 0.7, "reasoning": "acute onset requires evaluation"}
                ],
                "secondary_specialists": [
                    {"specialty": "pulmonology", "confidence": 0.6, "reasoning": "breathing symptoms"}
                ],
                "urgency_assessment": {
                    "level": "high",
                    "reasoning": "acute chest pain requires immediate evaluation",
                    "time_frame": "within 2 hours",
                    "red_flags": ["chest pain", "shortness of breath"]
                },
                "overall_confidence": 0.85,
                "triage_notes": "Patient should be seen urgently due to cardiac symptoms"
            }"""
            
            # Prepare enhanced user message
            demographics = ""
            if patient_age:
                demographics += f"Age: {patient_age} years old\n"
            if patient_gender:
                demographics += f"Gender: {patient_gender}\n"
            
            user_message = f"""
            Patient Demographics:
            {demographics if demographics else "Not provided"}
            
            Reported Urgency Level: {urgency_level}
            
            Symptoms Description:
            "{symptoms_summary}"
            
            Please analyze these symptoms and provide specialist recommendations with confidence scores.
            """
            
            # Call Groq API
            groq_client = Groq(api_key=settings.groq_api_key)
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                model=settings.groq_model,
                max_tokens=800,
                temperature=0.1
            )
            
            # Parse AI response
            try:
                ai_analysis = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError:
                # Fallback to keyword-based analysis
                ai_analysis = await cls._keyword_based_analysis(symptoms_summary, urgency_level)
            
            # Enhance with local keyword matching
            keyword_matches = cls._analyze_symptom_keywords(symptoms_summary)
            
            # Combine AI and keyword analysis
            final_analysis = cls._combine_analyses(ai_analysis, keyword_matches)
            
            logger.info(
                "Symptom analysis completed",
                symptoms_length=len(symptoms_summary),
                primary_specialists=len(final_analysis.get("primary_specialists", [])),
                urgency=final_analysis.get("urgency_assessment", {}).get("level", "unknown")
            )
            
            return final_analysis
            
        except Exception as e:
            logger.error("Symptom analysis failed", error=str(e))
            # Return basic fallback analysis
            return await cls._keyword_based_analysis(symptoms_summary, urgency_level)
    
    @classmethod
    def _analyze_symptom_keywords(cls, symptoms: str) -> Dict[str, float]:
        """Analyze symptoms using keyword matching."""
        
        symptoms_lower = symptoms.lower()
        specialty_scores = {}
        
        for specialty, keywords in cls.SPECIALTY_KEYWORDS.items():
            score = 0
            matches = []
            
            for keyword in keywords:
                if keyword in symptoms_lower:
                    score += 1
                    matches.append(keyword)
            
            if score > 0:
                # Normalize score by number of keywords for the specialty
                normalized_score = min(score / len(keywords) * 2, 1.0)
                specialty_scores[specialty] = {
                    "score": normalized_score,
                    "matches": matches,
                    "match_count": score
                }
        
        return specialty_scores
    
    @classmethod
    def _combine_analyses(cls, ai_analysis: Dict, keyword_analysis: Dict) -> Dict[str, Any]:
        """Combine AI and keyword-based analyses."""
        
        # Start with AI analysis as base
        combined = ai_analysis.copy()
        
        # Enhance primary specialists with keyword scores
        ai_specialists = {spec["specialty"]: spec for spec in combined.get("primary_specialists", [])}
        
        # Add keyword-based specialists not found by AI
        for specialty, data in keyword_analysis.items():
            if specialty not in ai_specialists and data["score"] > 0.3:
                combined.setdefault("primary_specialists", []).append({
                    "specialty": specialty,
                    "confidence": data["score"],
                    "reasoning": f"Keyword matches: {', '.join(data['matches'][:3])}"
                })
        
        # Sort by confidence
        if "primary_specialists" in combined:
            combined["primary_specialists"] = sorted(
                combined["primary_specialists"],
                key=lambda x: x["confidence"],
                reverse=True
            )[:5]  # Top 5 specialists
        
        return combined
    
    @classmethod
    async def _keyword_based_analysis(cls, symptoms: str, urgency: str) -> Dict[str, Any]:
        """Fallback keyword-based analysis."""
        
        keyword_matches = cls._analyze_symptom_keywords(symptoms)
        
        # Convert to expected format
        primary_specialists = []
        for specialty, data in sorted(keyword_matches.items(), key=lambda x: x[1]["score"], reverse=True)[:3]:
            primary_specialists.append({
                "specialty": specialty,
                "confidence": data["score"],
                "reasoning": f"Keyword analysis: {', '.join(data['matches'][:2])}"
            })
        
        # Default urgency assessment
        urgency_map = {
            "low": {"level": "low", "time_frame": "within 1 week"},
            "medium": {"level": "medium", "time_frame": "within 2-3 days"},
            "high": {"level": "high", "time_frame": "within 24 hours"},
            "critical": {"level": "critical", "time_frame": "immediately"}
        }
        
        return {
            "primary_specialists": primary_specialists,
            "secondary_specialists": [],
            "urgency_assessment": {
                **urgency_map.get(urgency, urgency_map["medium"]),
                "reasoning": "Based on reported urgency level",
                "red_flags": []
            },
            "overall_confidence": 0.6,
            "triage_notes": "Analysis based on keyword matching"
        }
    
    @classmethod
    async def find_matching_doctors(
        cls,
        specialist_analysis: Dict[str, Any],
        location: Optional[str] = None,
        max_budget: Optional[float] = None,
        consultation_type: str = "video"
    ) -> List[Dict[str, Any]]:
        """Find doctors matching the specialist requirements."""
        
        try:
            matching_doctors = []
            
            # Get primary specialists
            primary_specialists = specialist_analysis.get("primary_specialists", [])
            
            for spec_info in primary_specialists[:3]:  # Top 3 specialties
                specialty = spec_info["specialty"]
                confidence = spec_info["confidence"]
                
                # Build query conditions
                where_conditions = [
                    "d.is_verified = true",
                    "d.is_accepting_patients = true", 
                    "u.is_active = true"
                ]
                params = []
                param_count = 0
                
                # Specialty matching
                param_count += 1
                where_conditions.append(f"(d.specialization ILIKE ${param_count} OR ${param_count} = ANY(d.sub_specializations))")
                params.append(f"%{specialty}%")
                
                # Budget filter
                if max_budget:
                    param_count += 1
                    where_conditions.append(f"d.base_consultation_fee <= ${param_count}")
                    params.append(max_budget)
                
                # Location filter (simplified - would use geolocation in production)
                if location:
                    param_count += 1
                    where_conditions.append(f"d.location ILIKE ${param_count}")
                    params.append(f"%{location}%")
                
                # Query doctors
                query = f"""
                    SELECT d.id, d.user_id, u.first_name, u.last_name, d.specialization,
                           d.sub_specializations, d.years_experience, d.rating, d.total_reviews,
                           d.base_consultation_fee, d.bio, d.languages, d.is_verified,
                           d.is_accepting_patients, d.location, d.consultation_types
                    FROM doctors d
                    JOIN users u ON d.user_id = u.id
                    WHERE {' AND '.join(where_conditions)}
                    ORDER BY d.rating DESC, d.total_reviews DESC
                    LIMIT 10
                """
                
                doctors = await DatabaseService.execute_query(query, *params, fetch_all=True)
                
                # Add match metadata
                for doctor in doctors:
                    doctor_dict = dict(doctor)
                    doctor_dict.update({
                        "match_specialty": specialty,
                        "match_confidence": confidence,
                        "match_reasoning": spec_info.get("reasoning", ""),
                        "consultation_available": consultation_type in doctor_dict.get("consultation_types", ["video"])
                    })
                    matching_doctors.append(doctor_dict)
            
            # Remove duplicates and sort by overall match quality
            unique_doctors = {}
            for doctor in matching_doctors:
                doctor_id = doctor["id"]
                if doctor_id not in unique_doctors or doctor["match_confidence"] > unique_doctors[doctor_id]["match_confidence"]:
                    unique_doctors[doctor_id] = doctor
            
            # Sort by match quality and rating
            sorted_doctors = sorted(
                unique_doctors.values(),
                key=lambda x: (x["match_confidence"], x["rating"], x["total_reviews"]),
                reverse=True
            )
            
            logger.info(
                "Doctor matching completed",
                total_matches=len(sorted_doctors),
                specialties_searched=len(primary_specialists)
            )
            
            return sorted_doctors[:15]  # Top 15 matches
            
        except Exception as e:
            logger.error("Doctor matching failed", error=str(e))
            return []
    
    @classmethod
    async def calculate_bid_recommendations(
        cls,
        appointment_id: str,
        doctor_id: str,
        urgency_level: str,
        consultation_type: str
    ) -> Dict[str, Any]:
        """Calculate recommended bid amount and duration for doctor."""
        
        try:
            # Get doctor's base fee and stats
            doctor_stats = await DatabaseService.execute_query(
                """
                SELECT d.base_consultation_fee, d.rating, d.total_reviews, d.years_experience,
                       AVG(ab.bid_amount) as avg_bid, COUNT(ab.id) as total_bids,
                       COUNT(CASE WHEN ab.is_selected THEN 1 END) as successful_bids
                FROM doctors d
                LEFT JOIN appointment_bids ab ON d.id = ab.doctor_id
                WHERE d.id = $1
                GROUP BY d.id, d.base_consultation_fee, d.rating, d.total_reviews, d.years_experience
                """,
                doctor_id,
                fetch_one=True
            )
            
            if not doctor_stats:
                return {"error": "Doctor not found"}
            
            base_fee = float(doctor_stats["base_consultation_fee"])
            
            # Get market rates for similar appointments
            market_data = await DatabaseService.execute_query(
                """
                SELECT AVG(ab.bid_amount) as market_avg, MIN(ab.bid_amount) as market_min,
                       MAX(ab.bid_amount) as market_max, COUNT(*) as sample_size
                FROM appointment_bids ab
                JOIN appointments a ON ab.appointment_id = a.id
                WHERE a.urgency_level = $1 AND a.consultation_type = $2
                AND ab.created_at > NOW() - INTERVAL '30 days'
                """,
                urgency_level,
                consultation_type,
                fetch_one=True
            )
            
            # Calculate recommendations
            market_avg = float(market_data["market_avg"] or base_fee)
            market_min = float(market_data["market_min"] or base_fee * 0.8)
            market_max = float(market_data["market_max"] or base_fee * 1.5)
            
            # Urgency multipliers
            urgency_multipliers = {
                "low": 0.9,
                "medium": 1.0,
                "high": 1.2,
                "critical": 1.5
            }
            
            urgency_multiplier = urgency_multipliers.get(urgency_level, 1.0)
            
            # Experience bonus
            experience_bonus = min(doctor_stats["years_experience"] * 0.02, 0.2)  # Max 20% bonus
            
            # Rating bonus
            rating_bonus = max((doctor_stats["rating"] - 3.0) * 0.1, 0)  # Bonus for rating > 3.0
            
            # Success rate factor
            success_rate = (
                doctor_stats["successful_bids"] / max(doctor_stats["total_bids"], 1)
                if doctor_stats["total_bids"] else 0.5
            )
            
            # Calculate recommended bid
            base_recommendation = base_fee * urgency_multiplier * (1 + experience_bonus + rating_bonus)
            
            # Adjust based on market conditions
            if market_avg > 0:
                market_factor = min(market_avg / base_fee, 1.5)  # Don't go crazy with market rates
                base_recommendation = (base_recommendation + market_avg * market_factor) / 2
            
            # Provide range
            competitive_bid = base_recommendation * 0.9  # 10% below recommendation
            premium_bid = base_recommendation * 1.1     # 10% above recommendation
            
            # Duration recommendations
            duration_map = {
                "low": 30,
                "medium": 45,
                "high": 60,
                "critical": 90
            }
            
            recommended_duration = duration_map.get(urgency_level, 45)
            
            return {
                "recommended_bid": round(base_recommendation, 2),
                "competitive_bid": round(competitive_bid, 2),
                "premium_bid": round(premium_bid, 2),
                "market_range": {
                    "min": round(market_min, 2),
                    "avg": round(market_avg, 2),
                    "max": round(market_max, 2)
                },
                "recommended_duration": recommended_duration,
                "success_rate": round(success_rate, 2),
                "factors": {
                    "urgency_multiplier": urgency_multiplier,
                    "experience_bonus": round(experience_bonus, 2),
                    "rating_bonus": round(rating_bonus, 2),
                    "market_factor": round(market_avg / base_fee if market_avg > 0 else 1.0, 2)
                },
                "market_sample_size": market_data["sample_size"]
            }
            
        except Exception as e:
            logger.error("Bid calculation failed", doctor_id=doctor_id, error=str(e))
            return {"error": "Failed to calculate bid recommendations"}
    
    @classmethod
    async def get_appointment_analytics(
        cls,
        appointment_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive analytics for an appointment."""
        
        try:
            # Get appointment details
            appointment = await DatabaseService.execute_query(
                """
                SELECT a.*, u.first_name || ' ' || u.last_name as patient_name,
                       u.date_of_birth, u.gender
                FROM appointments a
                JOIN users u ON a.patient_id = u.id
                WHERE a.id = $1
                """,
                appointment_id,
                fetch_one=True
            )
            
            if not appointment:
                return {"error": "Appointment not found"}
            
            # Get bid statistics
            bid_stats = await DatabaseService.execute_query(
                """
                SELECT COUNT(*) as total_bids, AVG(bid_amount) as avg_bid,
                       MIN(bid_amount) as min_bid, MAX(bid_amount) as max_bid,
                       AVG(estimated_duration) as avg_duration
                FROM appointment_bids
                WHERE appointment_id = $1
                """,
                appointment_id,
                fetch_one=True
            )
            
            # Get bidding doctors info
            bidding_doctors = await DatabaseService.execute_query(
                """
                SELECT ab.bid_amount, ab.estimated_duration, ab.created_at,
                       d.specialization, d.rating, d.years_experience,
                       u.first_name || ' ' || u.last_name as doctor_name
                FROM appointment_bids ab
                JOIN doctors d ON ab.doctor_id = d.id
                JOIN users u ON d.user_id = u.id
                WHERE ab.appointment_id = $1
                ORDER BY ab.bid_amount ASC
                """,
                appointment_id,
                fetch_all=True
            )
            
            # Calculate patient age if DOB available
            patient_age = None
            if appointment["date_of_birth"]:
                today = datetime.now().date()
                birth_date = appointment["date_of_birth"]
                patient_age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            return {
                "appointment": {
                    "id": appointment["id"],
                    "patient_name": appointment["patient_name"],
                    "patient_age": patient_age,
                    "patient_gender": appointment["gender"],
                    "symptoms": appointment["symptoms_summary"],
                    "urgency": appointment["urgency_level"],
                    "consultation_type": appointment["consultation_type"],
                    "status": appointment["status"],
                    "created_at": appointment["created_at"]
                },
                "bidding_stats": {
                    "total_bids": bid_stats["total_bids"],
                    "average_bid": round(float(bid_stats["avg_bid"] or 0), 2),
                    "bid_range": {
                        "min": round(float(bid_stats["min_bid"] or 0), 2),
                        "max": round(float(bid_stats["max_bid"] or 0), 2)
                    },
                    "average_duration": round(float(bid_stats["avg_duration"] or 0), 0)
                },
                "bidding_doctors": [
                    {
                        "doctor_name": doctor["doctor_name"],
                        "specialization": doctor["specialization"],
                        "rating": float(doctor["rating"]),
                        "experience": doctor["years_experience"],
                        "bid_amount": float(doctor["bid_amount"]),
                        "duration": doctor["estimated_duration"],
                        "bid_time": doctor["created_at"]
                    }
                    for doctor in bidding_doctors
                ]
            }
            
        except Exception as e:
            logger.error("Analytics calculation failed", appointment_id=appointment_id, error=str(e))
            return {"error": "Failed to calculate analytics"}