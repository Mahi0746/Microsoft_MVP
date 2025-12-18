# HealthSync AI - Doctor Marketplace Tests
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from services.marketplace_service import MarketplaceService
from services.db_service import DatabaseService


class TestMarketplaceService:
    """Comprehensive tests for the doctor marketplace service."""
    
    @pytest.fixture
    def sample_symptoms(self):
        """Sample symptom descriptions for testing."""
        return {
            "cardiac": "I've been experiencing chest pain and shortness of breath for the past 2 days. The pain is sharp and gets worse when I exercise.",
            "dermatology": "I have a strange rash on my arm that appeared last week. It's red, itchy, and seems to be spreading.",
            "neurology": "I've been having severe headaches and dizziness. Sometimes I feel numbness in my left hand.",
            "orthopedic": "My knee has been hurting for weeks after I fell while jogging. It's swollen and I can barely walk.",
            "gastro": "I've been having severe stomach pain, nausea, and diarrhea for 3 days. I can't keep food down.",
            "complex": "I have chest pain, difficulty breathing, stomach issues, and a skin rash. I'm very worried."
        }
    
    @pytest.fixture
    def mock_groq_response(self):
        """Mock Groq API response."""
        return {
            "primary_specialists": [
                {
                    "specialty": "cardiology",
                    "confidence": 0.9,
                    "reasoning": "Chest pain and shortness of breath indicate cardiac evaluation needed"
                },
                {
                    "specialty": "emergency_medicine", 
                    "confidence": 0.8,
                    "reasoning": "Acute symptoms require immediate assessment"
                }
            ],
            "secondary_specialists": [
                {
                    "specialty": "pulmonology",
                    "confidence": 0.6,
                    "reasoning": "Breathing difficulties may indicate respiratory issues"
                }
            ],
            "urgency_assessment": {
                "level": "high",
                "reasoning": "Acute chest pain requires urgent evaluation",
                "time_frame": "within 2 hours",
                "red_flags": ["chest pain", "shortness of breath"]
            },
            "overall_confidence": 0.85,
            "triage_notes": "Patient should be seen urgently due to cardiac symptoms"
        }
    
    @pytest.mark.asyncio
    async def test_keyword_analysis(self, sample_symptoms):
        """Test keyword-based symptom analysis."""
        
        # Test cardiac symptoms
        cardiac_analysis = MarketplaceService._analyze_symptom_keywords(sample_symptoms["cardiac"])
        
        assert "cardiology" in cardiac_analysis
        assert cardiac_analysis["cardiology"]["score"] > 0
        assert "chest pain" in cardiac_analysis["cardiology"]["matches"]
        assert "shortness of breath" in cardiac_analysis["cardiology"]["matches"]
        
        # Test dermatology symptoms
        derma_analysis = MarketplaceService._analyze_symptom_keywords(sample_symptoms["dermatology"])
        
        assert "dermatology" in derma_analysis
        assert derma_analysis["dermatology"]["score"] > 0
        assert "rash" in derma_analysis["dermatology"]["matches"]
        
        # Test complex symptoms (multiple specialties)
        complex_analysis = MarketplaceService._analyze_symptom_keywords(sample_symptoms["complex"])
        
        assert len(complex_analysis) >= 2  # Should match multiple specialties
        assert "cardiology" in complex_analysis
        assert "dermatology" in complex_analysis
    
    @pytest.mark.asyncio
    @patch('services.marketplace_service.Groq')
    async def test_ai_symptom_analysis(self, mock_groq, sample_symptoms, mock_groq_response):
        """Test AI-powered symptom analysis."""
        
        # Mock Groq client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps(mock_groq_response)
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        # Test analysis
        result = await MarketplaceService.analyze_symptoms_for_specialists(
            sample_symptoms["cardiac"],
            urgency_level="high",
            patient_age=45,
            patient_gender="male"
        )
        
        # Verify structure
        assert "primary_specialists" in result
        assert "urgency_assessment" in result
        assert "overall_confidence" in result
        
        # Verify content
        assert len(result["primary_specialists"]) > 0
        assert result["primary_specialists"][0]["specialty"] == "cardiology"
        assert result["urgency_assessment"]["level"] == "high"
        
        # Verify Groq was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert "Age: 45 years old" in call_args[1]["messages"][1]["content"]
        assert "Gender: male" in call_args[1]["messages"][1]["content"]
    
    @pytest.mark.asyncio
    @patch('services.marketplace_service.Groq')
    async def test_fallback_analysis(self, mock_groq, sample_symptoms):
        """Test fallback when AI analysis fails."""
        
        # Mock Groq to raise exception
        mock_groq.side_effect = Exception("API Error")
        
        # Should fall back to keyword analysis
        result = await MarketplaceService.analyze_symptoms_for_specialists(
            sample_symptoms["cardiac"],
            urgency_level="medium"
        )
        
        # Should still return valid structure
        assert "primary_specialists" in result
        assert "urgency_assessment" in result
        assert result["urgency_assessment"]["level"] == "medium"
        
        # Should have cardiology from keyword matching
        specialties = [spec["specialty"] for spec in result["primary_specialists"]]
        assert "cardiology" in specialties
    
    @pytest.mark.asyncio
    @patch('services.db_service.DatabaseService.execute_query')
    async def test_find_matching_doctors(self, mock_db, mock_groq_response):
        """Test doctor matching functionality."""
        
        # Mock database response
        mock_doctors = [
            {
                "id": "doc1",
                "user_id": "user1", 
                "first_name": "John",
                "last_name": "Smith",
                "specialization": "Cardiology",
                "sub_specializations": ["Interventional Cardiology"],
                "years_experience": 15,
                "rating": 4.8,
                "total_reviews": 120,
                "base_consultation_fee": 200.0,
                "bio": "Experienced cardiologist",
                "languages": ["English", "Spanish"],
                "is_verified": True,
                "is_accepting_patients": True,
                "location": "New York",
                "consultation_types": ["video", "in_person"]
            },
            {
                "id": "doc2",
                "user_id": "user2",
                "first_name": "Sarah", 
                "last_name": "Johnson",
                "specialization": "Emergency Medicine",
                "sub_specializations": ["Critical Care"],
                "years_experience": 10,
                "rating": 4.6,
                "total_reviews": 85,
                "base_consultation_fee": 180.0,
                "bio": "Emergency medicine specialist",
                "languages": ["English"],
                "is_verified": True,
                "is_accepting_patients": True,
                "location": "New York", 
                "consultation_types": ["video", "audio"]
            }
        ]
        
        mock_db.return_value = mock_doctors
        
        # Test matching
        result = await MarketplaceService.find_matching_doctors(
            mock_groq_response,
            location="New York",
            max_budget=250.0,
            consultation_type="video"
        )
        
        # Verify results
        assert len(result) == 2
        assert result[0]["match_specialty"] == "cardiology"
        assert result[0]["match_confidence"] == 0.9
        assert result[0]["consultation_available"] == True
        
        # Verify database was called correctly
        mock_db.assert_called()
    
    @pytest.mark.asyncio
    @patch('services.db_service.DatabaseService.execute_query')
    async def test_bid_recommendations(self, mock_db):
        """Test bid recommendation calculations."""
        
        # Mock doctor stats
        mock_db.side_effect = [
            # Doctor stats query
            {
                "base_consultation_fee": 200.0,
                "rating": 4.5,
                "total_reviews": 50,
                "years_experience": 12,
                "avg_bid": 220.0,
                "total_bids": 25,
                "successful_bids": 15
            },
            # Market data query
            {
                "market_avg": 210.0,
                "market_min": 150.0,
                "market_max": 300.0,
                "sample_size": 100
            }
        ]
        
        # Test calculation
        result = await MarketplaceService.calculate_bid_recommendations(
            "appointment123",
            "doctor456", 
            "high",
            "video"
        )
        
        # Verify structure
        assert "recommended_bid" in result
        assert "competitive_bid" in result
        assert "premium_bid" in result
        assert "market_range" in result
        assert "recommended_duration" in result
        assert "factors" in result
        
        # Verify calculations make sense
        assert result["competitive_bid"] < result["recommended_bid"] < result["premium_bid"]
        assert result["recommended_duration"] == 60  # High urgency = 60 minutes
        assert 0 <= result["success_rate"] <= 1
        
        # Verify urgency affects duration
        high_urgency_result = result["recommended_duration"]
        
        # Test low urgency
        mock_db.side_effect = [
            {
                "base_consultation_fee": 200.0,
                "rating": 4.5,
                "total_reviews": 50,
                "years_experience": 12,
                "avg_bid": 220.0,
                "total_bids": 25,
                "successful_bids": 15
            },
            {
                "market_avg": 210.0,
                "market_min": 150.0,
                "market_max": 300.0,
                "sample_size": 100
            }
        ]
        
        low_urgency_result = await MarketplaceService.calculate_bid_recommendations(
            "appointment123",
            "doctor456",
            "low", 
            "video"
        )
        
        assert low_urgency_result["recommended_duration"] == 30  # Low urgency = 30 minutes
        assert low_urgency_result["recommended_bid"] < result["recommended_bid"]  # Lower urgency = lower bid
    
    @pytest.mark.asyncio
    @patch('services.db_service.DatabaseService.execute_query')
    async def test_appointment_analytics(self, mock_db):
        """Test appointment analytics generation."""
        
        # Mock database responses
        mock_db.side_effect = [
            # Appointment details
            {
                "id": "apt123",
                "patient_name": "Jane Doe",
                "date_of_birth": datetime(1985, 5, 15).date(),
                "gender": "female",
                "symptoms_summary": "Chest pain and shortness of breath",
                "urgency_level": "high",
                "consultation_type": "video",
                "status": "bidding",
                "created_at": datetime.now()
            },
            # Bid statistics
            {
                "total_bids": 5,
                "avg_bid": 225.0,
                "min_bid": 180.0,
                "max_bid": 280.0,
                "avg_duration": 50.0
            },
            # Bidding doctors
            [
                {
                    "doctor_name": "Dr. Smith",
                    "specialization": "Cardiology",
                    "rating": 4.8,
                    "years_experience": 15,
                    "bid_amount": 250.0,
                    "estimated_duration": 60,
                    "created_at": datetime.now()
                },
                {
                    "doctor_name": "Dr. Johnson", 
                    "specialization": "Emergency Medicine",
                    "rating": 4.6,
                    "years_experience": 10,
                    "bid_amount": 200.0,
                    "estimated_duration": 45,
                    "created_at": datetime.now()
                }
            ]
        ]
        
        # Test analytics
        result = await MarketplaceService.get_appointment_analytics("apt123")
        
        # Verify structure
        assert "appointment" in result
        assert "bidding_stats" in result
        assert "bidding_doctors" in result
        
        # Verify appointment data
        appointment = result["appointment"]
        assert appointment["patient_name"] == "Jane Doe"
        assert appointment["patient_age"] == 39  # Calculated from DOB
        assert appointment["urgency"] == "high"
        
        # Verify bidding stats
        stats = result["bidding_stats"]
        assert stats["total_bids"] == 5
        assert stats["average_bid"] == 225.0
        assert stats["bid_range"]["min"] == 180.0
        assert stats["bid_range"]["max"] == 280.0
        
        # Verify doctor data
        doctors = result["bidding_doctors"]
        assert len(doctors) == 2
        assert doctors[0]["doctor_name"] == "Dr. Smith"
        assert doctors[0]["specialization"] == "Cardiology"
    
    @pytest.mark.asyncio
    async def test_specialty_keyword_coverage(self):
        """Test that specialty keywords cover common medical terms."""
        
        # Test coverage of major medical specialties
        specialties = MarketplaceService.SPECIALTY_KEYWORDS
        
        # Should have major specialties
        expected_specialties = [
            "cardiology", "dermatology", "neurology", "orthopedics",
            "gastroenterology", "pulmonology", "endocrinology", 
            "psychiatry", "gynecology", "urology"
        ]
        
        for specialty in expected_specialties:
            assert specialty in specialties
            assert len(specialties[specialty]) >= 5  # At least 5 keywords per specialty
        
        # Test common symptoms are covered
        common_symptoms = [
            "chest pain", "headache", "rash", "back pain", 
            "stomach pain", "cough", "anxiety", "diabetes"
        ]
        
        for symptom in common_symptoms:
            found = False
            for specialty_keywords in specialties.values():
                if any(symptom in keyword for keyword in specialty_keywords):
                    found = True
                    break
            assert found, f"Symptom '{symptom}' not covered in any specialty keywords"
    
    @pytest.mark.asyncio
    async def test_urgency_level_handling(self, sample_symptoms):
        """Test different urgency levels are handled correctly."""
        
        urgency_levels = ["low", "medium", "high", "critical"]
        
        for urgency in urgency_levels:
            result = await MarketplaceService._keyword_based_analysis(
                sample_symptoms["cardiac"],
                urgency
            )
            
            # Should return valid structure for all urgency levels
            assert "urgency_assessment" in result
            assert result["urgency_assessment"]["level"] == urgency
            
            # Time frames should be appropriate
            time_frame = result["urgency_assessment"]["time_frame"]
            if urgency == "critical":
                assert "immediately" in time_frame
            elif urgency == "high":
                assert "24 hours" in time_frame
            elif urgency == "low":
                assert "week" in time_frame
    
    def test_combine_analyses(self, mock_groq_response):
        """Test combining AI and keyword analyses."""
        
        # Mock keyword analysis
        keyword_analysis = {
            "cardiology": {
                "score": 0.8,
                "matches": ["chest pain", "heart"],
                "match_count": 2
            },
            "pulmonology": {
                "score": 0.6,
                "matches": ["shortness of breath", "breathing"],
                "match_count": 2
            },
            "neurology": {  # Not in AI analysis
                "score": 0.4,
                "matches": ["dizziness"],
                "match_count": 1
            }
        }
        
        # Test combination
        result = MarketplaceService._combine_analyses(mock_groq_response, keyword_analysis)
        
        # Should preserve AI analysis structure
        assert "primary_specialists" in result
        assert "urgency_assessment" in result
        
        # Should add neurology from keyword analysis (score > 0.3)
        specialties = [spec["specialty"] for spec in result["primary_specialists"]]
        assert "neurology" in specialties
        
        # Should be sorted by confidence
        confidences = [spec["confidence"] for spec in result["primary_specialists"]]
        assert confidences == sorted(confidences, reverse=True)
        
        # Should limit to top 5
        assert len(result["primary_specialists"]) <= 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])