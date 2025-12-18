# HealthSync AI - Future Simulator Service Tests (Simplified)
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
import base64

# Mock the settings to avoid configuration issues
@pytest.fixture(autouse=True)
def mock_settings():
    with patch('services.future_simulator_service.settings') as mock_settings:
        mock_settings.replicate_api_token = "test_token"
        mock_settings.groq_api_key = "test_groq_key"
        yield mock_settings

@pytest.fixture
def mock_replicate():
    with patch('services.future_simulator_service.replicate') as mock_rep:
        yield mock_rep

@pytest.fixture
def mock_groq():
    with patch('services.future_simulator_service.Groq') as mock_groq_class:
        mock_client = Mock()
        mock_groq_class.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_supabase():
    with patch('services.future_simulator_service.get_supabase_client') as mock_supabase:
        mock_client = Mock()
        mock_supabase.return_value = mock_client
        yield mock_client

class TestFutureSimulatorService:
    """Test suite for Future Simulator Service"""
    
    @pytest.fixture
    def service(self, mock_replicate, mock_groq, mock_supabase):
        """Create service instance with mocked dependencies"""
        from services.future_simulator_service import FutureSimulatorService
        return FutureSimulatorService()
    
    @pytest.mark.asyncio
    async def test_validate_image_success(self, service):
        """Test successful image validation"""
        # Create a mock image file
        mock_image_data = b"fake_image_data"
        
        with patch('services.future_simulator_service.Image.open') as mock_image_open:
            mock_image = Mock()
            mock_image.format = "JPEG"
            mock_image.size = (800, 600)
            mock_image_open.return_value = mock_image
            
            result = await service.validate_image(mock_image_data)
            
            assert result["valid"] is True
            assert result["format"] == "JPEG"
            assert result["dimensions"] == (800, 600)
    
    @pytest.mark.asyncio
    async def test_validate_image_invalid_format(self, service):
        """Test image validation with invalid format"""
        mock_image_data = b"fake_image_data"
        
        with patch('services.future_simulator_service.Image.open') as mock_image_open:
            mock_image = Mock()
            mock_image.format = "BMP"  # Invalid format
            mock_image.size = (800, 600)
            mock_image_open.return_value = mock_image
            
            result = await service.validate_image(mock_image_data)
            
            assert result["valid"] is False
            assert "Invalid image format" in result["error"]
    
    @pytest.mark.asyncio
    async def test_validate_image_too_large(self, service):
        """Test image validation with oversized image"""
        mock_image_data = b"fake_image_data"
        
        with patch('services.future_simulator_service.Image.open') as mock_image_open:
            mock_image = Mock()
            mock_image.format = "JPEG"
            mock_image.size = (5000, 5000)  # Too large
            mock_image_open.return_value = mock_image
            
            result = await service.validate_image(mock_image_data)
            
            assert result["valid"] is False
            assert "Image dimensions too large" in result["error"]
    
    @pytest.mark.asyncio
    async def test_generate_age_progression_success(self, service, mock_replicate):
        """Test successful age progression generation"""
        # Mock Replicate API response
        mock_replicate.run.return_value = ["https://example.com/aged_image.jpg"]
        
        image_data = b"fake_image_data"
        target_age = 65
        
        result = await service.generate_age_progression(image_data, target_age)
        
        assert result["success"] is True
        assert result["aged_image_url"] == "https://example.com/aged_image.jpg"
        assert result["target_age"] == target_age
        
        # Verify Replicate was called with correct parameters
        mock_replicate.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_age_progression_failure(self, service, mock_replicate):
        """Test age progression generation failure"""
        # Mock Replicate API failure
        mock_replicate.run.side_effect = Exception("API Error")
        
        image_data = b"fake_image_data"
        target_age = 65
        
        result = await service.generate_age_progression(image_data, target_age)
        
        assert result["success"] is False
        assert "Failed to generate age progression" in result["error"]
    
    @pytest.mark.asyncio
    async def test_predict_health_outcomes(self, service):
        """Test health outcome predictions"""
        user_data = {
            "age": 35,
            "gender": "male",
            "height": 175,
            "weight": 80,
            "lifestyle": "moderate",
            "medical_history": ["hypertension"]
        }
        
        result = await service.predict_health_outcomes(user_data, years_ahead=30)
        
        assert "cardiovascular_risk" in result
        assert "diabetes_risk" in result
        assert "life_expectancy" in result
        assert "health_score" in result
        
        # Verify all values are within expected ranges
        assert 0 <= result["cardiovascular_risk"] <= 100
        assert 0 <= result["diabetes_risk"] <= 100
        assert 0 <= result["health_score"] <= 100
    
    @pytest.mark.asyncio
    async def test_generate_lifestyle_scenarios(self, service):
        """Test lifestyle scenario generation"""
        base_health = {
            "cardiovascular_risk": 25,
            "diabetes_risk": 15,
            "life_expectancy": 78,
            "health_score": 75
        }
        
        scenarios = await service.generate_lifestyle_scenarios(base_health)
        
        assert "improved" in scenarios
        assert "current" in scenarios
        assert "declined" in scenarios
        
        # Verify improved scenario is better than current
        assert scenarios["improved"]["health_score"] > scenarios["current"]["health_score"]
        assert scenarios["improved"]["life_expectancy"] > scenarios["current"]["life_expectancy"]
        
        # Verify declined scenario is worse than current
        assert scenarios["declined"]["health_score"] < scenarios["current"]["health_score"]
        assert scenarios["declined"]["life_expectancy"] < scenarios["current"]["life_expectancy"]
    
    @pytest.mark.asyncio
    async def test_generate_ai_narrative_success(self, service, mock_groq):
        """Test successful AI narrative generation"""
        # Mock Groq API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "This is a test narrative about your future health."
        
        mock_groq.chat.completions.create.return_value = mock_response
        
        simulation_data = {
            "current_age": 35,
            "target_age": 65,
            "health_predictions": {"health_score": 75},
            "lifestyle_scenarios": {"current": {"health_score": 75}}
        }
        
        result = await service.generate_ai_narrative(simulation_data)
        
        assert result["success"] is True
        assert "narrative" in result
        assert len(result["narrative"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_ai_narrative_failure(self, service, mock_groq):
        """Test AI narrative generation failure"""
        # Mock Groq API failure
        mock_groq.chat.completions.create.side_effect = Exception("API Error")
        
        simulation_data = {
            "current_age": 35,
            "target_age": 65,
            "health_predictions": {"health_score": 75},
            "lifestyle_scenarios": {"current": {"health_score": 75}}
        }
        
        result = await service.generate_ai_narrative(simulation_data)
        
        assert result["success"] is False
        assert "Failed to generate AI narrative" in result["error"]
    
    @pytest.mark.asyncio
    async def test_create_simulation_success(self, service, mock_supabase, mock_replicate, mock_groq):
        """Test successful simulation creation"""
        # Mock all external API calls
        mock_replicate.run.return_value = ["https://example.com/aged_image.jpg"]
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Test narrative"
        mock_groq.chat.completions.create.return_value = mock_response
        
        # Mock Supabase insert
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "test_simulation_id"}
        ]
        
        simulation_data = {
            "user_id": "test_user",
            "image_data": base64.b64encode(b"fake_image_data").decode(),
            "current_age": 35,
            "target_age": 65,
            "lifestyle_factors": {
                "exercise": "moderate",
                "diet": "balanced",
                "smoking": False,
                "alcohol": "moderate"
            }
        }
        
        with patch.object(service, 'validate_image') as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "format": "JPEG",
                "dimensions": (800, 600)
            }
            
            result = await service.create_simulation(simulation_data)
            
            assert result["success"] is True
            assert "simulation_id" in result
            assert "aged_image_url" in result
            assert "health_predictions" in result
            assert "lifestyle_scenarios" in result
            assert "ai_narrative" in result
    
    @pytest.mark.asyncio
    async def test_create_simulation_invalid_image(self, service):
        """Test simulation creation with invalid image"""
        simulation_data = {
            "user_id": "test_user",
            "image_data": base64.b64encode(b"fake_image_data").decode(),
            "current_age": 35,
            "target_age": 65,
            "lifestyle_factors": {}
        }
        
        with patch.object(service, 'validate_image') as mock_validate:
            mock_validate.return_value = {
                "valid": False,
                "error": "Invalid image format"
            }
            
            result = await service.create_simulation(simulation_data)
            
            assert result["success"] is False
            assert "Invalid image format" in result["error"]
    
    @pytest.mark.asyncio
    async def test_get_simulation_success(self, service, mock_supabase):
        """Test successful simulation retrieval"""
        # Mock Supabase select
        mock_simulation_data = {
            "id": "test_simulation_id",
            "user_id": "test_user",
            "aged_image_url": "https://example.com/aged_image.jpg",
            "health_predictions": {"health_score": 75},
            "lifestyle_scenarios": {"current": {"health_score": 75}},
            "ai_narrative": "Test narrative",
            "created_at": datetime.now().isoformat()
        }
        
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = mock_simulation_data
        
        result = await service.get_simulation("test_simulation_id", "test_user")
        
        assert result["success"] is True
        assert result["simulation"]["id"] == "test_simulation_id"
        assert result["simulation"]["aged_image_url"] == "https://example.com/aged_image.jpg"
    
    @pytest.mark.asyncio
    async def test_get_simulation_not_found(self, service, mock_supabase):
        """Test simulation retrieval when not found"""
        # Mock Supabase returning no data
        mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value.data = None
        
        result = await service.get_simulation("nonexistent_id", "test_user")
        
        assert result["success"] is False
        assert "Simulation not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_get_user_simulations(self, service, mock_supabase):
        """Test retrieving user's simulations"""
        # Mock Supabase select
        mock_simulations = [
            {
                "id": "sim1",
                "target_age": 65,
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "sim2", 
                "target_age": 70,
                "created_at": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ]
        
        mock_supabase.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = mock_simulations
        
        result = await service.get_user_simulations("test_user", limit=10)
        
        assert result["success"] is True
        assert len(result["simulations"]) == 2
        assert result["simulations"][0]["id"] == "sim1"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])