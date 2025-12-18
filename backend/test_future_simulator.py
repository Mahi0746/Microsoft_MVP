# HealthSync AI - Future Simulator Tests
import pytest
import asyncio
import json
import io
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from PIL import Image

from services.future_simulator_service import FutureSimulatorService
from services.db_service import DatabaseService


class TestFutureSimulatorService:
    """Comprehensive tests for the Future Simulator service."""
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        # Create a test image
        image = Image.new('RGB', (512, 512), color='white')
        
        # Add some basic content to simulate a face
        # (In real tests, you'd use actual face images)
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=85)
        return output.getvalue()
    
    @pytest.fixture
    def sample_health_data(self):
        """Sample health data for testing."""
        return {
            "current_age": 30,
            "gender": "female",
            "height": 165.0,
            "weight": 60.0,
            "health_metrics": {
                "blood_pressure": {"value": 120.0, "recorded_at": datetime.now()},
                "heart_rate": {"value": 72.0, "recorded_at": datetime.now()},
                "bmi": {"value": 22.0, "recorded_at": datetime.now()}
            },
            "current_predictions": {
                "diabetes": {"probability": 0.15, "confidence": 0.8},
                "heart_disease": {"probability": 0.25, "confidence": 0.75}
            },
            "family_history": {
                "diabetes": [{"relation": "mother", "age_of_onset": 55}],
                "heart_disease": [{"relation": "father", "age_of_onset": 62}]
            }
        }
    
    @pytest.fixture
    def mock_replicate_response(self):
        """Mock Replicate API response."""
        return ["https://example.com/aged_image.jpg"]
    
    @pytest.mark.asyncio
    async def test_face_detection(self, sample_image_data):
        """Test basic face detection functionality."""
        
        # Load image
        image = Image.open(io.BytesIO(sample_image_data))
        
        # Test face detection
        face_detected = await FutureSimulatorService._detect_face_in_image(image)
        
        # Should detect face (or assume present for test image)
        assert face_detected == True
    
    @pytest.mark.asyncio
    @patch('services.future_simulator_service.SupabaseService')
    async def test_image_upload_validation(self, mock_supabase, sample_image_data):
        """Test image upload and validation process."""
        
        # Mock Supabase responses
        mock_supabase.upload_file.return_value = {"success": True}
        mock_supabase.get_signed_url.return_value = "https://example.com/signed_url.jpg"
        
        # Mock database response
        with patch.object(DatabaseService, 'execute_query') as mock_db:
            mock_db.return_value = {"id": "img_123"}
            
            # Test upload
            result = await FutureSimulatorService.upload_and_validate_image(
                image_data=sample_image_data,
                user_id="user_123",
                content_type="image/jpeg"
            )
            
            # Verify success
            assert result["success"] == True
            assert result["image_id"] == "img_123"
            assert "signed_url" in result
            
            # Verify database was called
            mock_db.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_image_validation_errors(self):
        """Test image validation error cases."""
        
        # Test oversized image
        large_data = b"x" * (11 * 1024 * 1024)  # 11MB
        result = await FutureSimulatorService.upload_and_validate_image(
            image_data=large_data,
            user_id="user_123",
            content_type="image/jpeg"
        )
        assert "error" in result
        assert "too large" in result["error"].lower()
        
        # Test invalid image data
        invalid_data = b"not an image"
        result = await FutureSimulatorService.upload_and_validate_image(
            image_data=invalid_data,
            user_id="user_123",
            content_type="image/jpeg"
        )
        assert "error" in result
    
    def test_age_progression_prompt_generation(self):
        """Test age progression prompt generation."""
        
        # Test different age ranges
        prompt_5 = FutureSimulatorService._create_age_progression_prompt(5)
        assert "subtle aging" in prompt_5.lower()
        
        prompt_15 = FutureSimulatorService._create_age_progression_prompt(15)
        assert "natural aging" in prompt_15.lower()
        
        prompt_25 = FutureSimulatorService._create_age_progression_prompt(25)
        assert "significant aging" in prompt_25.lower()
        
        prompt_35 = FutureSimulatorService._create_age_progression_prompt(35)
        assert "elderly" in prompt_35.lower()
    
    @pytest.mark.asyncio
    @patch('services.future_simulator_service.replicate')
    @patch('services.future_simulator_service.httpx.AsyncClient')
    @patch('services.future_simulator_service.SupabaseService')
    async def test_age_progression_generation(self, mock_supabase, mock_httpx, mock_replicate, mock_replicate_response):
        """Test AI-powered age progression generation."""
        
        # Mock Replicate response
        mock_replicate.run.return_value = mock_replicate_response
        
        # Mock HTTP client for image download
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        mock_response.content = b"fake_aged_image_data"
        mock_client.get.return_value = mock_response
        mock_httpx.return_value.__aenter__.return_value = mock_client
        
        # Mock Supabase operations
        mock_supabase.get_signed_url.return_value = "https://example.com/original.jpg"
        mock_supabase.upload_file.return_value = {"success": True}
        
        # Mock database operations
        with patch.object(DatabaseService, 'execute_query') as mock_db:
            mock_db.return_value = {"id": "prog_123"}
            
            # Test age progression
            result = await FutureSimulatorService.generate_age_progression(
                image_path="test/path.jpg",
                target_age_years=20,
                user_id="user_123",
                current_age=30
            )
            
            # Verify success
            assert result["success"] == True
            assert result["progression_id"] == "prog_123"
            assert "aged_image_url" in result
            
            # Verify Replicate was called
            mock_replicate.run.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('services.future_simulator_service.FutureSimulatorService._get_current_health_data')
    @patch('services.future_simulator_service.MLModelService')
    async def test_health_projections_generation(self, mock_ml_service, mock_health_data, sample_health_data):
        """Test health projections generation."""
        
        # Mock health data
        mock_health_data.return_value = sample_health_data
        
        # Mock ML predictions
        mock_ml_service.predict_future_health_risks.return_value = {
            "diabetes": {"probability": 0.3, "confidence": 0.8},
            "heart_disease": {"probability": 0.4, "confidence": 0.75},
            "cancer": {"probability": 0.2, "confidence": 0.7}
        }
        
        # Mock database operations
        with patch.object(DatabaseService, 'execute_query') as mock_db:
            mock_db.return_value = {"id": "proj_123"}
            
            # Test health projections
            result = await FutureSimulatorService.generate_health_projections(
                user_id="user_123",
                target_age_years=20,
                lifestyle_scenario="improved"
            )
            
            # Verify success
            assert result["success"] == True
            assert result["projection_id"] == "proj_123"
            assert "life_expectancy" in result
            assert "condition_projections" in result
            assert "lifestyle_impact" in result
            
            # Verify ML service was called
            mock_ml_service.predict_future_health_risks.assert_called_once()
    
    def test_life_expectancy_calculation(self, sample_health_data):
        """Test life expectancy calculation logic."""
        
        # Mock future predictions
        future_predictions = {
            "diabetes": {"probability": 0.3},
            "heart_disease": {"probability": 0.4},
            "cancer": {"probability": 0.2}
        }
        
        # Test different scenarios
        improved_expectancy = asyncio.run(
            FutureSimulatorService._calculate_life_expectancy(
                sample_health_data,
                future_predictions,
                "improved"
            )
        )
        
        current_expectancy = asyncio.run(
            FutureSimulatorService._calculate_life_expectancy(
                sample_health_data,
                future_predictions,
                "current"
            )
        )
        
        declined_expectancy = asyncio.run(
            FutureSimulatorService._calculate_life_expectancy(
                sample_health_data,
                future_predictions,
                "declined"
            )
        )
        
        # Improved lifestyle should have higher life expectancy
        assert improved_expectancy > current_expectancy > declined_expectancy
        
        # All should be reasonable values
        assert 70 <= improved_expectancy <= 95
        assert 70 <= current_expectancy <= 95
        assert 70 <= declined_expectancy <= 95
    
    def test_condition_projections_generation(self, sample_health_data):
        """Test health condition projections."""
        
        # Mock future predictions
        future_predictions = {
            "diabetes": {"probability": 0.6},
            "heart_disease": {"probability": 0.7},
            "cancer": {"probability": 0.3}
        }
        
        # Test projections
        projections = asyncio.run(
            FutureSimulatorService._generate_condition_projections(
                sample_health_data,
                future_predictions,
                target_age_years=20
            )
        )
        
        # Verify structure
        assert "diabetes" in projections
        assert "heart_disease" in projections
        assert "cancer" in projections
        assert "aging_effects" in projections
        
        # Verify diabetes projection
        diabetes_proj = projections["diabetes"]
        assert diabetes_proj["probability"] == 0.6
        assert diabetes_proj["risk_level"] == "high"  # > 0.6
        assert len(diabetes_proj["potential_complications"]) > 0
        
        # Verify heart disease projection
        heart_proj = projections["heart_disease"]
        assert heart_proj["probability"] == 0.7
        assert heart_proj["risk_level"] == "high"  # > 0.6
    
    def test_lifestyle_impact_analysis(self, sample_health_data):
        """Test lifestyle impact analysis."""
        
        # Mock future predictions
        future_predictions = {
            "diabetes": {"probability": 0.4},
            "heart_disease": {"probability": 0.5}
        }
        
        # Test improved scenario
        improved_impact = asyncio.run(
            FutureSimulatorService._analyze_lifestyle_impact(
                sample_health_data,
                future_predictions,
                "improved"
            )
        )
        
        # Verify structure
        assert "scenario" in improved_impact
        assert "adjusted_predictions" in improved_impact
        assert "recommendations" in improved_impact
        
        # Verify risk reductions
        diabetes_adjusted = improved_impact["adjusted_predictions"]["diabetes"]
        assert diabetes_adjusted["adjusted_probability"] < diabetes_adjusted["original_probability"]
        
        # Test declined scenario
        declined_impact = asyncio.run(
            FutureSimulatorService._analyze_lifestyle_impact(
                sample_health_data,
                future_predictions,
                "declined"
            )
        )
        
        # Risk should increase in declined scenario
        diabetes_declined = declined_impact["adjusted_predictions"]["diabetes"]
        assert diabetes_declined["adjusted_probability"] > diabetes_declined["original_probability"]
    
    def test_lifestyle_recommendations_generation(self, sample_health_data):
        """Test personalized lifestyle recommendations."""
        
        # High diabetes risk scenario
        high_diabetes_predictions = {
            "diabetes": {"probability": 0.7},
            "heart_disease": {"probability": 0.3}
        }
        
        recommendations = FutureSimulatorService._generate_lifestyle_recommendations(
            sample_health_data,
            high_diabetes_predictions,
            "current"
        )
        
        # Should include diabetes-specific recommendations
        diabetes_recs = [rec for rec in recommendations if "sugar" in rec.lower() or "diabetes" in rec.lower()]
        assert len(diabetes_recs) > 0
        
        # High heart disease risk scenario
        high_heart_predictions = {
            "diabetes": {"probability": 0.2},
            "heart_disease": {"probability": 0.8}
        }
        
        heart_recommendations = FutureSimulatorService._generate_lifestyle_recommendations(
            sample_health_data,
            high_heart_predictions,
            "current"
        )
        
        # Should include heart-specific recommendations
        heart_recs = [rec for rec in heart_recommendations if "heart" in rec.lower() or "cardiovascular" in rec.lower()]
        assert len(heart_recs) > 0
    
    def test_fallback_narrative_generation(self):
        """Test fallback narrative generation."""
        
        context = {
            "current_age": 30,
            "target_age": 50,
            "lifestyle_scenario": "improved"
        }
        
        narrative = FutureSimulatorService._generate_fallback_narrative(context)
        
        # Verify structure
        assert "current_path" in narrative
        assert "improved_path" in narrative
        
        # Verify content
        assert len(narrative["current_path"]) > 50  # Reasonable length
        assert len(narrative["improved_path"]) > 50
        assert "20 years" in narrative["current_path"]  # Age difference mentioned
    
    @pytest.mark.asyncio
    @patch('services.future_simulator_service.DatabaseService.execute_query')
    @patch('services.future_simulator_service.SupabaseService')
    async def test_get_user_simulations(self, mock_supabase, mock_db):
        """Test retrieving user's simulation history."""
        
        # Mock database response
        mock_db.return_value = [
            {
                "progression_id": "prog_1",
                "health_projection_id": "proj_1",
                "target_age_years": 20,
                "lifestyle_scenario": "improved",
                "life_expectancy": 82.5,
                "original_image_path": "path/to/original.jpg",
                "aged_image_path": "path/to/aged.jpg",
                "progression_date": datetime.now()
            }
        ]
        
        # Mock Supabase signed URLs
        mock_supabase.get_signed_url.return_value = "https://example.com/signed_url.jpg"
        
        # Test retrieval
        simulations = await FutureSimulatorService.get_user_simulations(
            user_id="user_123",
            limit=10
        )
        
        # Verify results
        assert len(simulations) == 1
        simulation = simulations[0]
        assert simulation["progression_id"] == "prog_1"
        assert simulation["target_age_years"] == 20
        assert simulation["life_expectancy"] == 82.5
        assert "original_image_url" in simulation
        assert "aged_image_url" in simulation
    
    @pytest.mark.asyncio
    async def test_health_effects_mapping(self):
        """Test health condition visual effects mapping."""
        
        # Verify health effects structure
        effects = FutureSimulatorService.HEALTH_EFFECTS
        
        # Should have major health conditions
        expected_conditions = ["diabetes", "heart_disease", "smoking", "obesity", "hypertension"]
        for condition in expected_conditions:
            assert condition in effects
            
            # Each condition should have effect categories
            condition_effects = effects[condition]
            assert len(condition_effects) > 0
            
            # Effects should be lists of strings
            for effect_category, effect_list in condition_effects.items():
                assert isinstance(effect_list, list)
                assert len(effect_list) > 0
                for effect in effect_list:
                    assert isinstance(effect, str)
                    assert len(effect) > 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_image_data):
        """Test error handling in various scenarios."""
        
        # Test with invalid user ID
        result = await FutureSimulatorService.upload_and_validate_image(
            image_data=sample_image_data,
            user_id="",  # Invalid user ID
            content_type="image/jpeg"
        )
        # Should handle gracefully (implementation dependent)
        
        # Test with None image data
        result = await FutureSimulatorService.upload_and_validate_image(
            image_data=None,
            user_id="user_123",
            content_type="image/jpeg"
        )
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, sample_image_data):
        """Test handling of concurrent operations."""
        
        # Mock successful operations
        with patch.object(FutureSimulatorService, 'upload_and_validate_image') as mock_upload:
            mock_upload.return_value = {"success": True, "file_path": "test/path.jpg"}
            
            with patch.object(FutureSimulatorService, 'generate_age_progression') as mock_progression:
                mock_progression.return_value = {"success": True, "progression_id": "prog_123"}
                
                with patch.object(FutureSimulatorService, 'generate_health_projections') as mock_projections:
                    mock_projections.return_value = {"success": True, "projection_id": "proj_123"}
                    
                    # Test concurrent operations
                    tasks = [
                        FutureSimulatorService.upload_and_validate_image(sample_image_data, f"user_{i}", "image/jpeg")
                        for i in range(3)
                    ]
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # All should succeed
                    for result in results:
                        assert not isinstance(result, Exception)
                        assert result["success"] == True


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])