# Simple Marketplace Service Tests (No Config Dependencies)
import pytest
import json
from unittest.mock import MagicMock, patch

# Test the core functionality without config dependencies
def test_keyword_analysis():
    """Test keyword-based symptom analysis without config."""
    
    # Import the class directly to avoid config issues
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    # Mock the config and imports
    with patch.dict('sys.modules', {
        'config': MagicMock(),
        'services.db_service': MagicMock(),
        'services.ai_service': MagicMock()
    }):
        from services.marketplace_service import MarketplaceService
        
        # Test cardiac symptoms
        cardiac_symptoms = "I've been experiencing chest pain and shortness of breath for the past 2 days"
        cardiac_analysis = MarketplaceService._analyze_symptom_keywords(cardiac_symptoms)
        
        assert "cardiology" in cardiac_analysis
        assert cardiac_analysis["cardiology"]["score"] > 0
        assert "chest pain" in cardiac_analysis["cardiology"]["matches"]
        
        # Test dermatology symptoms
        derma_symptoms = "I have a strange rash on my arm that appeared last week"
        derma_analysis = MarketplaceService._analyze_symptom_keywords(derma_symptoms)
        
        assert "dermatology" in derma_analysis
        assert derma_analysis["dermatology"]["score"] > 0
        assert "rash" in derma_analysis["dermatology"]["matches"]
        
        print("✅ Keyword analysis tests passed!")


def test_specialty_keywords():
    """Test that specialty keywords are comprehensive."""
    
    with patch.dict('sys.modules', {
        'config': MagicMock(),
        'services.db_service': MagicMock(),
        'services.ai_service': MagicMock()
    }):
        from services.marketplace_service import MarketplaceService
        
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
        
        print("✅ Specialty keywords test passed!")


def test_combine_analyses():
    """Test combining AI and keyword analyses."""
    
    with patch.dict('sys.modules', {
        'config': MagicMock(),
        'services.db_service': MagicMock(),
        'services.ai_service': MagicMock()
    }):
        from services.marketplace_service import MarketplaceService
        
        # Mock AI analysis
        ai_analysis = {
            "primary_specialists": [
                {
                    "specialty": "cardiology",
                    "confidence": 0.9,
                    "reasoning": "Chest pain indicates cardiac evaluation"
                }
            ],
            "urgency_assessment": {
                "level": "high",
                "reasoning": "Acute symptoms"
            }
        }
        
        # Mock keyword analysis
        keyword_analysis = {
            "cardiology": {
                "score": 0.8,
                "matches": ["chest pain", "heart"],
                "match_count": 2
            },
            "neurology": {
                "score": 0.4,
                "matches": ["dizziness"],
                "match_count": 1
            }
        }
        
        # Test combination
        result = MarketplaceService._combine_analyses(ai_analysis, keyword_analysis)
        
        # Should preserve AI analysis structure
        assert "primary_specialists" in result
        assert "urgency_assessment" in result
        
        # Should add neurology from keyword analysis (score > 0.3)
        specialties = [spec["specialty"] for spec in result["primary_specialists"]]
        assert "neurology" in specialties
        
        print("✅ Combine analyses test passed!")


if __name__ == "__main__":
    test_keyword_analysis()
    test_specialty_keywords()
    test_combine_analyses()
    print("\n🎉 All marketplace tests passed successfully!")