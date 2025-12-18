# HealthSync AI - Future-You Simulator API Documentation

## Overview

The Future-You Simulator is an AI-powered feature that combines age progression technology with health projections to show users how they might look and feel in the future based on their current health data and lifestyle choices.

## Key Features

- **AI Age Progression**: Uses Stable Diffusion to generate realistic aged photos
- **Health Projections**: ML-powered predictions for disease risks and life expectancy
- **Lifestyle Scenarios**: Compare outcomes for different lifestyle choices
- **Visual Health Effects**: Map health conditions to visual changes
- **Personalized Narratives**: AI-generated stories about future health
- **Comprehensive Analytics**: Track simulation history and insights

## Authentication

All endpoints require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### 1. Image Upload & Validation

#### Upload Image for Future Simulation
```http
POST /api/v1/future-simulator/upload-image
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with image file

**Parameters:**
- `file` (required): Image file (JPEG, PNG, WebP)
- Maximum size: 10MB
- Minimum dimensions: 256x256 pixels
- Must contain a detectable face

**Response:**
```json
{
  "success": true,
  "image_id": "img_123456",
  "file_path": "future_sim/user_123/abc123_1703001234.jpg",
  "signed_url": "https://supabase.co/storage/v1/object/sign/user-images/...",
  "file_size": 2048576,
  "dimensions": "512x512"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "No face detected in image. Please upload a clear photo of your face."
}
```

### 2. Age Progression

#### Generate Age Progression
```http
POST /api/v1/future-simulator/age-progression
```

**Request Body:**
```json
{
  "image_path": "future_sim/user_123/abc123_1703001234.jpg",
  "target_age_years": 20,
  "current_age": 30
}
```

**Response:**
```json
{
  "success": true,
  "progression_id": "prog_789",
  "original_image_url": "https://supabase.co/storage/v1/object/sign/...",
  "aged_image_url": "https://supabase.co/storage/v1/object/sign/...",
  "target_age_years": 20,
  "generation_prompt": "Age this person by 20 years, significant aging, wrinkles, gray hair, mature features, realistic aging, high quality"
}
```

### 3. Health Projections

#### Generate Health Projections
```http
POST /api/v1/future-simulator/health-projections
```

**Request Body:**
```json
{
  "target_age_years": 20,
  "lifestyle_scenario": "improved"
}
```

**Response:**
```json
{
  "success": true,
  "projection_id": "proj_456",
  "target_age_years": 20,
  "lifestyle_scenario": "improved",
  "life_expectancy": 84.2,
  "condition_projections": {
    "diabetes": {
      "probability": 0.25,
      "risk_level": "low",
      "potential_complications": [],
      "visual_effects": {}
    },
    "heart_disease": {
      "probability": 0.35,
      "risk_level": "medium",
      "potential_complications": [
        "Chest pain",
        "Shortness of breath",
        "Fatigue"
      ],
      "visual_effects": {
        "facial_effects": ["pale_complexion", "fatigue_lines"],
        "physical_effects": ["reduced_vitality"]
      }
    },
    "aging_effects": {
      "probability": 0.8,
      "potential_effects": [
        "Reduced mobility",
        "Joint stiffness",
        "Decreased muscle mass"
      ],
      "visual_effects": {
        "posture_effects": ["slight_stoop"],
        "skin_effects": ["wrinkles", "age_spots"],
        "hair_effects": ["graying", "thinning"]
      }
    }
  },
  "lifestyle_impact": {
    "scenario": "improved",
    "description": "Healthy lifestyle improvements",
    "lifestyle_changes": [
      "Regular exercise (150 min/week)",
      "Balanced diet with reduced processed foods",
      "Adequate sleep (7-9 hours)",
      "Stress management techniques"
    ],
    "adjusted_predictions": {
      "diabetes": {
        "original_probability": 0.4,
        "adjusted_probability": 0.25,
        "change": -0.15
      }
    },
    "recommendations": [
      "Maintain a healthy weight through balanced diet and exercise",
      "Engage in regular cardiovascular exercise",
      "Stay physically active with age-appropriate exercises"
    ]
  },
  "health_narrative": {
    "current_path": "By maintaining your current lifestyle, you may experience typical age-related changes over the next 20 years.",
    "improved_path": "With healthy lifestyle improvements, you could significantly reduce your risk of chronic diseases and maintain better physical and mental health as you age."
  }
}
```

### 4. Complete Future Simulation

#### Generate Complete Simulation
```http
POST /api/v1/future-simulator/complete-simulation
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with image file and JSON parameters

**Parameters:**
- `image_file` (required): Image file for age progression
- `target_age_years` (required): Years to age (5-50)
- `lifestyle_scenario` (optional): "improved", "current", or "declined"
- `current_age` (optional): User's current age

**Response:**
```json
{
  "success": true,
  "simulation_id": "sim_999",
  "age_progression": {
    "success": true,
    "progression_id": "prog_789",
    "original_image_url": "https://...",
    "aged_image_url": "https://...",
    "target_age_years": 20
  },
  "health_projections": {
    "success": true,
    "projection_id": "proj_456",
    "life_expectancy": 84.2,
    "condition_projections": { /* ... */ },
    "lifestyle_impact": { /* ... */ }
  },
  "combined_analysis": {
    "simulation_summary": {
      "target_age_years": 20,
      "lifestyle_scenario": "improved",
      "has_age_progression": true,
      "has_health_projections": true
    },
    "visual_health_effects": [
      {
        "condition": "heart_disease",
        "probability": 0.35,
        "effects": {
          "facial_effects": ["pale_complexion", "fatigue_lines"]
        }
      }
    ],
    "lifestyle_comparison": {
      "current_scenario": "improved",
      "potential_benefits": {
        "diabetes": "Risk reduced by 15.0%",
        "heart_disease": "Risk reduced by 12.0%"
      },
      "key_recommendations": [
        "Maintain a healthy weight through balanced diet and exercise",
        "Engage in regular cardiovascular exercise"
      ]
    },
    "health_score": {
      "overall_score": 78.5,
      "life_expectancy": 84.2,
      "score_interpretation": "Good"
    }
  }
}
```

### 5. Simulation History & Analytics

#### Get Simulation History
```http
GET /api/v1/future-simulator/history?limit=10
```

**Response:**
```json
[
  {
    "progression_id": "prog_789",
    "health_projection_id": "proj_456",
    "target_age_years": 20,
    "lifestyle_scenario": "improved",
    "life_expectancy": 84.2,
    "original_image_url": "https://...",
    "aged_image_url": "https://...",
    "created_at": "2025-12-17T10:30:00Z"
  }
]
```

#### Get Simulation Details
```http
GET /api/v1/future-simulator/simulation/{simulation_id}
```

**Response:**
```json
{
  "simulation_id": "sim_999",
  "target_age_years": 20,
  "lifestyle_scenario": "improved",
  "life_expectancy": 84.2,
  "original_image_url": "https://...",
  "aged_image_url": "https://...",
  "projections_data": {
    "condition_projections": { /* ... */ },
    "lifestyle_impact": { /* ... */ }
  },
  "created_at": "2025-12-17T10:30:00Z"
}
```

### 6. Lifestyle Scenario Comparison

#### Compare Lifestyle Scenarios
```http
POST /api/v1/future-simulator/compare-scenarios?target_age_years=20
```

**Response:**
```json
{
  "target_age_years": 20,
  "scenario_comparisons": {
    "improved": {
      "life_expectancy": 84.2,
      "condition_projections": { /* ... */ },
      "recommendations": [ /* ... */ ]
    },
    "current": {
      "life_expectancy": 81.5,
      "condition_projections": { /* ... */ },
      "recommendations": [ /* ... */ ]
    },
    "declined": {
      "life_expectancy": 78.8,
      "condition_projections": { /* ... */ },
      "recommendations": [ /* ... */ ]
    }
  },
  "scenario_differences": {
    "improvement_benefit": {
      "life_expectancy_gain": 2.7,
      "risk_reductions": {
        "diabetes": {
          "current_risk": 0.4,
          "improved_risk": 0.25,
          "risk_reduction": 0.15
        },
        "heart_disease": {
          "current_risk": 0.5,
          "improved_risk": 0.35,
          "risk_reduction": 0.15
        }
      }
    }
  },
  "recommendation": "Choose the 'improved' lifestyle scenario for the best health outcomes"
}
```

### 7. Personalized Health Insights

#### Get Health Insights
```http
GET /api/v1/future-simulator/health-insights
```

**Response:**
```json
{
  "total_simulations": 5,
  "insights": [
    {
      "type": "life_expectancy_trend",
      "title": "Life Expectancy Trend: Improving",
      "description": "Your projected life expectancy is improving based on recent simulations.",
      "action": "Continue healthy habits"
    },
    {
      "type": "planning_horizon",
      "title": "Planning Focus: 20 Years Ahead",
      "description": "You're most interested in your health 20 years from now.",
      "action": "Consider both short-term and long-term health goals"
    }
  ],
  "recommendations": [
    "Regular simulations help track your health trajectory",
    "Compare different lifestyle scenarios to make informed decisions",
    "Use insights to set realistic health goals"
  ]
}
```

## Data Models

### Lifestyle Scenarios

The system supports three lifestyle scenarios:

1. **Improved**: Healthy lifestyle with regular exercise, balanced diet, adequate sleep
2. **Current**: Continue existing lifestyle patterns
3. **Declined**: Unhealthy lifestyle with poor diet, sedentary behavior, high stress

### Health Conditions Tracked

- **Diabetes**: Type 2 diabetes risk and complications
- **Heart Disease**: Cardiovascular disease risk and symptoms
- **Cancer**: General cancer risk factors
- **Aging Effects**: Natural aging processes and mobility changes
- **Obesity**: Weight-related health impacts
- **Hypertension**: Blood pressure-related conditions

### Visual Health Effects

Each health condition maps to specific visual effects:

```json
{
  "diabetes": {
    "vision_effects": ["blurred_vision", "eye_damage"],
    "skin_effects": ["slow_healing", "infections"],
    "weight_effects": ["weight_gain", "fatigue_appearance"]
  },
  "heart_disease": {
    "facial_effects": ["pale_complexion", "fatigue_lines"],
    "physical_effects": ["reduced_vitality"],
    "posture_effects": ["slouched_posture"]
  }
}
```

## Error Handling

### Common Error Codes

- `400` - Bad Request (invalid parameters, validation errors)
- `401` - Unauthorized (invalid or missing token)
- `404` - Not Found (image or simulation not found)
- `413` - Payload Too Large (image file too large)
- `422` - Unprocessable Entity (validation errors)
- `500` - Internal Server Error (AI service failures, processing errors)

### Error Response Format

```json
{
  "error": true,
  "message": "Detailed error description",
  "status_code": 400,
  "timestamp": "2025-12-17T10:30:00Z",
  "path": "/api/v1/future-simulator/upload-image",
  "request_id": "req_123456"
}
```

### Specific Error Cases

#### Image Upload Errors
```json
{
  "success": false,
  "error": "Image too large",
  "max_size_mb": 10
}
```

#### Face Detection Errors
```json
{
  "success": false,
  "error": "No face detected in image. Please upload a clear photo of your face."
}
```

#### Health Data Errors
```json
{
  "success": false,
  "error": "No health data available for projections"
}
```

## Rate Limits

- Image upload endpoints: 5 requests per minute
- General endpoints: 100 requests per minute
- Comparison endpoints: 30 requests per minute

## Validation Rules

### Image Upload
- File types: JPEG, PNG, WebP
- Maximum size: 10MB
- Minimum dimensions: 256x256 pixels
- Maximum dimensions: 2048x2048 pixels
- Must contain detectable face

### Age Progression
- `target_age_years`: 5-50 years
- `current_age`: Optional, 18-100 years

### Health Projections
- `target_age_years`: 5-50 years
- `lifestyle_scenario`: "improved", "current", or "declined"

## AI Models & Services

### Age Progression
- **Primary**: Stable Diffusion via Replicate API
- **Fallback**: PIL-based image processing
- **Prompt Engineering**: Age-specific prompts for realistic results

### Health Projections
- **ML Models**: scikit-learn models for disease prediction
- **Life Expectancy**: Actuarial calculations with risk adjustments
- **Narrative Generation**: Groq LLM for personalized stories

### Image Processing
- **Face Detection**: Basic heuristic-based detection
- **Image Enhancement**: PIL filters and adjustments
- **Compression**: Automatic optimization for storage

## Security & Privacy

### Data Protection
- Images stored securely in Supabase Storage
- Signed URLs with 1-hour expiration
- User data isolation with RLS policies
- Automatic cleanup of temporary files

### Medical Disclaimers
- All projections include medical disclaimers
- Results are for educational purposes only
- Users advised to consult healthcare professionals
- No medical diagnosis or treatment recommendations

### Privacy Controls
- Users can delete simulation history
- Images can be removed from storage
- Health data anonymized in analytics
- No sharing without explicit consent

## Integration Examples

### JavaScript/React Example
```javascript
// Upload image and generate complete simulation
const generateFutureSimulation = async (imageFile, targetAge) => {
  const formData = new FormData();
  formData.append('image_file', imageFile);
  formData.append('target_age_years', targetAge);
  formData.append('lifestyle_scenario', 'improved');
  
  const response = await fetch('/api/v1/future-simulator/complete-simulation', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`
    },
    body: formData
  });
  
  return await response.json();
};

// Compare lifestyle scenarios
const compareScenarios = async (targetAge) => {
  const response = await fetch(`/api/v1/future-simulator/compare-scenarios?target_age_years=${targetAge}`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    }
  });
  
  return await response.json();
};
```

### Python Example
```python
import requests
from pathlib import Path

class FutureSimulator:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}'
        }
    
    def upload_image(self, image_path):
        url = f'{self.base_url}/api/v1/future-simulator/upload-image'
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, headers=self.headers)
        
        return response.json()
    
    def generate_age_progression(self, image_path, target_age):
        url = f'{self.base_url}/api/v1/future-simulator/age-progression'
        data = {
            'image_path': image_path,
            'target_age_years': target_age
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()
    
    def get_health_projections(self, target_age, scenario='current'):
        url = f'{self.base_url}/api/v1/future-simulator/health-projections'
        data = {
            'target_age_years': target_age,
            'lifestyle_scenario': scenario
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()
```

## Performance Considerations

### Response Times
- Image upload: 2-5 seconds
- Age progression: 10-30 seconds (AI processing)
- Health projections: 1-3 seconds
- Complete simulation: 15-45 seconds

### Optimization Strategies
- Image compression and resizing
- Caching of ML model predictions
- Asynchronous processing for heavy operations
- CDN delivery for generated images

### Scalability
- Horizontal scaling support
- Queue-based processing for AI operations
- Database indexing for fast queries
- Rate limiting to prevent abuse

## Monitoring & Analytics

### Health Metrics
- API response times
- Success/failure rates
- AI model performance
- User engagement metrics

### Error Tracking
- Structured logging with request IDs
- Automatic error reporting
- Performance monitoring
- Usage analytics

## Future Enhancements

### Planned Features
- Video age progression
- 3D face modeling
- Genetic risk integration
- Wearable device data
- Social sharing capabilities
- Professional consultations

### API Versioning
- Current version: v1
- Backward compatibility maintained
- Deprecation notices for breaking changes
- Migration guides for updates

## Support

For API support and questions:
- Email: api-support@healthsync.ai
- Documentation: https://docs.healthsync.ai/future-simulator
- Status Page: https://status.healthsync.ai