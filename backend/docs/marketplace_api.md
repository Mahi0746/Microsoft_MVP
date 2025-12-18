# HealthSync AI - Doctor Marketplace API Documentation

## Overview

The Doctor Marketplace is an AI-powered platform that connects patients with healthcare specialists through an intelligent bidding system. It uses advanced symptom analysis, specialist matching, and dynamic pricing to ensure optimal patient-doctor connections.

## Key Features

- **AI-Powered Symptom Analysis**: Uses Groq LLM and keyword matching for accurate specialist recommendations
- **Intelligent Doctor Matching**: Matches patients with specialists based on symptoms, location, budget, and availability
- **Dynamic Bidding System**: Allows doctors to bid on appointments with AI-powered bid recommendations
- **Comprehensive Analytics**: Provides detailed insights into appointment trends and marketplace statistics
- **Real-time Notifications**: Notifies relevant doctors about new appointments instantly

## Authentication

All endpoints require JWT authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your_jwt_token>
```

## API Endpoints

### 1. Doctor Search & Discovery

#### Search Doctors
```http
GET /api/v1/doctors/search
```

**Query Parameters:**
- `specialization` (optional): Filter by medical specialization
- `location` (optional): Filter by doctor location
- `max_price` (optional): Maximum consultation fee
- `min_rating` (optional): Minimum doctor rating (0-5)
- `limit` (optional): Number of results (max 50, default 20)
- `offset` (optional): Pagination offset (default 0)

**Response:**
```json
[
  {
    "id": "doc_123",
    "user_id": "user_456",
    "first_name": "John",
    "last_name": "Smith",
    "specialization": "Cardiology",
    "sub_specializations": ["Interventional Cardiology", "Heart Surgery"],
    "years_experience": 15,
    "rating": 4.8,
    "total_reviews": 120,
    "base_consultation_fee": 200.0,
    "bio": "Experienced cardiologist specializing in interventional procedures",
    "languages": ["English", "Spanish"],
    "is_verified": true,
    "is_accepting_patients": true
  }
]
```

#### Enhanced Specialist Matching
```http
POST /api/v1/doctors/match-specialists
```

**Request Body:**
```json
{
  "symptoms_summary": "I've been experiencing chest pain and shortness of breath for the past 2 days",
  "urgency_level": "high",
  "preferred_date": "2025-12-20T10:00:00Z",
  "consultation_type": "video",
  "max_budget": 300.0
}
```

**Response:**
```json
{
  "analysis": {
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
  },
  "matching_doctors": [
    {
      "id": "doc_123",
      "first_name": "John",
      "last_name": "Smith",
      "specialization": "Cardiology",
      "rating": 4.8,
      "base_consultation_fee": 200.0,
      "match_confidence": 0.9,
      "match_specialty": "cardiology",
      "match_reasoning": "Chest pain and shortness of breath indicate cardiac evaluation needed",
      "consultation_available": true
    }
  ],
  "match_summary": {
    "total_matches": 8,
    "top_specialties": ["cardiology", "emergency_medicine", "pulmonology"],
    "urgency_level": "high",
    "recommended_timeframe": "within 2 hours"
  }
}
```

#### Get Doctor Profile
```http
GET /api/v1/doctors/{doctor_id}/profile
```

**Response:**
```json
{
  "id": "doc_123",
  "user_id": "user_456",
  "first_name": "John",
  "last_name": "Smith",
  "specialization": "Cardiology",
  "sub_specializations": ["Interventional Cardiology"],
  "years_experience": 15,
  "rating": 4.8,
  "total_reviews": 120,
  "base_consultation_fee": 200.0,
  "bio": "Experienced cardiologist with expertise in interventional procedures",
  "languages": ["English", "Spanish"],
  "is_verified": true,
  "is_accepting_patients": true
}
```

### 2. Appointment Management

#### Create Appointment Request
```http
POST /api/v1/doctors/appointments
```

**Request Body:**
```json
{
  "symptoms_summary": "Severe chest pain and difficulty breathing",
  "urgency_level": "high",
  "preferred_date": "2025-12-20T10:00:00Z",
  "consultation_type": "video",
  "max_budget": 300.0
}
```

**Response:**
```json
{
  "id": "apt_789",
  "patient_id": "user_123",
  "doctor_id": null,
  "status": "pending",
  "symptoms_summary": "Severe chest pain and difficulty breathing",
  "urgency_level": "high",
  "preferred_date": "2025-12-20T10:00:00Z",
  "consultation_type": "video",
  "final_fee": null,
  "scheduled_at": null,
  "created_at": "2025-12-17T10:30:00Z"
}
```

#### Get Appointment Bids (Patient)
```http
GET /api/v1/doctors/appointments/{appointment_id}/bids
```

**Response:**
```json
[
  {
    "id": "bid_456",
    "appointment_id": "apt_789",
    "doctor_id": "doc_123",
    "doctor_name": "Dr. John Smith",
    "specialization": "Cardiology",
    "bid_amount": 250.0,
    "estimated_duration": 60,
    "available_slots": ["2025-12-20T10:00:00Z", "2025-12-20T14:00:00Z"],
    "message": "I can see you urgently today. Experienced with cardiac emergencies.",
    "is_selected": false,
    "created_at": "2025-12-17T10:35:00Z"
  }
]
```

#### Accept Appointment Bid
```http
PUT /api/v1/doctors/appointments/{appointment_id}/accept-bid/{bid_id}
```

**Response:**
```json
{
  "message": "Bid accepted successfully",
  "appointment_id": "apt_789",
  "doctor_name": "Dr. John Smith",
  "final_fee": 250.0
}
```

### 3. Doctor Bidding System

#### Get Bid Recommendations (Doctors Only)
```http
GET /api/v1/doctors/bid-recommendations/{appointment_id}
```

**Response:**
```json
{
  "recommended_bid": 245.50,
  "competitive_bid": 221.00,
  "premium_bid": 270.00,
  "market_range": {
    "min": 180.00,
    "avg": 230.00,
    "max": 320.00
  },
  "recommended_duration": 60,
  "success_rate": 0.75,
  "factors": {
    "urgency_multiplier": 1.2,
    "experience_bonus": 0.15,
    "rating_bonus": 0.08,
    "market_factor": 1.1
  },
  "market_sample_size": 45
}
```

#### Submit Appointment Bid (Doctors Only)
```http
POST /api/v1/doctors/appointments/bid
```

**Request Body:**
```json
{
  "appointment_id": "apt_789",
  "bid_amount": 250.0,
  "estimated_duration": 60,
  "available_slots": ["2025-12-20T10:00:00Z", "2025-12-20T14:00:00Z"],
  "message": "I can see you urgently today. Experienced with cardiac emergencies."
}
```

**Response:**
```json
{
  "id": "bid_456",
  "appointment_id": "apt_789",
  "doctor_id": "doc_123",
  "doctor_name": "Dr. John Smith",
  "specialization": "Cardiology",
  "bid_amount": 250.0,
  "estimated_duration": 60,
  "available_slots": ["2025-12-20T10:00:00Z", "2025-12-20T14:00:00Z"],
  "message": "I can see you urgently today. Experienced with cardiac emergencies.",
  "is_selected": false,
  "created_at": "2025-12-17T10:35:00Z"
}
```

### 4. Analytics & Insights

#### Get Appointment Analytics
```http
GET /api/v1/doctors/appointments/{appointment_id}/analytics
```

**Response:**
```json
{
  "appointment": {
    "id": "apt_789",
    "patient_name": "Jane Doe",
    "patient_age": 35,
    "patient_gender": "female",
    "symptoms": "Severe chest pain and difficulty breathing",
    "urgency": "high",
    "consultation_type": "video",
    "status": "bidding",
    "created_at": "2025-12-17T10:30:00Z"
  },
  "bidding_stats": {
    "total_bids": 5,
    "average_bid": 245.00,
    "bid_range": {
      "min": 200.00,
      "max": 300.00
    },
    "average_duration": 55
  },
  "bidding_doctors": [
    {
      "doctor_name": "Dr. John Smith",
      "specialization": "Cardiology",
      "rating": 4.8,
      "experience": 15,
      "bid_amount": 250.00,
      "duration": 60,
      "bid_time": "2025-12-17T10:35:00Z"
    }
  ]
}
```

#### Get Marketplace Statistics
```http
GET /api/v1/doctors/marketplace/stats
```

**Response:**
```json
{
  "overview": {
    "total_doctors": 1250,
    "active_doctors": 980,
    "avg_doctor_rating": 4.3,
    "total_appointments": 5420,
    "completed_appointments": 4890,
    "avg_appointment_fee": 185.50,
    "completion_rate": 90.2
  },
  "specialties": [
    {
      "specialty": "Primary Care",
      "doctor_count": 245,
      "avg_rating": 4.2,
      "avg_fee": 120.00
    },
    {
      "specialty": "Cardiology",
      "doctor_count": 89,
      "avg_rating": 4.6,
      "avg_fee": 250.00
    }
  ],
  "bidding_trends": [
    {
      "date": "2025-12-17",
      "daily_bids": 156,
      "avg_bid_amount": 195.50
    }
  ],
  "urgency_distribution": [
    {
      "urgency_level": "low",
      "appointment_count": 1200,
      "avg_fee": 145.00
    },
    {
      "urgency_level": "high",
      "appointment_count": 320,
      "avg_fee": 285.00
    }
  ]
}
```

### 5. Reviews & Ratings

#### Submit Appointment Review
```http
POST /api/v1/doctors/appointments/{appointment_id}/review
```

**Request Body:**
```json
{
  "appointment_id": "apt_789",
  "rating": 5,
  "review_text": "Excellent consultation. Dr. Smith was very thorough and professional.",
  "is_anonymous": false
}
```

**Response:**
```json
{
  "message": "Review submitted successfully",
  "review_id": "rev_123"
}
```

#### Get Appointment Reviews
```http
GET /api/v1/doctors/appointments/{appointment_id}/reviews
```

**Response:**
```json
[
  {
    "id": "rev_123",
    "rating": 5,
    "review_text": "Excellent consultation. Very thorough and professional.",
    "is_anonymous": false,
    "reviewer_name": "Jane Doe",
    "reviewee_name": "Dr. John Smith",
    "created_at": "2025-12-17T11:00:00Z"
  }
]
```

### 6. Doctor Dashboard

#### Get Doctor Appointments (Doctors Only)
```http
GET /api/v1/doctors/dashboard/appointments
```

**Query Parameters:**
- `status_filter` (optional): Filter by appointment status
- `limit` (optional): Number of results (max 50, default 20)
- `offset` (optional): Pagination offset (default 0)

**Response:**
```json
[
  {
    "id": "apt_789",
    "patient_id": "user_123",
    "status": "confirmed",
    "symptoms_summary": "Chest pain and shortness of breath",
    "urgency_level": "high",
    "preferred_date": "2025-12-20T10:00:00Z",
    "consultation_type": "video",
    "final_fee": 250.0,
    "scheduled_at": "2025-12-20T10:00:00Z",
    "created_at": "2025-12-17T10:30:00Z",
    "patient_name": "Jane Doe"
  }
]
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": true,
  "message": "Error description",
  "status_code": 400,
  "timestamp": "2025-12-17T10:30:00Z",
  "path": "/api/v1/doctors/search",
  "request_id": "req_123456"
}
```

### Common Error Codes

- `400` - Bad Request (validation errors)
- `401` - Unauthorized (invalid or missing token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `422` - Unprocessable Entity (validation errors)
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

## Rate Limits

- General endpoints: 100 requests per minute
- Search endpoints: 50 requests per minute
- Bidding endpoints: 20 requests per minute
- Analytics endpoints: 30 requests per minute

## Validation Rules

### Appointment Request
- `symptoms_summary`: Minimum 10 characters, maximum 1000 characters
- `urgency_level`: Must be one of: "low", "medium", "high", "critical"
- `consultation_type`: Must be one of: "video", "audio", "chat", "in_person"
- `max_budget`: Must be positive number, maximum 1000

### Appointment Bid
- `bid_amount`: Must be between 0 and 1000
- `estimated_duration`: Must be between 15 and 180 minutes
- `available_slots`: Must be valid ISO datetime strings
- `message`: Maximum 500 characters

### Review
- `rating`: Must be integer between 1 and 5
- `review_text`: Maximum 1000 characters

## WebSocket Events

The marketplace uses WebSocket connections for real-time updates:

### Events Sent to Clients

#### New Appointment Notification (Doctors)
```json
{
  "type": "new_appointment",
  "data": {
    "appointment_id": "apt_789",
    "urgency_level": "high",
    "match_confidence": 0.85,
    "symptoms_preview": "Chest pain and shortness of breath",
    "created_at": "2025-12-17T10:30:00Z"
  }
}
```

#### Bid Accepted Notification (Doctors)
```json
{
  "type": "bid_accepted",
  "data": {
    "appointment_id": "apt_789",
    "patient_name": "Jane Doe",
    "final_fee": 250.0,
    "scheduled_at": "2025-12-20T10:00:00Z"
  }
}
```

#### New Bid Notification (Patients)
```json
{
  "type": "new_bid",
  "data": {
    "appointment_id": "apt_789",
    "doctor_name": "Dr. John Smith",
    "bid_amount": 250.0,
    "specialization": "Cardiology",
    "created_at": "2025-12-17T10:35:00Z"
  }
}
```

## AI-Powered Features

### Symptom Analysis
The marketplace uses advanced AI to analyze patient symptoms:

1. **Groq LLM Analysis**: Primary analysis using Llama 3.1 model
2. **Keyword Matching**: Fallback system using medical terminology
3. **Confidence Scoring**: Each recommendation includes confidence levels
4. **Urgency Assessment**: Automatic triage based on symptom severity

### Specialist Matching
Intelligent matching considers:

- Symptom-to-specialty mapping
- Doctor availability and preferences
- Patient location and budget
- Historical success rates
- Real-time market conditions

### Bid Recommendations
AI-powered bid suggestions factor in:

- Doctor experience and ratings
- Market rates for similar appointments
- Urgency level multipliers
- Historical success rates
- Competition analysis

## Security & Privacy

- All data is encrypted in transit and at rest
- Patient information is anonymized in analytics
- Doctors only see necessary patient information
- All API calls are logged for audit purposes
- Rate limiting prevents abuse
- Input validation prevents injection attacks

## Integration Examples

### JavaScript/React Example
```javascript
// Search for specialists
const searchSpecialists = async (symptoms) => {
  const response = await fetch('/api/v1/doctors/match-specialists', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      symptoms_summary: symptoms,
      urgency_level: 'medium',
      consultation_type: 'video'
    })
  });
  
  return await response.json();
};

// Submit a bid (doctor)
const submitBid = async (appointmentId, bidData) => {
  const response = await fetch('/api/v1/doctors/appointments/bid', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      appointment_id: appointmentId,
      ...bidData
    })
  });
  
  return await response.json();
};
```

### Python Example
```python
import requests

class HealthSyncMarketplace:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def match_specialists(self, symptoms, urgency='medium'):
        url = f'{self.base_url}/api/v1/doctors/match-specialists'
        data = {
            'symptoms_summary': symptoms,
            'urgency_level': urgency,
            'consultation_type': 'video'
        }
        
        response = requests.post(url, json=data, headers=self.headers)
        return response.json()
    
    def get_bid_recommendations(self, appointment_id):
        url = f'{self.base_url}/api/v1/doctors/bid-recommendations/{appointment_id}'
        response = requests.get(url, headers=self.headers)
        return response.json()
```

## Support

For API support and questions:
- Email: api-support@healthsync.ai
- Documentation: https://docs.healthsync.ai
- Status Page: https://status.healthsync.ai