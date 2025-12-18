# HealthSync AI - System Architecture Design

## Executive Summary

HealthSync AI is a comprehensive healthcare platform built with a microservices architecture using 100% FREE cloud services. The system handles real-time voice analysis, AR medical scanning, ML-powered health predictions, gamified therapy, doctor marketplace, and future health simulation.

## Architecture Principles

### 1. Cost Optimization (FREE Tier Strategy)
- **Supabase**: PostgreSQL + Auth + Storage (500MB DB, 1GB storage)
- **MongoDB Atlas**: Document storage (512MB free)
- **Railway/Render**: Backend hosting ($5 credit/month)
- **Vercel**: Frontend hosting (unlimited for personal)
- **Groq**: LLM inference (30 req/min free)
- **Replicate**: AI models (50 predictions/month)
- **Hugging Face**: Vision/OCR models (rate-limited free)

### 2. Scalability Design
- **Horizontal scaling**: Stateless FastAPI services
- **Caching strategy**: Redis for session data and API responses
- **Queue system**: Background job processing for AI tasks
- **CDN**: Supabase Storage for global file distribution

### 3. Security Framework
- **Authentication**: Supabase Auth with JWT tokens
- **Authorization**: Role-based access control (RBAC)
- **Data protection**: Row Level Security (RLS) in PostgreSQL
- **API security**: Rate limiting, input validation, HTTPS only

## Detailed Component Architecture

### Frontend Layer

#### React Native Mobile App (Expo)
```typescript
// Core Technologies
- Expo SDK 50+
- TypeScript (strict mode)
- React Navigation 6
- Expo Camera & Audio
- Three.js for AR overlays
- Socket.io for real-time communication

// Key Features
- Voice recording with real-time streaming
- Camera capture with AR overlays
- MediaPipe pose detection
- Offline-first data synchronization
- Push notifications
```

#### Next.js Admin Dashboard
```typescript
// Core Technologies
- Next.js 14 (App Router)
- TypeScript + Tailwind CSS
- D3.js for data visualization
- Chart.js for analytics
- Socket.io for real-time updates

// Key Features
- Doctor management interface
- Real-time appointment bidding
- Family health graph visualization
- Analytics dashboard
- Bulk data operations
```

### Backend Layer

#### FastAPI Application Server
```python
# Core Architecture
- FastAPI 0.109+ with async/await
- Pydantic for data validation
- SQLAlchemy ORM for PostgreSQL
- Motor for MongoDB async operations
- WebSocket support for real-time features

# Service Organization
api/
├── routes/          # HTTP endpoints
├── websocket/       # Real-time connections
├── middleware/      # Auth, CORS, rate limiting
├── models/          # Database models
├── schemas/         # Request/response validation
└── services/        # Business logic
```

#### AI/ML Processing Pipeline
```python
# Voice Analysis Pipeline
Audio Input → Whisper (Speech-to-Text) → librosa (Feature Extraction) 
→ Groq LLM (Medical Analysis) → Structured Response

# Image Processing Pipeline
Image Upload → Supabase Storage → Hugging Face APIs 
→ TrOCR (Text Extraction) + BLIP-2 (Image Description) → Structured Data

# Health Prediction Pipeline
User Data → Feature Engineering → scikit-learn Models 
→ Risk Calculations → MongoDB Storage → Real-time Updates
```

### Database Layer

#### PostgreSQL (Supabase) - Relational Data
```sql
-- Core Tables
users (id, email, role, profile_data, created_at)
doctors (id, user_id, specialization, rating, price_range, availability)
health_metrics (id, user_id, metric_type, value, date, source)
symptoms (id, user_id, description, severity, voice_analysis, timestamp)
appointments (id, user_id, doctor_id, status, bid_amount, scheduled_at)
therapy_sessions (id, user_id, exercise_type, score, duration, pain_level)
predictions (id, user_id, disease, probability, confidence, created_at)

-- Indexes for Performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_health_metrics_user_date ON health_metrics(user_id, date);
CREATE INDEX idx_appointments_status ON appointments(status, scheduled_at);
```

#### MongoDB Atlas - Document Storage
```javascript
// Collections Schema
family_graph: {
  user_id: ObjectId,
  family_members: [{
    id: String,
    relation: String,
    health_conditions: [String],
    age_of_onset: Object,
    genetic_markers: Object
  }],
  inherited_risks: Object,
  risk_calculations: Object,
  updated_at: Date
}

health_events: {
  user_id: ObjectId,
  event_type: String,
  data: Object,
  timestamp: Date,
  source: String,
  confidence: Number
}

ml_models: {
  user_id: ObjectId,
  model_type: String,
  model_data: Binary,
  accuracy_metrics: Object,
  training_date: Date,
  version: Number
}
```

#### Redis Cache - Session & Performance
```redis
# Cache Keys Structure
user:session:{user_id} → JWT session data (TTL: 30min)
health:predictions:{user_id} → Cached predictions (TTL: 1hour)
doctors:search:{specialty}:{location} → Doctor search results (TTL: 30min)
api:rate_limit:{user_id}:{endpoint} → Rate limiting counters (TTL: 1min)
```

## Data Flow Diagrams

### 1. Voice AI Doctor Flow
```
User speaks → Mobile App records audio chunks → WebSocket stream to Backend
→ Whisper API (speech-to-text) → librosa (voice stress analysis)
→ Groq LLM (medical reasoning) → Response with risk assessment
→ Store in PostgreSQL → Real-time update to mobile app
```

### 2. AR Medical Scanner Flow
```
User captures image → Expo Camera → Compress & upload to Supabase Storage
→ Backend downloads image → Hugging Face APIs (TrOCR + BLIP-2)
→ Extract structured data → Store results → AR overlay with detected text
```

### 3. Health Twin Prediction Flow
```
User health data → Feature engineering → scikit-learn models
→ Disease risk calculations → Combine with family history (MongoDB)
→ Generate comprehensive report → Cache in Redis → Update mobile app
```

### 4. Doctor Marketplace Flow
```
Patient submits symptoms → Groq LLM matches specialists → Query doctor database
→ Send notifications to matching doctors → Doctors submit bids
→ Real-time updates via WebSocket → Patient selects doctor → Booking confirmation
```

## API Design Patterns

### RESTful Endpoints
```python
# Resource-based URLs
GET    /api/v1/health/metrics          # List user's health data
POST   /api/v1/health/metrics          # Add new health metric
GET    /api/v1/doctors/search          # Search doctors by specialty
POST   /api/v1/appointments            # Create appointment request
PUT    /api/v1/appointments/{id}       # Update appointment status

# Consistent response format
{
  "success": true,
  "data": {...},
  "message": "Operation completed successfully",
  "timestamp": "2025-12-17T10:30:00Z"
}
```

### WebSocket Events
```javascript
// Client → Server Events
voice_stream_start    → Initialize voice analysis session
voice_chunk          → Send audio data chunk
therapy_pose_data    → Send pose detection results
appointment_bid      → Doctor submits bid

// Server → Client Events
voice_analysis_result → Real-time voice analysis
therapy_feedback     → Exercise form correction
appointment_update   → New bid or status change
health_alert         → Critical health notification
```

## Security Architecture

### Authentication Flow
```
1. User registers → Supabase Auth creates account
2. Email verification → Account activation
3. Login → Supabase returns JWT token
4. Mobile app stores token in SecureStore
5. API requests include Authorization header
6. Backend validates JWT with Supabase
7. Token refresh before expiration
```

### Authorization Levels
```python
# Role-based permissions
@require_auth                    # Any authenticated user
@require_role("patient")         # Patient-specific endpoints
@require_role("doctor")          # Doctor-specific endpoints  
@require_role("admin")           # Admin dashboard access

# Resource-level security (RLS in PostgreSQL)
CREATE POLICY user_health_data ON health_metrics
  FOR ALL TO authenticated
  USING (user_id = auth.uid());
```

### Data Protection
```python
# Input validation with Pydantic
class HealthMetricCreate(BaseModel):
    metric_type: str = Field(..., regex="^[a-zA-Z_]+$")
    value: float = Field(..., ge=0, le=1000)
    date: datetime = Field(..., le=datetime.now())

# Rate limiting by endpoint
@limiter.limit("10/minute")
async def voice_analysis(request: Request):
    pass

@limiter.limit("5/minute") 
async def image_upload(request: Request):
    pass
```

## Performance Optimization

### Caching Strategy
```python
# Multi-level caching
1. Browser cache (static assets) → 24 hours
2. CDN cache (Supabase Storage) → 7 days  
3. Redis cache (API responses) → 30 minutes
4. Application cache (ML models) → 1 hour
5. Database query cache → 5 minutes
```

### Database Optimization
```sql
-- Partitioning for large tables
CREATE TABLE health_metrics_2025 PARTITION OF health_metrics
FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Materialized views for analytics
CREATE MATERIALIZED VIEW doctor_performance AS
SELECT doctor_id, AVG(rating), COUNT(*) as total_appointments
FROM appointments WHERE status = 'completed'
GROUP BY doctor_id;
```

### AI Service Optimization
```python
# Batch processing for efficiency
async def batch_process_images(images: List[str]):
    # Process multiple images in single API call
    # Reduces API calls and improves throughput
    
# Model caching
@lru_cache(maxsize=10)
def load_ml_model(model_type: str):
    # Cache trained models in memory
    # Avoid repeated disk I/O
```

## Monitoring & Observability

### Health Checks
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "services": {
            "database": await check_db_connection(),
            "redis": await check_redis_connection(),
            "ai_services": await check_ai_apis()
        }
    }
```

### Logging Strategy
```python
# Structured logging with correlation IDs
logger.info(
    "Voice analysis completed",
    extra={
        "user_id": user_id,
        "session_id": session_id,
        "processing_time": elapsed_time,
        "confidence_score": confidence
    }
)
```

### Error Handling
```python
# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": request.state.request_id}
    )
```

## Deployment Architecture

### Development Environment
```yaml
# docker-compose.yml
- Backend: FastAPI with hot reload
- Database: Local PostgreSQL + MongoDB
- Cache: Local Redis
- Frontend: Expo development server
- Admin: Next.js development server
```

### Production Environment
```yaml
# Railway.app (Backend)
- FastAPI with Gunicorn
- Auto-scaling based on CPU/memory
- Health checks and auto-restart
- Environment variable management

# Vercel (Frontend)
- Static site generation for admin dashboard
- Edge functions for API routes
- Global CDN distribution
- Automatic HTTPS
```

This architecture ensures scalability, security, and cost-effectiveness while maintaining high performance for all HealthSync AI features.