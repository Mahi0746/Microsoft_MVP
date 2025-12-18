-- HealthSync AI - PostgreSQL Database Schema
-- Supabase PostgreSQL with Row Level Security (RLS)

-- =============================================================================
-- EXTENSIONS & SETUP
-- =============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create custom types
CREATE TYPE user_role AS ENUM ('patient', 'doctor', 'admin');
CREATE TYPE appointment_status AS ENUM ('pending', 'bidding', 'confirmed', 'completed', 'cancelled');
CREATE TYPE health_metric_type AS ENUM ('blood_pressure', 'heart_rate', 'weight', 'bmi', 'blood_sugar', 'temperature', 'oxygen_saturation');
CREATE TYPE severity_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE exercise_type AS ENUM ('arm_raises', 'knee_bends', 'neck_rotations', 'shoulder_rolls');

-- =============================================================================
-- CORE USER MANAGEMENT
-- =============================================================================

-- Users table (extends Supabase auth.users)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    role user_role DEFAULT 'patient',
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    date_of_birth DATE,
    gender VARCHAR(20),
    phone VARCHAR(20),
    address JSONB,
    emergency_contact JSONB,
    medical_history JSONB,
    profile_image_url TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Doctors table (professional profiles)
CREATE TABLE doctors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    license_number VARCHAR(50) UNIQUE NOT NULL,
    specialization VARCHAR(100) NOT NULL,
    sub_specializations TEXT[],
    years_experience INTEGER DEFAULT 0,
    education JSONB,
    certifications JSONB,
    languages TEXT[],
    base_consultation_fee DECIMAL(10,2),
    rating DECIMAL(3,2) DEFAULT 0.00,
    total_reviews INTEGER DEFAULT 0,
    availability JSONB, -- Weekly schedule
    bio TEXT,
    clinic_address JSONB,
    is_verified BOOLEAN DEFAULT false,
    is_accepting_patients BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- HEALTH DATA MANAGEMENT
-- =============================================================================

-- Health metrics (vital signs, measurements)
CREATE TABLE health_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    metric_type health_metric_type NOT NULL,
    value DECIMAL(10,2) NOT NULL,
    unit VARCHAR(20),
    measured_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    source VARCHAR(50), -- 'manual', 'device', 'doctor'
    device_info JSONB,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Symptoms and voice analysis
CREATE TABLE symptoms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    description TEXT NOT NULL,
    severity severity_level DEFAULT 'medium',
    duration_hours INTEGER,
    voice_transcript TEXT,
    voice_analysis JSONB, -- stress_level, speech_rate, pause_frequency
    ai_assessment JSONB, -- risk_level, urgency, recommendations
    location_on_body VARCHAR(100),
    triggers TEXT[],
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Disease predictions and risk assessments
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    disease VARCHAR(100) NOT NULL,
    probability DECIMAL(5,4) NOT NULL, -- 0.0000 to 1.0000
    confidence_score DECIMAL(5,4),
    risk_factors JSONB,
    model_version VARCHAR(20),
    based_on_data JSONB, -- What data was used for prediction
    recommendations TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE -- Predictions expire after 30 days
);

-- =============================================================================
-- APPOINTMENT & MARKETPLACE
-- =============================================================================

-- Appointment requests and bookings
CREATE TABLE appointments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES users(id) ON DELETE CASCADE,
    doctor_id UUID REFERENCES doctors(id) ON DELETE SET NULL,
    status appointment_status DEFAULT 'pending',
    symptoms_summary TEXT NOT NULL,
    urgency_level severity_level DEFAULT 'medium',
    preferred_date TIMESTAMP WITH TIME ZONE,
    duration_minutes INTEGER DEFAULT 30,
    consultation_type VARCHAR(50) DEFAULT 'video', -- 'video', 'audio', 'chat'
    final_fee DECIMAL(10,2),
    payment_status VARCHAR(20) DEFAULT 'pending',
    scheduled_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Doctor bids for appointments
CREATE TABLE appointment_bids (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    appointment_id UUID REFERENCES appointments(id) ON DELETE CASCADE,
    doctor_id UUID REFERENCES doctors(id) ON DELETE CASCADE,
    bid_amount DECIMAL(10,2) NOT NULL,
    estimated_duration INTEGER DEFAULT 30,
    available_slots JSONB, -- Array of available time slots
    message TEXT,
    is_selected BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(appointment_id, doctor_id)
);

-- Appointment reviews and ratings
CREATE TABLE appointment_reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    appointment_id UUID REFERENCES appointments(id) ON DELETE CASCADE,
    reviewer_id UUID REFERENCES users(id) ON DELETE CASCADE,
    reviewee_id UUID REFERENCES users(id) ON DELETE CASCADE,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    review_text TEXT,
    is_anonymous BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- THERAPY & GAMIFICATION
-- =============================================================================

-- Therapy game sessions
CREATE TABLE therapy_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exercise_type exercise_type NOT NULL,
    duration_seconds INTEGER NOT NULL,
    repetitions_completed INTEGER DEFAULT 0,
    repetitions_target INTEGER DEFAULT 10,
    form_accuracy DECIMAL(5,2), -- Percentage of correct form
    pain_level INTEGER CHECK (pain_level >= 0 AND pain_level <= 10),
    pain_detected BOOLEAN DEFAULT false,
    points_earned INTEGER DEFAULT 0,
    streak_count INTEGER DEFAULT 0,
    pose_data JSONB, -- MediaPipe pose analysis results
    session_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Therapy progress tracking
CREATE TABLE therapy_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    exercise_type exercise_type NOT NULL,
    week_start_date DATE NOT NULL,
    sessions_completed INTEGER DEFAULT 0,
    total_points INTEGER DEFAULT 0,
    average_accuracy DECIMAL(5,2),
    average_pain_level DECIMAL(3,1),
    improvement_percentage DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, exercise_type, week_start_date)
);

-- Leaderboard (anonymized)
CREATE TABLE therapy_leaderboard (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    anonymous_name VARCHAR(50) NOT NULL,
    total_points INTEGER DEFAULT 0,
    current_streak INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    level_achieved INTEGER DEFAULT 1,
    badges JSONB, -- Array of earned badges
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- AR SCANNER & MEDICAL DOCUMENTS
-- =============================================================================

-- AR scan results
CREATE TABLE ar_scans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    scan_type VARCHAR(50), -- 'prescription', 'lab_report', 'medical_document'
    ocr_results JSONB, -- Extracted text and confidence scores
    image_analysis JSONB, -- BLIP-2 image description
    detected_medications JSONB, -- Structured medication data
    confidence_score DECIMAL(5,4),
    processing_status VARCHAR(20) DEFAULT 'completed',
    warnings TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- FUTURE SIMULATOR
-- =============================================================================

-- User uploaded images for various purposes
CREATE TABLE user_images (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    original_filename VARCHAR(255),
    file_size INTEGER NOT NULL,
    content_type VARCHAR(100) NOT NULL,
    purpose VARCHAR(50) NOT NULL, -- 'future_simulation', 'ar_scan', 'profile'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Age progression results
CREATE TABLE age_progressions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    original_image_path VARCHAR(500) NOT NULL,
    aged_image_path VARCHAR(500) NOT NULL,
    target_age_years INTEGER NOT NULL,
    generation_prompt TEXT,
    model_used VARCHAR(100) DEFAULT 'stable_diffusion',
    processing_status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Health projections for future scenarios
CREATE TABLE health_projections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    target_age_years INTEGER NOT NULL,
    lifestyle_scenario VARCHAR(50) NOT NULL, -- 'improved', 'current', 'declined'
    projections_data JSONB NOT NULL, -- Condition projections and lifestyle impact
    life_expectancy DECIMAL(5,2),
    health_narrative JSONB, -- AI-generated narratives
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Complete future simulations (combines age progression + health projections)
CREATE TABLE future_simulations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    progression_id UUID REFERENCES age_progressions(id) ON DELETE SET NULL,
    projection_id UUID REFERENCES health_projections(id) ON DELETE SET NULL,
    target_age_years INTEGER NOT NULL,
    lifestyle_scenario VARCHAR(50) NOT NULL,
    combined_analysis JSONB, -- Combined insights and recommendations
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- SYSTEM TABLES
-- =============================================================================

-- API usage tracking
CREATE TABLE api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    response_time_ms INTEGER,
    ai_service_used VARCHAR(50), -- 'groq', 'replicate', 'huggingface'
    tokens_used INTEGER,
    cost_estimate DECIMAL(10,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Notifications
CREATE TABLE notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    type VARCHAR(50), -- 'appointment', 'health_alert', 'system'
    is_read BOOLEAN DEFAULT false,
    action_url TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- User indexes
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_created_at ON users(created_at);

-- Doctor indexes
CREATE INDEX idx_doctors_user_id ON doctors(user_id);
CREATE INDEX idx_doctors_specialization ON doctors(specialization);
CREATE INDEX idx_doctors_rating ON doctors(rating DESC);
CREATE INDEX idx_doctors_verified ON doctors(is_verified, is_accepting_patients);

-- Health metrics indexes
CREATE INDEX idx_health_metrics_user_date ON health_metrics(user_id, measured_at DESC);
CREATE INDEX idx_health_metrics_type ON health_metrics(metric_type, measured_at DESC);

-- Symptoms indexes
CREATE INDEX idx_symptoms_user_timestamp ON symptoms(user_id, timestamp DESC);
CREATE INDEX idx_symptoms_severity ON symptoms(severity, timestamp DESC);

-- Predictions indexes
CREATE INDEX idx_predictions_user_disease ON predictions(user_id, disease);
CREATE INDEX idx_predictions_created ON predictions(created_at DESC);
CREATE INDEX idx_predictions_expires ON predictions(expires_at);

-- Appointment indexes
CREATE INDEX idx_appointments_patient ON appointments(patient_id, created_at DESC);
CREATE INDEX idx_appointments_doctor ON appointments(doctor_id, scheduled_at);
CREATE INDEX idx_appointments_status ON appointments(status, created_at DESC);

-- Therapy indexes
CREATE INDEX idx_therapy_sessions_user ON therapy_sessions(user_id, created_at DESC);
CREATE INDEX idx_therapy_progress_user_week ON therapy_progress(user_id, week_start_date DESC);
CREATE INDEX idx_therapy_leaderboard_points ON therapy_leaderboard(total_points DESC);

-- Future Simulator indexes
CREATE INDEX idx_user_images_user_purpose ON user_images(user_id, purpose);
CREATE INDEX idx_age_progressions_user ON age_progressions(user_id, created_at DESC);
CREATE INDEX idx_health_projections_user_scenario ON health_projections(user_id, lifestyle_scenario, created_at DESC);
CREATE INDEX idx_future_simulations_user ON future_simulations(user_id, created_at DESC);

-- System indexes
CREATE INDEX idx_api_usage_user_endpoint ON api_usage(user_id, endpoint, created_at DESC);
CREATE INDEX idx_notifications_user_unread ON notifications(user_id, is_read, created_at DESC);

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_doctors_updated_at BEFORE UPDATE ON doctors FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_appointments_updated_at BEFORE UPDATE ON appointments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update doctor ratings automatically
CREATE OR REPLACE FUNCTION update_doctor_rating()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE doctors 
    SET rating = (
        SELECT AVG(rating)::DECIMAL(3,2)
        FROM appointment_reviews 
        WHERE reviewee_id = NEW.reviewee_id
    ),
    total_reviews = (
        SELECT COUNT(*)
        FROM appointment_reviews 
        WHERE reviewee_id = NEW.reviewee_id
    )
    WHERE user_id = NEW.reviewee_id;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_doctor_rating_trigger 
    AFTER INSERT OR UPDATE ON appointment_reviews 
    FOR EACH ROW EXECUTE FUNCTION update_doctor_rating();

-- Clean up expired predictions
CREATE OR REPLACE FUNCTION cleanup_expired_predictions()
RETURNS void AS $$
BEGIN
    DELETE FROM predictions WHERE expires_at < NOW();
END;
$$ language 'plpgsql';

-- Schedule cleanup (run daily)
-- Note: This would be handled by a cron job in production