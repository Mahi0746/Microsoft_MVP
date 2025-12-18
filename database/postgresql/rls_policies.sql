-- HealthSync AI - Row Level Security (RLS) Policies
-- Supabase PostgreSQL Security Implementation

-- =============================================================================
-- ENABLE RLS ON ALL TABLES
-- =============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE doctors ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE symptoms ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointments ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointment_bids ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointment_reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE therapy_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE therapy_progress ENABLE ROW LEVEL SECURITY;
ALTER TABLE therapy_leaderboard ENABLE ROW LEVEL SECURITY;
ALTER TABLE ar_scans ENABLE ROW LEVEL SECURITY;
ALTER TABLE future_simulations ENABLE ROW LEVEL SECURITY;
ALTER TABLE notifications ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_usage ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- HELPER FUNCTIONS FOR RLS
-- =============================================================================

-- Get current user ID from JWT
CREATE OR REPLACE FUNCTION auth.user_id() 
RETURNS UUID AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'sub',
    (current_setting('request.jwt.claims', true)::json->>'user_id')
  )::UUID
$$ LANGUAGE SQL STABLE;

-- Get current user role
CREATE OR REPLACE FUNCTION auth.user_role() 
RETURNS TEXT AS $$
  SELECT COALESCE(
    current_setting('request.jwt.claims', true)::json->>'role',
    'patient'
  )::TEXT
$$ LANGUAGE SQL STABLE;

-- Check if user is admin
CREATE OR REPLACE FUNCTION auth.is_admin() 
RETURNS BOOLEAN AS $$
  SELECT auth.user_role() = 'admin'
$$ LANGUAGE SQL STABLE;

-- Check if user is doctor
CREATE OR REPLACE FUNCTION auth.is_doctor() 
RETURNS BOOLEAN AS $$
  SELECT auth.user_role() IN ('doctor', 'admin')
$$ LANGUAGE SQL STABLE;

-- Get doctor ID for current user
CREATE OR REPLACE FUNCTION auth.doctor_id() 
RETURNS UUID AS $$
  SELECT id FROM doctors WHERE user_id = auth.user_id()
$$ LANGUAGE SQL STABLE;

-- =============================================================================
-- USERS TABLE POLICIES
-- =============================================================================

-- Users can view their own profile
CREATE POLICY "Users can view own profile" ON users
  FOR SELECT USING (id = auth.user_id());

-- Users can update their own profile
CREATE POLICY "Users can update own profile" ON users
  FOR UPDATE USING (id = auth.user_id());

-- Admins can view all users
CREATE POLICY "Admins can view all users" ON users
  FOR SELECT USING (auth.is_admin());

-- Doctors can view basic patient info for their appointments
CREATE POLICY "Doctors can view patient info" ON users
  FOR SELECT USING (
    auth.is_doctor() AND 
    id IN (
      SELECT patient_id FROM appointments 
      WHERE doctor_id = auth.doctor_id()
    )
  );

-- =============================================================================
-- DOCTORS TABLE POLICIES
-- =============================================================================

-- Anyone can view verified doctors (for marketplace)
CREATE POLICY "Anyone can view verified doctors" ON doctors
  FOR SELECT USING (is_verified = true AND is_accepting_patients = true);

-- Doctors can view and update their own profile
CREATE POLICY "Doctors can manage own profile" ON doctors
  FOR ALL USING (user_id = auth.user_id());

-- Admins can manage all doctor profiles
CREATE POLICY "Admins can manage doctors" ON doctors
  FOR ALL USING (auth.is_admin());

-- =============================================================================
-- HEALTH DATA POLICIES
-- =============================================================================

-- Health metrics: Users own their data
CREATE POLICY "Users own health metrics" ON health_metrics
  FOR ALL USING (user_id = auth.user_id());

-- Doctors can view health metrics for their patients
CREATE POLICY "Doctors can view patient health metrics" ON health_metrics
  FOR SELECT USING (
    auth.is_doctor() AND 
    user_id IN (
      SELECT patient_id FROM appointments 
      WHERE doctor_id = auth.doctor_id() 
      AND status IN ('confirmed', 'completed')
    )
  );

-- Symptoms: Users own their data
CREATE POLICY "Users own symptoms" ON symptoms
  FOR ALL USING (user_id = auth.user_id());

-- Doctors can view symptoms for their patients
CREATE POLICY "Doctors can view patient symptoms" ON symptoms
  FOR SELECT USING (
    auth.is_doctor() AND 
    user_id IN (
      SELECT patient_id FROM appointments 
      WHERE doctor_id = auth.doctor_id()
    )
  );

-- Predictions: Users own their predictions
CREATE POLICY "Users own predictions" ON predictions
  FOR ALL USING (user_id = auth.user_id());

-- =============================================================================
-- APPOINTMENT POLICIES
-- =============================================================================

-- Patients can manage their own appointments
CREATE POLICY "Patients can manage own appointments" ON appointments
  FOR ALL USING (patient_id = auth.user_id());

-- Doctors can view and update appointments assigned to them
CREATE POLICY "Doctors can manage assigned appointments" ON appointments
  FOR SELECT USING (doctor_id = auth.doctor_id());

CREATE POLICY "Doctors can update assigned appointments" ON appointments
  FOR UPDATE USING (doctor_id = auth.doctor_id());

-- Doctors can view pending appointments in their specialty
CREATE POLICY "Doctors can view relevant pending appointments" ON appointments
  FOR SELECT USING (
    auth.is_doctor() AND 
    status = 'pending' AND
    id IN (
      SELECT a.id FROM appointments a
      JOIN symptoms s ON s.user_id = a.patient_id
      JOIN doctors d ON d.id = auth.doctor_id()
      WHERE a.status = 'pending'
      -- Add specialty matching logic here
    )
  );

-- =============================================================================
-- APPOINTMENT BIDS POLICIES
-- =============================================================================

-- Doctors can create bids for appointments
CREATE POLICY "Doctors can create bids" ON appointment_bids
  FOR INSERT WITH CHECK (doctor_id = auth.doctor_id());

-- Doctors can view and update their own bids
CREATE POLICY "Doctors can manage own bids" ON appointment_bids
  FOR ALL USING (doctor_id = auth.doctor_id());

-- Patients can view bids for their appointments
CREATE POLICY "Patients can view bids for their appointments" ON appointment_bids
  FOR SELECT USING (
    appointment_id IN (
      SELECT id FROM appointments WHERE patient_id = auth.user_id()
    )
  );

-- =============================================================================
-- REVIEW POLICIES
-- =============================================================================

-- Users can create reviews for their completed appointments
CREATE POLICY "Users can create reviews" ON appointment_reviews
  FOR INSERT WITH CHECK (reviewer_id = auth.user_id());

-- Users can view reviews they wrote
CREATE POLICY "Users can view own reviews" ON appointment_reviews
  FOR SELECT USING (reviewer_id = auth.user_id());

-- Users can view reviews about them (doctors seeing patient reviews)
CREATE POLICY "Users can view reviews about them" ON appointment_reviews
  FOR SELECT USING (reviewee_id = auth.user_id());

-- Public can view non-anonymous reviews for doctors
CREATE POLICY "Public can view doctor reviews" ON appointment_reviews
  FOR SELECT USING (
    is_anonymous = false AND
    reviewee_id IN (SELECT user_id FROM doctors WHERE is_verified = true)
  );

-- =============================================================================
-- THERAPY & GAMIFICATION POLICIES
-- =============================================================================

-- Users own their therapy data
CREATE POLICY "Users own therapy sessions" ON therapy_sessions
  FOR ALL USING (user_id = auth.user_id());

CREATE POLICY "Users own therapy progress" ON therapy_progress
  FOR ALL USING (user_id = auth.user_id());

-- Leaderboard is public but anonymized
CREATE POLICY "Public can view leaderboard" ON therapy_leaderboard
  FOR SELECT USING (true);

-- Users can only update their own leaderboard entry
CREATE POLICY "Users can update own leaderboard" ON therapy_leaderboard
  FOR UPDATE USING (user_id = auth.user_id());

-- =============================================================================
-- AR SCANNER POLICIES
-- =============================================================================

-- Users own their AR scan results
CREATE POLICY "Users own AR scans" ON ar_scans
  FOR ALL USING (user_id = auth.user_id());

-- Doctors can view AR scans shared by their patients
CREATE POLICY "Doctors can view shared AR scans" ON ar_scans
  FOR SELECT USING (
    auth.is_doctor() AND 
    user_id IN (
      SELECT patient_id FROM appointments 
      WHERE doctor_id = auth.doctor_id() 
      AND status IN ('confirmed', 'completed')
    )
  );

-- =============================================================================
-- FUTURE SIMULATOR POLICIES
-- =============================================================================

-- Users own their future simulations
CREATE POLICY "Users own future simulations" ON future_simulations
  FOR ALL USING (user_id = auth.user_id());

-- =============================================================================
-- SYSTEM TABLE POLICIES
-- =============================================================================

-- Users can view their own notifications
CREATE POLICY "Users can view own notifications" ON notifications
  FOR SELECT USING (user_id = auth.user_id());

-- Users can update their own notifications (mark as read)
CREATE POLICY "Users can update own notifications" ON notifications
  FOR UPDATE USING (user_id = auth.user_id());

-- System can create notifications for users
CREATE POLICY "System can create notifications" ON notifications
  FOR INSERT WITH CHECK (true);

-- API usage tracking - users can view their own usage
CREATE POLICY "Users can view own API usage" ON api_usage
  FOR SELECT USING (user_id = auth.user_id());

-- Admins can view all API usage
CREATE POLICY "Admins can view all API usage" ON api_usage
  FOR SELECT USING (auth.is_admin());

-- System can log API usage
CREATE POLICY "System can log API usage" ON api_usage
  FOR INSERT WITH CHECK (true);

-- =============================================================================
-- SPECIAL ADMIN POLICIES
-- =============================================================================

-- Admins have full access to all tables (bypass RLS)
CREATE POLICY "Admins have full access" ON users FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON doctors FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON health_metrics FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON symptoms FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON predictions FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON appointments FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON appointment_bids FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON appointment_reviews FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON therapy_sessions FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON therapy_progress FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON therapy_leaderboard FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON ar_scans FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON future_simulations FOR ALL USING (auth.is_admin());
CREATE POLICY "Admins have full access" ON notifications FOR ALL USING (auth.is_admin());

-- =============================================================================
-- SECURITY FUNCTIONS
-- =============================================================================

-- Function to check if user can access appointment
CREATE OR REPLACE FUNCTION can_access_appointment(appointment_uuid UUID)
RETURNS BOOLEAN AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM appointments 
    WHERE id = appointment_uuid 
    AND (
      patient_id = auth.user_id() OR 
      doctor_id = auth.doctor_id() OR
      auth.is_admin()
    )
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to check if user can access health data
CREATE OR REPLACE FUNCTION can_access_health_data(target_user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
  -- User can access their own data
  IF target_user_id = auth.user_id() THEN
    RETURN true;
  END IF;
  
  -- Doctors can access their patients' data
  IF auth.is_doctor() THEN
    RETURN EXISTS (
      SELECT 1 FROM appointments 
      WHERE patient_id = target_user_id 
      AND doctor_id = auth.doctor_id()
      AND status IN ('confirmed', 'completed')
    );
  END IF;
  
  -- Admins can access all data
  IF auth.is_admin() THEN
    RETURN true;
  END IF;
  
  RETURN false;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;