-- HealthSync AI - Sample Seed Data
-- PostgreSQL Test Data for Development

-- =============================================================================
-- SAMPLE USERS (PATIENTS)
-- =============================================================================

INSERT INTO users (id, email, role, first_name, last_name, date_of_birth, gender, phone, address, emergency_contact, medical_history) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'john.doe@example.com', 'patient', 'John', 'Doe', '1985-03-15', 'Male', '+1-555-0101', 
 '{"street": "123 Main St", "city": "San Francisco", "state": "CA", "zip": "94102", "country": "USA"}',
 '{"name": "Jane Doe", "relationship": "Spouse", "phone": "+1-555-0102"}',
 '{"allergies": ["Penicillin"], "chronic_conditions": [], "surgeries": [], "family_history": ["Diabetes", "Hypertension"]}'),

('550e8400-e29b-41d4-a716-446655440002', 'sarah.johnson@example.com', 'patient', 'Sarah', 'Johnson', '1990-07-22', 'Female', '+1-555-0201',
 '{"street": "456 Oak Ave", "city": "Los Angeles", "state": "CA", "zip": "90210", "country": "USA"}',
 '{"name": "Mike Johnson", "relationship": "Brother", "phone": "+1-555-0202"}',
 '{"allergies": [], "chronic_conditions": ["Asthma"], "surgeries": ["Appendectomy"], "family_history": ["Heart Disease"]}'),

('550e8400-e29b-41d4-a716-446655440003', 'mike.chen@example.com', 'patient', 'Mike', 'Chen', '1978-11-08', 'Male', '+1-555-0301',
 '{"street": "789 Pine St", "city": "Seattle", "state": "WA", "zip": "98101", "country": "USA"}',
 '{"name": "Lisa Chen", "relationship": "Wife", "phone": "+1-555-0302"}',
 '{"allergies": ["Shellfish"], "chronic_conditions": ["Type 2 Diabetes"], "surgeries": [], "family_history": ["Diabetes", "Cancer"]}'),

('550e8400-e29b-41d4-a716-446655440004', 'emma.wilson@example.com', 'patient', 'Emma', 'Wilson', '1995-02-14', 'Female', '+1-555-0401',
 '{"street": "321 Elm St", "city": "Austin", "state": "TX", "zip": "73301", "country": "USA"}',
 '{"name": "Tom Wilson", "relationship": "Father", "phone": "+1-555-0402"}',
 '{"allergies": [], "chronic_conditions": [], "surgeries": [], "family_history": ["Hypertension"]}'),

('550e8400-e29b-41d4-a716-446655440005', 'david.brown@example.com', 'patient', 'David', 'Brown', '1982-09-30', 'Male', '+1-555-0501',
 '{"street": "654 Maple Dr", "city": "Denver", "state": "CO", "zip": "80202", "country": "USA"}',
 '{"name": "Amy Brown", "relationship": "Sister", "phone": "+1-555-0502"}',
 '{"allergies": ["Latex"], "chronic_conditions": ["Hypertension"], "surgeries": ["Knee Surgery"], "family_history": ["Heart Disease", "Stroke"]}');

-- =============================================================================
-- SAMPLE DOCTORS
-- =============================================================================

INSERT INTO users (id, email, role, first_name, last_name, date_of_birth, gender, phone, address) VALUES
('550e8400-e29b-41d4-a716-446655440101', 'dr.smith@healthsync.com', 'doctor', 'Robert', 'Smith', '1975-05-12', 'Male', '+1-555-1001',
 '{"street": "100 Medical Plaza", "city": "San Francisco", "state": "CA", "zip": "94102", "country": "USA"}'),

('550e8400-e29b-41d4-a716-446655440102', 'dr.garcia@healthsync.com', 'doctor', 'Maria', 'Garcia', '1980-08-25', 'Female', '+1-555-1002',
 '{"street": "200 Health Center", "city": "Los Angeles", "state": "CA", "zip": "90210", "country": "USA"}'),

('550e8400-e29b-41d4-a716-446655440103', 'dr.patel@healthsync.com', 'doctor', 'Raj', 'Patel', '1972-12-03', 'Male', '+1-555-1003',
 '{"street": "300 Cardiology Clinic", "city": "Seattle", "state": "WA", "zip": "98101", "country": "USA"}'),

('550e8400-e29b-41d4-a716-446655440104', 'dr.lee@healthsync.com', 'doctor', 'Jennifer', 'Lee', '1985-04-18', 'Female', '+1-555-1004',
 '{"street": "400 Dermatology Center", "city": "Austin", "state": "TX", "zip": "73301", "country": "USA"}'),

('550e8400-e29b-41d4-a716-446655440105', 'dr.johnson@healthsync.com', 'doctor', 'Michael', 'Johnson', '1978-10-07', 'Male', '+1-555-1005',
 '{"street": "500 Orthopedic Clinic", "city": "Denver", "state": "CO", "zip": "80202", "country": "USA"}');

INSERT INTO doctors (id, user_id, license_number, specialization, sub_specializations, years_experience, education, certifications, languages, base_consultation_fee, rating, total_reviews, availability, bio, clinic_address, is_verified, is_accepting_patients) VALUES
('650e8400-e29b-41d4-a716-446655440101', '550e8400-e29b-41d4-a716-446655440101', 'CA-MD-12345', 'Internal Medicine', 
 ARRAY['Preventive Care', 'Chronic Disease Management'], 15,
 '{"medical_school": "UCSF School of Medicine", "residency": "UCSF Internal Medicine", "fellowship": null}',
 '{"board_certifications": ["American Board of Internal Medicine"], "additional_certs": ["CPR", "ACLS"]}',
 ARRAY['English', 'Spanish'], 150.00, 4.8, 127,
 '{"monday": ["09:00-17:00"], "tuesday": ["09:00-17:00"], "wednesday": ["09:00-17:00"], "thursday": ["09:00-17:00"], "friday": ["09:00-15:00"], "saturday": [], "sunday": []}',
 'Dr. Smith is a board-certified internist with over 15 years of experience in preventive care and chronic disease management.',
 '{"street": "100 Medical Plaza", "city": "San Francisco", "state": "CA", "zip": "94102", "country": "USA"}', true, true),

('650e8400-e29b-41d4-a716-446655440102', '550e8400-e29b-41d4-a716-446655440102', 'CA-MD-23456', 'Pediatrics',
 ARRAY['Adolescent Medicine', 'Developmental Pediatrics'], 12,
 '{"medical_school": "UCLA School of Medicine", "residency": "Childrens Hospital LA Pediatrics", "fellowship": "Adolescent Medicine"}',
 '{"board_certifications": ["American Board of Pediatrics"], "additional_certs": ["PALS", "NRP"]}',
 ARRAY['English', 'Spanish', 'Portuguese'], 120.00, 4.9, 89,
 '{"monday": ["08:00-16:00"], "tuesday": ["08:00-16:00"], "wednesday": ["08:00-16:00"], "thursday": ["08:00-16:00"], "friday": ["08:00-14:00"], "saturday": ["09:00-13:00"], "sunday": []}',
 'Dr. Garcia specializes in pediatric care with a focus on adolescent health and development.',
 '{"street": "200 Health Center", "city": "Los Angeles", "state": "CA", "zip": "90210", "country": "USA"}', true, true),

('650e8400-e29b-41d4-a716-446655440103', '550e8400-e29b-41d4-a716-446655440103', 'WA-MD-34567', 'Cardiology',
 ARRAY['Interventional Cardiology', 'Heart Failure'], 20,
 '{"medical_school": "University of Washington School of Medicine", "residency": "UW Internal Medicine", "fellowship": "UW Cardiology"}',
 '{"board_certifications": ["American Board of Internal Medicine", "American Board of Cardiovascular Disease"], "additional_certs": ["ACLS", "Cardiac Catheterization"]}',
 ARRAY['English', 'Hindi'], 250.00, 4.7, 156,
 '{"monday": ["07:00-18:00"], "tuesday": ["07:00-18:00"], "wednesday": ["07:00-18:00"], "thursday": ["07:00-18:00"], "friday": ["07:00-16:00"], "saturday": [], "sunday": []}',
 'Dr. Patel is a leading cardiologist specializing in interventional procedures and heart failure management.',
 '{"street": "300 Cardiology Clinic", "city": "Seattle", "state": "WA", "zip": "98101", "country": "USA"}', true, true),

('650e8400-e29b-41d4-a716-446655440104', '550e8400-e29b-41d4-a716-446655440104', 'TX-MD-45678', 'Dermatology',
 ARRAY['Cosmetic Dermatology', 'Dermatopathology'], 8,
 '{"medical_school": "UT Southwestern Medical School", "residency": "UT Southwestern Dermatology", "fellowship": "Dermatopathology"}',
 '{"board_certifications": ["American Board of Dermatology"], "additional_certs": ["Mohs Surgery", "Laser Therapy"]}',
 ARRAY['English', 'Korean'], 180.00, 4.6, 73,
 '{"monday": ["09:00-17:00"], "tuesday": ["09:00-17:00"], "wednesday": ["09:00-17:00"], "thursday": ["09:00-17:00"], "friday": ["09:00-15:00"], "saturday": [], "sunday": []}',
 'Dr. Lee is a skilled dermatologist offering both medical and cosmetic dermatology services.',
 '{"street": "400 Dermatology Center", "city": "Austin", "state": "TX", "zip": "73301", "country": "USA"}', true, true),

('650e8400-e29b-41d4-a716-446655440105', '550e8400-e29b-41d4-a716-446655440105', 'CO-MD-56789', 'Orthopedic Surgery',
 ARRAY['Sports Medicine', 'Joint Replacement'], 10,
 '{"medical_school": "University of Colorado School of Medicine", "residency": "UC Orthopedic Surgery", "fellowship": "Sports Medicine"}',
 '{"board_certifications": ["American Board of Orthopedic Surgery"], "additional_certs": ["Arthroscopy", "Joint Replacement"]}',
 ARRAY['English'], 300.00, 4.5, 94,
 '{"monday": ["06:00-16:00"], "tuesday": ["06:00-16:00"], "wednesday": ["06:00-16:00"], "thursday": ["06:00-16:00"], "friday": ["06:00-14:00"], "saturday": [], "sunday": []}',
 'Dr. Johnson specializes in orthopedic surgery with expertise in sports injuries and joint replacement.',
 '{"street": "500 Orthopedic Clinic", "city": "Denver", "state": "CO", "zip": "80202", "country": "USA"}', true, true);

-- =============================================================================
-- SAMPLE HEALTH METRICS
-- =============================================================================

INSERT INTO health_metrics (user_id, metric_type, value, unit, measured_at, source, notes) VALUES
-- John Doe's metrics
('550e8400-e29b-41d4-a716-446655440001', 'blood_pressure', 120, 'mmHg systolic', '2025-12-15 08:30:00+00', 'manual', 'Morning reading'),
('550e8400-e29b-41d4-a716-446655440001', 'heart_rate', 72, 'bpm', '2025-12-15 08:30:00+00', 'manual', 'Resting heart rate'),
('550e8400-e29b-41d4-a716-446655440001', 'weight', 175.5, 'lbs', '2025-12-15 07:00:00+00', 'manual', 'Weekly weigh-in'),
('550e8400-e29b-41d4-a716-446655440001', 'bmi', 24.2, 'kg/m²', '2025-12-15 07:00:00+00', 'manual', 'Calculated BMI'),

-- Sarah Johnson's metrics
('550e8400-e29b-41d4-a716-446655440002', 'blood_pressure', 110, 'mmHg systolic', '2025-12-14 19:15:00+00', 'device', 'Home monitor'),
('550e8400-e29b-41d4-a716-446655440002', 'heart_rate', 68, 'bpm', '2025-12-14 19:15:00+00', 'device', 'Fitness tracker'),
('550e8400-e29b-41d4-a716-446655440002', 'oxygen_saturation', 98, '%', '2025-12-14 19:15:00+00', 'device', 'Pulse oximeter'),

-- Mike Chen's metrics (diabetic)
('550e8400-e29b-41d4-a716-446655440003', 'blood_sugar', 145, 'mg/dL', '2025-12-16 12:00:00+00', 'device', 'Post-meal reading'),
('550e8400-e29b-41d4-a716-446655440003', 'blood_pressure', 135, 'mmHg systolic', '2025-12-16 08:00:00+00', 'manual', 'Morning reading'),
('550e8400-e29b-41d4-a716-446655440003', 'weight', 190.2, 'lbs', '2025-12-16 07:30:00+00', 'manual', 'Daily monitoring'),

-- Emma Wilson's metrics
('550e8400-e29b-41d4-a716-446655440004', 'heart_rate', 65, 'bpm', '2025-12-17 06:45:00+00', 'device', 'Morning workout'),
('550e8400-e29b-41d4-a716-446655440004', 'weight', 125.0, 'lbs', '2025-12-17 06:30:00+00', 'manual', 'Pre-workout'),

-- David Brown's metrics (hypertensive)
('550e8400-e29b-41d4-a716-446655440005', 'blood_pressure', 145, 'mmHg systolic', '2025-12-16 20:00:00+00', 'device', 'Evening reading'),
('550e8400-e29b-41d4-a716-446655440005', 'heart_rate', 78, 'bpm', '2025-12-16 20:00:00+00', 'device', 'With BP reading'),
('550e8400-e29b-41d4-a716-446655440005', 'weight', 185.8, 'lbs', '2025-12-16 07:00:00+00', 'manual', 'Morning weigh-in');

-- =============================================================================
-- SAMPLE SYMPTOMS & VOICE ANALYSIS
-- =============================================================================

INSERT INTO symptoms (user_id, description, severity, duration_hours, voice_transcript, voice_analysis, ai_assessment, location_on_body, triggers) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Chest pain and shortness of breath during exercise', 'high', 2,
 'I have been experiencing chest pain and shortness of breath when I exercise. It started about two hours ago during my morning jog.',
 '{"stress_level": 0.75, "speech_rate": 140, "pause_frequency": "high", "voice_tremor": 0.3}',
 '{"risk_level": "high", "urgency": true, "recommended_actions": ["Seek immediate medical attention", "Stop physical activity"], "suggested_specialists": ["Cardiologist", "Emergency Medicine"]}',
 'chest', ARRAY['physical_exertion']),

('550e8400-e29b-41d4-a716-446655440002', 'Persistent cough and wheezing', 'medium', 24,
 'I have had a persistent cough and wheezing for about a day now. It seems to get worse at night.',
 '{"stress_level": 0.45, "speech_rate": 120, "pause_frequency": "medium", "voice_tremor": 0.1}',
 '{"risk_level": "medium", "urgency": false, "recommended_actions": ["Monitor symptoms", "Use rescue inhaler if available"], "suggested_specialists": ["Pulmonologist", "Primary Care"]}',
 'chest', ARRAY['nighttime', 'allergens']),

('550e8400-e29b-41d4-a716-446655440003', 'Frequent urination and excessive thirst', 'medium', 72,
 'I have been urinating frequently and feeling very thirsty for the past three days. I am also feeling more tired than usual.',
 '{"stress_level": 0.55, "speech_rate": 110, "pause_frequency": "medium", "voice_tremor": 0.2}',
 '{"risk_level": "medium", "urgency": false, "recommended_actions": ["Check blood sugar levels", "Schedule appointment with doctor"], "suggested_specialists": ["Endocrinologist", "Primary Care"]}',
 'systemic', ARRAY['dehydration', 'high_blood_sugar']),

('550e8400-e29b-41d4-a716-446655440004', 'Headache and neck stiffness', 'medium', 6,
 'I woke up with a severe headache and my neck feels very stiff. The pain is getting worse.',
 '{"stress_level": 0.65, "speech_rate": 95, "pause_frequency": "high", "voice_tremor": 0.4}',
 '{"risk_level": "medium", "urgency": true, "recommended_actions": ["Seek medical evaluation", "Monitor for fever"], "suggested_specialists": ["Neurologist", "Emergency Medicine"]}',
 'head_neck', ARRAY['stress', 'poor_sleep']),

('550e8400-e29b-41d4-a716-446655440005', 'Knee pain and swelling after surgery', 'low', 12,
 'My knee has been painful and swollen since yesterday. I had knee surgery three weeks ago.',
 '{"stress_level": 0.35, "speech_rate": 125, "pause_frequency": "low", "voice_tremor": 0.1}',
 '{"risk_level": "low", "urgency": false, "recommended_actions": ["Apply ice", "Elevate leg", "Contact surgeon if worsening"], "suggested_specialists": ["Orthopedic Surgery", "Physical Therapy"]}',
 'knee', ARRAY['post_surgery', 'physical_activity']);

-- =============================================================================
-- SAMPLE PREDICTIONS
-- =============================================================================

INSERT INTO predictions (user_id, disease, probability, confidence_score, risk_factors, model_version, based_on_data, recommendations, expires_at) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Heart Disease', 0.3250, 0.8500,
 '{"age": 39, "family_history": true, "exercise_frequency": "moderate", "smoking": false, "bmi": 24.2}',
 'v1.2.0', '{"health_metrics": ["blood_pressure", "heart_rate", "bmi"], "family_history": ["heart_disease"], "lifestyle": ["exercise", "diet"]}',
 ARRAY['Maintain regular exercise', 'Monitor blood pressure', 'Annual cardiac screening'], '2026-01-17 00:00:00+00'),

('550e8400-e29b-41d4-a716-446655440002', 'Asthma Exacerbation', 0.6750, 0.9200,
 '{"existing_asthma": true, "allergen_exposure": "high", "medication_compliance": "good", "recent_symptoms": true}',
 'v1.1.5', '{"symptoms": ["cough", "wheezing"], "medical_history": ["asthma"], "environmental": ["allergens"]}',
 ARRAY['Continue rescue inhaler', 'Avoid known triggers', 'Consider allergy testing'], '2026-01-17 00:00:00+00'),

('550e8400-e29b-41d4-a716-446655440003', 'Diabetes Complications', 0.4500, 0.8800,
 '{"existing_diabetes": true, "blood_sugar_control": "fair", "duration_years": 5, "complications_present": false}',
 'v1.3.1', '{"health_metrics": ["blood_sugar", "bmi"], "medical_history": ["diabetes"], "symptoms": ["frequent_urination"]}',
 ARRAY['Improve blood sugar control', 'Regular eye exams', 'Foot care monitoring'], '2026-01-17 00:00:00+00'),

('550e8400-e29b-41d4-a716-446655440004', 'Migraine', 0.7200, 0.7500,
 '{"headache_frequency": "increasing", "stress_level": "high", "sleep_quality": "poor", "family_history": false}',
 'v1.0.8', '{"symptoms": ["headache", "neck_stiffness"], "lifestyle": ["stress", "sleep"], "demographics": ["age", "gender"]}',
 ARRAY['Stress management techniques', 'Improve sleep hygiene', 'Keep headache diary'], '2026-01-17 00:00:00+00'),

('550e8400-e29b-41d4-a716-446655440005', 'Hypertension', 0.8100, 0.9100,
 '{"existing_hypertension": true, "medication_compliance": "good", "lifestyle_factors": "improving", "family_history": true}',
 'v1.2.3', '{"health_metrics": ["blood_pressure"], "medical_history": ["hypertension"], "family_history": ["heart_disease"]}',
 ARRAY['Continue medication', 'Low sodium diet', 'Regular exercise', 'Weight management'], '2026-01-17 00:00:00+00');

-- =============================================================================
-- SAMPLE APPOINTMENTS & BIDS
-- =============================================================================

INSERT INTO appointments (id, patient_id, doctor_id, status, symptoms_summary, urgency_level, preferred_date, final_fee, scheduled_at) VALUES
('750e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', '650e8400-e29b-41d4-a716-446655440103', 'completed',
 'Chest pain during exercise, shortness of breath', 'high', '2025-12-16 14:00:00+00', 250.00, '2025-12-16 14:00:00+00'),

('750e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', '650e8400-e29b-41d4-a716-446655440102', 'confirmed',
 'Persistent cough and wheezing, asthma-related', 'medium', '2025-12-18 10:00:00+00', 120.00, '2025-12-18 10:00:00+00'),

('750e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440003', NULL, 'bidding',
 'Frequent urination, excessive thirst, diabetes management', 'medium', '2025-12-19 15:00:00+00', NULL, NULL),

('750e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440004', NULL, 'pending',
 'Severe headache with neck stiffness', 'medium', '2025-12-20 09:00:00+00', NULL, NULL),

('750e8400-e29b-41d4-a716-446655440005', '550e8400-e29b-41d4-a716-446655440005', '650e8400-e29b-41d4-a716-446655440105', 'confirmed',
 'Post-surgical knee pain and swelling', 'low', '2025-12-21 11:00:00+00', 300.00, '2025-12-21 11:00:00+00');

INSERT INTO appointment_bids (appointment_id, doctor_id, bid_amount, estimated_duration, available_slots, message) VALUES
('750e8400-e29b-41d4-a716-446655440003', '650e8400-e29b-41d4-a716-446655440101', 140.00, 30,
 '["2025-12-19T15:00:00Z", "2025-12-19T16:00:00Z", "2025-12-20T09:00:00Z"]',
 'I have extensive experience with diabetes management and can help optimize your treatment plan.'),

('750e8400-e29b-41d4-a716-446655440003', '650e8400-e29b-41d4-a716-446655440102', 160.00, 45,
 '["2025-12-19T14:00:00Z", "2025-12-19T15:30:00Z"]',
 'As an endocrinology specialist, I can provide comprehensive diabetes care and monitoring.'),

('750e8400-e29b-41d4-a716-446655440004', '650e8400-e29b-41d4-a716-446655440101', 150.00, 30,
 '["2025-12-20T09:00:00Z", "2025-12-20T10:00:00Z", "2025-12-20T14:00:00Z"]',
 'Headaches with neck stiffness require careful evaluation. I can assess and provide appropriate treatment.');

-- =============================================================================
-- SAMPLE THERAPY SESSIONS
-- =============================================================================

INSERT INTO therapy_sessions (user_id, exercise_type, duration_seconds, repetitions_completed, repetitions_target, form_accuracy, pain_level, pain_detected, points_earned, streak_count, pose_data, session_notes) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'arm_raises', 300, 10, 10, 85.5, 2, false, 120, 5,
 '{"accuracy_per_rep": [0.9, 0.85, 0.88, 0.82, 0.87, 0.89, 0.84, 0.86, 0.83, 0.81], "average_angle": 165, "consistency": 0.85}',
 'Good session, slight fatigue towards the end'),

('550e8400-e29b-41d4-a716-446655440002', 'knee_bends', 240, 8, 10, 92.0, 1, false, 100, 3,
 '{"accuracy_per_rep": [0.95, 0.92, 0.94, 0.89, 0.91, 0.93, 0.88, 0.90], "average_angle": 90, "consistency": 0.92}',
 'Excellent form, completed 8 reps with high accuracy'),

('550e8400-e29b-41d4-a716-446655440003', 'neck_rotations', 180, 12, 12, 78.3, 3, true, 80, 2,
 '{"accuracy_per_rep": [0.82, 0.75, 0.79, 0.81, 0.76, 0.83, 0.74, 0.80, 0.77, 0.85, 0.79, 0.73], "range_of_motion": 0.78, "consistency": 0.78}',
 'Some discomfort detected, reduced intensity automatically'),

('550e8400-e29b-41d4-a716-446655440004', 'arm_raises', 360, 15, 15, 95.2, 0, false, 180, 7,
 '{"accuracy_per_rep": [0.96, 0.94, 0.97, 0.95, 0.96, 0.93, 0.98, 0.94, 0.95, 0.97, 0.92, 0.96, 0.94, 0.98, 0.95], "average_angle": 170, "consistency": 0.95}',
 'Perfect session! Excellent form throughout'),

('550e8400-e29b-41d4-a716-446655440005', 'knee_bends', 420, 6, 10, 65.8, 5, true, 40, 1,
 '{"accuracy_per_rep": [0.70, 0.65, 0.68, 0.62, 0.64, 0.67], "average_angle": 75, "consistency": 0.66}',
 'Post-surgery limitations, pain detected and session modified');

-- =============================================================================
-- SAMPLE THERAPY PROGRESS
-- =============================================================================

INSERT INTO therapy_progress (user_id, exercise_type, week_start_date, sessions_completed, total_points, average_accuracy, average_pain_level, improvement_percentage) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'arm_raises', '2025-12-16', 5, 600, 85.5, 2.0, 15.2),
('550e8400-e29b-41d4-a716-446655440002', 'knee_bends', '2025-12-16', 3, 300, 92.0, 1.0, 8.5),
('550e8400-e29b-41d4-a716-446655440003', 'neck_rotations', '2025-12-16', 2, 160, 78.3, 3.0, -2.1),
('550e8400-e29b-41d4-a716-446655440004', 'arm_raises', '2025-12-16', 7, 1260, 95.2, 0.0, 22.8),
('550e8400-e29b-41d4-a716-446655440005', 'knee_bends', '2025-12-16', 1, 40, 65.8, 5.0, -10.5);

-- =============================================================================
-- SAMPLE LEADERBOARD
-- =============================================================================

INSERT INTO therapy_leaderboard (user_id, anonymous_name, total_points, current_streak, longest_streak, level_achieved, badges, last_activity) VALUES
('550e8400-e29b-41d4-a716-446655440004', 'HealthWarrior22', 1260, 7, 12, 3, '["Perfect Form", "Week Warrior", "Consistency King"]', '2025-12-17 08:00:00+00'),
('550e8400-e29b-41d4-a716-446655440001', 'FitnessFan85', 600, 5, 8, 2, '["First Steps", "Week Warrior"]', '2025-12-17 07:30:00+00'),
('550e8400-e29b-41d4-a716-446655440002', 'ActiveAmy90', 300, 3, 5, 1, '["First Steps"]', '2025-12-16 19:00:00+00'),
('550e8400-e29b-41d4-a716-446655440003', 'ProgressPro78', 160, 2, 4, 1, '["First Steps"]', '2025-12-16 15:00:00+00'),
('550e8400-e29b-41d4-a716-446655440005', 'RecoveryRock82', 40, 1, 2, 1, '[]', '2025-12-16 10:00:00+00');

-- =============================================================================
-- SAMPLE NOTIFICATIONS
-- =============================================================================

INSERT INTO notifications (user_id, title, message, type, is_read, action_url, metadata) VALUES
('550e8400-e29b-41d4-a716-446655440001', 'Appointment Confirmed', 'Your appointment with Dr. Patel has been confirmed for Dec 16 at 2:00 PM', 'appointment', true, '/appointments/750e8400-e29b-41d4-a716-446655440001', '{"appointment_id": "750e8400-e29b-41d4-a716-446655440001"}'),

('550e8400-e29b-41d4-a716-446655440002', 'New Health Prediction', 'Your asthma risk assessment has been updated based on recent symptoms', 'health_alert', false, '/health/predictions', '{"prediction_type": "asthma_exacerbation"}'),

('550e8400-e29b-41d4-a716-446655440003', 'New Bid Received', 'Dr. Smith has placed a bid on your appointment request', 'appointment', false, '/appointments/750e8400-e29b-41d4-a716-446655440003/bids', '{"appointment_id": "750e8400-e29b-41d4-a716-446655440003", "doctor_name": "Dr. Smith"}'),

('550e8400-e29b-41d4-a716-446655440004', 'Therapy Milestone', 'Congratulations! You have reached Level 3 in your therapy program', 'system', false, '/therapy/progress', '{"achievement": "level_3", "points_earned": 180}'),

('550e8400-e29b-41d4-a716-446655440005', 'Recovery Update', 'Your post-surgery recovery is progressing well. Keep up the gentle exercises!', 'health_alert', false, '/therapy/sessions', '{"recovery_stage": "week_3", "recommendation": "continue_gentle_exercises"}')