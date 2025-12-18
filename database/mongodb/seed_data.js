// HealthSync AI - MongoDB Sample Data
// Sample documents for development and testing

// =============================================================================
// FAMILY GRAPH SAMPLE DATA
// =============================================================================

// John Doe's family health graph
db.family_graph.insertOne({
  user_id: "550e8400-e29b-41d4-a716-446655440001",
  family_members: [
    {
      id: "fm_001_father",
      relation: "father",
      name: "Robert Doe",
      birth_year: 1955,
      health_conditions: ["diabetes", "hypertension", "heart_disease"],
      age_of_onset: {
        "diabetes": 45,
        "hypertension": 40,
        "heart_disease": 58
      },
      genetic_markers: {
        "brca1": "negative",
        "brca2": "unknown",
        "apoe4": "unknown"
      },
      lifestyle_factors: {
        "smoking": "former",
        "alcohol": "moderate",
        "exercise": "light",
        "diet": "average"
      }
    },
    {
      id: "fm_001_mother",
      relation: "mother",
      name: "Mary Doe",
      birth_year: 1958,
      health_conditions: ["osteoporosis", "depression"],
      age_of_onset: {
        "osteoporosis": 62,
        "depression": 35
      },
      genetic_markers: {
        "brca1": "negative",
        "brca2": "negative",
        "apoe4": "unknown"
      },
      lifestyle_factors: {
        "smoking": "never",
        "alcohol": "light",
        "exercise": "moderate",
        "diet": "good"
      }
    },
    {
      id: "fm_001_grandfather_paternal",
      relation: "grandfather_paternal",
      name: "William Doe",
      birth_year: 1925,
      death_year: 1995,
      health_conditions: ["diabetes", "stroke", "heart_disease"],
      age_of_onset: {
        "diabetes": 50,
        "stroke": 68,
        "heart_disease": 65
      },
      lifestyle_factors: {
        "smoking": "current",
        "alcohol": "heavy",
        "exercise": "sedentary",
        "diet": "poor"
      }
    },
    {
      id: "fm_001_grandmother_maternal",
      relation: "grandmother_maternal",
      name: "Helen Smith",
      birth_year: 1930,
      death_year: 2010,
      health_conditions: ["alzheimers", "hypertension"],
      age_of_onset: {
        "alzheimers": 75,
        "hypertension": 55
      },
      lifestyle_factors: {
        "smoking": "never",
        "alcohol": "none",
        "exercise": "light",
        "diet": "good"
      }
    }
  ],
  inherited_risks: {
    "diabetes": 0.65,
    "heart_disease": 0.72,
    "hypertension": 0.58,
    "stroke": 0.35,
    "alzheimers": 0.25,
    "depression": 0.30,
    "osteoporosis": 0.20,
    "cancer": 0.15
  },
  risk_calculations: {
    algorithm_version: "v2.1.0",
    calculation_date: new Date("2025-12-17T10:00:00Z"),
    factors_considered: [
      "direct_family_history",
      "age_of_onset_patterns",
      "lifestyle_factors",
      "genetic_markers",
      "user_current_health"
    ],
    confidence_scores: {
      "diabetes": 0.85,
      "heart_disease": 0.90,
      "hypertension": 0.80,
      "stroke": 0.70,
      "alzheimers": 0.60,
      "depression": 0.75,
      "osteoporosis": 0.65,
      "cancer": 0.50
    }
  },
  family_tree_visualization: {
    nodes: [
      { id: "user", name: "John Doe", generation: 0, x: 400, y: 300 },
      { id: "fm_001_father", name: "Robert Doe", generation: 1, x: 200, y: 150 },
      { id: "fm_001_mother", name: "Mary Doe", generation: 1, x: 600, y: 150 },
      { id: "fm_001_grandfather_paternal", name: "William Doe", generation: 2, x: 100, y: 50 },
      { id: "fm_001_grandmother_maternal", name: "Helen Smith", generation: 2, x: 700, y: 50 }
    ],
    links: [
      { source: "user", target: "fm_001_father", type: "parent" },
      { source: "user", target: "fm_001_mother", type: "parent" },
      { source: "fm_001_father", target: "fm_001_grandfather_paternal", type: "parent" },
      { source: "fm_001_mother", target: "fm_001_grandmother_maternal", type: "parent" }
    ],
    layout_config: {
      width: 800,
      height: 400,
      node_radius: 30,
      link_distance: 100
    }
  },
  updated_at: new Date("2025-12-17T10:00:00Z"),
  created_at: new Date("2025-12-15T08:00:00Z")
});

// Sarah Johnson's family health graph
db.family_graph.insertOne({
  user_id: "550e8400-e29b-41d4-a716-446655440002",
  family_members: [
    {
      id: "fm_002_mother",
      relation: "mother",
      name: "Linda Johnson",
      birth_year: 1965,
      health_conditions: ["asthma", "allergies"],
      age_of_onset: {
        "asthma": 12,
        "allergies": 8
      },
      lifestyle_factors: {
        "smoking": "never",
        "alcohol": "light",
        "exercise": "active",
        "diet": "excellent"
      }
    },
    {
      id: "fm_002_brother",
      relation: "brother",
      name: "Mike Johnson",
      birth_year: 1988,
      health_conditions: ["asthma"],
      age_of_onset: {
        "asthma": 15
      },
      lifestyle_factors: {
        "smoking": "never",
        "alcohol": "moderate",
        "exercise": "active",
        "diet": "good"
      }
    }
  ],
  inherited_risks: {
    "asthma": 0.80,
    "allergies": 0.70,
    "heart_disease": 0.15,
    "diabetes": 0.10,
    "hypertension": 0.20,
    "cancer": 0.12,
    "stroke": 0.08,
    "depression": 0.15
  },
  risk_calculations: {
    algorithm_version: "v2.1.0",
    calculation_date: new Date("2025-12-16T15:30:00Z"),
    factors_considered: [
      "direct_family_history",
      "respiratory_conditions",
      "lifestyle_factors"
    ],
    confidence_scores: {
      "asthma": 0.95,
      "allergies": 0.90,
      "heart_disease": 0.60,
      "diabetes": 0.55
    }
  },
  updated_at: new Date("2025-12-16T15:30:00Z"),
  created_at: new Date("2025-12-14T12:00:00Z")
});

// =============================================================================
// HEALTH EVENTS SAMPLE DATA
// =============================================================================

// John Doe's health events
db.health_events.insertMany([
  {
    user_id: "550e8400-e29b-41d4-a716-446655440001",
    event_type: "voice_analysis",
    timestamp: new Date("2025-12-17T09:30:00Z"),
    data: {
      session_id: "voice_session_001",
      transcript: "I have been experiencing chest pain and shortness of breath when I exercise",
      duration_seconds: 45,
      symptoms_mentioned: ["chest_pain", "shortness_of_breath"]
    },
    ai_analysis: {
      risk_assessment: "high",
      confidence_score: 0.85,
      recommendations: [
        "Seek immediate medical attention",
        "Stop physical activity until evaluated",
        "Monitor symptoms closely"
      ],
      follow_up_needed: true,
      specialist_referral: "Cardiologist"
    },
    source: "user_input",
    metadata: {
      stress_level: 0.75,
      location: {
        type: "Point",
        coordinates: [-122.4194, 37.7749] // San Francisco
      }
    }
  },
  {
    user_id: "550e8400-e29b-41d4-a716-446655440001",
    event_type: "health_metric",
    timestamp: new Date("2025-12-17T08:30:00Z"),
    data: {
      metric_type: "blood_pressure",
      value: 145,
      unit: "mmHg systolic",
      reading_context: "morning_routine"
    },
    ai_analysis: {
      risk_assessment: "medium",
      confidence_score: 0.70,
      recommendations: [
        "Monitor blood pressure regularly",
        "Consider lifestyle modifications",
        "Discuss with primary care physician"
      ],
      follow_up_needed: false
    },
    source: "device",
    metadata: {
      device_info: {
        type: "blood_pressure_monitor",
        model: "Omron HEM-7120",
        calibration_date: "2025-12-01"
      }
    }
  },
  {
    user_id: "550e8400-e29b-41d4-a716-446655440001",
    event_type: "prediction_update",
    timestamp: new Date("2025-12-17T10:00:00Z"),
    data: {
      disease: "heart_disease",
      new_probability: 0.42,
      previous_probability: 0.35,
      change_reason: "new_symptoms_and_family_history"
    },
    ai_analysis: {
      risk_assessment: "medium",
      confidence_score: 0.88,
      recommendations: [
        "Schedule cardiac evaluation",
        "Implement heart-healthy lifestyle changes",
        "Monitor cardiovascular symptoms"
      ],
      follow_up_needed: true,
      specialist_referral: "Cardiologist"
    },
    correlations: [
      {
        correlation_type: "symptom_pattern",
        strength: 0.85
      }
    ],
    source: "ai_analysis"
  }
]);

// Sarah Johnson's health events
db.health_events.insertMany([
  {
    user_id: "550e8400-e29b-41d4-a716-446655440002",
    event_type: "symptom_report",
    timestamp: new Date("2025-12-16T20:15:00Z"),
    data: {
      symptoms: ["persistent_cough", "wheezing"],
      severity: "medium",
      duration_hours: 24,
      triggers: ["nighttime", "allergens"]
    },
    ai_analysis: {
      risk_assessment: "medium",
      confidence_score: 0.80,
      recommendations: [
        "Use rescue inhaler as needed",
        "Avoid known allergens",
        "Monitor for worsening symptoms"
      ],
      follow_up_needed: false,
      specialist_referral: "Pulmonologist"
    },
    source: "user_input",
    metadata: {
      stress_level: 0.45,
      weather: {
        pollen_count: "high",
        air_quality_index: 85,
        temperature: 68
      }
    }
  }
]);

// =============================================================================
// ML MODELS SAMPLE DATA
// =============================================================================

// Disease prediction model for diabetes
db.ml_models.insertOne({
  user_id: null, // Global model
  model_type: "disease_prediction",
  model_data: new BinData(0, ""), // Placeholder for actual binary model data
  model_config: {
    algorithm: "RandomForestClassifier",
    hyperparameters: {
      n_estimators: 100,
      max_depth: 10,
      min_samples_split: 5,
      random_state: 42
    },
    feature_columns: [
      "age", "bmi", "blood_pressure_systolic", "blood_pressure_diastolic",
      "family_history_diabetes", "exercise_frequency", "diet_quality",
      "smoking_status", "alcohol_consumption"
    ],
    target_column: "diabetes_risk",
    preprocessing_steps: [
      "StandardScaler",
      "LabelEncoder for categorical variables",
      "Feature selection using SelectKBest"
    ]
  },
  training_data: {
    sample_count: 10000,
    feature_count: 9,
    data_sources: ["synthetic_health_data", "anonymized_patient_records"],
    date_range: {
      start_date: new Date("2024-01-01T00:00:00Z"),
      end_date: new Date("2025-12-01T00:00:00Z")
    }
  },
  accuracy_metrics: {
    accuracy: 0.87,
    precision: 0.85,
    recall: 0.82,
    f1_score: 0.83,
    auc_roc: 0.91,
    cross_validation_scores: [0.86, 0.88, 0.85, 0.89, 0.87]
  },
  validation_results: {
    test_accuracy: 0.85,
    confusion_matrix: [[1800, 200], [150, 1850]],
    feature_importance: {
      "family_history_diabetes": 0.25,
      "bmi": 0.20,
      "age": 0.18,
      "blood_pressure_systolic": 0.15,
      "exercise_frequency": 0.12,
      "diet_quality": 0.10
    },
    prediction_examples: [
      {
        input: {"age": 45, "bmi": 28.5, "family_history_diabetes": 1},
        prediction: 0.72,
        actual: 1
      }
    ]
  },
  training_date: new Date("2025-12-15T00:00:00Z"),
  version: "v1.2.0",
  is_active: true,
  deployment_info: {
    deployed_at: new Date("2025-12-15T12:00:00Z"),
    deployment_environment: "production",
    api_endpoint: "/api/v1/predict/diabetes",
    performance_monitoring: {
      requests_per_day: 150,
      average_response_time_ms: 45,
      error_rate: 0.02
    }
  }
});

// Voice stress analysis model
db.ml_models.insertOne({
  user_id: null, // Global model
  model_type: "voice_stress",
  model_data: new BinData(0, ""), // Placeholder for actual binary model data
  model_config: {
    algorithm: "SupportVectorMachine",
    hyperparameters: {
      kernel: "rbf",
      C: 1.0,
      gamma: "scale"
    },
    feature_columns: [
      "mfcc_mean", "mfcc_std", "pitch_mean", "pitch_variance",
      "energy_rms", "zero_crossing_rate", "spectral_centroid",
      "jitter", "shimmer", "speech_rate", "pause_frequency"
    ],
    target_column: "stress_level",
    preprocessing_steps: [
      "MinMaxScaler",
      "PCA for dimensionality reduction",
      "Feature normalization"
    ]
  },
  training_data: {
    sample_count: 5000,
    feature_count: 11,
    data_sources: ["voice_recordings", "stress_annotations"],
    date_range: {
      start_date: new Date("2024-06-01T00:00:00Z"),
      end_date: new Date("2025-12-01T00:00:00Z")
    }
  },
  accuracy_metrics: {
    accuracy: 0.78,
    precision: 0.76,
    recall: 0.80,
    f1_score: 0.78,
    auc_roc: 0.84,
    cross_validation_scores: [0.76, 0.79, 0.77, 0.80, 0.78]
  },
  training_date: new Date("2025-12-10T00:00:00Z"),
  version: "v1.1.5",
  is_active: true
});

// =============================================================================
// VOICE SESSIONS SAMPLE DATA
// =============================================================================

db.voice_sessions.insertOne({
  user_id: "550e8400-e29b-41d4-a716-446655440001",
  session_id: "voice_session_001",
  audio_metadata: {
    duration_seconds: 45.2,
    sample_rate: 44100,
    channels: 1,
    format: "wav",
    file_size_bytes: 1992600
  },
  audio_features: {
    mfcc: [
      [-12.5, -8.2, 3.1, -2.4, 1.8, -0.9, 2.3, -1.1, 0.7, -0.3, 1.2, -0.8, 0.4],
      [-11.8, -7.9, 3.4, -2.1, 1.9, -0.7, 2.1, -1.3, 0.8, -0.2, 1.0, -0.9, 0.3]
    ],
    pitch: {
      mean: 145.6,
      std: 28.3,
      min: 98.2,
      max: 210.4,
      variance: 801.2
    },
    energy: {
      rms_energy: 0.045,
      zero_crossing_rate: 0.078,
      spectral_centroid: 2150.3
    },
    rhythm: {
      tempo: 120.5,
      beat_frames: [22, 44, 66, 88, 110],
      onset_frames: [15, 35, 58, 82, 105]
    },
    voice_quality: {
      jitter: 0.012,
      shimmer: 0.089,
      harmonics_to_noise_ratio: 15.2
    }
  },
  speech_analysis: {
    transcript: "I have been experiencing chest pain and shortness of breath when I exercise. It started about two hours ago during my morning jog.",
    word_count: 22,
    speech_rate: 140.5,
    pause_analysis: {
      total_pauses: 8,
      average_pause_duration: 0.85,
      pause_frequency: "high"
    },
    disfluencies: {
      filler_words: 2,
      repetitions: 1,
      false_starts: 0
    }
  },
  stress_indicators: {
    stress_level: 0.75,
    anxiety_score: 0.68,
    fatigue_score: 0.32,
    pain_indicators: 0.82,
    confidence_level: 0.45
  },
  analysis_results: {
    risk_assessment: "high",
    urgency_flag: true,
    recommended_actions: [
      "Seek immediate medical attention",
      "Stop physical activity until evaluated",
      "Call emergency services if symptoms worsen"
    ],
    suggested_specialists: ["Cardiologist", "Emergency Medicine"],
    confidence_score: 0.85,
    reasoning: "High stress indicators combined with cardiovascular symptoms during exercise suggest potential cardiac event. Voice analysis shows elevated anxiety and pain markers."
  },
  timestamp: new Date("2025-12-17T09:30:00Z"),
  processing_metadata: {
    processing_time_ms: 2850,
    ai_services_used: ["whisper", "groq_llm", "librosa"],
    model_versions: {
      "whisper": "v3",
      "groq_llm": "llama-3.1-70b-versatile",
      "voice_stress": "v1.1.5"
    },
    api_costs: {
      "whisper": 0.006,
      "groq_llm": 0.012,
      "total_usd": 0.018
    }
  }
});

// =============================================================================
// AR ANALYSIS SAMPLE DATA
// =============================================================================

db.ar_analysis.insertOne({
  user_id: "550e8400-e29b-41d4-a716-446655440003",
  scan_id: "ar_scan_001",
  image_metadata: {
    width: 1920,
    height: 1080,
    format: "jpeg",
    file_size_bytes: 2456789,
    camera_info: {
      make: "Apple",
      model: "iPhone 15 Pro",
      focal_length: 26,
      iso: 100
    },
    location: {
      type: "Point",
      coordinates: [-104.9903, 39.7392] // Denver
    }
  },
  image_analysis: {
    description: "A prescription paper with medication details and doctor information",
    detected_objects: ["prescription_pad", "text", "medical_logo", "signature"],
    scene_classification: "medical_document",
    confidence_scores: {
      "prescription_pad": 0.95,
      "text": 0.92,
      "medical_logo": 0.78,
      "signature": 0.65
    },
    visual_features: {
      "dominant_colors": ["white", "blue", "black"],
      "text_regions": 4,
      "layout_type": "structured_document"
    }
  },
  ocr_results: {
    raw_text: "Dr. Michael Johnson, MD\nOrthopedic Surgery\n500 Orthopedic Clinic, Denver, CO\n\nPatient: David Brown\nDOB: 09/30/1982\nDate: 12/16/2025\n\nRx:\nIbuprofen 600mg\nTake 1 tablet every 8 hours with food\nQuantity: 30 tablets\nRefills: 2\n\nFor post-surgical pain management\n\nSignature: Dr. M. Johnson",
    structured_text: [
      {
        text: "Dr. Michael Johnson, MD",
        confidence: 0.98,
        bounding_box: {"x": 50, "y": 20, "width": 200, "height": 25},
        text_type: "doctor_name"
      },
      {
        text: "Ibuprofen 600mg",
        confidence: 0.95,
        bounding_box: {"x": 50, "y": 180, "width": 150, "height": 20},
        text_type: "medication"
      },
      {
        text: "Take 1 tablet every 8 hours with food",
        confidence: 0.92,
        bounding_box: {"x": 50, "y": 200, "width": 280, "height": 20},
        text_type: "instruction"
      },
      {
        text: "12/16/2025",
        confidence: 0.89,
        bounding_box: {"x": 250, "y": 120, "width": 80, "height": 18},
        text_type: "date"
      }
    ],
    language_detected: "en",
    overall_confidence: 0.93
  },
  medication_extraction: {
    medications: [
      {
        name: "Ibuprofen",
        generic_name: "Ibuprofen",
        dosage: "600mg",
        frequency: "every 8 hours",
        instructions: "Take with food",
        duration: "30 tablets (10 days supply)",
        confidence: 0.95
      }
    ],
    prescriber_info: {
      doctor_name: "Dr. Michael Johnson, MD",
      clinic_name: "500 Orthopedic Clinic",
      date_prescribed: "12/16/2025"
    }
  },
  safety_analysis: {
    warnings: [
      "Take with food to reduce stomach irritation",
      "Do not exceed recommended dosage",
      "May cause drowsiness"
    ],
    drug_interactions: [
      {
        drug: "Warfarin",
        severity: "moderate",
        description: "May increase bleeding risk"
      }
    ],
    allergy_alerts: [],
    dosage_concerns: []
  },
  timestamp: new Date("2025-12-16T14:30:00Z"),
  processing_metadata: {
    processing_time_ms: 4200,
    ai_services_used: ["blip2", "trocr", "drug_database"],
    model_versions: {
      "blip2": "v2.0",
      "trocr": "v1.1",
      "drug_database": "v2025.12"
    },
    api_costs: {
      "blip2": 0.008,
      "trocr": 0.012,
      "total_usd": 0.020
    }
  }
});

print("MongoDB sample data inserted successfully!");
print("Collections created:");
print("- family_graph: " + db.family_graph.countDocuments() + " documents");
print("- health_events: " + db.health_events.countDocuments() + " documents");
print("- ml_models: " + db.ml_models.countDocuments() + " documents");
print("- voice_sessions: " + db.voice_sessions.countDocuments() + " documents");
print("- ar_analysis: " + db.ar_analysis.countDocuments() + " documents");