// HealthSync AI - MongoDB Collections Schema
// MongoDB Atlas Document Database Design

// =============================================================================
// FAMILY HEALTH GRAPH COLLECTION
// =============================================================================

// Collection: family_graph
// Purpose: Store complex family relationships and inherited health risks
db.createCollection("family_graph", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "family_members", "inherited_risks", "updated_at"],
      properties: {
        user_id: {
          bsonType: "string",
          description: "UUID of the user (matches PostgreSQL users.id)"
        },
        family_members: {
          bsonType: "array",
          description: "Array of family member objects",
          items: {
            bsonType: "object",
            required: ["id", "relation", "health_conditions"],
            properties: {
              id: {
                bsonType: "string",
                description: "Unique identifier for family member"
              },
              relation: {
                bsonType: "string",
                enum: ["father", "mother", "brother", "sister", "son", "daughter", 
                       "grandfather_paternal", "grandmother_paternal", 
                       "grandfather_maternal", "grandmother_maternal",
                       "uncle_paternal", "aunt_paternal", "uncle_maternal", "aunt_maternal",
                       "cousin", "spouse"],
                description: "Relationship to the user"
              },
              name: {
                bsonType: "string",
                description: "Name of family member (optional for privacy)"
              },
              birth_year: {
                bsonType: "int",
                minimum: 1900,
                maximum: 2025,
                description: "Birth year for age calculations"
              },
              death_year: {
                bsonType: "int",
                minimum: 1900,
                maximum: 2025,
                description: "Death year if deceased"
              },
              health_conditions: {
                bsonType: "array",
                description: "List of diagnosed health conditions",
                items: {
                  bsonType: "string"
                }
              },
              age_of_onset: {
                bsonType: "object",
                description: "Age when each condition was diagnosed",
                additionalProperties: {
                  bsonType: "int",
                  minimum: 0,
                  maximum: 120
                }
              },
              genetic_markers: {
                bsonType: "object",
                description: "Known genetic test results",
                properties: {
                  brca1: { bsonType: "string", enum: ["positive", "negative", "unknown"] },
                  brca2: { bsonType: "string", enum: ["positive", "negative", "unknown"] },
                  apoe4: { bsonType: "string", enum: ["positive", "negative", "unknown"] },
                  factor_v_leiden: { bsonType: "string", enum: ["positive", "negative", "unknown"] }
                }
              },
              lifestyle_factors: {
                bsonType: "object",
                properties: {
                  smoking: { bsonType: "string", enum: ["never", "former", "current"] },
                  alcohol: { bsonType: "string", enum: ["none", "light", "moderate", "heavy"] },
                  exercise: { bsonType: "string", enum: ["sedentary", "light", "moderate", "active"] },
                  diet: { bsonType: "string", enum: ["poor", "average", "good", "excellent"] }
                }
              }
            }
          }
        },
        inherited_risks: {
          bsonType: "object",
          description: "Calculated inherited disease risks",
          properties: {
            diabetes: { bsonType: "double", minimum: 0, maximum: 1 },
            heart_disease: { bsonType: "double", minimum: 0, maximum: 1 },
            cancer: { bsonType: "double", minimum: 0, maximum: 1 },
            hypertension: { bsonType: "double", minimum: 0, maximum: 1 },
            stroke: { bsonType: "double", minimum: 0, maximum: 1 },
            alzheimers: { bsonType: "double", minimum: 0, maximum: 1 },
            depression: { bsonType: "double", minimum: 0, maximum: 1 },
            osteoporosis: { bsonType: "double", minimum: 0, maximum: 1 }
          }
        },
        risk_calculations: {
          bsonType: "object",
          description: "Detailed risk calculation metadata",
          properties: {
            algorithm_version: { bsonType: "string" },
            calculation_date: { bsonType: "date" },
            factors_considered: { bsonType: "array", items: { bsonType: "string" } },
            confidence_scores: {
              bsonType: "object",
              additionalProperties: { bsonType: "double", minimum: 0, maximum: 1 }
            }
          }
        },
        family_tree_visualization: {
          bsonType: "object",
          description: "D3.js visualization data",
          properties: {
            nodes: { bsonType: "array" },
            links: { bsonType: "array" },
            layout_config: { bsonType: "object" }
          }
        },
        updated_at: {
          bsonType: "date",
          description: "Last update timestamp"
        },
        created_at: {
          bsonType: "date",
          description: "Creation timestamp"
        }
      }
    }
  }
});

// =============================================================================
// HEALTH EVENTS TIMELINE COLLECTION
// =============================================================================

// Collection: health_events
// Purpose: Store chronological health events and AI analysis results
db.createCollection("health_events", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "event_type", "timestamp", "data"],
      properties: {
        user_id: {
          bsonType: "string",
          description: "UUID of the user"
        },
        event_type: {
          bsonType: "string",
          enum: ["symptom_report", "voice_analysis", "ar_scan", "therapy_session", 
                 "health_metric", "prediction_update", "appointment", "medication_change"],
          description: "Type of health event"
        },
        timestamp: {
          bsonType: "date",
          description: "When the event occurred"
        },
        data: {
          bsonType: "object",
          description: "Event-specific data payload"
        },
        ai_analysis: {
          bsonType: "object",
          description: "AI-generated insights for this event",
          properties: {
            risk_assessment: { bsonType: "string", enum: ["low", "medium", "high", "critical"] },
            confidence_score: { bsonType: "double", minimum: 0, maximum: 1 },
            recommendations: { bsonType: "array", items: { bsonType: "string" } },
            follow_up_needed: { bsonType: "bool" },
            specialist_referral: { bsonType: "string" }
          }
        },
        correlations: {
          bsonType: "array",
          description: "Related events or patterns",
          items: {
            bsonType: "object",
            properties: {
              event_id: { bsonType: "objectId" },
              correlation_type: { bsonType: "string" },
              strength: { bsonType: "double", minimum: 0, maximum: 1 }
            }
          }
        },
        source: {
          bsonType: "string",
          enum: ["user_input", "device", "ai_analysis", "doctor_note", "system"],
          description: "Source of the event data"
        },
        metadata: {
          bsonType: "object",
          description: "Additional metadata",
          properties: {
            device_info: { bsonType: "object" },
            location: { bsonType: "object" },
            weather: { bsonType: "object" },
            stress_level: { bsonType: "double", minimum: 0, maximum: 1 }
          }
        }
      }
    }
  }
});

// =============================================================================
// ML MODELS COLLECTION
// =============================================================================

// Collection: ml_models
// Purpose: Store trained ML models and their metadata
db.createCollection("ml_models", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "model_type", "model_data", "training_date", "version"],
      properties: {
        user_id: {
          bsonType: "string",
          description: "UUID of the user (null for global models)"
        },
        model_type: {
          bsonType: "string",
          enum: ["disease_prediction", "voice_stress", "pose_analysis", "risk_assessment", "symptom_classifier"],
          description: "Type of ML model"
        },
        model_data: {
          bsonType: "binData",
          description: "Serialized model (joblib/pickle format)"
        },
        model_config: {
          bsonType: "object",
          description: "Model configuration and hyperparameters",
          properties: {
            algorithm: { bsonType: "string" },
            hyperparameters: { bsonType: "object" },
            feature_columns: { bsonType: "array", items: { bsonType: "string" } },
            target_column: { bsonType: "string" },
            preprocessing_steps: { bsonType: "array" }
          }
        },
        training_data: {
          bsonType: "object",
          description: "Training dataset metadata",
          properties: {
            sample_count: { bsonType: "int", minimum: 1 },
            feature_count: { bsonType: "int", minimum: 1 },
            data_sources: { bsonType: "array", items: { bsonType: "string" } },
            date_range: {
              bsonType: "object",
              properties: {
                start_date: { bsonType: "date" },
                end_date: { bsonType: "date" }
              }
            }
          }
        },
        accuracy_metrics: {
          bsonType: "object",
          description: "Model performance metrics",
          properties: {
            accuracy: { bsonType: "double", minimum: 0, maximum: 1 },
            precision: { bsonType: "double", minimum: 0, maximum: 1 },
            recall: { bsonType: "double", minimum: 0, maximum: 1 },
            f1_score: { bsonType: "double", minimum: 0, maximum: 1 },
            auc_roc: { bsonType: "double", minimum: 0, maximum: 1 },
            cross_validation_scores: { bsonType: "array", items: { bsonType: "double" } }
          }
        },
        validation_results: {
          bsonType: "object",
          description: "Validation and testing results",
          properties: {
            test_accuracy: { bsonType: "double", minimum: 0, maximum: 1 },
            confusion_matrix: { bsonType: "array" },
            feature_importance: { bsonType: "object" },
            prediction_examples: { bsonType: "array" }
          }
        },
        training_date: {
          bsonType: "date",
          description: "When the model was trained"
        },
        version: {
          bsonType: "string",
          description: "Model version (semantic versioning)"
        },
        is_active: {
          bsonType: "bool",
          description: "Whether this model is currently in use"
        },
        deployment_info: {
          bsonType: "object",
          properties: {
            deployed_at: { bsonType: "date" },
            deployment_environment: { bsonType: "string" },
            api_endpoint: { bsonType: "string" },
            performance_monitoring: { bsonType: "object" }
          }
        }
      }
    }
  }
});

// =============================================================================
// VOICE ANALYSIS SESSIONS COLLECTION
// =============================================================================

// Collection: voice_sessions
// Purpose: Store detailed voice analysis sessions and results
db.createCollection("voice_sessions", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "session_id", "audio_features", "analysis_results", "timestamp"],
      properties: {
        user_id: {
          bsonType: "string",
          description: "UUID of the user"
        },
        session_id: {
          bsonType: "string",
          description: "Unique session identifier"
        },
        audio_metadata: {
          bsonType: "object",
          properties: {
            duration_seconds: { bsonType: "double", minimum: 0 },
            sample_rate: { bsonType: "int" },
            channels: { bsonType: "int" },
            format: { bsonType: "string" },
            file_size_bytes: { bsonType: "long" }
          }
        },
        audio_features: {
          bsonType: "object",
          description: "Extracted audio features using librosa",
          properties: {
            mfcc: { bsonType: "array", description: "Mel-frequency cepstral coefficients" },
            pitch: {
              bsonType: "object",
              properties: {
                mean: { bsonType: "double" },
                std: { bsonType: "double" },
                min: { bsonType: "double" },
                max: { bsonType: "double" },
                variance: { bsonType: "double" }
              }
            },
            energy: {
              bsonType: "object",
              properties: {
                rms_energy: { bsonType: "double" },
                zero_crossing_rate: { bsonType: "double" },
                spectral_centroid: { bsonType: "double" }
              }
            },
            rhythm: {
              bsonType: "object",
              properties: {
                tempo: { bsonType: "double" },
                beat_frames: { bsonType: "array" },
                onset_frames: { bsonType: "array" }
              }
            },
            voice_quality: {
              bsonType: "object",
              properties: {
                jitter: { bsonType: "double" },
                shimmer: { bsonType: "double" },
                harmonics_to_noise_ratio: { bsonType: "double" }
              }
            }
          }
        },
        speech_analysis: {
          bsonType: "object",
          description: "Speech pattern analysis",
          properties: {
            transcript: { bsonType: "string" },
            word_count: { bsonType: "int" },
            speech_rate: { bsonType: "double", description: "Words per minute" },
            pause_analysis: {
              bsonType: "object",
              properties: {
                total_pauses: { bsonType: "int" },
                average_pause_duration: { bsonType: "double" },
                pause_frequency: { bsonType: "string", enum: ["low", "medium", "high"] }
              }
            },
            disfluencies: {
              bsonType: "object",
              properties: {
                filler_words: { bsonType: "int" },
                repetitions: { bsonType: "int" },
                false_starts: { bsonType: "int" }
              }
            }
          }
        },
        stress_indicators: {
          bsonType: "object",
          description: "Calculated stress and emotion indicators",
          properties: {
            stress_level: { bsonType: "double", minimum: 0, maximum: 1 },
            anxiety_score: { bsonType: "double", minimum: 0, maximum: 1 },
            fatigue_score: { bsonType: "double", minimum: 0, maximum: 1 },
            pain_indicators: { bsonType: "double", minimum: 0, maximum: 1 },
            confidence_level: { bsonType: "double", minimum: 0, maximum: 1 }
          }
        },
        analysis_results: {
          bsonType: "object",
          description: "AI analysis results from Groq LLM",
          properties: {
            risk_assessment: { bsonType: "string", enum: ["low", "medium", "high", "critical"] },
            urgency_flag: { bsonType: "bool" },
            recommended_actions: { bsonType: "array", items: { bsonType: "string" } },
            suggested_specialists: { bsonType: "array", items: { bsonType: "string" } },
            confidence_score: { bsonType: "double", minimum: 0, maximum: 1 },
            reasoning: { bsonType: "string" }
          }
        },
        timestamp: {
          bsonType: "date",
          description: "Session timestamp"
        },
        processing_metadata: {
          bsonType: "object",
          properties: {
            processing_time_ms: { bsonType: "int" },
            ai_services_used: { bsonType: "array", items: { bsonType: "string" } },
            model_versions: { bsonType: "object" },
            api_costs: { bsonType: "object" }
          }
        }
      }
    }
  }
});

// =============================================================================
// AR SCAN ANALYSIS COLLECTION
// =============================================================================

// Collection: ar_analysis
// Purpose: Store detailed AR scan analysis and OCR results
db.createCollection("ar_analysis", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["user_id", "scan_id", "image_analysis", "ocr_results", "timestamp"],
      properties: {
        user_id: {
          bsonType: "string",
          description: "UUID of the user"
        },
        scan_id: {
          bsonType: "string",
          description: "Unique scan identifier (matches PostgreSQL ar_scans.id)"
        },
        image_metadata: {
          bsonType: "object",
          properties: {
            width: { bsonType: "int" },
            height: { bsonType: "int" },
            format: { bsonType: "string" },
            file_size_bytes: { bsonType: "long" },
            camera_info: { bsonType: "object" },
            location: { bsonType: "object" }
          }
        },
        image_analysis: {
          bsonType: "object",
          description: "BLIP-2 image analysis results",
          properties: {
            description: { bsonType: "string" },
            detected_objects: { bsonType: "array", items: { bsonType: "string" } },
            scene_classification: { bsonType: "string" },
            confidence_scores: { bsonType: "object" },
            visual_features: { bsonType: "object" }
          }
        },
        ocr_results: {
          bsonType: "object",
          description: "TrOCR text extraction results",
          properties: {
            raw_text: { bsonType: "string" },
            structured_text: {
              bsonType: "array",
              items: {
                bsonType: "object",
                properties: {
                  text: { bsonType: "string" },
                  confidence: { bsonType: "double", minimum: 0, maximum: 1 },
                  bounding_box: { bsonType: "object" },
                  text_type: { bsonType: "string", enum: ["medication", "dosage", "instruction", "date", "doctor_name", "other"] }
                }
              }
            },
            language_detected: { bsonType: "string" },
            overall_confidence: { bsonType: "double", minimum: 0, maximum: 1 }
          }
        },
        medication_extraction: {
          bsonType: "object",
          description: "Structured medication information",
          properties: {
            medications: {
              bsonType: "array",
              items: {
                bsonType: "object",
                properties: {
                  name: { bsonType: "string" },
                  generic_name: { bsonType: "string" },
                  dosage: { bsonType: "string" },
                  frequency: { bsonType: "string" },
                  instructions: { bsonType: "string" },
                  duration: { bsonType: "string" },
                  confidence: { bsonType: "double", minimum: 0, maximum: 1 }
                }
              }
            },
            prescriber_info: {
              bsonType: "object",
              properties: {
                doctor_name: { bsonType: "string" },
                clinic_name: { bsonType: "string" },
                date_prescribed: { bsonType: "string" }
              }
            }
          }
        },
        safety_analysis: {
          bsonType: "object",
          description: "Safety warnings and drug interactions",
          properties: {
            warnings: { bsonType: "array", items: { bsonType: "string" } },
            drug_interactions: { bsonType: "array", items: { bsonType: "object" } },
            allergy_alerts: { bsonType: "array", items: { bsonType: "string" } },
            dosage_concerns: { bsonType: "array", items: { bsonType: "string" } }
          }
        },
        timestamp: {
          bsonType: "date",
          description: "Analysis timestamp"
        },
        processing_metadata: {
          bsonType: "object",
          properties: {
            processing_time_ms: { bsonType: "int" },
            ai_services_used: { bsonType: "array", items: { bsonType: "string" } },
            model_versions: { bsonType: "object" },
            api_costs: { bsonType: "object" }
          }
        }
      }
    }
  }
});

// =============================================================================
// INDEXES FOR PERFORMANCE
// =============================================================================

// Family Graph Indexes
db.family_graph.createIndex({ "user_id": 1 }, { unique: true });
db.family_graph.createIndex({ "updated_at": -1 });
db.family_graph.createIndex({ "family_members.relation": 1 });
db.family_graph.createIndex({ "family_members.health_conditions": 1 });

// Health Events Indexes
db.health_events.createIndex({ "user_id": 1, "timestamp": -1 });
db.health_events.createIndex({ "event_type": 1, "timestamp": -1 });
db.health_events.createIndex({ "ai_analysis.risk_assessment": 1 });
db.health_events.createIndex({ "source": 1, "timestamp": -1 });

// ML Models Indexes
db.ml_models.createIndex({ "user_id": 1, "model_type": 1 });
db.ml_models.createIndex({ "model_type": 1, "is_active": 1 });
db.ml_models.createIndex({ "training_date": -1 });
db.ml_models.createIndex({ "version": 1 });

// Voice Sessions Indexes
db.voice_sessions.createIndex({ "user_id": 1, "timestamp": -1 });
db.voice_sessions.createIndex({ "session_id": 1 }, { unique: true });
db.voice_sessions.createIndex({ "analysis_results.risk_assessment": 1 });
db.voice_sessions.createIndex({ "stress_indicators.stress_level": 1 });

// AR Analysis Indexes
db.ar_analysis.createIndex({ "user_id": 1, "timestamp": -1 });
db.ar_analysis.createIndex({ "scan_id": 1 }, { unique: true });
db.ar_analysis.createIndex({ "medication_extraction.medications.name": 1 });
db.ar_analysis.createIndex({ "safety_analysis.warnings": 1 });

// =============================================================================
// AGGREGATION PIPELINE EXAMPLES
// =============================================================================

// Example: Get family health risk summary
/*
db.family_graph.aggregate([
  { $match: { user_id: "550e8400-e29b-41d4-a716-446655440001" } },
  { $unwind: "$family_members" },
  { $unwind: "$family_members.health_conditions" },
  { $group: {
    _id: "$family_members.health_conditions",
    count: { $sum: 1 },
    relations: { $addToSet: "$family_members.relation" }
  }},
  { $sort: { count: -1 } }
]);
*/

// Example: Get recent high-risk health events
/*
db.health_events.aggregate([
  { $match: { 
    user_id: "550e8400-e29b-41d4-a716-446655440001",
    "ai_analysis.risk_assessment": { $in: ["high", "critical"] },
    timestamp: { $gte: new Date(Date.now() - 30*24*60*60*1000) }
  }},
  { $sort: { timestamp: -1 } },
  { $limit: 10 }
]);
*/

// Example: Get voice stress trends
/*
db.voice_sessions.aggregate([
  { $match: { user_id: "550e8400-e29b-41d4-a716-446655440001" } },
  { $group: {
    _id: { 
      year: { $year: "$timestamp" },
      month: { $month: "$timestamp" },
      day: { $dayOfMonth: "$timestamp" }
    },
    avg_stress: { $avg: "$stress_indicators.stress_level" },
    session_count: { $sum: 1 }
  }},
  { $sort: { "_id.year": 1, "_id.month": 1, "_id.day": 1 } }
]);
*/