# AR Medical Scanner API Documentation

The HealthSync AI AR Medical Scanner provides advanced medical document analysis, OCR capabilities, and augmented reality overlays for healthcare applications. This API supports prescription analysis, medical report processing, medication identification, and medical device reading interpretation.

## Features

- **Medical Document OCR**: Extract text from prescriptions and medical reports using TrOCR
- **Prescription Analysis**: Parse medication details, dosages, and safety warnings
- **Medical Report Processing**: Analyze lab results and identify abnormal values
- **Medication Identification**: Identify pills and medications using computer vision
- **Medical Device Reading**: Interpret readings from blood pressure monitors, thermometers, etc.
- **AR Overlays**: Generate augmented reality annotations and highlights
- **Safety Analysis**: Detect drug interactions and provide safety warnings

## Base URL

```
/api/v1/ar-scanner
```

## Authentication

All endpoints require JWT authentication via the `Authorization: Bearer <token>` header.

## Scan Types

### Medical Document Scanning

| Scan Type | Description | AI Model | Use Case |
|-----------|-------------|----------|----------|
| `prescription_ocr` | Extract and analyze prescription text | TrOCR | Prescription verification, medication tracking |
| `medical_report_ocr` | Process medical reports and lab results | TrOCR | Lab result analysis, report digitization |
| `pill_identification` | Identify medications from images | BLIP-2 | Medication verification, pill identification |
| `medical_device_scan` | Read medical device displays | BLIP-2 | Device reading interpretation, health monitoring |

### Traditional Medical Scanning

| Scan Type | Description | AI Model | Use Case |
|-----------|-------------|----------|----------|
| `skin_analysis` | Skin condition and mole analysis | Dermatology | Skin health monitoring, mole tracking |
| `wound_assessment` | Wound healing progress tracking | Wound Analysis | Wound care, healing progress |
| `rash_detection` | Rash pattern and severity analysis | Dermatology | Rash diagnosis, treatment tracking |
| `eye_examination` | Basic eye health screening | Ophthalmology | Eye health monitoring |
| `posture_analysis` | Posture and spine alignment check | Orthopedic | Posture correction, ergonomics |
| `vitals_estimation` | Heart rate and breathing estimation | Vitals | Basic health monitoring |

## Endpoints

### Perform AR Scan

#### Base64 Image Scan
```http
POST /scan
```

**Request Body:**
```json
{
  "scan_type": "prescription_ocr",
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
  "scan_metadata": {
    "source": "camera_capture",
    "timestamp": "2025-12-17T10:30:00Z",
    "device_info": "iPhone 15 Pro"
  }
}
```

**Response:**
```json
{
  "scan_id": "scan_67890abcdef",
  "user_id": "user_12345",
  "scan_type": "prescription_ocr",
  "timestamp": "2025-12-17T10:30:00Z",
  "confidence_score": 0.85,
  "medical_assessment": {
    "findings": [
      "Prescription contains 2 medication(s)",
      "Medication: Amoxicillin - 500mg",
      "Medication: Ibuprofen - 200mg"
    ],
    "severity_assessment": "low",
    "recommendations": [
      "Verify all medication details with prescribing physician",
      "Follow dosage instructions exactly as prescribed",
      "Report any side effects to healthcare provider"
    ],
    "follow_up_required": false,
    "urgency_level": "routine"
  },
  "ar_overlay": {
    "annotations": [
      {
        "type": "text_annotation",
        "position": {"x": 10, "y": 50},
        "text": "Prescription contains 2 medication(s)",
        "color": "#FFFFFF",
        "background_color": "#000000AA"
      }
    ],
    "confidence_indicators": [
      {
        "type": "confidence_bar",
        "position": {"x": 10, "y": 10},
        "value": 0.85,
        "color": "#00FF00",
        "label": "Analysis Confidence: 85.0%"
      }
    ],
    "highlights": [
      {
        "type": "border_highlight",
        "color": "#00FF00",
        "thickness": 3,
        "style": "dashed"
      }
    ]
  },
  "recommendations": [
    "Verify all medication details with prescribing physician",
    "Follow dosage instructions exactly as prescribed"
  ],
  "follow_up_required": false
}
```

#### File Upload Scan
```http
POST /scan/upload?scan_type=prescription_ocr
```

**Request:**
- Content-Type: `multipart/form-data`
- File: Image file (JPEG, PNG, WebP)
- Max size: 10MB

**Response:** Same as base64 scan response

### Prescription OCR Analysis

**Example Request:**
```json
{
  "scan_type": "prescription_ocr",
  "image_data": "base64_encoded_prescription_image"
}
```

**Detailed Response:**
```json
{
  "scan_id": "scan_prescription_001",
  "confidence_score": 0.88,
  "medical_assessment": {
    "findings": [
      "Prescription contains 1 medication(s)",
      "Medication: Amoxicillin - 500mg"
    ],
    "medications_detected": [
      {
        "name": "Amoxicillin",
        "dosage": "500mg",
        "form": "tablet",
        "instructions": "Take 1 tablet 3 times daily",
        "quantity": "21 tablets",
        "refills": 2
      }
    ],
    "prescriber_info": {
      "name": "Dr. Sarah Smith, MD",
      "practice": "Smith Medical Center"
    },
    "safety_analysis": {
      "interaction_warnings": [],
      "allergy_alerts": [],
      "risk_level": "low",
      "general_safety": [
        "Take medications as prescribed",
        "Report side effects to healthcare provider"
      ]
    },
    "warnings": [
      "This analysis is for informational purposes only",
      "Always verify medication details with your healthcare provider"
    ]
  }
}
```

### Medical Report OCR Analysis

**Example Request:**
```json
{
  "scan_type": "medical_report_ocr",
  "image_data": "base64_encoded_lab_report_image"
}
```

**Detailed Response:**
```json
{
  "scan_id": "scan_report_001",
  "confidence_score": 0.90,
  "medical_assessment": {
    "findings": [
      "Lab values appear within normal limits",
      "5 test results analyzed"
    ],
    "lab_analysis": {
      "total_tests": 5,
      "abnormal_values": [],
      "normal_values": [
        {
          "test_name": "Glucose",
          "value": 95,
          "unit": "mg/dL",
          "reference_range": {"low": 70, "high": 100},
          "status": "normal"
        },
        {
          "test_name": "Cholesterol",
          "value": 180,
          "unit": "mg/dL",
          "reference_range": {"low": 150, "high": 200},
          "status": "normal"
        }
      ],
      "critical_values": [],
      "summary": "All values within normal limits"
    },
    "report_info": {
      "report_type": "Blood Test Report",
      "patient_info": "John Doe",
      "report_date": "12/17/2025"
    }
  }
}
```

### Medication Identification

**Example Request:**
```json
{
  "scan_type": "pill_identification",
  "image_data": "base64_encoded_pill_image"
}
```

**Detailed Response:**
```json
{
  "scan_id": "scan_pill_001",
  "confidence_score": 0.75,
  "medical_assessment": {
    "findings": [
      "Medication identified as: Aspirin",
      "Identification confidence: 75.0%"
    ],
    "identification": {
      "medication_name": "Aspirin",
      "generic_name": "Acetylsalicylic acid",
      "brand_names": ["Bayer", "Bufferin", "Excedrin"],
      "dosage_strength": "325mg",
      "medication_class": "NSAID"
    },
    "visual_analysis": {
      "shape": "round",
      "color": "white",
      "size": "medium",
      "markings": ["A325"],
      "imprint": "A325"
    },
    "safety_information": {
      "common_uses": ["Pain relief", "Fever reduction", "Anti-inflammatory"],
      "side_effects": ["Stomach upset", "Bleeding risk", "Allergic reactions"],
      "warnings": ["Do not exceed recommended dose", "Consult doctor if pregnant"],
      "interactions": ["Blood thinners", "Other NSAIDs"]
    },
    "disclaimers": [
      "Medication identification is not 100% accurate",
      "Always verify with pharmacist or healthcare provider",
      "Do not take unknown medications"
    ]
  }
}
```

### Medical Device Reading

**Example Request:**
```json
{
  "scan_type": "medical_device_scan",
  "image_data": "base64_encoded_device_image"
}
```

**Detailed Response:**
```json
{
  "scan_id": "scan_device_001",
  "confidence_score": 0.82,
  "medical_assessment": {
    "findings": [
      "Device type: Blood Pressure Monitor",
      "Reading status: normal"
    ],
    "device_info": {
      "device_type": "blood_pressure_monitor",
      "brand": "Omron",
      "model": "HEM-7120",
      "status": "normal"
    },
    "readings": {
      "systolic": 120,
      "diastolic": 80,
      "pulse": 72
    },
    "interpretation": {
      "status": "normal",
      "normal_ranges": {
        "systolic": "90-120 mmHg",
        "diastolic": "60-80 mmHg"
      },
      "recommendations": [
        "Blood pressure within normal range",
        "Continue regular monitoring"
      ]
    },
    "disclaimers": [
      "Device readings should be verified with healthcare provider",
      "Multiple readings may be needed for accurate assessment"
    ]
  }
}
```

### Scan History

#### Get Scan History
```http
GET /history?scan_type=prescription_ocr&limit=10
```

**Response:**
```json
[
  {
    "scan_id": "scan_67890abcdef",
    "scan_type": "prescription_ocr",
    "timestamp": "2025-12-17T10:30:00Z",
    "confidence_score": 0.85,
    "urgency_level": "routine",
    "findings_count": 3,
    "follow_up_required": false
  }
]
```

#### Get Scan Details
```http
GET /scan/{scan_id}
```

**Response:**
```json
{
  "scan_id": "scan_67890abcdef",
  "scan_type": "prescription_ocr",
  "timestamp": "2025-12-17T10:30:00Z",
  "confidence_score": 0.85,
  "medical_assessment": {
    "findings": ["Prescription contains 2 medication(s)"],
    "recommendations": ["Verify medication details"]
  },
  "ar_overlay": {
    "annotations": [],
    "highlights": []
  }
}
```

### Analytics

#### Get Scan Analytics
```http
GET /analytics
```

**Response:**
```json
{
  "total_scans": 25,
  "scan_types": {
    "prescription_ocr": 10,
    "medical_report_ocr": 8,
    "pill_identification": 4,
    "skin_analysis": 3
  },
  "average_confidence": 0.83,
  "urgency_distribution": {
    "routine": 20,
    "medium": 4,
    "high": 1,
    "urgent": 0
  },
  "latest_scan": {
    "scan_id": "scan_67890abcdef",
    "scan_type": "prescription_ocr",
    "timestamp": "2025-12-17T10:30:00Z"
  }
}
```

### Scan Comparison

#### Compare Two Scans
```http
GET /compare/{scan_id1}/{scan_id2}
```

**Response:**
```json
{
  "scan_type": "wound_assessment",
  "scan1": {
    "id": "scan_001",
    "timestamp": "2025-12-10T10:30:00Z",
    "confidence": 0.82,
    "findings": ["Wound in inflammatory stage"]
  },
  "scan2": {
    "id": "scan_002",
    "timestamp": "2025-12-17T10:30:00Z",
    "confidence": 0.85,
    "findings": ["Wound in proliferative stage"]
  },
  "time_difference_days": 7,
  "progress_analysis": {
    "confidence_change": 0.03,
    "findings_change": 0,
    "overall_trend": "improving",
    "healing_status": "good_healing"
  },
  "recommendations": [
    "Continue current treatment approach - showing positive progress",
    "Wound healing progressing well - continue current care"
  ]
}
```

### Available Scan Types

#### Get Scan Types
```http
GET /scan-types
```

**Response:**
```json
{
  "available_scan_types": [
    {
      "type": "prescription_ocr",
      "name": "Prescription OCR",
      "description": "Medical prescription text extraction and analysis",
      "ai_model": "trocr",
      "confidence_threshold": 0.8,
      "estimated_processing_time": 4.0,
      "supported_conditions": ["medication_verification", "dosage_analysis"]
    }
  ],
  "total_types": 10,
  "service_status": "active"
}
```

## AR Overlay Format

The AR overlay data provides information for rendering augmented reality elements:

### Annotation Types

```json
{
  "annotations": [
    {
      "type": "text_annotation",
      "position": {"x": 10, "y": 50},
      "text": "Medication: Aspirin 325mg",
      "color": "#FFFFFF",
      "background_color": "#000000AA",
      "font_size": 14
    }
  ],
  "measurements": [
    {
      "type": "dimension_lines",
      "length": 25.0,
      "width": 15.0,
      "area": 3.75,
      "color": "#00FFFF",
      "unit": "mm"
    }
  ],
  "highlights": [
    {
      "type": "border_highlight",
      "color": "#00FF00",
      "thickness": 3,
      "style": "solid"
    },
    {
      "type": "area_highlight",
      "area": "medication_region",
      "color": "#FFFF00",
      "opacity": 0.3,
      "label": "Amoxicillin (500mg)"
    }
  ],
  "confidence_indicators": [
    {
      "type": "confidence_bar",
      "position": {"x": 10, "y": 10},
      "value": 0.85,
      "color": "#00FF00",
      "label": "Analysis Confidence: 85.0%"
    }
  ]
}
```

## Safety Features

### Medical Disclaimers

All medical analysis includes appropriate disclaimers:

- "This analysis is for informational purposes only"
- "Always verify with healthcare provider"
- "Do not make medical decisions based on this analysis alone"
- "Consult pharmacist for medication verification"

### Confidence Scoring

- **High Confidence (≥80%)**: Green indicators, reliable analysis
- **Medium Confidence (60-79%)**: Yellow indicators, verification recommended
- **Low Confidence (<60%)**: Red indicators, manual review required

### Urgency Levels

- **Routine**: Standard follow-up, no immediate action needed
- **Low**: Monitor, schedule regular check-up
- **Medium**: Follow up with healthcare provider within days
- **High**: Seek medical attention soon
- **Urgent**: Immediate medical attention required

## Error Handling

### Common Error Responses

```json
{
  "error": true,
  "message": "Invalid scan type: invalid_type",
  "status_code": 400,
  "timestamp": "2025-12-17T10:30:00Z",
  "path": "/api/v1/ar-scanner/scan"
}
```

### Error Codes

- **400**: Invalid request (bad scan type, invalid image data)
- **401**: Authentication required
- **413**: File too large (>10MB)
- **415**: Unsupported media type
- **429**: Rate limit exceeded
- **500**: Internal server error
- **503**: Service unavailable

## Rate Limits

- **Image processing**: 5 requests/minute
- **General endpoints**: 100 requests/minute
- **File uploads**: 3 requests/minute

## SDK Examples

### JavaScript/TypeScript
```javascript
const scanner = new HealthSyncARScanner({
  apiKey: 'your-jwt-token',
  baseUrl: 'https://api.healthsync.ai/api/v1/ar-scanner'
});

// Scan prescription
const result = await scanner.scanPrescription(imageData);
console.log('Medications found:', result.medical_assessment.medications_detected);

// Get scan history
const history = await scanner.getHistory('prescription_ocr', 10);
```

### Python
```python
from healthsync import ARScannerClient

client = ARScannerClient(api_key='your-jwt-token')

# Scan medical report
result = await client.scan_medical_report(image_data)
print(f"Lab values: {result.medical_assessment.lab_analysis.total_tests}")

# Compare scans
comparison = await client.compare_scans(scan_id1, scan_id2)
print(f"Progress: {comparison.progress_analysis.overall_trend}")
```

### React Native Integration
```javascript
import { Camera } from 'expo-camera';
import { ARScannerService } from './services/ar-scanner';

const ScanPrescription = () => {
  const takePicture = async () => {
    const photo = await camera.takePictureAsync({
      base64: true,
      quality: 0.8
    });
    
    const result = await ARScannerService.scanPrescription(photo.base64);
    
    // Display AR overlay
    setOverlayData(result.ar_overlay);
    setMedications(result.medical_assessment.medications_detected);
  };
};
```

## Best Practices

### Image Quality
- Use good lighting conditions
- Ensure text is clearly visible
- Avoid shadows and glare
- Keep camera steady
- Use appropriate distance (6-12 inches)

### Security
- Never store sensitive medical images
- Use HTTPS for all API calls
- Implement proper authentication
- Follow HIPAA guidelines for production use

### User Experience
- Show confidence indicators
- Provide clear disclaimers
- Allow manual verification
- Implement offline fallbacks
- Use progressive loading for AR overlays

## Limitations

### Current MVP Limitations
- English language only
- Basic OCR accuracy (85-90%)
- Limited medication database
- No real-time video processing
- Simplified drug interaction checking

### Future Enhancements
- Multi-language support
- Improved OCR accuracy (95%+)
- Comprehensive drug database integration
- Real-time AR video overlays
- Advanced clinical decision support
- Integration with electronic health records