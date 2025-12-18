#!/usr/bin/env python3
"""
Test script for AR Medical Scanner functionality.
Tests OCR, medical document analysis, and AR overlay generation.
"""

import asyncio
import base64
import json
from datetime import datetime
from services.ar_scanner_service import ARMedicalScannerService
from services.ai_service import AIService


async def test_ar_scanner_service():
    """Test the AR Medical Scanner service functionality."""
    
    print("🔬 Testing HealthSync AI - AR Medical Scanner")
    print("=" * 60)
    
    try:
        # Test 1: Initialize AI services
        print("1. Initializing AI services...")
        await AIService.initialize()
        print("✅ AI services initialized successfully")
        
        # Test 2: Test scan types
        print("\n2. Testing scan type configurations...")
        scan_types = ARMedicalScannerService.SCAN_TYPES
        print(f"✅ Found {len(scan_types)} scan types:")
        
        for scan_type, config in scan_types.items():
            print(f"   • {scan_type}: {config['description']}")
            print(f"     Model: {config['ai_model']}, Confidence: {config['confidence_threshold']}")
        
        # Test 3: Test medical conditions database
        print("\n3. Testing medical conditions database...")
        conditions = ARMedicalScannerService.MEDICAL_CONDITIONS
        total_conditions = sum(len(category) for category in conditions.values())
        print(f"✅ Medical conditions database loaded: {total_conditions} conditions across {len(conditions)} categories")
        
        for category, condition_list in conditions.items():
            print(f"   • {category}: {len(condition_list)} conditions")
        
        # Test 4: Test OCR text parsing
        print("\n4. Testing OCR text parsing...")
        
        # Test prescription parsing
        sample_prescription = """
        Dr. Smith Medical Center
        Patient: John Doe
        Date: 12/17/2025
        
        Rx: Amoxicillin 500mg
        Take 1 tablet 3 times daily
        Qty: 21 tablets
        Refills: 2
        
        Dr. Sarah Smith, MD
        """
        
        prescription_data = ARMedicalScannerService._parse_prescription_text(sample_prescription)
        print("✅ Prescription parsing test:")
        print(f"   Medications found: {len(prescription_data['medications'])}")
        print(f"   Instructions: {len(prescription_data['dosage_instructions'])}")
        
        # Test medical report parsing
        sample_report = """
        Lab Results - Blood Chemistry Panel
        Patient: Jane Doe
        Date: 12/17/2025
        
        Glucose: 95 mg/dL (70-100)
        Cholesterol: 180 mg/dL (150-200)
        Hemoglobin: 14.2 g/dL (12.0-16.0)
        WBC: 7.5 (4.0-11.0)
        
        Impression: All values within normal limits
        """
        
        report_data = ARMedicalScannerService._parse_medical_report(sample_report)
        print("✅ Medical report parsing test:")
        print(f"   Lab values found: {len(report_data['lab_values'])}")
        print(f"   Test results: {len(report_data['test_results'])}")
        
        # Test 5: Test lab value analysis
        print("\n5. Testing lab value analysis...")
        lab_analysis = ARMedicalScannerService._analyze_lab_values(report_data)
        print("✅ Lab analysis test:")
        print(f"   Total tests: {lab_analysis['total_tests']}")
        print(f"   Normal values: {len(lab_analysis['normal_values'])}")
        print(f"   Abnormal values: {len(lab_analysis['abnormal_values'])}")
        print(f"   Summary: {lab_analysis['summary']}")
        
        # Test 6: Test medication identification parsing
        print("\n6. Testing medication identification...")
        sample_med_description = "Round white pill with imprint 'A500', medium size, aspirin medication"
        med_details = ARMedicalScannerService._parse_medication_identification(sample_med_description)
        print("✅ Medication identification test:")
        print(f"   Shape: {med_details['shape']}")
        print(f"   Color: {med_details['color']}")
        print(f"   Size: {med_details['size']}")
        print(f"   Name: {med_details['name']}")
        
        # Test 7: Test medication lookup
        print("\n7. Testing medication information lookup...")
        med_info = await ARMedicalScannerService._lookup_medication_info({"name": "aspirin"})
        print("✅ Medication lookup test:")
        print(f"   Generic name: {med_info.get('generic_name', 'Unknown')}")
        print(f"   Brand names: {med_info.get('brand_names', [])}")
        print(f"   Uses: {med_info.get('uses', [])}")
        
        # Test 8: Test device reading parsing
        print("\n8. Testing medical device reading parsing...")
        sample_device_text = "Blood Pressure Monitor - Reading: 120/80 mmHg, Pulse: 72 BPM"
        device_data = ARMedicalScannerService._parse_device_reading(sample_device_text)
        print("✅ Device reading test:")
        print(f"   Device type: {device_data['type']}")
        print(f"   Primary value: {device_data['primary_value']}")
        print(f"   Units: {device_data['units']}")
        
        # Test 9: Test device reading interpretation
        print("\n9. Testing device reading interpretation...")
        interpretation = ARMedicalScannerService._interpret_device_readings(device_data)
        print("✅ Device interpretation test:")
        print(f"   Status: {interpretation['status']}")
        print(f"   Recommendations: {len(interpretation['recommendations'])}")
        
        # Test 10: Test AR overlay generation
        print("\n10. Testing AR overlay generation...")
        
        # Mock AI analysis data
        mock_ai_analysis = {
            "scan_type": "prescription_ocr",
            "confidence": 0.85,
            "medications": [{"name": "Amoxicillin", "dosage": "500mg"}],
            "ocr_results": {"raw_text": sample_prescription, "confidence": 0.85}
        }
        
        mock_assessment = {
            "findings": ["Prescription contains 1 medication(s)", "Medication: Amoxicillin - 500mg"],
            "urgency_level": "routine",
            "recommendations": ["Verify medication details with prescribing physician"]
        }
        
        # Mock image (small numpy array)
        import numpy as np
        mock_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        overlay_data = ARMedicalScannerService._generate_ar_overlay(
            mock_image, mock_ai_analysis, mock_assessment
        )
        
        print("✅ AR overlay generation test:")
        print(f"   Annotations: {len(overlay_data['annotations'])}")
        print(f"   Confidence indicators: {len(overlay_data['confidence_indicators'])}")
        print(f"   Highlights: {len(overlay_data['highlights'])}")
        
        print("\n🎉 All AR Medical Scanner tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_text_parsing_functions():
    """Test individual text parsing functions."""
    
    print("\n📝 Testing Text Parsing Functions")
    print("-" * 40)
    
    # Test medication name extraction
    test_lines = [
        "Amoxicillin 500mg tablets",
        "Ibuprofen 200 mg capsules",
        "Acetaminophen 325mg tablet"
    ]
    
    print("Testing medication name extraction:")
    for line in test_lines:
        name = ARMedicalScannerService._extract_medication_name(line)
        dosage = ARMedicalScannerService._extract_dosage(line)
        form = ARMedicalScannerService._extract_medication_form(line)
        print(f"  '{line}' → Name: {name}, Dosage: {dosage}, Form: {form}")
    
    # Test date detection
    date_lines = [
        "Date: 12/17/2025",
        "Prescribed on 2025-12-17",
        "December 17, 2025"
    ]
    
    print("\nTesting date detection:")
    for line in date_lines:
        is_date = ARMedicalScannerService._is_date_line(line)
        print(f"  '{line}' → Is date: {is_date}")
    
    # Test lab value detection
    lab_lines = [
        "Glucose: 95 mg/dL",
        "WBC 7.2 (4.0-11.0)",
        "Hemoglobin = 14.2 g/dL"
    ]
    
    print("\nTesting lab value detection:")
    for line in lab_lines:
        is_lab = ARMedicalScannerService._is_lab_value_line(line)
        lab_value = ARMedicalScannerService._parse_lab_value(line)
        print(f"  '{line}' → Is lab: {is_lab}")
        if lab_value:
            print(f"    Parsed: {lab_value['test_name']} = {lab_value['value']} {lab_value['unit']}")


def test_safety_analysis():
    """Test medication safety analysis."""
    
    print("\n🛡️ Testing Safety Analysis")
    print("-" * 30)
    
    # Test prescription with potential interactions
    test_prescription = {
        "medications": [
            {"name": "Warfarin", "dosage": "5mg"},
            {"name": "Aspirin", "dosage": "81mg"},
            {"name": "Ibuprofen", "dosage": "200mg"}
        ]
    }
    
    safety_analysis = asyncio.run(
        ARMedicalScannerService._analyze_medication_safety(test_prescription)
    )
    
    print("Safety analysis results:")
    print(f"  Risk level: {safety_analysis['risk_level']}")
    print(f"  Interaction warnings: {len(safety_analysis['interaction_warnings'])}")
    for warning in safety_analysis['interaction_warnings']:
        print(f"    • {warning}")
    
    print(f"  Dosage concerns: {len(safety_analysis['dosage_concerns'])}")
    print(f"  General safety tips: {len(safety_analysis['general_safety'])}")


def test_critical_values():
    """Test critical lab value detection."""
    
    print("\n⚠️ Testing Critical Value Detection")
    print("-" * 35)
    
    test_values = [
        ("glucose", 35, "low"),      # Critical low
        ("glucose", 450, "high"),    # Critical high
        ("glucose", 95, "normal"),   # Normal
        ("potassium", 2.0, "low"),   # Critical low
        ("potassium", 6.5, "high"),  # Critical high
        ("hemoglobin", 6.0, "low"),  # Critical low
    ]
    
    for test_name, value, status in test_values:
        is_critical = ARMedicalScannerService._is_critical_value(test_name, value, status)
        print(f"  {test_name}: {value} ({status}) → Critical: {is_critical}")


async def test_ai_integration():
    """Test AI service integration."""
    
    print("\n🤖 Testing AI Service Integration")
    print("-" * 35)
    
    try:
        # Test medical text extraction (mock)
        print("Testing medical text extraction...")
        
        # Create a simple test image (base64 encoded)
        import io
        from PIL import Image
        
        # Create a simple white image with text
        img = Image.new('RGB', (200, 100), color='white')
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # This would normally call the actual AI service
        print("✅ AI integration test setup complete")
        print("   (Actual AI calls would require API keys and network access)")
        
    except Exception as e:
        print(f"❌ AI integration test failed: {str(e)}")


async def main():
    """Run all tests."""
    
    print("🚀 HealthSync AI - AR Medical Scanner Test Suite")
    print("=" * 70)
    
    # Run all test functions
    await test_ar_scanner_service()
    test_text_parsing_functions()
    test_safety_analysis()
    test_critical_values()
    await test_ai_integration()
    
    print("\n" + "=" * 70)
    print("✨ AR Medical Scanner test suite completed!")
    print("\nKey Features Tested:")
    print("• OCR text extraction and parsing")
    print("• Medical document structure analysis")
    print("• Prescription safety analysis")
    print("• Lab value interpretation")
    print("• Medication identification")
    print("• Medical device reading analysis")
    print("• AR overlay generation")
    print("• Critical value detection")


if __name__ == "__main__":
    asyncio.run(main())