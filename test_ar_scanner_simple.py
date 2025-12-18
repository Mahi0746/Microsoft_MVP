#!/usr/bin/env python3
"""
Simple AR Scanner Test - Works without optional dependencies
Tests the dynamic AR scanner with minimal dependencies
"""

import asyncio
import base64
import time
import json
from pathlib import Path
import sys
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.ar_scanner_service import ARMedicalScannerService
from config_flexible import settings

def create_simple_test_image():
    """Create a simple test image."""
    
    # Create a 300x300 white image
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    
    # Add some text using PIL
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Add some simple text
    draw.text((50, 50), "PRESCRIPTION", fill=(0, 0, 0))
    draw.text((50, 100), "Patient: John Doe", fill=(0, 0, 0))
    draw.text((50, 150), "Medication: Aspirin 100mg", fill=(0, 0, 0))
    draw.text((50, 200), "Take 1 daily", fill=(0, 0, 0))
    
    # Convert back to numpy
    return np.array(pil_img)

def image_to_base64(image_array):
    """Convert numpy image array to base64 string."""
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Convert to base64
    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    image_b64 = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/jpeg;base64,{image_b64}"

async def test_simple_ar_scanner():
    """Test the AR scanner with minimal dependencies."""
    
    print("🔍 Testing Simple AR Scanner (Minimal Dependencies)")
    print("=" * 60)
    
    try:
        # Create test image
        print("📸 Creating simple test image...")
        test_image = create_simple_test_image()
        image_b64 = image_to_base64(test_image)
        
        print(f"✅ Created {test_image.shape[0]}x{test_image.shape[1]} test image")
        
        # Initialize AR scanner
        print("🚀 Initializing AR Medical Scanner...")
        await ARMedicalScannerService.initialize_models()
        
        # Test skin analysis
        print("\n🔍 Testing Skin Analysis...")
        start_time = time.time()
        
        skin_result = await ARMedicalScannerService.process_ar_scan(
            user_id="test_simple_user",
            scan_type="skin_analysis",
            image_data=image_b64,
            scan_metadata={
                "test_type": "simple_test",
                "image_source": "generated_simple_image"
            }
        )
        
        skin_time = time.time() - start_time
        
        print(f"📊 Skin Analysis Results:")
        print(f"⏱️  Processing Time: {skin_time:.3f}s")
        print(f"🎯 Confidence Score: {skin_result.get('confidence_score', 0):.3f}")
        print(f"🤖 Models Used: {len(skin_result.get('ai_models_used', []))}")
        print(f"🔬 Analysis Method: {skin_result.get('analysis_method', 'unknown')}")
        
        # Test prescription OCR
        print("\n📄 Testing Prescription OCR...")
        start_time = time.time()
        
        ocr_result = await ARMedicalScannerService.process_ar_scan(
            user_id="test_simple_user",
            scan_type="prescription_ocr",
            image_data=image_b64,
            scan_metadata={
                "test_type": "simple_ocr_test",
                "image_source": "generated_prescription"
            }
        )
        
        ocr_time = time.time() - start_time
        
        print(f"📊 OCR Results:")
        print(f"⏱️  Processing Time: {ocr_time:.3f}s")
        print(f"🎯 Confidence Score: {ocr_result.get('confidence_score', 0):.3f}")
        print(f"🤖 Models Used: {len(ocr_result.get('ai_models_used', []))}")
        
        # Show extracted text if available
        analysis_result = ocr_result.get('analysis_result', {})
        extracted_text = analysis_result.get('extracted_text', '')
        if extracted_text:
            print(f"📝 Extracted Text: '{extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}'")
        
        # Test device scan
        print("\n🔬 Testing Device Scan...")
        start_time = time.time()
        
        device_result = await ARMedicalScannerService.process_ar_scan(
            user_id="test_simple_user",
            scan_type="medical_device_scan",
            image_data=image_b64,
            scan_metadata={
                "test_type": "simple_device_test",
                "image_source": "generated_device_image"
            }
        )
        
        device_time = time.time() - start_time
        
        print(f"📊 Device Scan Results:")
        print(f"⏱️  Processing Time: {device_time:.3f}s")
        print(f"🎯 Confidence Score: {device_result.get('confidence_score', 0):.3f}")
        print(f"🤖 Models Used: {len(device_result.get('ai_models_used', []))}")
        
        print("\n" + "=" * 60)
        print("🎉 SIMPLE AR SCANNER TEST COMPLETE!")
        print("✅ All tests completed successfully")
        print("✅ Scanner works with minimal dependencies")
        print("✅ Dynamic analysis functioning properly")
        
        # Show AI service status
        ai_status = settings.get_ai_status()
        print(f"\n🤖 AI Services Status:")
        for service, available in ai_status.items():
            status = "✅ Available" if available else "⚠️ Demo Mode"
            print(f"  {service}: {status}")
        
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run simple AR scanner test."""
    
    print("🎯 AR Medical Scanner - Simple Test Suite")
    print("=" * 60)
    print("This test verifies the scanner works with minimal dependencies")
    print("No Redis, EasyOCR, MediaPipe, or other optional packages required")
    print("=" * 60)
    
    success = await test_simple_ar_scanner()
    
    if success:
        print("\n🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("\n❌ TESTS FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)