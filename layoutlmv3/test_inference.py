#!/usr/bin/env python3
"""
Simple test script for LayoutLMv3 inference
"""

import os
import sys
from inference import LayoutLMv3Inference

def test_layoutlmv3():
    """Test LayoutLMv3 with a sample image"""
    
    print("🚀 Starting LayoutLMv3 inference test...")
    
    try:
        # Initialize LayoutLMv3
        print("📥 Loading LayoutLMv3 model...")
        layoutlmv3 = LayoutLMv3Inference()
        print("✅ Model loaded successfully!")
        
        # Test with your PDF file
        image_path = "../Solaras-test-18.pdf"
        
        if os.path.exists(image_path):
            print(f"📄 Processing file: {image_path}")
            
            # Make prediction
            result = layoutlmv3.predict_invoice_type(image_path)
            
            # Display results
            print("\n" + "="*50)
            print("📊 LAYOUTLMV3 RESULTS")
            print("="*50)
            
            if "error" in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"📋 Is Invoice: {result['is_invoice']}")
                print(f"🎯 Confidence: {result['confidence']:.2%}")
                print(f"🏷️  Type: {result['invoice_type']}")
                
                # Display extracted fields
                fields = result.get('invoice_fields', {})
                print(f"\n📝 Extracted Fields:")
                for field, value in fields.items():
                    if value:
                        print(f"   {field}: {value}")
                
                # Display text count
                text_count = len(result.get('extracted_text', []))
                print(f"\n📝 Total text elements: {text_count}")
            
            print("="*50)
            
        else:
            print(f"❌ File not found: {image_path}")
            print("Please make sure the PDF file exists in the parent directory")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_layoutlmv3()
