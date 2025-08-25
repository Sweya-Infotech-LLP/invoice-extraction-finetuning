#!/usr/bin/env python3
"""
Example script demonstrating LayoutLMv3 usage for invoice extraction
"""

import os
import json
from datetime import datetime
from inference import LayoutLMv3Inference

def example_single_invoice():
    """Example: Process a single invoice"""
    print("📄 Example 1: Single Invoice Processing")
    print("=" * 50)
    
    try:
        # Initialize LayoutLMv3
        layoutlmv3 = LayoutLMv3Inference()
        
        # Process your PDF file
        image_path = "../Solaras-test-18.pdf"
        
        if os.path.exists(image_path):
            print(f"Processing: {image_path}")
            
            # Extract invoice information
            result = layoutlmv3.predict_invoice_type(image_path)
            
            # Display results
            if "error" not in result:
                print(f"✅ Successfully processed invoice")
                print(f"📋 Is Invoice: {result['is_invoice']}")
                print(f"🎯 Confidence: {result['confidence']:.2%}")
                print(f"🏷️  Type: {result['invoice_type']}")
                
                # Show extracted fields
                fields = result.get('invoice_fields', {})
                if any(fields.values()):
                    print(f"\n📝 Extracted Fields:")
                    for field, value in fields.items():
                        if value:
                            print(f"   {field}: {value}")
                
                # Show text extraction stats
                text_count = len(result.get('extracted_text', []))
                print(f"\n📝 Text Elements: {text_count}")
                
                # Save results
                output_file = f"example_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\n💾 Results saved to: {output_file}")
                
            else:
                print(f"❌ Error: {result['error']}")
                
        else:
            print(f"❌ File not found: {image_path}")
            print("Please make sure the PDF file exists in the parent directory")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_batch_processing():
    """Example: Process multiple files"""
    print("\n📚 Example 2: Batch Processing")
    print("=" * 50)
    
    try:
        # Initialize LayoutLMv3
        layoutlmv3 = LayoutLMv3Inference()
        
        # List of files to process (adjust paths as needed)
        files_to_process = [
            "../Solaras-test-18.pdf",
            # Add more files here when available
        ]
        
        # Filter existing files
        existing_files = [f for f in files_to_process if os.path.exists(f)]
        
        if existing_files:
            print(f"Processing {len(existing_files)} files...")
            
            # Process files in batch
            results = layoutlmv3.batch_predict(existing_files)
            
            # Display batch results
            successful = 0
            for i, result in enumerate(results):
                filename = os.path.basename(existing_files[i])
                print(f"\n📄 File {i+1}: {filename}")
                
                if "error" in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   ✅ Success: {result['is_invoice']} (Confidence: {result['confidence']:.2%})")
                    successful += 1
            
            print(f"\n📊 Batch Summary:")
            print(f"   Total files: {len(existing_files)}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {len(existing_files) - successful}")
            
        else:
            print("No files found to process")
            print("Please add file paths to the files_to_process list")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def example_custom_processing():
    """Example: Custom processing with specific fields"""
    print("\n🔧 Example 3: Custom Field Processing")
    print("=" * 50)
    
    try:
        # Initialize LayoutLMv3
        layoutlmv3 = LayoutLMv3Inference()
        
        # Process file
        image_path = "../Solaras-test-18.pdf"
        
        if os.path.exists(image_path):
            print(f"Processing: {image_path}")
            
            # Get OCR results first
            text_boxes = layoutlmv3.extract_text_with_ocr(image_path)
            
            print(f"📝 Extracted {len(text_boxes)} text elements")
            
            # Show first few text elements
            if text_boxes:
                print("\n🔍 Sample Text Elements:")
                for i, box in enumerate(text_boxes[:5]):  # Show first 5
                    print(f"   {i+1}. '{box['text']}' (Confidence: {box['confidence']}%)")
                    print(f"      BBox: {box['bbox']}")
            
            # Custom field detection
            print("\n🎯 Custom Field Detection:")
            
            # Look for specific patterns
            amounts = []
            dates = []
            
            for box in text_boxes:
                text = box['text'].lower()
                
                # Look for amounts (simple pattern)
                if any(char.isdigit() for char in text) and any(char in '$€£' for char in text):
                    amounts.append(box['text'])
                
                # Look for dates (simple pattern)
                if any(char.isdigit() for char in text) and ('/' in text or '-' in text):
                    dates.append(box['text'])
            
            if amounts:
                print(f"   💰 Amounts found: {', '.join(amounts[:3])}")
            if dates:
                print(f"   📅 Dates found: {', '.join(dates[:3])}")
                
        else:
            print(f"❌ File not found: {image_path}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def main():
    """Run all examples"""
    print("🚀 LayoutLMv3 Invoice Extraction Examples")
    print("=" * 60)
    
    # Run examples
    example_single_invoice()
    example_batch_processing()
    example_custom_processing()
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("\n💡 Tips:")
    print("   - Check the generated JSON files for detailed results")
    print("   - Adjust file paths in the examples as needed")
    print("   - Use the API (api.py) for web-based processing")
    print("   - Check the README.md for more information")

if __name__ == "__main__":
    main()
