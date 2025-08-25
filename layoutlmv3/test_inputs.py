#!/usr/bin/env python3
"""
Test script to show what input types work best with LayoutLMv3
"""

import os
import sys
from PIL import Image
import numpy as np
import io

def test_input_types():
    """Test different input types and show recommendations"""
    
    print("🧪 LayoutLMv3 Input Type Testing")
    print("=" * 50)
    
    # Test file paths
    test_files = [
        "../Solaras-test-18.pdf",
        # Add more test files here
    ]
    
    print("📁 Testing available files:")
    for file_path in test_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"   ✅ {file_path} ({file_size:.2f} MB)")
        else:
            print(f"   ❌ {file_path} (not found)")
    
    print("\n🎯 RECOMMENDED INPUT TYPES:")
    print("   📸 Images (PNG, JPEG, TIFF):")
    print("      - High resolution (300+ DPI)")
    print("      - Clear, well-lit")
    print("      - RGB color space")
    print("      - Minimum 224x224 pixels")
    
    print("\n   📄 PDFs:")
    print("      - Single page (first page will be used)")
    print("      - Text-based (not scanned images)")
    print("      - Clear, readable content")
    print("      - Reasonable file size (< 50MB)")
    
    print("\n⚠️  INPUT LIMITATIONS:")
    print("   - Multi-page PDFs: Only first page processed")
    print("   - Very large images: Will be resized to 224x224")
    print("   - Rotated documents: May affect accuracy")
    print("   - Low-quality scans: Poor OCR results")
    
    print("\n🔧 OPTIMAL INPUT SPECIFICATIONS:")
    print("   Format: PNG or JPEG")
    print("   Resolution: 300-600 DPI")
    print("   Size: 1000x1000 to 4000x4000 pixels")
    print("   Color: RGB (not grayscale)")
    print("   Quality: Clear, readable text")
    
    print("\n💡 TIPS FOR BEST RESULTS:")
    print("   1. Use high-quality scans or photos")
    print("   2. Ensure good lighting and contrast")
    print("   3. Keep documents flat and aligned")
    print("   4. Avoid shadows or reflections")
    print("   5. Convert PDFs to images if possible")

def test_pdf_conversion():
    """Test PDF to image conversion capabilities"""
    
    print("\n🔄 PDF Conversion Testing")
    print("=" * 50)
    
    pdf_path = "../Solaras-test-18.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Testing PDF: {pdf_path}")
    
    # Test different conversion methods
    methods = [
        ("pdf2image", "convert_from_path"),
        ("PyMuPDF", "fitz.open"),
        ("PIL direct", "Image.open")
    ]
    
    for method_name, method_desc in methods:
        try:
            if method_name == "pdf2image":
                from pdf2image import convert_from_path
                images = convert_from_path(pdf_path, first_page=1, last_page=1)
                if images:
                    img = images[0]
                    print(f"   ✅ {method_name}: Success ({img.size[0]}x{img.size[1]})")
                else:
                    print(f"   ❌ {method_name}: No images extracted")
            elif method_name == "PyMuPDF":
                import fitz
                doc = fitz.open(pdf_path)
                page = doc[0]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                print(f"   ✅ {method_name}: Success ({img.size[0]}x{img.size[1]})")
                doc.close()
            elif method_name == "PIL direct":
                img = Image.open(pdf_path)
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    img.seek(0)
                print(f"   ✅ {method_name}: Success ({img.size[0]}x{img.size[1]})")
        except ImportError:
            print(f"   ❌ {method_name}: Library not installed")
        except Exception as e:
            print(f"   ❌ {method_name}: Failed - {str(e)}")

def create_sample_inputs():
    """Create sample input files for testing"""
    
    print("\n🛠️  Creating Sample Inputs")
    print("=" * 50)
    
    # Create a simple test image
    try:
        # Create a simple invoice-like image
        img = Image.new('RGB', (800, 600), color='white')
        
        # Save as different formats
        test_formats = [
            ("sample_invoice.png", "PNG"),
            ("sample_invoice.jpg", "JPEG"),
            ("sample_invoice.tiff", "TIFF")
        ]
        
        for filename, format_name in test_formats:
            img.save(filename, format=format_name)
            print(f"   ✅ Created: {filename}")
        
        print("\n💡 Sample files created! Use these to test LayoutLMv3:")
        print("   - sample_invoice.png (best quality)")
        print("   - sample_invoice.jpg (smaller size)")
        print("   - sample_invoice.tiff (high quality)")
        
    except Exception as e:
        print(f"   ❌ Error creating samples: {str(e)}")

if __name__ == "__main__":
    test_input_types()
    test_pdf_conversion()
    create_sample_inputs()
    
    print("\n" + "=" * 50)
    print("✅ Input testing completed!")
    print("\n🚀 Next steps:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Test with your PDF: python test_inference.py")
    print("   3. Try the API: python api.py")
