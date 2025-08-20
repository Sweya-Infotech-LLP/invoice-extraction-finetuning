#!/usr/bin/env python3
"""
Example usage script for the Donut Invoice Processing API
Demonstrates how to use the new invoice type functionality
"""

import requests
import json
import os
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def check_api_status():
    """Check if the API is running and get folder structure"""
    try:
        print("🔍 Checking API status...")
        
        # Health check
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is running")
        else:
            print("❌ API is not responding")
            return False
        
        # Get folder structure
        response = requests.get(f"{BASE_URL}/folder-structure")
        if response.status_code == 200:
            folders = response.json()["folder_structure"]
            print("\n📁 Folder Structure:")
            for folder_name, folder_info in folders.items():
                status = "✅" if folder_info["exists"] else "❌"
                print(f"   {status} {folder_name}: {folder_info['path']}")
        else:
            print("❌ Could not retrieve folder structure")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking API status: {e}")
        return False

def train_model_example():
    """Example of training a Donut model"""
    print("\n🚀 Training Model Example")
    print("=" * 40)
    
    # Note: This will fail if no dataset exists
    payload = {
        "dataset_path": "./donut_dataset.json",
        "num_epochs": 1
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/train-donut",
            json=payload
        )
        
        if response.status_code == 200:
            print("✅ Training started successfully")
            print(f"Response: {response.json()}")
        else:
            print(f"⚠️ Training request failed (expected if no dataset): {response.status_code}")
            print(f"Response: {response.json()}")
            
    except Exception as e:
        print(f"❌ Error during training request: {e}")

def process_utility_invoice_example():
    """Example of processing a utility invoice"""
    print("\n🔌 Utility Invoice Processing Example")
    print("=" * 40)
    
    # This is a mock example - you would need an actual image file
    print("Note: This example shows the API call structure")
    print("You would need to provide an actual invoice image file")
    
    # Example API call structure
    print("\nAPI Call Structure:")
    print("POST /process-invoice")
    print("Form Data:")
    print("  - file: [invoice_image.png]")
    print("  - query: utility bill analysis")
    print("  - invoice_type: utility")
    
    print("\nExpected Response Structure:")
    expected_response = {
        "status": "success",
        "donut_output": "Extracted invoice text...",
        "retrieved_documents": [],
        "structured_response": "Structured utility JSON...",
        "donut_response_file": "./responses/donut_response_20241201_143022.json",
        "llm_response_file": "./llm_responses/llm_response_20241201_143022.json",
        "invoice_type": "utility"
    }
    print(json.dumps(expected_response, indent=2))

def process_telecom_invoice_example():
    """Example of processing a telecom invoice"""
    print("\n📱 Telecom Invoice Processing Example")
    print("=" * 40)
    
    print("Note: This example shows the API call structure")
    print("You would need to provide an actual invoice image file")
    
    # Example API call structure
    print("\nAPI Call Structure:")
    print("POST /process-invoice")
    print("Form Data:")
    print("  - file: [telecom_invoice.png]")
    print("  - query: telecom service analysis")
    print("  - invoice_type: telecom")
    
    print("\nExpected Response Structure:")
    expected_response = {
        "status": "success",
        "donut_output": "Extracted invoice text...",
        "retrieved_documents": [],
        "structured_response": "Structured telecom JSON...",
        "donut_response_file": "./responses/donut_response_20241201_143023.json",
        "llm_response_file": "./llm_responses/llm_response_20241201_143023.json",
        "invoice_type": "telecom"
    }
    print(json.dumps(expected_response, indent=2))

def show_curl_examples():
    """Show curl examples for the API endpoints"""
    print("\n📋 cURL Examples")
    print("=" * 40)
    
    print("1. Train Model:")
    print("curl -X POST \"http://localhost:8000/train-donut\" \\")
    print("     -H \"Content-Type: application/json\" \\")
    print("     -d '{\"dataset_path\": \"./donut_dataset.json\", \"num_epochs\": 5}'")
    
    print("\n2. Process Utility Invoice:")
    print("curl -X POST \"http://localhost:8000/process-invoice\" \\")
    print("     -F \"file=@utility_invoice.png\" \\")
    print("     -F \"query=utility bill analysis\" \\")
    print("     -F \"invoice_type=utility\"")
    
    print("\n3. Process Telecom Invoice:")
    print("curl -X POST \"http://localhost:8000/process-invoice\" \\")
    print("     -F \"file=@telecom_invoice.png\" \\")
    print("     -F \"query=telecom service analysis\" \\")
    print("     -F \"invoice_type=telecom\"")
    
    print("\n4. Check Folder Structure:")
    print("curl -X GET \"http://localhost:8000/folder-structure\"")

def main():
    """Main function to run all examples"""
    print("🚀 Donut Invoice Processing API - Usage Examples")
    print("=" * 60)
    
    # Check API status
    if not check_api_status():
        print("\n❌ Cannot proceed - API is not available")
        print("Please start the API server first using: python run.py")
        return
    
    # Show examples
    train_model_example()
    process_utility_invoice_example()
    process_telecom_invoice_example()
    show_curl_examples()
    
    print("\n" + "=" * 60)
    print("📖 For more information, visit:")
    print(f"   API Documentation: {BASE_URL}/docs")
    print(f"   ReDoc: {BASE_URL}/redoc")
    print("\n💡 Tips:")
    print("   - Make sure you have trained a model before processing invoices")
    print("   - Use 'utility' invoice type for utility bills")
    print("   - Use 'telecom' invoice type for telecom bills")
    print("   - Check the folder structure endpoint to monitor your setup")
    print("   - All responses are automatically saved with timestamps")

if __name__ == "__main__":
    main()
