#!/usr/bin/env python3
"""
Test script for the Donut Invoice Processing API
"""

import requests
import json
import time
import os

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("\nTesting root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_model_status():
    """Test the model status endpoint"""
    print("\nTesting model status...")
    try:
        response = requests.get(f"{BASE_URL}/model-status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_training_endpoint():
    """Test the training endpoint (will fail if no dataset exists)"""
    print("\nTesting training endpoint...")
    try:
        # This will likely fail since we don't have a dataset
        payload = {
            "dataset_path": "./donut_dataset.json",
            "num_epochs": 1
        }
        response = requests.post(
            f"{BASE_URL}/train-donut",
            json=payload
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code in [200, 400]  # 400 is expected if no dataset
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_invoice_processing():
    """Test the invoice processing endpoint (will fail if no model exists)"""
    print("\nTesting invoice processing endpoint...")
    try:
        # This will likely fail since we don't have a trained model
        files = {
            'file': ('test_invoice.png', b'fake_image_data', 'image/png')
        }
        data = {
            'query': 'test query'
        }
        response = requests.post(
            f"{BASE_URL}/process-invoice",
            files=files,
            data=data
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code in [200, 400]  # 400 is expected if no model
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting API tests...")
    print("=" * 50)
    
    # Test basic endpoints
    tests = [
        ("Health Check", test_health_check),
        ("Root Endpoint", test_root_endpoint),
        ("Model Status", test_model_status),
        ("Training Endpoint", test_training_endpoint),
        ("Invoice Processing", test_invoice_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)  # Small delay between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
        if success:
            passed += 1
    
    print(f"\nTotal: {len(results)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {len(results) - passed}")
    
    if passed == len(results):
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed")
        print("Note: Some failures are expected if the model hasn't been trained yet")

if __name__ == "__main__":
    main()
