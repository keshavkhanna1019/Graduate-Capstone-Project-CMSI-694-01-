#!/usr/bin/env python3
"""
API Testing Script for Face Recognition Backend
Run this script to test all endpoints
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def print_response(title, response):
    """Pretty print API response"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")

def test_apis():
    """Test all API endpoints"""
    
    # Test 1: Root endpoint
    response = requests.get(f"{BASE_URL}/")
    print_response("1. Root Endpoint (GET /)", response)
    
    # Test 2: Health check
    response = requests.get(f"{BASE_URL}/health")
    print_response("2. Health Check (GET /health)", response)
    
    # Test 3: Give consent
    response = requests.post(
        f"{BASE_URL}/api/consent",
        json={
            "user_id": "user_123",
            "consent_version": "v1.0"
        }
    )
    print_response("3. Give Consent (POST /api/consent)", response)
    
    # Test 4: Get consent
    response = requests.get(f"{BASE_URL}/api/consent/user_123")
    print_response("4. Get Consent (GET /api/consent/user_123)", response)
    
    # Test 5: Register device
    response = requests.post(
        f"{BASE_URL}/api/devices",
        json={
            "device_id": "kiosk_01",
            "store_id": "store_100"
        }
    )
    print_response("5. Register Device (POST /api/devices)", response)
    
    # Test 6: Enroll embedding
    # Using a sample embedding vector (typically 128 or 512 dimensions)
    sample_embedding = [0.1] * 128  # 128-dimensional embedding
    response = requests.post(
        f"{BASE_URL}/api/enroll-embedding",
        json={
            "user_id": "user_123",
            "embedding": sample_embedding
        }
    )
    print_response("6. Enroll Embedding (POST /api/enroll-embedding)", response)
    
    # Test 7: Recognize embedding
    response = requests.post(
        f"{BASE_URL}/api/recognize-embedding",
        json={
            "embedding": sample_embedding,
            "top_k": 1,
            "device_id": "kiosk_01"
        }
    )
    print_response("7. Recognize Embedding (POST /api/recognize-embedding)", response)
    
    # Test 8: Deactivate embedding
    response = requests.post(f"{BASE_URL}/api/embeddings/user_123/deactivate")
    print_response("8. Deactivate Embedding (POST /api/embeddings/user_123/deactivate)", response)
    
    print(f"\n{'='*50}")
    print("Testing complete!")
    print(f"{'='*50}")

if __name__ == "__main__":
    try:
        test_apis()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the server.")
        print("Make sure the server is running on http://localhost:8000")
        print("\nStart the server with:")
        print("  ./venv/bin/uvicorn app.main:app --reload")
    except Exception as e:
        print(f"ERROR: {e}")
