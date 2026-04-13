"""
Quick test script to diagnose enrollment issues
Run this to see what's wrong
"""
import requests
import os

API_BASE = "http://localhost:8000/api"

def test_enrollment():
    print("Testing enrollment...")
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{API_BASE.replace('/api', '')}/health", timeout=2)
        print(f"✅ Server is running: {response.status_code}")
    except Exception as e:
        print(f"❌ Server not running: {e}")
        print("   Start server with: uvicorn app.main:app --reload")
        return
    
    # Test 2: Give consent
    user_id = "test_user_123"
    try:
        response = requests.post(
            f"{API_BASE}/consent",
            json={"user_id": user_id, "consent_version": "v1.0"}
        )
        print(f"✅ Consent given: {response.status_code}")
    except Exception as e:
        print(f"❌ Consent failed: {e}")
        return
    
    # Test 3: Check if we have a test image
    test_image = "test_face.jpg"
    if not os.path.exists(test_image):
        print(f"⚠️  No test image found at {test_image}")
        print("   Create a test image or use the web UI")
        return
    
    # Test 4: Try enrollment
    try:
        with open(test_image, 'rb') as f:
            files = {'file': (test_image, f, 'image/jpeg')}
            response = requests.post(
                f"{API_BASE}/enroll-face?user_id={user_id}",
                files=files
            )
        
        if response.status_code == 200:
            print(f"✅ Enrollment successful!")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Enrollment failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Enrollment error: {e}")

if __name__ == "__main__":
    test_enrollment()
