#!/usr/bin/env python3
"""
Super simple test script - just run: python3 quick_test.py
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("🧪 Testing Face Recognition API\n")
print("=" * 50)

# Test 1: Health Check
print("1. Testing Health Check...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")
    print("   Make sure your server is running!")
    exit(1)

# Test 2: Give Consent
print("2. Giving Consent...")
try:
    response = requests.post(
        f"{BASE_URL}/api/consent",
        json={"user_id": "test_user", "consent_version": "v1.0"}
    )
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 3: Register Device
print("3. Registering Device...")
try:
    response = requests.post(
        f"{BASE_URL}/api/devices",
        json={"device_id": "test_kiosk", "store_id": "test_store"}
    )
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 4: Enroll Embedding
print("4. Enrolling Embedding...")
try:
    # Simple 10-dimensional embedding for testing
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    response = requests.post(
        f"{BASE_URL}/api/enroll-embedding",
        json={"user_id": "test_user", "embedding": embedding}
    )
    print(f"   ✅ Status: {response.status_code}")
    print(f"   Response: {response.json()}\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 5: Recognize Embedding
print("5. Recognizing Embedding...")
try:
    embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    response = requests.post(
        f"{BASE_URL}/api/recognize-embedding",
        json={
            "embedding": embedding,
            "top_k": 1,
            "device_id": "test_kiosk"
        }
    )
    print(f"   ✅ Status: {response.status_code}")
    result = response.json()
    print(f"   Matches found: {len(result.get('matches', []))}")
    if result.get('matches'):
        print(f"   Top match: {result['matches'][0]}\n")
    else:
        print(f"   (No matches - this is OK if similarity is low)\n")
except Exception as e:
    print(f"   ❌ Error: {e}\n")

# Test 6: Check Database
print("6. Checking Database...")
try:
    import sqlite3
    conn = sqlite3.connect('face.db')
    cur = conn.cursor()
    
    cur.execute("SELECT COUNT(*) FROM consent")
    consent_count = cur.fetchone()[0]
    print(f"   ✅ Consent records: {consent_count}")
    
    cur.execute("SELECT COUNT(*) FROM devices")
    device_count = cur.fetchone()[0]
    print(f"   ✅ Device records: {device_count}")
    
    cur.execute("SELECT COUNT(*) FROM embeddings")
    embedding_count = cur.fetchone()[0]
    print(f"   ✅ Embedding records: {embedding_count}")
    
    cur.execute("SELECT COUNT(*) FROM recognition_logs")
    log_count = cur.fetchone()[0]
    print(f"   ✅ Recognition logs: {log_count}")
    
    conn.close()
    print()
except Exception as e:
    print(f"   ⚠️  Could not check database: {e}\n")

print("=" * 50)
print("✅ Testing complete!")
print("\n💡 Tip: Use http://localhost:8000/docs for interactive testing")
