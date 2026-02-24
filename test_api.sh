#!/bin/bash

# API Testing Script for Face Recognition Backend
# Make sure your server is running on http://localhost:8000

BASE_URL="http://localhost:8000"

echo "=========================================="
echo "Testing Face Recognition API"
echo "=========================================="
echo ""

# Test 1: Root endpoint
echo "1. Testing root endpoint (GET /)"
curl -s "$BASE_URL/" | python3 -m json.tool
echo -e "\n"

# Test 2: Health check
echo "2. Testing health check (GET /health)"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo -e "\n"

# Test 3: Give consent
echo "3. Testing consent endpoint (POST /api/consent)"
curl -s -X POST "$BASE_URL/api/consent" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "consent_version": "v1.0"
  }' | python3 -m json.tool
echo -e "\n"

# Test 4: Get consent
echo "4. Testing get consent (GET /api/consent/user_123)"
curl -s "$BASE_URL/api/consent/user_123" | python3 -m json.tool
echo -e "\n"

# Test 5: Register device
echo "5. Testing device registration (POST /api/devices)"
curl -s -X POST "$BASE_URL/api/devices" \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "kiosk_01",
    "store_id": "store_100"
  }' | python3 -m json.tool
echo -e "\n"

# Test 6: Enroll embedding
echo "6. Testing enroll embedding (POST /api/enroll-embedding)"
curl -s -X POST "$BASE_URL/api/enroll-embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  }' | python3 -m json.tool
echo -e "\n"

# Test 7: Recognize embedding
echo "7. Testing recognize embedding (POST /api/recognize-embedding)"
curl -s -X POST "$BASE_URL/api/recognize-embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "top_k": 1,
    "device_id": "kiosk_01"
  }' | python3 -m json.tool
echo -e "\n"

# Test 8: Deactivate embedding
echo "8. Testing deactivate embedding (POST /api/embeddings/user_123/deactivate)"
curl -s -X POST "$BASE_URL/api/embeddings/user_123/deactivate" | python3 -m json.tool
echo -e "\n"

echo "=========================================="
echo "Testing complete!"
echo "=========================================="
