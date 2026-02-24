# API Testing Guide

This guide shows you how to test all the endpoints in your Face Recognition Backend API.

## Prerequisites

1. Make sure your server is running:
   ```bash
   ./venv/bin/uvicorn app.main:app --reload
   ```

2. Install the `requests` library if using the Python test script:
   ```bash
   ./venv/bin/pip install requests
   ```

## Method 1: Interactive API Documentation (Easiest)

FastAPI automatically generates interactive API documentation:

1. **Swagger UI** (Recommended): Open in your browser:
   ```
   http://localhost:8000/docs
   ```
   - Click on any endpoint to expand it
   - Click "Try it out" to test the endpoint
   - Fill in the request body and click "Execute"
   - See the response below

2. **ReDoc**: Alternative documentation format:
   ```
   http://localhost:8000/redoc
   ```

## Method 2: Using the Test Scripts

### Python Script (Recommended)
```bash
./venv/bin/python test_api.py
```

### Bash Script
```bash
./test_api.sh
```

## Method 3: Manual Testing with curl

### 1. Root Endpoint
```bash
curl http://localhost:8000/
```

### 2. Health Check
```bash
curl http://localhost:8000/health
```

### 3. Give Consent
```bash
curl -X POST http://localhost:8000/api/consent \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "consent_version": "v1.0"
  }'
```

### 4. Get Consent
```bash
curl http://localhost:8000/api/consent/user_123
```

### 5. Register Device
```bash
curl -X POST http://localhost:8000/api/devices \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "kiosk_01",
    "store_id": "store_100"
  }'
```

### 6. Enroll Embedding
```bash
curl -X POST http://localhost:8000/api/enroll-embedding \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  }'
```

**Note:** In production, embeddings are typically 128 or 512-dimensional vectors. The example above uses 10 values for simplicity.

### 7. Recognize Embedding
```bash
curl -X POST http://localhost:8000/api/recognize-embedding \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "top_k": 1,
    "device_id": "kiosk_01"
  }'
```

### 8. Deactivate Embedding
```bash
curl -X POST http://localhost:8000/api/embeddings/user_123/deactivate
```

## Method 4: Using Python requests library

```python
import requests

BASE_URL = "http://localhost:8000"

# Enroll an embedding
response = requests.post(
    f"{BASE_URL}/api/enroll-embedding",
    json={
        "user_id": "user_123",
        "embedding": [0.1] * 128  # 128-dimensional vector
    }
)
print(response.json())

# Recognize an embedding
response = requests.post(
    f"{BASE_URL}/api/recognize-embedding",
    json={
        "embedding": [0.1] * 128,
        "top_k": 1,
        "device_id": "kiosk_01"
    }
)
print(response.json())
```

## Available Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/api/consent` | Give consent for a user |
| GET | `/api/consent/{user_id}` | Get consent status for a user |
| POST | `/api/devices` | Register a device |
| POST | `/api/enroll-embedding` | Enroll a face embedding |
| POST | `/api/recognize-embedding` | Recognize a face from embedding |
| POST | `/api/embeddings/{user_id}/deactivate` | Deactivate an embedding |

## Testing Tips

1. **Start with Swagger UI** (`/docs`) - It's the easiest way to explore and test endpoints
2. **Check the database** - After testing, you can check `face.db` to see stored data
3. **Test error cases** - Try invalid user_ids, missing fields, etc.
4. **Test the flow** - Give consent → Enroll embedding → Recognize embedding

## Troubleshooting

- **Connection refused**: Make sure the server is running
- **404 errors**: Check that you're using the correct endpoint paths
- **422 errors**: Check that your request body matches the expected schema
- **Database errors**: Make sure `face.db` exists (run `python3 -c "from app.init_db import *"` to initialize)
