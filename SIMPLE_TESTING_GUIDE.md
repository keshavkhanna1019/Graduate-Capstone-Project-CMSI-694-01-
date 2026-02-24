# Simple Testing Guide

## Step 1: Start Your Server

Open a terminal and run:
```bash
cd /Users/keshav/Desktop/CMSCI-694
./venv/bin/uvicorn app.main:app --reload
```

You should see: `Uvicorn running on http://127.0.0.1:8000`

---

## Step 2: Test Using Your Browser (Easiest Way!)

### Open the Interactive API Docs
1. Open your web browser
2. Go to: **http://localhost:8000/docs**
3. You'll see a list of all your API endpoints

### Test Each Endpoint:

#### 1. Health Check
- Find **GET /health**
- Click on it
- Click **"Try it out"**
- Click **"Execute"**
- ✅ Should return: `{"status": "ok"}`

#### 2. Give Consent
- Find **POST /api/consent**
- Click **"Try it out"**
- Fill in:
  ```json
  {
    "user_id": "test_user_1",
    "consent_version": "v1.0"
  }
  ```
- Click **"Execute"**
- ✅ Should return consent with timestamp

#### 3. Register a Device
- Find **POST /api/devices**
- Click **"Try it out"**
- Fill in:
  ```json
  {
    "device_id": "kiosk_01",
    "store_id": "store_100"
  }
  ```
- Click **"Execute"**
- ✅ Should return device info with status "active"

#### 4. Enroll an Embedding
- Find **POST /api/enroll-embedding**
- Click **"Try it out"**
- Fill in:
  ```json
  {
    "user_id": "test_user_1",
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  }
  ```
- Click **"Execute"**
- ✅ Should return: `{"status": "accepted", "user_id": "test_user_1"}`

#### 5. Recognize an Embedding
- Find **POST /api/recognize-embedding**
- Click **"Try it out"**
- Fill in:
  ```json
  {
    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "top_k": 1,
    "device_id": "kiosk_01"
  }
  ```
- Click **"Execute"**
- ✅ Should return matches (might be empty if similarity is low)

#### 6. Deactivate Embedding
- Find **POST /api/embeddings/{user_id}/deactivate**
- Click **"Try it out"**
- Enter: `test_user_1` in the user_id field
- Click **"Execute"**
- ✅ Should return deactivation info with timestamp

---

## Step 3: Check Your Database

### Option A: Using Python (Simple)
```bash
python3
```

Then paste this:
```python
import sqlite3

# Connect to database
conn = sqlite3.connect('face.db')
cur = conn.cursor()

# Check consent table
print("=== CONSENT TABLE ===")
cur.execute("SELECT * FROM consent")
for row in cur.fetchall():
    print(row)

# Check devices table
print("\n=== DEVICES TABLE ===")
cur.execute("SELECT * FROM devices")
for row in cur.fetchall():
    print(row)

# Check embeddings table
print("\n=== EMBEDDINGS TABLE ===")
cur.execute("SELECT user_id, is_active, deactivated_at FROM embeddings")
for row in cur.fetchall():
    print(row)

# Check recognition logs
print("\n=== RECOGNITION LOGS ===")
cur.execute("SELECT * FROM recognition_logs ORDER BY timestamp DESC LIMIT 5")
for row in cur.fetchall():
    print(row)

conn.close()
```

### Option B: Using SQLite Command Line
```bash
sqlite3 face.db
```

Then run:
```sql
SELECT * FROM consent;
SELECT * FROM devices;
SELECT user_id, is_active, deactivated_at FROM embeddings;
SELECT * FROM recognition_logs ORDER BY timestamp DESC LIMIT 5;
```

Type `.quit` to exit.

---

## Quick Test Flow (Test Everything Together)

Follow this order to test the complete flow:

1. **Give Consent** → `POST /api/consent` with `user_id: "alice"`
2. **Register Device** → `POST /api/devices` with `device_id: "kiosk_01"`
3. **Enroll Embedding** → `POST /api/enroll-embedding` with `user_id: "alice"` and an embedding
4. **Recognize** → `POST /api/recognize-embedding` with same embedding → Should find "alice"!
5. **Deactivate** → `POST /api/embeddings/alice/deactivate`
6. **Recognize Again** → Should NOT find "alice" (deactivated)

---

## What to Look For (Success Signs)

✅ **API Works If:**
- All endpoints return 200 status codes
- You see JSON responses (not errors)
- Recognition finds matches when embeddings are similar
- Deactivation returns a timestamp

✅ **Database Works If:**
- You see data in the tables after API calls
- Consent records appear after giving consent
- Embeddings show `is_active = 1` when active, `0` when deactivated
- Recognition logs appear after recognition requests

❌ **If Something Fails:**
- Check server terminal for error messages
- Make sure server is running on port 8000
- Check that `face.db` file exists in your project folder
- Try running: `python3 -c "from app.init_db import *"` to initialize database

---

## Super Quick Test (30 seconds)

1. Open: http://localhost:8000/docs
2. Click **GET /health** → Try it out → Execute
3. If you see `{"status": "ok"}`, your API is working! 🎉

That's it! Your API is ready to use.
