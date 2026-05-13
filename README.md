# Zero-Friction Face Recognition System

A locally-hosted web application for face enrollment and recognition using **ArcFace** embeddings
and an **SVM classifier**. Built with FastAPI (Python) and a plain HTML/JS frontend — no cloud
dependencies, no API keys.

---

## How it works

1. **Enroll** — upload or capture a photo; the backend extracts a 512-dim ArcFace embedding and
   stores it in SQLite (no raw images stored).
2. **Recognize** — upload or capture a photo; the backend extracts an embedding and runs it through
   the SVM classifier (or cosine similarity fallback) to identify the person.
3. **Optional: Train SVM** — collect a few labeled photos per user, then hit Train SVM for sharper
   separation and unknown-face rejection.

Recognition works immediately after enrollment with no training step required.

---

## Requirements

- **Python 3.9 or 3.11** (3.11 recommended; matches the Docker image)
- pip
- A webcam (optional — all flows also accept uploaded images)

---

## Quick start (local)

```bash
# 1. Clone the repo
git clone <repo-url>
cd "Image Recognition App"

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The database (`face.db`) is created automatically on first run.

| URL | What it is |
|---|---|
| http://localhost:8000 | Web UI |
| http://localhost:8000/docs | Interactive API docs (Swagger) |
| http://localhost:8000/redoc | API docs (ReDoc) |

---

## Local artifacts (not in the repo)

These files are **git-ignored** and are generated locally at runtime:

| File / folder | How to get it |
|---|---|
| `face.db` | Created automatically when the server first starts |
| `svm_classifier.pkl` | Generated when you click Train SVM in the UI (or call `POST /api/train-svm`) |
| `data/` | Collected via the Collect Data tab — used only for SVM training |

No model weights need to be downloaded or copied. ArcFace weights are fetched automatically by
DeepFace on first inference and cached in `~/.deepface/`.

---

## Happy-path demo (no training required)

This is the shortest path from a clean state to a working recognition result.

### Step 1 — Enroll a user
1. Open the **Enroll** tab
2. Enter a User ID (e.g. `user_keshav`)
3. Upload a clear face photo or use the webcam capture
4. Click **Enroll Face** — consent is recorded automatically, embedding is saved

### Step 2 — Recognize
1. Open the **Recognize** tab
2. Upload or capture a photo of the enrolled person
3. Click **Recognize Face** — the matched user and confidence score appear

> First inference takes 10–30 s while ArcFace weights load. Subsequent requests are fast.

---

## Full demo (with SVM for better accuracy)

Use this flow when you have multiple enrolled users and want sharper separation and unknown-face
rejection.

### Step 1 — Enroll all users
Follow the Enroll steps above for each user.

### Step 2 — Collect training photos
1. Open the **Collect Data** tab
2. Select a User ID and choose **Positive** (photos of that person)
3. Capture via webcam **or** drag-and-drop / upload images from your gallery
4. Repeat for **Negative** photos (other people), then repeat for each enrolled user
5. Aim for 10–20 positive photos per user

### Step 3 — Train SVM
1. Open the **Train SVM** tab
2. Click **Train SVM** — requires at least 2 enrolled users with training photos
3. Wait for the status badge to show `Trained` (~5–15 s)

### Step 4 — Recognize
Recognition now uses SVM probability scores. Results show a per-user confidence bar and a
`SVM` or `Cosine` classifier badge. Faces with no confident match are returned as `Unknown`.

---

## Docker

```bash
# Build and run
docker compose up --build
```

App available at http://localhost:8000. No model files need to be mounted — ArcFace weights are
fetched automatically on first inference.

### Known limitations

| Limitation | Detail |
|---|---|
| **No GPU** | ArcFace inference runs on CPU. ~1–3 s per request after warm-up. |
| **Cold start** | First inference downloads ArcFace weights (~100 MB) and takes 10–30 s. |
| **SQLite** | Single-file DB — not suitable for multi-replica deployments. |
| **Camera** | Webcam features work when the container is on the same machine as the browser. For remote hosting, serve over HTTPS. |

---

## Smoke check

Verify all core API endpoints are healthy (server must be running first):

```bash
python smoke_test.py

# Different port
API_ORIGIN=http://127.0.0.1:8001 python smoke_test.py

# Also test enroll + recognize
python smoke_test.py --image path/to/face.jpg
```

Exits `0` on pass, `1` on failure.

---

## Project structure

```
Image Recognition App/
├── app/
│   ├── main.py              # FastAPI app, mounts static files
│   ├── db.py                # SQLite connection (face.db)
│   ├── init_db.py           # Creates tables on first run
│   ├── models.py            # SQLAlchemy models
│   ├── repositories/        # DB access layer (embeddings, consent, logs)
│   └── services/
│       ├── face_extraction.py   # ArcFace embedding extraction (DeepFace)
│       ├── svm_classifier.py    # SVM classifier with unknown-face rejection
│       ├── similarity.py        # Cosine similarity (fallback)
│       └── siamese_network.py   # Legacy — kept for reference only
├── routes/                  # FastAPI routers (one per domain)
│   ├── recognition.py       # Enroll + recognize endpoints
│   ├── training.py          # Train SVM + status endpoints
│   ├── embeddings.py        # Embedding management
│   └── data_collection.py   # Training image save/stats
├── static/index.html        # Single-file frontend (HTML/CSS/JS)
├── smoke_test.py            # API smoke check script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/` | Web UI |
| GET | `/docs` | Swagger API docs |
| POST | `/api/consent` | Grant biometric consent |
| GET | `/api/consent/{user_id}` | Check consent status |
| GET | `/api/consents` | List all consents |
| DELETE | `/api/consent/{user_id}` | Revoke consent |
| POST | `/api/enroll-face` | Enroll from uploaded image (ArcFace) |
| POST | `/api/enroll-embedding` | Enroll from raw embedding vector |
| POST | `/api/recognize-face` | Recognize from uploaded image (SVM → cosine) |
| POST | `/api/recognize-embedding` | Recognize from raw embedding vector |
| POST | `/api/extract-face-embedding` | Extract ArcFace embedding without enrolling |
| GET | `/api/embeddings` | List enrolled users |
| DELETE | `/api/embeddings/{user_id}` | Delete enrollment |
| POST | `/api/train-svm` | Train SVM classifier on stored embeddings |
| GET | `/api/svm-status` | SVM training status and user count |
| GET | `/api/training-status` | Background training job status |
| POST | `/api/save-training-image` | Save a labeled training image |
| GET | `/api/training-data-stats` | Image counts per user/category |

---

## Troubleshooting

**`uvicorn: command not found`**
```bash
source venv/bin/activate   # activate the venv first
```

**First inference is very slow (30+ s)**
ArcFace weights are being downloaded on first use. This is a one-time download cached at
`~/.deepface/`. Subsequent requests are fast.

**"No face detected" error**
Ensure the photo has a clearly visible, well-lit face. The system uses OpenCV face detection —
extreme angles, heavy blur, or very dark images may fail detection. `enforce_detection=False` is
set, so it will attempt extraction even with low-confidence detection.

**SVM returns "Unknown" for a known user**
The SVM confidence threshold is 45%. Re-enroll the user with a clearer photo, or collect more
training photos and retrain the SVM.

**SVM "needs at least 2 users" error**
The SVM requires at least 2 enrolled users with training photos to learn class boundaries.
Enroll a second user before training.

**Database errors**
```bash
rm face.db
python -m app.init_db
```

---

## Author

Keshav Khanna — CMSI 694 Graduate Capstone Project
