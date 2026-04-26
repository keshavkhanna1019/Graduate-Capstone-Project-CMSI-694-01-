# Face Recognition System

A locally-hosted web application for face recognition using a Siamese Neural Network.
Built with FastAPI (Python) + a plain HTML/JS frontend.

---

## Requirements

- **Python 3.9 or 3.11** (3.11 recommended; matches the Docker image)
- pip
- A webcam (for data collection and live capture features)

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

These files are **git-ignored** and must be generated locally:

| File / folder | How to get it |
|---|---|
| `face.db` | Created automatically when the server first starts |
| `siamese_model.h5`, `siamese_model_user_*.h5` | Train via the web UI or API (see Model Training below) |
| `data/` | Collected via the Collect Data tab in the UI |

---

## Happy-path demo (consent → enroll → recognize)

This is the complete flow from a clean state:

### Step 1 — Collect training data
1. Open **Advanced → Collect Data**
2. Enter a User ID (e.g. `user_keshav`)
3. Start the camera and capture **20–50 Positive** photos (of the target person) and **10–20 Negative** photos (of different people)

### Step 2 — Train the model
1. Open **Advanced → Train Model**
2. Enter the same User ID, set Epochs to **100–200**, click **Start Training**
3. Wait for training to complete (~5–15 min depending on hardware)

### Step 3 — Enroll
1. Open **Enroll Face** (consent is recorded automatically)
2. Enter the User ID, upload or capture a clear face photo
3. Click **Enroll Face** — you should see a success message

### Step 4 — Recognize
1. Open **Recognize Face**
2. Upload or capture a photo of the enrolled person
3. Click **Recognize Face** — the top match and similarity score appear

> **Score guide:** scores above ~0.75 are a strong match. All scores will be
> high if the model was undertrained (embedding collapse) — train for more epochs.

---

## Quick-path demo (pre-trained model required)

If `siamese_model.h5` already exists (e.g. copied from another machine):

```bash
# 1. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Open http://localhost:8000
# 3. Enroll Face → enter user ID → upload photo → Enroll Face
# 4. Recognize Face → upload photo → Recognize Face
```

---

## Model Training

Model files (`*.h5`) are large (~107 MB each) and git-ignored. Train locally:

### Option 1 — Web UI (recommended)
1. **Advanced → Collect Data**: capture Positive and Negative images for a User ID
2. **Advanced → Train Model**: enter the User ID, 100–200 epochs, Start Training
3. Model saved as `siamese_model_<user_id>.h5` in the project root

### Option 2 — API
```bash
curl -X POST http://localhost:8000/api/train-model \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "epochs": 100, "batch_size": 16, "data_dir": "data"}'
```

Training data layout (git-ignored):
```
data/
└── user_123/
    ├── positive/   # 20–50 images of the target person
    └── negative/   # 10–20 images of other people
```

---

## Docker

```bash
# Build and run
docker compose up --build
```

App available at http://localhost:8000. See Docker section notes below.

### Mounting models
Model files are not bundled in the image. Copy `.h5` files to the project root,
then add a volume line per model in `docker-compose.yml`:

```yaml
volumes:
  - ./siamese_model.h5:/app/siamese_model.h5
  - ./siamese_model_user_keshav.h5:/app/siamese_model_user_keshav.h5
```

### Known limitations

| Limitation | Detail |
|---|---|
| **No GPU** | TensorFlow runs on CPU in the container. ~2–5s per inference. |
| **Model size** | Each `*.h5` is ~107 MB — train locally and bind-mount. |
| **Cold start** | First inference takes 10–30s while TensorFlow loads the model. |
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

Exits `0` on pass, `1` on failure. Enroll/recognize are skipped if no model or image is present.

---

## Project structure

```
Image Recognition App/
├── app/
│   ├── main.py              # FastAPI app, mounts static files
│   ├── db.py                # SQLite connection (face.db)
│   ├── init_db.py           # Creates tables on first run
│   ├── models.py            # SQLAlchemy models
│   ├── repositories/        # DB access layer
│   └── services/
│       ├── siamese_network.py   # Siamese network (TF/Keras)
│       ├── face_extraction.py   # Embedding extraction
│       └── similarity.py        # Cosine similarity
├── routes/                  # FastAPI routers (one per domain)
├── static/index.html        # Single-file frontend (HTML/CSS/JS)
├── smoke_test.py            # API smoke check script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Troubleshooting

**`uvicorn: command not found`**
```bash
source venv/bin/activate   # activate the venv first
```

**`No trained model found at siamese_model.h5`**
Train a model first (see Model Training above).

**All recognition scores are very high (>95%) for everyone**
The model has embedding collapse from insufficient training. Re-train with 100–200 epochs and more training images.

**Database errors**
```bash
rm face.db
python -m app.init_db
```

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/` | Web UI |
| GET | `/docs` | Swagger API docs |
| POST | `/api/consent` | Grant consent |
| GET | `/api/consent/{user_id}` | Check consent |
| GET | `/api/consents` | List all consents |
| DELETE | `/api/consent/{user_id}` | Delete consent |
| POST | `/api/enroll-face` | Enroll a face |
| POST | `/api/recognize-face` | Recognize a face |
| GET | `/api/embeddings` | List enrolled users |
| DELETE | `/api/embeddings/{user_id}` | Delete enrollment |
| POST | `/api/train-model` | Start model training |
| GET | `/api/training-status` | Poll training progress |
| POST | `/api/save-training-image` | Save a training image |
| GET | `/api/training-data-stats` | Image counts per category |

---

## Author

Keshav Khanna — CMSI 6352 Deep Learning Assignment #2
