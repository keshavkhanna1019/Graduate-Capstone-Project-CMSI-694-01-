# Face Recognition System - Deep Learning Assignment #2

A locally-hosted web application for face recognition using Siamese Neural Networks. This system allows users to enroll faces, perform real-time recognition, and manage consent for data collection.

## 🎯 Project Overview

This project implements a **face recognition system** that uses a **Siamese Neural Network** to generate face embeddings and perform similarity matching. The system includes:

- **Face Enrollment**: Upload images to create face embeddings for users
- **Face Recognition**: Match uploaded images against enrolled faces
- **Web Interface**: User-friendly HTML interface for interaction
- **REST API**: FastAPI-based backend with automatic documentation
- **Data Management**: SQLite database for storing embeddings, consent, and recognition logs

## 🏗️ Architecture

- **Backend**: FastAPI (Python)
- **Frontend**: HTML/CSS/JavaScript (static files)
- **Database**: SQLite
- **Model**: Siamese Neural Network (TensorFlow/Keras)
- **Image Processing**: OpenCV, PIL

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment support (venv)

## 🚀 Installation

### Step 1: Clone or Navigate to Project Directory

```bash
cd "/Users/keshav/Desktop/Deep Learning Web App"
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
```

### Step 3: Activate Virtual Environment

**On macOS/Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `python-multipart` - File upload support
- `deepface` - Face recognition utilities
- `opencv-python` - Image processing
- `pillow` - Image manipulation
- `numpy` - Numerical operations
- `tensorflow` - Deep learning framework (included via deepface)

### Step 5: Initialize Database

The database will be automatically created on first run. If you need to manually initialize:

```bash
python -m app.init_db
```

## 🎮 Running the Application

### Option 1: Using Uvicorn Directly

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option 2: Using the Start Script

```bash
chmod +x start_server.sh
./start_server.sh
```

### Option 3: Using Python Module

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at:
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs (Swagger UI)
- **Alternative API Docs**: http://localhost:8000/redoc (ReDoc)

## 📖 Usage Guide

### Web Interface

1. **Open your browser** and navigate to `http://localhost:8000`
2. **Give Consent**: First, provide consent for data collection (required for enrollment)
3. **Enroll a Face**: 
   - Enter a user ID
   - Upload a face image (JPG, PNG, etc.)
   - Click "Enroll Face"
4. **Recognize a Face**:
   - Upload an image
   - Click "Recognize Face"
   - View the matched user and similarity score

### API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Give Consent
```bash
curl -X POST http://localhost:8000/api/consent \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "consent_version": "v1.0"}'
```

#### Enroll Face
```bash
curl -X POST "http://localhost:8000/api/enroll-face?user_id=user_123" \
  -F "file=@path/to/image.jpg"
```

#### Recognize Face
```bash
curl -X POST "http://localhost:8000/api/recognize-face?device_id=kiosk_01" \
  -F "file=@path/to/image.jpg"
```

#### Extract Embedding
```bash
curl -X POST "http://localhost:8000/api/extract-embedding" \
  -F "file=@path/to/image.jpg"
```

For complete API documentation, visit `http://localhost:8000/docs` after starting the server.

## 📁 Project Structure

```
Deep Learning Web App/
├── app/                          # Main application package
│   ├── main.py                   # FastAPI application entry point
│   ├── db.py                     # Database connection
│   ├── init_db.py                # Database initialization
│   ├── repositories/             # Data access layer
│   │   ├── consent_repo.py       # Consent management
│   │   ├── device_repo.py        # Device management
│   │   ├── embeddings_repo.py    # Face embeddings storage
│   │   └── recognition_logs_repo.py  # Recognition event logging
│   └── services/                 # Business logic
│       ├── face_extraction.py    # Face embedding extraction
│       ├── siamese_network.py    # Siamese network implementation
│       ├── siamese_network_v2.py # Alternative implementation
│       └── similarity.py         # Similarity computation
├── routes/                        # API route handlers
│   ├── health.py                 # Health check endpoint
│   ├── recognition.py            # Face recognition endpoints
│   ├── consent.py                # Consent endpoints
│   ├── devices.py                # Device endpoints
│   ├── embeddings.py             # Embedding management
│   ├── training.py               # Model training endpoints
│   └── data_collection.py        # Data collection endpoints
├── static/                        # Frontend files
│   └── index.html                # Web interface
├── data/                          # Training and test data
│   ├── positive/                 # Positive face samples
│   └── negative/                 # Negative face samples
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── TECHNICAL_REPORT.md            # Technical documentation
├── siamese_model.h5               # Trained model (if available)
└── face.db                        # SQLite database (created on first run)
```

## 🧪 Testing

### Test Enrollment Script

A test script is provided to verify the system:

```bash
python test_enrollment.py
```

This script will:
1. Check if the server is running (`API_ORIGIN` defaults to `http://127.0.0.1:8000`; override if you use another port)
2. Give consent for a test user
3. Attempt enrollment with `test_face.jpg` in the project root
4. Call `GET /api/embeddings` to confirm the user appears (ticket verification)

### Manual Testing

1. **Start the server** (see Running the Application)
2. **Open the web interface** at `http://127.0.0.1:<port>/` (same port as the server)
3. **Enroll** with User ID + upload or camera; on success click **Verify — open Manage Embeddings** or check **GET /api/embeddings** in `/docs`
4. **Test recognition** with the same or similar image
5. **Verify results** in the web interface or API response

## 🔧 Configuration

### Model Path

The default model path is `siamese_model.h5`. To use a different model:

1. Place your model file in the project root
2. Update the model path in `app/services/face_extraction.py` or pass it as a parameter

### Database

The database file (`face.db`) is created automatically. To reset:

```bash
rm face.db
python -m app.init_db
```

### Port Configuration

Change the port by modifying the uvicorn command:

```bash
uvicorn app.main:app --reload --port 8080
```

## 🐛 Troubleshooting

### Issue: "uvicorn: command not found"

**Solution**: Make sure your virtual environment is activated:
```bash
source .venv/bin/activate
```

If the issue persists, reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "python-multipart not installed"

**Solution**: Install the missing package:
```bash
pip install python-multipart
```

### Issue: Model not found

**Solution**: 
1. Ensure `siamese_model.h5` exists in the project root
2. Or train a new model using `train_siamese.py`:
```bash
python train_siamese.py
```

### Issue: Database errors

**Solution**: Reinitialize the database:
```bash
rm face.db
python -m app.init_db
```

### Issue: Import errors

**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📊 Model Training

Trained model files (`*.h5`) and training images (`data/`) are **not committed** to this repository — they are large, user-specific, and must be generated locally.

### Option 1: Web UI (recommended)

1. Open the app at `http://localhost:8000`
2. Go to **Advanced → Collect Data**: start the camera and capture positive images (the target person) and negative images (other people) for a given User ID
3. Go to **Advanced → Train Model**: enter the same User ID and click **Start Training** (100–200 epochs recommended)
4. The trained model is saved as `siamese_model_{user_id}.h5` in the project root

### Option 2: API

```bash
# Trigger training via the REST API
curl -X POST http://localhost:8000/api/train-model \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_123", "epochs": 100, "batch_size": 16, "data_dir": "data"}'
```

Training data must be in `data/{user_id}/positive/` and `data/{user_id}/negative/` before calling this endpoint.

### Data layout (local only, git-ignored)

```
data/
└── user_123/
    ├── positive/   # 20–50 images of the target person
    └── negative/   # 10–20 images of other people
```

## 🔒 Privacy and Consent

This system implements consent management:
- Users must provide consent before enrollment
- Consent is stored in the database
- Recognition events are logged for audit purposes

## 📝 API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/api/consent` | POST | Give consent |
| `/api/enroll-face` | POST | Enroll a face |
| `/api/recognize-face` | POST | Recognize a face |
| `/api/extract-embedding` | POST | Extract face embedding |
| `/api/embeddings/{user_id}` | GET | Get user embeddings |
| `/api/train` | POST | Train model |
| `/docs` | GET | API documentation (Swagger) |

## 🎓 Assignment Requirements Checklist

- ✅ **Model Selection**: Siamese Neural Network for face recognition
- ✅ **Web Interface**: HTML/CSS/JavaScript frontend with FastAPI backend
- ✅ **Preprocessing**: Image resizing, normalization, face extraction
- ✅ **Inference Pipeline**: Complete enrollment and recognition workflow
- ✅ **Error Handling**: Input validation, error messages, graceful failures
- ✅ **Documentation**: README, technical report, code comments
- ✅ **Requirements File**: Complete `requirements.txt`
- ✅ **Local Execution**: Runs on localhost without cloud services

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **TensorFlow/Keras**: https://www.tensorflow.org/
- **OpenCV**: https://opencv.org/
- **Siamese Networks**: See `TECHNICAL_REPORT.md` for architecture details

## 👤 Author

Keshav - CMSI 6352 Deep Learning Assignment #2

## 📄 License

This project is for educational purposes as part of CMSI 6352.

---

**Note**: This application is designed for local use and educational purposes. For production deployment, additional security measures, authentication, and infrastructure considerations would be required.
