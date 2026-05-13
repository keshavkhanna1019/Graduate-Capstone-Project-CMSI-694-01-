from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.db import ensure_sqlite_schema

from routes.health import router as health_router
from routes.recognition import router as recognition_router
from routes.consent import router as consent_router
from routes.devices import router as devices_router
from routes.embeddings import router as embeddings_router
from routes.training import router as training_router
from routes.data_collection import router as data_collection_router
from routes.recognition_logs import router as recognition_logs_router

ensure_sqlite_schema()

app = FastAPI(title="Face Recognition Backend", description="Deep Learning Assignment #2 - Face Recognition System")

# Add CORS middleware to allow frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # Must be False when allow_origins is "*"; True + "*" is invalid CORS and breaks browsers on cross-origin requests.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(health_router)
app.include_router(recognition_router)
app.include_router(consent_router)
app.include_router(devices_router)
app.include_router(embeddings_router)
app.include_router(training_router)
app.include_router(data_collection_router)
app.include_router(recognition_logs_router)


@app.get("/")
async def root():
    """Root endpoint - serves the frontend"""
    from pathlib import Path
    html_path = Path("static/index.html")
    if not html_path.exists():
        return {"error": "Frontend not found. Make sure static/index.html exists."}
    return FileResponse(html_path)

