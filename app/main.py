from fastapi import FastAPI

from routes.health import router as health_router
from routes.recognition import router as recognition_router
from routes.consent import router as consent_router
from routes.devices import router as devices_router
from routes.embeddings import router as embeddings_router

app = FastAPI(title="Face Recognition Backend")

app.include_router(health_router)
app.include_router(recognition_router)
app.include_router(consent_router)
app.include_router(devices_router)
app.include_router(embeddings_router)


@app.get("/")
def root():
    return {"message": "API is running"}

