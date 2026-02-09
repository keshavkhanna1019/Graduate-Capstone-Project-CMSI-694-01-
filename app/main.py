from fastapi import FastAPI
from routes.health import router as health_router

app = FastAPI(title="Face Recognition Backend")

app.include_router(health_router)


@app.get("/")
def root():
    return {"message": "API is running"}

# GCPC60-6 FastAPI skeleton
