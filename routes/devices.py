# routes/devices.py

from fastapi import APIRouter
from pydantic import BaseModel, Field
from app.repositories.device_repo import create_device


router = APIRouter(prefix="/api", tags=["devices"])


class RegisterDeviceRequest(BaseModel):
    device_id: str = Field(..., example="kiosk_01")
    store_id: str = Field(..., example="store_100")


class RegisterDeviceResponse(BaseModel):
    device_id: str
    store_id: str
    status: str


@router.post("/devices", response_model=RegisterDeviceResponse)
def register_device(request: RegisterDeviceRequest):

    create_device(
        device_id=request.device_id,
        store_id=request.store_id
    )

    return RegisterDeviceResponse(
        device_id=request.device_id,
        store_id=request.store_id,
        status="active"
    )