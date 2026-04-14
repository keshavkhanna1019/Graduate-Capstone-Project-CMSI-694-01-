from datetime import datetime, timezone

from app.db import SessionLocal
from app.models import Device


def create_device(device_id: str, store_id: str):
    db = SessionLocal()
    try:
        record = db.query(Device).filter(Device.device_id == device_id).first()
        if record:
            record.store_id = store_id
            record.status = "active"
        else:
            db.add(Device(
                device_id=device_id,
                store_id=store_id,
                status="active",
                created_at=datetime.now(timezone.utc).isoformat()
            ))
        db.commit()
    finally:
        db.close()
