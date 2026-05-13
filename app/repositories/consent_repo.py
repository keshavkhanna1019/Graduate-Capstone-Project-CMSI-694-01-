from datetime import datetime, timezone

from app.db import SessionLocal
from app.models import Consent


def create_consent(user_id: str, consent_version: str):
    db = SessionLocal()
    try:
        record = db.query(Consent).filter(Consent.user_id == user_id).first()
        if record:
            record.consent_version = consent_version
            record.timestamp = datetime.now(timezone.utc).isoformat()
        else:
            db.add(Consent(
                user_id=user_id,
                consent_version=consent_version,
                timestamp=datetime.now(timezone.utc).isoformat()
            ))
        db.commit()
    finally:
        db.close()


def has_consent(user_id: str) -> bool:
    db = SessionLocal()
    try:
        return db.query(Consent).filter(Consent.user_id == user_id).first() is not None
    finally:
        db.close()


def get_consent_details(user_id: str):
    db = SessionLocal()
    try:
        record = db.query(Consent).filter(Consent.user_id == user_id).first()
        if record is None:
            return None
        return {
            "user_id": record.user_id,
            "consent_version": record.consent_version,
            "timestamp": record.timestamp
        }
    finally:
        db.close()


def list_all_consents():
    db = SessionLocal()
    try:
        records = db.query(Consent).order_by(Consent.timestamp.desc()).all()
        return [
            {"user_id": r.user_id, "consent_version": r.consent_version, "timestamp": r.timestamp}
            for r in records
        ]
    finally:
        db.close()


def delete_consent(user_id: str) -> bool:
    db = SessionLocal()
    try:
        record = db.query(Consent).filter(Consent.user_id == user_id).first()
        if record is None:
            return False
        db.delete(record)
        db.commit()
        return True
    finally:
        db.close()
