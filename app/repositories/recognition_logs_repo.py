from datetime import datetime, timezone

from app.db import SessionLocal
from app.models import RecognitionLog

FAILURE_THRESHOLD = 0.5  # Scores below this are considered failed recognition


def log_recognition_event(
    device_id: str,
    top_k: int,
    matched_user_id: str = None,
    similarity_score: float = None
):
    is_failed = int(
        matched_user_id is None or
        similarity_score is None or
        similarity_score < FAILURE_THRESHOLD
    )

    db = SessionLocal()
    try:
        db.add(RecognitionLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            device_id=device_id,
            top_k=top_k,
            matched_user_id=matched_user_id,
            similarity_score=similarity_score,
            is_failed=is_failed
        ))
        db.commit()
    finally:
        db.close()


def get_recognition_logs(
    device_id: str = None,
    start_time: str = None,
    end_time: str = None,
    failed_only: bool = False,
    limit: int = 100
):
    db = SessionLocal()
    try:
        query = db.query(RecognitionLog)

        if device_id:
            query = query.filter(RecognitionLog.device_id == device_id)
        if start_time:
            query = query.filter(RecognitionLog.timestamp >= start_time)
        if end_time:
            query = query.filter(RecognitionLog.timestamp <= end_time)
        if failed_only:
            query = query.filter(RecognitionLog.is_failed == 1)

        rows = query.order_by(RecognitionLog.timestamp.desc()).limit(limit).all()

        return [
            {
                "id": row.id,
                "timestamp": row.timestamp,
                "device_id": row.device_id,
                "top_k": row.top_k,
                "matched_user_id": row.matched_user_id,
                "similarity_score": row.similarity_score,
                "is_failed": bool(row.is_failed)
            }
            for row in rows
        ]
    finally:
        db.close()
