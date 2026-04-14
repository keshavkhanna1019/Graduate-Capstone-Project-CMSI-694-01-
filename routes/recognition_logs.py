from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.repositories.recognition_logs_repo import get_recognition_logs

router = APIRouter(prefix="/api", tags=["recognition-logs"])


class RecognitionLogEntry(BaseModel):
    id: int
    timestamp: str
    device_id: str
    top_k: int
    matched_user_id: Optional[str]
    similarity_score: Optional[float]
    is_failed: bool


@router.get("/recognition-logs", response_model=List[RecognitionLogEntry])
def list_recognition_logs(
    device_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    failed_only: bool = False,
    limit: int = 100
):
    """
    Query recognition logs.

    - **device_id**: filter by device (optional)
    - **start_time**: ISO 8601 datetime string, inclusive lower bound (optional)
    - **end_time**: ISO 8601 datetime string, inclusive upper bound (optional)
    - **failed_only**: if true, return only failed recognition attempts (optional)
    - **limit**: max number of results (default 100)
    """
    return get_recognition_logs(
        device_id=device_id,
        start_time=start_time,
        end_time=end_time,
        failed_only=failed_only,
        limit=limit
    )
