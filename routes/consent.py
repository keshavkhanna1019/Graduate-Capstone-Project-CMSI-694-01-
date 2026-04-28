# routes/consent.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from app.repositories.consent_repo import create_consent, has_consent, get_consent_details, list_all_consents, delete_consent




router = APIRouter(prefix="/api", tags=["consent"])


class ConsentRequest(BaseModel):
    user_id: str = Field(..., example="user_123")
    consent_version: str = Field(..., example="v1.0")


class ConsentResponse(BaseModel):
    user_id: str
    consent_given: bool
    timestamp: datetime


@router.post("/consent", response_model=ConsentResponse)
def give_consent(request: ConsentRequest):

    # Store in DB
    create_consent(
        user_id=request.user_id,
        consent_version=request.consent_version
    )

    # Get the actual timestamp from database
    consent_details = get_consent_details(request.user_id)
    try:
        db_timestamp = datetime.fromisoformat(consent_details["timestamp"].replace(" ", "T"))
    except:
        db_timestamp = datetime.utcnow()

    return ConsentResponse(
        user_id=request.user_id,
        consent_given=True,
        timestamp=db_timestamp
    )


@router.get("/consents")
def get_all_consents():
    return list_all_consents()


@router.delete("/consent/{user_id}")
def remove_consent(user_id: str):
    deleted = delete_consent(user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"No consent record found for '{user_id}'")
    return {"status": "deleted", "user_id": user_id}


@router.get("/consent/{user_id}", response_model=ConsentResponse)
def get_consent(user_id: str):
    consent_details = get_consent_details(user_id)

    if consent_details is None:
        return ConsentResponse(
            user_id=user_id,
            consent_given=False,
            timestamp=datetime.utcnow()
        )

    # Parse the timestamp from database (SQLite datetime format)
    try:
        # SQLite stores timestamp as string, parse it
        db_timestamp = datetime.fromisoformat(consent_details["timestamp"].replace(" ", "T"))
    except:
        # Fallback to current time if parsing fails
        db_timestamp = datetime.utcnow()

    return ConsentResponse(
        user_id=user_id,
        consent_given=True,
        timestamp=db_timestamp
    )

