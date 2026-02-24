# routes/embeddings.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

from app.repositories.embeddings_repo import (
    deactivate_embedding,
    is_embedding_active,
    embedding_exists,
    get_deactivation_timestamp
)

router = APIRouter(prefix="/api", tags=["embeddings"])


class DeactivateEmbeddingResponse(BaseModel):
    user_id: str
    is_active: bool
    deleted_at: datetime


@router.post(
    "/embeddings/{user_id}/deactivate",
    response_model=DeactivateEmbeddingResponse
)
def deactivate_embedding_route(user_id: str):
    # Check if embedding exists at all
    if not embedding_exists(user_id):
        raise HTTPException(
            status_code=404,
            detail="Embedding not found for this user"
        )

    # Check if already deactivated - return not found
    if not is_embedding_active(user_id):
        raise HTTPException(
            status_code=404,
            detail="Embedding not found for this user (already deactivated)"
        )

    # Deactivate the embedding
    success = deactivate_embedding(user_id)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to deactivate embedding"
        )

    # Get the actual deactivation timestamp from database
    deactivated_at = get_deactivation_timestamp(user_id)
    try:
        db_timestamp = datetime.fromisoformat(deactivated_at.replace(" ", "T"))
    except:
        db_timestamp = datetime.utcnow()

    return DeactivateEmbeddingResponse(
        user_id=user_id,
        is_active=False,
        deleted_at=db_timestamp
    )

