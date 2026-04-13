# routes/embeddings.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import sqlite3

from app.repositories.embeddings_repo import (
    deactivate_embedding,
    is_embedding_active,
    embedding_exists,
    get_deactivation_timestamp
)
from app.db import get_connection

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


class EmbeddingInfo(BaseModel):
    user_id: str
    is_active: bool
    enrolled_at: Optional[str] = None
    deactivated_at: Optional[str] = None


@router.get("/embeddings", response_model=List[EmbeddingInfo])
def list_all_embeddings():
    """
    List all enrolled users (both active and inactive)
    """
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT user_id, is_active, 
               NULL as enrolled_at,
               datetime(deactivated_at) as deactivated_at
        FROM embeddings
        ORDER BY user_id
    """)
    
    rows = cur.fetchall()
    conn.close()
    
    return [
        EmbeddingInfo(
            user_id=row[0],
            is_active=bool(row[1]),
            enrolled_at=row[2],
            deactivated_at=row[3]
        )
        for row in rows
    ]


@router.delete("/embeddings/{user_id}")
def delete_embedding_permanently(user_id: str):
    """
    Permanently delete an embedding from the database
    """
    # Check if embedding exists
    if not embedding_exists(user_id):
        raise HTTPException(
            status_code=404,
            detail=f"Embedding not found for user: {user_id}"
        )
    
    # Permanently delete from database
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("DELETE FROM embeddings WHERE user_id = ?", (user_id,))
    rows_affected = cur.rowcount
    
    conn.commit()
    conn.close()
    
    if rows_affected == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Embedding not found for user: {user_id}"
        )
    
    return {
        "status": "success",
        "message": f"Embedding for {user_id} has been permanently deleted",
        "user_id": user_id
    }

