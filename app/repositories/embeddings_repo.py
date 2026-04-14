import json
from datetime import datetime, timezone

import numpy as np

from app.db import SessionLocal
from app.models import Embedding


def save_embedding(user_id: str, embedding: list[float]):
    db = SessionLocal()
    try:
        record = db.query(Embedding).filter(Embedding.user_id == user_id).first()
        if record:
            record.embedding = json.dumps(embedding)
            record.is_active = 1
            record.deactivated_at = None
        else:
            db.add(Embedding(
                user_id=user_id,
                embedding=json.dumps(embedding),
                is_active=1,
                deactivated_at=None
            ))
        db.commit()
    finally:
        db.close()


def load_active_embeddings():
    db = SessionLocal()
    try:
        rows = db.query(Embedding).filter(Embedding.is_active == 1).all()
        results = []
        for row in rows:
            embedding = json.loads(row.embedding)
            embedding_array = np.array(embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            else:
                embedding_array = np.zeros_like(embedding_array)
            results.append((row.user_id, embedding_array.tolist()))
        return results
    finally:
        db.close()


def deactivate_embedding(user_id: str) -> bool:
    db = SessionLocal()
    try:
        record = db.query(Embedding).filter(Embedding.user_id == user_id).first()
        if record is None:
            return False
        record.is_active = 0
        record.deactivated_at = datetime.now(timezone.utc).isoformat()
        db.commit()
        return True
    finally:
        db.close()


def reactivate_embedding(user_id: str):
    db = SessionLocal()
    try:
        record = db.query(Embedding).filter(Embedding.user_id == user_id).first()
        if record:
            record.is_active = 1
            record.deactivated_at = None
            db.commit()
    finally:
        db.close()


def get_deactivation_timestamp(user_id: str):
    db = SessionLocal()
    try:
        record = db.query(Embedding).filter(Embedding.user_id == user_id).first()
        if record is None:
            return None
        return record.deactivated_at
    finally:
        db.close()


def embedding_exists(user_id: str) -> bool:
    db = SessionLocal()
    try:
        return db.query(Embedding).filter(Embedding.user_id == user_id).first() is not None
    finally:
        db.close()


def is_embedding_active(user_id: str) -> bool:
    db = SessionLocal()
    try:
        record = db.query(Embedding).filter(Embedding.user_id == user_id).first()
        if record is None:
            return False
        return record.is_active == 1
    finally:
        db.close()
