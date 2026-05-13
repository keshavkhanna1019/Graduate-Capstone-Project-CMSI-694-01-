from sqlalchemy import Column, Integer, Text, Float
from app.db import Base


class Embedding(Base):
    __tablename__ = "embeddings"

    user_id = Column(Text, primary_key=True)
    embedding = Column(Text, nullable=False)
    is_active = Column(Integer, default=1)
    deactivated_at = Column(Text, nullable=True)
    extraction_model_path = Column(Text, nullable=True)


class Consent(Base):
    __tablename__ = "consent"

    user_id = Column(Text, primary_key=True)
    consent_version = Column(Text, nullable=False)
    timestamp = Column(Text, nullable=False)


class Device(Base):
    __tablename__ = "devices"

    device_id = Column(Text, primary_key=True)
    store_id = Column(Text, nullable=False)
    status = Column(Text, default="active")
    created_at = Column(Text)


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Text, nullable=False)
    device_id = Column(Text, nullable=False)
    top_k = Column(Integer, nullable=False)
    matched_user_id = Column(Text, nullable=True)
    similarity_score = Column(Float, nullable=True)
    is_failed = Column(Integer, default=0)  # 1 if no match or score < threshold
