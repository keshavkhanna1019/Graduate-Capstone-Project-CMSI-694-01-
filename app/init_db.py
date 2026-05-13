from sqlalchemy import text

from app.db import engine, Base

# Import all models so SQLAlchemy registers them before create_all
import app.models  # noqa: F401

Base.metadata.create_all(bind=engine)

# Migrate existing tables: add new columns if they don't exist
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE recognition_logs ADD COLUMN is_failed INTEGER DEFAULT 0"))
        conn.commit()
    except Exception:
        pass  # Column already exists
