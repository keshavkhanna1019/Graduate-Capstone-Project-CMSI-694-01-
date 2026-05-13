from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import sqlite3

DATABASE_URL = "sqlite:///./face.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

SQLITE_DB_PATH = "face.db"


def ensure_sqlite_schema() -> None:
    """Add columns missing on older face.db files."""
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(embeddings)").fetchall()}
        if "extraction_model_path" not in cols:
            conn.execute(
                "ALTER TABLE embeddings ADD COLUMN extraction_model_path TEXT"
            )
            conn.commit()
    finally:
        conn.close()


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
