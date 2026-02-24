from app.db import get_connection
import sqlite3

conn = get_connection()
cur = conn.cursor()

# Migrate existing databases: add deactivated_at column if it doesn't exist
try:
    cur.execute("ALTER TABLE embeddings ADD COLUMN deactivated_at TEXT")
except sqlite3.OperationalError:
    # Column already exists or table doesn't exist yet - that's fine
    pass

cur.execute("""
CREATE TABLE IF NOT EXISTS embeddings (
    user_id TEXT PRIMARY KEY,
    embedding TEXT,
    is_active INTEGER,
    deactivated_at TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS consent (
    user_id TEXT PRIMARY KEY,
    consent_version TEXT,
    timestamp TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    store_id TEXT,
    status TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS recognition_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    device_id TEXT,
    top_k INTEGER,
    matched_user_id TEXT,
    similarity_score REAL
)
""")


conn.commit()
conn.close()
