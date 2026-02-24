"""
Database migration script to add new columns and tables.
Run this if you have an existing database that needs updating.
"""
from app.db import get_connection

def migrate_database():
    conn = get_connection()
    cur = conn.cursor()

    # Add deactivated_at column to embeddings table if it doesn't exist
    try:
        cur.execute("ALTER TABLE embeddings ADD COLUMN deactivated_at TEXT")
        print("✓ Added deactivated_at column to embeddings table")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("✓ deactivated_at column already exists in embeddings table")
        else:
            print(f"⚠ Warning: {e}")

    # Create recognition_logs table if it doesn't exist
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
    print("✓ Created recognition_logs table")

    conn.commit()
    conn.close()
    print("✓ Database migration complete!")

if __name__ == "__main__":
    import sqlite3
    migrate_database()
