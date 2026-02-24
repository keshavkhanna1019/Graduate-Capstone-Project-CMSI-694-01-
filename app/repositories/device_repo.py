from app.db import get_connection

def create_device(device_id: str, store_id: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO devices (device_id, store_id, status)
        VALUES (?, ?, ?)
    """, (device_id, store_id, "active"))

    conn.commit()
    conn.close()
