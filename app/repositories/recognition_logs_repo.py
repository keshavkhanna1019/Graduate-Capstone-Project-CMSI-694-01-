from app.db import get_connection


def log_recognition_event(
    device_id: str,
    top_k: int,
    matched_user_id: str = None,
    similarity_score: float = None
):
    """
    Log a recognition event to the database.
    This function is designed to be fast and non-blocking.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO recognition_logs (device_id, top_k, matched_user_id, similarity_score)
        VALUES (?, ?, ?, ?)
    """, (device_id, top_k, matched_user_id, similarity_score))

    conn.commit()
    conn.close()


def get_recognition_logs(device_id: str = None, limit: int = 100):
    """
    Query recognition logs. Optionally filter by device_id.
    """
    conn = get_connection()
    cur = conn.cursor()

    if device_id:
        cur.execute("""
            SELECT timestamp, device_id, top_k, matched_user_id, similarity_score
            FROM recognition_logs
            WHERE device_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (device_id, limit))
    else:
        cur.execute("""
            SELECT timestamp, device_id, top_k, matched_user_id, similarity_score
            FROM recognition_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))

    rows = cur.fetchall()
    conn.close()

    return [
        {
            "timestamp": row[0],
            "device_id": row[1],
            "top_k": row[2],
            "matched_user_id": row[3],
            "similarity_score": row[4]
        }
        for row in rows
    ]
