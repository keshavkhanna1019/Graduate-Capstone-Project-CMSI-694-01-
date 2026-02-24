from app.db import get_connection


def create_consent(user_id: str, consent_version: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO consent (user_id, consent_version, timestamp)
        VALUES (?, ?, datetime('now'))
    """, (user_id, consent_version))

    conn.commit()
    conn.close()


def has_consent(user_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT 1 FROM consent WHERE user_id = ?",
        (user_id,)
    )

    result = cur.fetchone()
    conn.close()

    return result is not None


def get_consent_details(user_id: str):
    """Get consent details including timestamp for a user"""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT user_id, consent_version, timestamp FROM consent WHERE user_id = ?",
        (user_id,)
    )

    result = cur.fetchone()
    conn.close()

    if result is None:
        return None

    return {
        "user_id": result[0],
        "consent_version": result[1],
        "timestamp": result[2]
    }
