import json
from app.db import get_connection


# ➜ Create or update an embedding (always activates it)
def save_embedding(user_id: str, embedding: list[float]):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO embeddings (user_id, embedding, is_active, deactivated_at)
        VALUES (?, ?, 1, NULL)
    """, (user_id, json.dumps(embedding)))

    conn.commit()
    conn.close()


# ➜ Load only active embeddings (used by recognition service)
def load_active_embeddings():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT user_id, embedding
        FROM embeddings
        WHERE is_active = 1
    """)

    rows = cur.fetchall()
    conn.close()

    results = []

    for user_id, emb_json in rows:
        results.append(
            (user_id, json.loads(emb_json))
        )

    return results


# ➜ Soft delete (revoke biometric usage)
def deactivate_embedding(user_id: str) -> bool:
    """
    Deactivate an embedding. Returns True if a row was updated, False otherwise.
    """
    conn = get_connection()
    cur = conn.cursor()

    # Store the deactivation timestamp
    cur.execute("""
        UPDATE embeddings
        SET is_active = 0, deactivated_at = datetime('now')
        WHERE user_id = ?
    """, (user_id,))

    rows_affected = cur.rowcount
    conn.commit()
    conn.close()

    return rows_affected > 0


# ➜ Reactivate embedding (user gives consent again)
def reactivate_embedding(user_id: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        UPDATE embeddings
        SET is_active = 1, deactivated_at = NULL
        WHERE user_id = ?
    """, (user_id,))

    conn.commit()
    conn.close()


# ➜ Get deactivation timestamp for an embedding
def get_deactivation_timestamp(user_id: str):
    """Get the deactivation timestamp for a user's embedding"""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT deactivated_at
        FROM embeddings
        WHERE user_id = ?
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if row is None or row[0] is None:
        return None

    return row[0]


# ➜ Check if embedding exists (regardless of active status)
def embedding_exists(user_id: str) -> bool:
    """Check if an embedding exists for a user, regardless of active status"""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT 1
        FROM embeddings
        WHERE user_id = ?
    """, (user_id,))

    result = cur.fetchone()
    conn.close()

    return result is not None


# ➜ Optional: check if embedding exists & active
def is_embedding_active(user_id: str) -> bool:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT is_active
        FROM embeddings
        WHERE user_id = ?
    """, (user_id,))

    row = cur.fetchone()
    conn.close()

    if row is None:
        return False

    return row[0] == 1
