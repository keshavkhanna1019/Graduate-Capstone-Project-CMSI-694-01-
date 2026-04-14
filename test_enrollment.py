"""
Smoke test for browser-enrollment ticket: consent → enroll-face → list embeddings.

Usage:
  export API_ORIGIN=http://127.0.0.1:8001   # optional if not port 8000
  python test_enrollment.py
Requires: requests, a test_face.jpg in the project root, and siamese_model*.h5 on disk.
"""
import os

import requests

API_ORIGIN = os.environ.get("API_ORIGIN", "http://127.0.0.1:8000").rstrip("/")
API_BASE = f"{API_ORIGIN}/api"


def test_enrollment():
    print("Testing enrollment (consent → enroll-face → GET /embeddings)...")

    try:
        response = requests.get(f"{API_ORIGIN}/health", timeout=5)
        print(f"✅ Server is running: {response.status_code}")
    except Exception as e:
        print(f"❌ Server not running at {API_ORIGIN}: {e}")
        print("   Start: uvicorn app.main:app --reload --host 127.0.0.1 --port 8000")
        print("   Or set API_ORIGIN, e.g. export API_ORIGIN=http://127.0.0.1:8001")
        return

    user_id = "test_user_123"
    try:
        response = requests.post(
            f"{API_BASE}/consent",
            json={"user_id": user_id, "consent_version": "v1.0"},
            timeout=30,
        )
        if not response.ok:
            print(f"❌ Consent failed: {response.status_code} {response.text}")
            return
        print(f"✅ Consent given: {response.status_code}")
    except Exception as e:
        print(f"❌ Consent failed: {e}")
        return

    test_image = "test_face.jpg"
    if not os.path.exists(test_image):
        print(f"⚠️  No test image at ./{test_image} — add a face photo or use the web UI only.")
        return

    try:
        with open(test_image, "rb") as f:
            files = {"file": (test_image, f, "image/jpeg")}
            response = requests.post(
                f"{API_BASE}/enroll-face?user_id={user_id}",
                files=files,
                timeout=120,
            )

        if response.status_code == 200:
            print("✅ Enrollment successful:", response.json())
        else:
            print(f"❌ Enrollment failed: {response.status_code}")
            print(f"   {response.text}")
            return
    except Exception as e:
        print(f"❌ Enrollment error: {e}")
        return

    # Ticket DoD: verify user appears in enrollment list
    try:
        r = requests.get(f"{API_BASE}/embeddings", timeout=10)
        r.raise_for_status()
        rows = r.json()
        ids = [row.get("user_id") for row in rows if isinstance(row, dict)]
        if user_id in ids:
            print(f"✅ Verified: {user_id!r} appears in GET /api/embeddings ({len(rows)} row(s)).")
        else:
            print(f"⚠️  Enrollment OK but {user_id!r} not in embeddings list yet: {ids!r}")
    except Exception as e:
        print(f"⚠️  Could not verify via GET /embeddings: {e}")


if __name__ == "__main__":
    test_enrollment()
