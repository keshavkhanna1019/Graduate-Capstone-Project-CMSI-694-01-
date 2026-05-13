"""
Smoke check for core API paths.

Covers: health, consent (grant / check / list / delete),
        embeddings list, enroll-face, recognize-face.

Enroll + recognize require a trained model (siamese_model*.h5) and a test
image. If either is absent the test is skipped with a clear message — this is
intentional so the script passes in a freshly-cloned repo.

Usage
-----
  # server must already be running:
  uvicorn app.main:app --host 0.0.0.0 --port 8000

  python smoke_test.py                         # default: port 8000
  API_ORIGIN=http://127.0.0.1:8001 python smoke_test.py
  python smoke_test.py --image path/to/face.jpg

Exit code: 0 if all attempted checks pass, 1 if any fail.
"""

import argparse
import glob
import os
import sys

try:
    import requests
except ImportError:
    sys.exit("requests not installed. Run: pip install requests")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Smoke check for core API paths")
parser.add_argument("--image", default="test_face.jpg",
                    help="Face image for enroll/recognize tests (default: test_face.jpg)")
args = parser.parse_args()

API_ORIGIN = os.environ.get("API_ORIGIN", "http://127.0.0.1:8000").rstrip("/")
API_BASE   = f"{API_ORIGIN}/api"
TEST_USER  = "smoke_test_user"
TEST_IMAGE = args.image

PASSED, FAILED, SKIPPED = [], [], []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def ok(name, detail=""):
    msg = f"  PASS  {name}" + (f" — {detail}" if detail else "")
    print(msg)
    PASSED.append(name)


def fail(name, detail=""):
    msg = f"  FAIL  {name}" + (f" — {detail}" if detail else "")
    print(msg)
    FAILED.append(name)


def skip(name, reason):
    print(f"  SKIP  {name} — {reason}")
    SKIPPED.append(name)


def get(path, **kw):
    return requests.get(f"{API_BASE}{path}", timeout=10, **kw)


def post(path, **kw):
    return requests.post(f"{API_BASE}{path}", timeout=120, **kw)


def delete(path, **kw):
    return requests.delete(f"{API_BASE}{path}", timeout=10, **kw)


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
print("\n── Health ──────────────────────────────────────────────")
try:
    r = requests.get(f"{API_ORIGIN}/health", timeout=5)
    if r.ok:
        ok("GET /health", f"status={r.status_code}")
    else:
        fail("GET /health", f"status={r.status_code}")
except Exception as e:
    fail("GET /health", str(e))
    print(f"\n  Server unreachable at {API_ORIGIN}.")
    print("  Start it with:  uvicorn app.main:app --host 0.0.0.0 --port 8000\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. Consent
# ---------------------------------------------------------------------------
print("\n── Consent ─────────────────────────────────────────────")

# Grant
try:
    r = post("/consent", json={"user_id": TEST_USER, "consent_version": "v1.0"})
    if r.ok:
        ok("POST /consent (grant)")
    else:
        fail("POST /consent (grant)", f"{r.status_code} {r.text[:120]}")
except Exception as e:
    fail("POST /consent (grant)", str(e))

# Check
try:
    r = get(f"/consent/{TEST_USER}")
    if r.ok and r.json().get("consent_given"):
        ok("GET /consent/{user_id} (check)")
    else:
        fail("GET /consent/{user_id} (check)", f"{r.status_code} {r.text[:120]}")
except Exception as e:
    fail("GET /consent/{user_id} (check)", str(e))

# List all
try:
    r = get("/consents")
    if r.ok and isinstance(r.json(), list):
        ok("GET /consents (list all)", f"{len(r.json())} record(s)")
    else:
        fail("GET /consents (list all)", f"{r.status_code} {r.text[:120]}")
except Exception as e:
    fail("GET /consents (list all)", str(e))

# ---------------------------------------------------------------------------
# 3. Embeddings list
# ---------------------------------------------------------------------------
print("\n── Embeddings ──────────────────────────────────────────")
try:
    r = get("/embeddings")
    if r.ok and isinstance(r.json(), list):
        ok("GET /embeddings", f"{len(r.json())} enrolled user(s)")
    else:
        fail("GET /embeddings", f"{r.status_code} {r.text[:120]}")
except Exception as e:
    fail("GET /embeddings", str(e))

# ---------------------------------------------------------------------------
# 4. Enroll + Recognize  (skipped if model or image is missing)
# ---------------------------------------------------------------------------
print("\n── Enroll / Recognize ──────────────────────────────────")

model_present = bool(glob.glob("siamese_model*.h5"))
image_present = os.path.isfile(TEST_IMAGE)

if not model_present:
    skip("POST /enroll-face",    "no siamese_model*.h5 found — train a model first")
    skip("POST /recognize-face", "no siamese_model*.h5 found — train a model first")
elif not image_present:
    skip("POST /enroll-face",    f"test image not found: {TEST_IMAGE}")
    skip("POST /recognize-face", f"test image not found: {TEST_IMAGE}")
else:
    # Enroll
    enroll_ok = False
    try:
        with open(TEST_IMAGE, "rb") as f:
            r = post(f"/enroll-face?user_id={TEST_USER}", files={"file": (TEST_IMAGE, f, "image/jpeg")})
        if r.ok:
            ok("POST /enroll-face", r.json().get("status", ""))
            enroll_ok = True
        else:
            fail("POST /enroll-face", f"{r.status_code} {r.text[:200]}")
    except Exception as e:
        fail("POST /enroll-face", str(e))

    # Recognize (only if enroll succeeded)
    if enroll_ok:
        try:
            with open(TEST_IMAGE, "rb") as f:
                r = post("/recognize-face?top_k=1&device_id=smoke_test",
                         files={"file": (TEST_IMAGE, f, "image/jpeg")})
            if r.ok:
                matches = r.json().get("matches", [])
                top = matches[0] if matches else {}
                ok("POST /recognize-face",
                   f"top match={top.get('user_id','?')} score={top.get('score',0):.3f}")
            else:
                fail("POST /recognize-face", f"{r.status_code} {r.text[:200]}")
        except Exception as e:
            fail("POST /recognize-face", str(e))
    else:
        skip("POST /recognize-face", "enroll step failed")

# ---------------------------------------------------------------------------
# 5. Consent cleanup  (delete the test user's consent)
# ---------------------------------------------------------------------------
print("\n── Cleanup ─────────────────────────────────────────────")
try:
    r = delete(f"/consent/{TEST_USER}")
    if r.ok:
        ok(f"DELETE /consent/{TEST_USER}")
    else:
        fail(f"DELETE /consent/{TEST_USER}", f"{r.status_code} {r.text[:120]}")
except Exception as e:
    fail(f"DELETE /consent/{TEST_USER}", str(e))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = len(PASSED) + len(FAILED) + len(SKIPPED)
print(f"\n{'─'*55}")
print(f"  Results: {len(PASSED)} passed  {len(FAILED)} failed  {len(SKIPPED)} skipped  ({total} total)")
if FAILED:
    print(f"  Failed:  {', '.join(FAILED)}")
print(f"{'─'*55}\n")

sys.exit(1 if FAILED else 0)
