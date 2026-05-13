from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional
import tempfile
import os
import time

from app.repositories.embeddings_repo import save_embedding, load_active_embeddings
from app.repositories.consent_repo import has_consent
from app.repositories.recognition_logs_repo import log_recognition_event
from app.services.similarity import cosine_similarity
from app.services.face_extraction import extract_face_embedding


router = APIRouter(prefix="/api", tags=["recognition"])


SHARED_MODEL_PATH = "siamese_model.h5"


class EnrollEmbeddingRequest(BaseModel):
    user_id: str = Field(..., example="user_123")
    embedding: List[float] = Field(..., example=[0.12, -0.03, 0.44])

'''
So when someone calls your API, they must send:
user_id
embedding
This is the input for enrolling a face.
'''


class EnrollEmbeddingResponse(BaseModel):
    status: str
    user_id: str

'''
This is the box you send back after enrollment.
It says:
👉 I will reply with:
status
user_id
'''


class RecognizeEmbeddingRequest(BaseModel):
    embedding: List[float]
    top_k: int = Field(default=1, example=1)
    device_id: str = Field(..., example="kiosk_01")


'''
This is the input when someone says:
👉 “Who is this person?”
They send:
the face embedding
how many matches they want

Ques. I dont undderstand top_k or how many matches they want. 

'''


class MatchResult(BaseModel):
    user_id: str
    score: float

'''
his is one single match result.
Meaning:
👉 user X matched with confidence Y
'''


class RecognizeEmbeddingResponse(BaseModel):
    matches: List[MatchResult]
    note: Optional[str] = None
    verdict: Optional[str] = None       # "match" | "unknown" | None (cosine fallback)
    classifier: Optional[str] = None    # "svm" | "cosine"

'''
This is the final answer for recognition.
It is:
👉 a list of match results
'''


@router.post("/enroll-embedding", response_model=EnrollEmbeddingResponse)
def enroll_embedding(request: EnrollEmbeddingRequest):

    # -----------------------------
    # 1. Check user consent
    # -----------------------------
    user_has_consent = has_consent(request.user_id)

    if not user_has_consent:
        raise HTTPException(
            status_code=403,
            detail="User has not given biometric consent"
        )

    # -----------------------------
    # 2. Store embedding
    # -----------------------------
    save_embedding(user_id=request.user_id, embedding=request.embedding, extraction_model_path=None)

    return EnrollEmbeddingResponse(
        status="accepted",
        user_id=request.user_id
    )


'''
This means:
👉 someone sends a POST request to:
/api/enroll-embedding
👉 the body must match:
EnrollEmbeddingRequest
FastAPI automatically gives you:
request.user_id
request.embedding
Inside:
return EnrollEmbeddingResponse(
    status="accepted",
    user_id=request.user_id
)
So right now your API is literally doing:
👉 “Thanks, I received your data.”
👉 “Here is a confirmation.”
It does NOT save anything yet.
It does NOT do recognition.
'''


@router.post("/recognize-embedding", response_model=RecognizeEmbeddingResponse)
def recognize_embedding(request: RecognizeEmbeddingRequest):

    # -----------------------------
    # 1. Load active embeddings
    # -----------------------------
    embeddings = load_active_embeddings()

    if not embeddings:
        return RecognizeEmbeddingResponse(matches=[])

    # -----------------------------
    # 2. Compute similarity
    # -----------------------------
    similarities = []
    for user_id, stored_embedding, _stored_model in embeddings:
        score = cosine_similarity(request.embedding, stored_embedding)
        similarities.append(MatchResult(user_id=user_id, score=score))

    # Sort by score (highest first) and take top_k
    similarities.sort(key=lambda x: x.score, reverse=True)
    matches = similarities[:request.top_k]

    # -----------------------------
    # 3. Log recognition event (non-blocking)
    # -----------------------------
    # Extract match info for logging
    matched_user_id = matches[0].user_id if matches else None
    similarity_score = matches[0].score if matches else None

    # Log the recognition event (this is fast and won't block the response)
    try:
        log_recognition_event(
            device_id=request.device_id,
            top_k=request.top_k,
            matched_user_id=matched_user_id,
            similarity_score=similarity_score
        )
    except Exception as e:
        # Log errors silently to avoid affecting API response
        # In production, you might want to use proper logging here
        pass

    return RecognizeEmbeddingResponse(
        matches=matches
    )


@router.post("/extract-face-embedding")
async def extract_embedding_from_image(file: UploadFile = File(...)):
    """
    Extract face embedding from uploaded image using ML model.
    This is the ML-powered endpoint that actually processes images.
    """
    try:
        # Read file content
        content = await file.read()
        
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Save uploaded file temporarily with proper extension
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            file_ext = '.jpg'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp.flush()  # Ensure data is written to disk
            tmp_path = tmp.name
        
        # Verify file exists and has content
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to save uploaded file"
            )
        
        try:
            # Extract embedding using ML model (with timing)
            start_time = time.time()
            embedding = extract_face_embedding(image_bytes=content)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "embedding": embedding,
                "dimension": len(embedding),
                "model": "Siamese Network",
                "inference_time_ms": round(inference_time, 2)
            }
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # Ignore cleanup errors
                
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/enroll-face")
async def enroll_face(
    user_id: str,
    file: UploadFile = File(...)
):
    """
    Enroll a face from an uploaded image.
    This combines ML face extraction and enrollment in one step.
    """
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id is required")

    user_has_consent = has_consent(uid)

    if not user_has_consent:
        raise HTTPException(
            status_code=403,
            detail="Enrollment blocked: no biometric consent on file for this User ID. Grant consent first (Enroll tab does this automatically, or use POST /api/consent).",
        )
    
    try:
        # Read file content
        content = await file.read()
        
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Save uploaded file temporarily with proper extension
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            file_ext = '.jpg'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp.flush()  # Ensure data is written to disk
            tmp_path = tmp.name
        
        # Verify file exists and has content
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to save uploaded file"
            )
        
        try:
            import numpy as np

            # Extract one embedding from the uploaded photo (ArcFace, no .h5 needed)
            raw_emb = extract_face_embedding(image_bytes=content)

            # If this user already has stored positive training images, blend them in
            # so the stored embedding is an average of multiple views → more robust
            extra_embeddings = []
            user_pos_dir = os.path.join("data", uid, "positive")
            if os.path.isdir(user_pos_dir):
                img_files = sorted([
                    os.path.join(user_pos_dir, f)
                    for f in os.listdir(user_pos_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])[:8]  # use up to 8 training photos
                for img_path in img_files:
                    try:
                        extra_embeddings.append(
                            extract_face_embedding(image_path=img_path, )
                        )
                    except Exception:
                        pass

            if extra_embeddings:
                all_embs = np.array([raw_emb] + extra_embeddings)
                avg_emb = np.mean(all_embs, axis=0)
                avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
                final_embedding = avg_emb.tolist()
            else:
                final_embedding = raw_emb

            save_embedding(
                user_id=uid,
                embedding=final_embedding,
                extraction_model_path=None,
            )

            return EnrollEmbeddingResponse(
                status="accepted",
                user_id=uid
            )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass  # Ignore cleanup errors
                
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@router.post("/recognize-face")
async def recognize_face(
    file: UploadFile = File(...),
    top_k: int = 1,
    device_id: str = "default"
):
    """
    Recognize a face from an uploaded image.
    This combines ML face extraction and recognition in one step.
    """
    try:
        # Read file content
        content = await file.read()
        
        if not content or len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Save uploaded file temporarily with proper extension
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.jpg'
        if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            file_ext = '.jpg'
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp.flush()  # Ensure data is written to disk
            tmp_path = tmp.name
        
        # Verify file exists and has content
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to save uploaded file"
            )
        
        try:
            start_time = time.time()
            embedding = extract_face_embedding(image_bytes=content)
            extraction_time = (time.time() - start_time) * 1000  # noqa: F841

            stored_embeddings = load_active_embeddings()
            if not stored_embeddings:
                return RecognizeEmbeddingResponse(matches=[], verdict=None, classifier=None)

            # ── Try SVM classifier first ───────────────────────────────────────
            from app.services.svm_classifier import get_svm_classifier
            svm = get_svm_classifier()

            note = None
            verdict = None
            classifier = None

            if svm.is_trained:
                try:
                    all_probs, best_user, best_prob, is_match = svm.predict(embedding)
                    classifier = "svm"
                    verdict = "match" if is_match else "unknown"

                    if verdict == "unknown":
                        # Return unknown result — face doesn't confidently match anyone
                        matches = [MatchResult(user_id="Unknown", score=round(best_prob, 4))]
                        note = (
                            f"No confident match found (best score {best_prob*100:.1f}% < 45% threshold). "
                            "This face is not enrolled or the enrollment photo is very different from the test photo."
                        )
                    else:
                        # Return SVM probabilities for all users, ranked
                        matches = [
                            MatchResult(user_id=uid, score=round(prob, 4))
                            for uid, prob in all_probs[:top_k]
                        ]
                except Exception as svm_err:
                    # SVM prediction failed — fall through to cosine
                    print(f"SVM prediction error, falling back to cosine: {svm_err}")
                    svm = None

            if not svm or not svm.is_trained:
                # ── Fallback: cosine similarity ────────────────────────────────
                classifier = "cosine"
                similarities = []
                for stored_user_id, stored_embedding, _ in stored_embeddings:
                    score = cosine_similarity(embedding, stored_embedding)
                    print(f"🔍 {stored_user_id}: cosine={score:.6f}")
                    similarities.append(MatchResult(user_id=stored_user_id, score=score))

                similarities.sort(key=lambda x: x.score, reverse=True)
                matches = similarities[:top_k]

                if len(similarities) >= 2:
                    s0, s1 = similarities[0].score, similarities[1].score
                    if s0 >= 0.995 and s1 >= 0.995:
                        note = (
                            "Two or more users scored almost identically — likely embedding collapse. "
                            "Re-enroll all users, or train the SVM for sharper separation."
                        )
                    elif (s0 - s1) < 0.005 and s0 > 0.85:
                        note = (
                            "Top matches are very close. Train the SVM classifier for better separation."
                        )

            matched_user_id = matches[0].user_id if matches else None
            similarity_score = matches[0].score if matches else None

            try:
                log_recognition_event(
                    device_id=device_id,
                    top_k=top_k,
                    matched_user_id=matched_user_id,
                    similarity_score=similarity_score,
                )
            except Exception:
                pass

            return RecognizeEmbeddingResponse(
                matches=matches,
                note=note,
                verdict=verdict,
                classifier=classifier,
            )
            
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


'''
👶 recognize endpoint
@router.post("/recognize-embedding")
def recognize_embedding(request: RecognizeEmbeddingRequest):
This means:
👉 someone sends a POST request to:
/api/recognize-embedding
with:
embedding
top_k
And your server answers:
return RecognizeEmbeddingResponse(matches=[])
Which means:
👉 "I have no matches yet."

'''
