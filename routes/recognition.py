from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List
import tempfile
import os
import time

from app.repositories.embeddings_repo import save_embedding, load_active_embeddings
from app.repositories.consent_repo import has_consent
from app.repositories.recognition_logs_repo import log_recognition_event
from app.services.similarity import cosine_similarity
from app.services.face_extraction import extract_face_embedding


router = APIRouter(prefix="/api", tags=["recognition"])


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
    save_embedding(
            user_id=request.user_id,
            embedding=request.embedding
    )

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
    for user_id, stored_embedding in embeddings:
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
    # Check consent first
    user_has_consent = has_consent(user_id)
    
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
            # Extract embedding using user-specific model
            # Model path format: siamese_model_{user_id}.h5
            user_model_path = f"siamese_model_{user_id}.h5"
            
            # Fallback to default model if user-specific model doesn't exist
            if not os.path.exists(user_model_path):
                if os.path.exists("siamese_model.h5"):
                    user_model_path = "siamese_model.h5"
                elif os.path.exists("siamese_model_v2.h5"):
                    user_model_path = "siamese_model_v2.h5"
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No trained model found for {user_id}. Please train a model first:\n"
                               f"1. Collect data for {user_id} (positive: your images, negative: others)\n"
                               f"2. Train model for {user_id} using 'Train Model' tab\n"
                               f"3. Then enroll your face"
                    )
            
            start_time = time.time()
            
            # Use bytes directly (more reliable than temp file path)
            embedding = extract_face_embedding(image_bytes=content, model_path=user_model_path)
            
            extraction_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Store embedding
            save_embedding(
                user_id=user_id,
                embedding=embedding
            )
            
            return EnrollEmbeddingResponse(
                status="accepted",
                user_id=user_id
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
            # Extract embedding for recognition
            # We'll compare against each enrolled user using their specific model
            start_time = time.time()
            
            # Try to find any available model (we'll use user-specific models during comparison)
            default_model_path = None
            if os.path.exists("siamese_model.h5"):
                default_model_path = "siamese_model.h5"
            elif os.path.exists("siamese_model_v2.h5"):
                default_model_path = "siamese_model_v2.h5"
            
            if default_model_path is None:
                raise HTTPException(
                    status_code=400,
                    detail="No trained model found. Please train at least one user model first."
                )
            
            # Extract embedding using default model (we'll re-extract with user models during comparison)
            embedding = extract_face_embedding(image_bytes=content, model_path=default_model_path)
            
            extraction_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Load active embeddings
            stored_embeddings = load_active_embeddings()
            
            if not stored_embeddings:
                return RecognizeEmbeddingResponse(matches=[])
            
            # Compute similarity (with timing)
            # Use user-specific models for accurate comparison
            similarity_start = time.time()
            similarities = []
            
            for stored_user_id, stored_embedding in stored_embeddings:
                # Try to use the user-specific model for this enrolled user
                user_model_path = f"siamese_model_{stored_user_id}.h5"
                
                # Re-extract embedding using the enrolled user's model for accurate comparison
                try:
                    if os.path.exists(user_model_path):
                        # Use user-specific model
                        user_embedding = extract_face_embedding(image_bytes=content, model_path=user_model_path)
                        score = cosine_similarity(user_embedding, stored_embedding)
                    else:
                        # Fallback to default model
                        score = cosine_similarity(embedding, stored_embedding)
                except Exception as e:
                    # If user model fails, use default
                    print(f"⚠️  Error using {user_model_path}, using default model: {e}")
                    score = cosine_similarity(embedding, stored_embedding)
                
                # Debug: Print all similarity scores to help diagnose
                print(f"🔍 Similarity with {stored_user_id}: {score:.4f} ({score*100:.2f}%)")
                
                # DIAGNOSIS-BASED THRESHOLD:
                # Based on actual model performance analysis:
                # - Well-trained user-specific models: Same person = 75-88%, Different = <50%
                # - Default models (collapsed): Everyone = 85-95% (WRONG - need user-specific model)
                # - user_keshav model works correctly (negative similarity with others)
                # - user_123/user_1234 using default model (92% similarity - collapsed!)
                #
                # Strategy: Accept reasonable scores, but log warnings for suspicious patterns
                
                # Check if we're using user-specific model
                using_user_model = os.path.exists(user_model_path)
                
                if using_user_model:
                    # User-specific model: Accept all scores and let user see them
                    # The model should produce good scores, but if it doesn't, we still show them
                    # This helps diagnose model issues
                    similarities.append(MatchResult(user_id=stored_user_id, score=score))
                    
                    # Log warnings for suspicious patterns
                    if score > 0.95:
                        print(f"⚠️  WARNING: Very high score {score:.4f} for {stored_user_id}")
                        print(f"   >95% suggests model collapse - model may need retraining")
                    elif score < 0.50:
                        print(f"⚠️  WARNING: Low score {score:.4f} for {stored_user_id}")
                        print(f"   <50% suggests different person or model needs better training")
                    else:
                        print(f"✅ Match found for {stored_user_id} with {score:.4f} ({score*100:.2f}%)")
                else:
                    # Default model: Likely has embedding collapse, but show all scores
                    # Accept all scores to help diagnose
                    similarities.append(MatchResult(user_id=stored_user_id, score=score))
                    
                    # Log warnings
                    if score > 0.85:
                        print(f"⚠️  WARNING: {stored_user_id} using default model with high score {score:.4f}")
                        print(f"   Default models often have embedding collapse. Train user-specific model for {stored_user_id}.")
                    print(f"✅ Match found for {stored_user_id} with {score:.4f} ({score*100:.2f}%) [using default model]")
            
            # Sort by score and take top_k
            similarities.sort(key=lambda x: x.score, reverse=True)
            matches = similarities[:top_k]
            similarity_time = (time.time() - similarity_start) * 1000  # Convert to ms
            
            # Log recognition event
            matched_user_id = matches[0].user_id if matches else None
            similarity_score = matches[0].score if matches else None
            
            try:
                log_recognition_event(
                    device_id=device_id,
                    top_k=top_k,
                    matched_user_id=matched_user_id,
                    similarity_score=similarity_score
                )
            except:
                pass  # Don't fail if logging fails
            
            return RecognizeEmbeddingResponse(matches=matches)
            
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
