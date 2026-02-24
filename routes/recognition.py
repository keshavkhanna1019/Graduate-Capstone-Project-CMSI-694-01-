from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from app.repositories.embeddings_repo import save_embedding, load_active_embeddings
from app.repositories.consent_repo import has_consent
from app.repositories.recognition_logs_repo import log_recognition_event
from app.services.similarity import cosine_similarity


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
👉 “I have no matches yet.”

'''
