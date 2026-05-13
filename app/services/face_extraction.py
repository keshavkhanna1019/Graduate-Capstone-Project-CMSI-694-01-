"""
Face embedding extraction using DeepFace (ArcFace backend).
ArcFace is trained on millions of face images — produces reliable 512-dim
embeddings that work correctly with cosine similarity out of the box.
"""
from typing import List, Optional
import os
import io
import tempfile
import numpy as np


def extract_face_embedding(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_pil=None,
    model_path: Optional[str] = None,   # kept for API compatibility, not used
) -> List[float]:
    """
    Extract a face embedding using DeepFace ArcFace.
    Returns a L2-normalised 512-dim vector.
    """
    try:
        from deepface import DeepFace
    except ImportError:
        raise ImportError("deepface is required: pip install deepface")

    # Resolve to a file path — DeepFace needs one
    tmp_path = None
    try:
        if image_path and os.path.exists(image_path):
            path_to_use = image_path
        elif image_bytes:
            suffix = ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(image_bytes)
                tmp_path = tmp.name
            path_to_use = tmp_path
        elif image_pil is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image_pil.save(tmp, format="JPEG")
                tmp_path = tmp.name
            path_to_use = tmp_path
        else:
            raise ValueError("Must provide image_path, image_bytes, or image_pil")

        result = DeepFace.represent(
            img_path=path_to_use,
            model_name="ArcFace",
            enforce_detection=False,   # don't crash if face detector is uncertain
            detector_backend="opencv",
        )

        embedding = np.array(result[0]["embedding"], dtype=np.float64)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding.tolist()

    except Exception as e:
        raise ValueError(f"Error extracting face embedding: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
