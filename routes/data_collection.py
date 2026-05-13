# routes/data_collection.py

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
import os
import base64
import uuid
from datetime import datetime

from app.repositories.consent_repo import has_consent

router = APIRouter(prefix="/api", tags=["data-collection"])


class SaveImageRequest(BaseModel):
    image_data: str = Field(..., description="Base64 encoded image data")
    category: str = Field(..., description="Image category: 'positive' or 'negative'")
    user_id: str = Field(..., description="User ID to save image for")
    filename: Optional[str] = Field(None, description="Optional filename")


@router.post("/save-training-image")
async def save_training_image(request: SaveImageRequest):
    """
    Save a training image to the user-specific directory.
    Positive images require biometric consent from the user being enrolled.
    """
    valid_categories = ['positive', 'negative']

    if request.category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {valid_categories}"
        )

    # Positive images are biometric data of a specific person — consent required.
    # Negative images are "other people" examples and don't map to a single identity.
    if request.category == 'positive' and not has_consent(request.user_id):
        raise HTTPException(
            status_code=403,
            detail=f"Cannot collect training data for '{request.user_id}': no biometric consent on file. "
                   "Grant consent first via the Consent tab or Enroll tab."
        )
    
    # Create user-specific directory structure: data/{user_id}/{category}/
    data_dir = "data"
    user_dir = os.path.join(data_dir, request.user_id)
    category_dir = os.path.join(user_dir, request.category)
    os.makedirs(category_dir, exist_ok=True)
    
    try:
        # Decode base64 image
        image_data = request.image_data
        # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        
        # Generate filename
        if request.filename:
            filename = request.filename
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'
        else:
            filename = f"{uuid.uuid4()}.jpg"
        
        filepath = os.path.join(category_dir, filename)
        
        # Save image
        with open(filepath, 'wb') as f:
            f.write(image_bytes)
        
        return {
            "status": "success",
            "message": f"Image saved to {request.category}/",
            "filepath": filepath,
            "category": request.category,
            "filename": filename
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving image: {str(e)}"
        )


@router.get("/training-data-stats")
def get_training_data_stats(user_id: Optional[str] = None):
    """
    Get statistics about collected training data for a specific user
    """
    data_dir = "data"
    stats = {
        "positive": 0,
        "negative": 0,
        "total": 0
    }
    
    if user_id:
        # Get stats for specific user
        user_dir = os.path.join(data_dir, user_id)
        for category in ['positive', 'negative']:
            category_dir = os.path.join(user_dir, category)
            if os.path.exists(category_dir):
                count = len([f for f in os.listdir(category_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.HEIC'))])
                stats[category] = count
                stats["total"] += count
    else:
        # Get stats for all users (legacy support)
        for category in ['positive', 'negative']:
            category_dir = os.path.join(data_dir, category)
            if os.path.exists(category_dir):
                count = len([f for f in os.listdir(category_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.HEIC'))])
                stats[category] = count
                stats["total"] += count
    
    return stats


@router.get("/training-data/{user_id}/gallery")
def get_training_gallery(user_id: str):
    """Return lists of image filenames for a user's positive and negative folders."""
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.heic'}
    result = {"positive": [], "negative": []}
    for category in ("positive", "negative"):
        folder = os.path.join("data", user_id, category)
        if os.path.isdir(folder):
            result[category] = sorted(
                f for f in os.listdir(folder)
                if os.path.splitext(f)[1].lower() in IMG_EXTS
            )
    return result


@router.get("/training-data/{user_id}/{category}/{filename}")
def serve_training_image(user_id: str, category: str, filename: str):
    """Serve a single training image."""
    from fastapi.responses import FileResponse
    path = os.path.join("data", user_id, category, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@router.delete("/training-data/{user_id}/{category}")
def delete_training_data(user_id: str, category: str):
    """
    Delete all images in a user-specific category
    """
    valid_categories = ['positive', 'negative']
    
    if category not in valid_categories:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Must be one of: {valid_categories}"
        )
    
    category_dir = os.path.join("data", user_id, category)
    
    if not os.path.exists(category_dir):
        return {"status": "success", "message": f"No {category} directory found"}
    
    try:
        deleted_count = 0
        for filename in os.listdir(category_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(category_dir, filename)
                os.remove(filepath)
                deleted_count += 1
        
        return {
            "status": "success",
            "message": f"Deleted {deleted_count} images from {category}/",
            "deleted_count": deleted_count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting images: {str(e)}"
        )
