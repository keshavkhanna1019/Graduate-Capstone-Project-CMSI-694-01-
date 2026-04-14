"""
Face Recognition Service - Extracts face embeddings from images
Uses the clean Siamese network implementation
"""
from typing import List, Optional
import os
import numpy as np

# Try to import ML dependencies
try:
    from app.services.siamese_network import (
        SiameseNetwork,
        preprocess_image_for_siamese
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    _ML_ERROR = str(e)


def extract_face_embedding(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_pil = None,
    model_path: Optional[str] = None
) -> List[float]:
    """
    Extract face embedding from an image using the Siamese network
    
    Args:
        image_path: Path to image file (optional)
        image_bytes: Image as bytes (optional)
        image_pil: PIL Image object (optional)
        model_path: Path to trained model (default: siamese_model.h5)
        
    Returns:
        List of floats representing the face embedding (L2 normalized)
    """
    if not ML_AVAILABLE:
        raise ImportError(
            f"ML features not available: {_ML_ERROR}\n"
            "Please install required packages: pip install tensorflow opencv-python"
        )
    
    # Default model path - ONLY use the new model
    if model_path is None:
        model_path = "siamese_model.h5"
    
    # Warn about old broken models (but don't block - let user decide)
    if os.path.exists("siamese_model_v2.h5"):
        print("⚠️  WARNING: Old model (siamese_model_v2.h5) detected. This may cause embedding collapse.")
        print("   Consider deleting it and retraining with the new architecture.")
    
    if not os.path.exists(model_path):
        raise ValueError(
            f"No trained model found at {model_path}.\n\n"
            "Please train a model first:\n"
            "1. Collect data: Use 'Collect Data' tab in the web interface\n"
            "2. Train model: Use 'Train Model' tab (100+ epochs recommended)\n"
            "3. Restart server"
        )
    
    try:
        # Load model
        siamese = SiameseNetwork(model_path=model_path)
        
        # Preprocess image
        if image_bytes:
            processed_img = preprocess_image_for_siamese(image_bytes=image_bytes)
        elif image_path:
            processed_img = preprocess_image_for_siamese(image_path=image_path)
        elif image_pil:
            # Convert PIL to numpy array
            img_array = np.array(image_pil)
            if len(img_array.shape) == 2:
                # Grayscale
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                # RGBA to RGB
                img_array = img_array[:, :, :3]
            processed_img = preprocess_image_for_siamese(image_array=img_array)
        else:
            raise ValueError("Must provide image_path, image_bytes, or image_pil")
        
        # Extract embedding (already L2 normalized in the model)
        embedding = siamese.extract_embedding(processed_img)
        
        # Return as list
        return embedding.tolist()
        
    except Exception as e:
        raise ValueError(f"Error extracting face embedding: {str(e)}")
