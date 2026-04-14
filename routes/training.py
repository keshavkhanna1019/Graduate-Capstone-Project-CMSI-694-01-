"""
Training routes for Siamese network
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import os
import threading
import time
import numpy as np

router = APIRouter(prefix="/api", tags=["training"])

# Global training status
training_status = {
    "is_training": False,
    "progress": 0,
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "accuracy": None,
    "val_loss": None,
    "val_accuracy": None,
    "message": "",
    "error": None,
    "stop_requested": False  # Flag to stop training gracefully
}


class TrainModelRequest(BaseModel):
    user_id: str = Field(..., description="User ID to train model for (e.g., 'user_123')")
    epochs: int = Field(default=50, ge=1, le=500, description="Number of training epochs")
    batch_size: int = Field(default=16, ge=1, le=64, description="Batch size for training")
    data_dir: str = Field(default="data", description="Directory containing user-specific positive/negative folders")
    embedding_dim: int = Field(default=128, ge=64, le=512, description="Embedding dimension")


class TrainModelResponse(BaseModel):
    status: str
    message: str
    training_id: Optional[str] = None


def train_model_background(user_id: str, epochs: int, batch_size: int, data_dir: str, embedding_dim: int):
    """
    Background training function that runs in a separate thread
    Trains a user-specific model using positive (user's images) and negative (other people's images)
    """
    global training_status
    
    try:
        training_status["is_training"] = True
        training_status["error"] = None
        training_status["stop_requested"] = False  # Reset stop flag
        training_status["total_epochs"] = epochs
        training_status["message"] = f"Starting training for {user_id}..."
        
        # Import here to avoid issues if dependencies aren't available
        from app.services.siamese_network import (
            SiameseNetwork,
            preprocess_image_for_siamese
        )
        import cv2
        
        # Load images - user-specific structure
        # positive: user's own images
        # negative: other people's images
        training_status["message"] = f"Loading training images for {user_id}..."
        positive_dir = os.path.join(data_dir, user_id, 'positive')
        negative_dir = os.path.join(data_dir, user_id, 'negative')
        
        # Fallback to old structure if new structure doesn't exist
        if not os.path.exists(positive_dir):
            # Try old structure: data/positive and data/negative
            positive_dir = os.path.join(data_dir, 'positive')
            negative_dir = os.path.join(data_dir, 'negative')
        
        def load_images_from_directory(directory: str) -> list:
            images = []
            if not os.path.exists(directory):
                return images
            for filename in os.listdir(directory):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic', '.HEIC')):
                    img_path = os.path.join(directory, filename)
                    try:
                        img = preprocess_image_for_siamese(image_path=img_path)
                        images.append(img)
                    except Exception as e:
                        print(f"⚠️  Error loading {img_path}: {e}")
            return images
        
        positive_images = load_images_from_directory(positive_dir)
        negative_images = load_images_from_directory(negative_dir)
        
        if len(positive_images) == 0:
            raise ValueError(f"No positive images found for {user_id}. Need at least one image in {positive_dir}")
        
        if len(negative_images) == 0:
            raise ValueError(f"No negative images found for {user_id}. Need at least one image in {negative_dir}")
        
        # Use minimum count to balance positive and negative pairs
        min_count = min(len(positive_images), len(negative_images))
        if min_count < 2:
            raise ValueError(f"Need at least 2 images in both positive and negative directories for {user_id}")
        
        # Convert to numpy arrays
        positive_array = np.array(positive_images[:min_count])
        negative_array = np.array(negative_images[:min_count])
        
        training_status["message"] = f"Prepared {min_count} positive and {min_count} negative images for {user_id}"
        
        # Build and compile model
        training_status["message"] = f"Building Siamese network for {user_id}..."
        siamese = SiameseNetwork(embedding_dim=embedding_dim)
        siamese.compile_model(learning_rate=0.0001)
        
        # Custom training loop with progress updates
        training_status["message"] = f"Training for {epochs} epochs..."
        
        # Create training data: positive pairs (same person) and negative pairs (different person)
        # Based on reference notebook: https://github.com/nicknochnack/FaceRecognition
        # 
        # Positive pairs: (positive_image_1, positive_image_2) -> label 1 (same person)
        # Negative pairs: (positive_image, negative_image) -> label 0 (different person)
        # 
        # CRITICAL: Create MANY diverse negative pairs to help model learn to distinguish
        
        pos_pairs = []
        pos_labels = []
        # Create all pairs from positive images (same person)
        for i in range(min_count):
            for j in range(i + 1, min_count):
                pos_pairs.append([positive_array[i], positive_array[j]])
                pos_labels.append(1)
        
        # Create diverse negative pairs (different person)
        # Use ALL combinations of positive vs negative, not just matching indices
        neg_pairs = []
        neg_labels = []
        for i in range(min_count):
            for j in range(min_count):
                # Create negative pair: positive[i] vs negative[j]
                neg_pairs.append([positive_array[i], negative_array[j]])
                neg_labels.append(0)
        
        # Balance the dataset - ensure roughly equal positive and negative examples
        num_pos_pairs = len(pos_pairs)
        num_neg_pairs = len(neg_pairs)
        
        # If we have too many negative pairs, sample them
        if num_neg_pairs > num_pos_pairs * 2:
            # Randomly sample negative pairs to balance
            neg_indices = np.random.choice(num_neg_pairs, size=min(num_pos_pairs * 2, num_neg_pairs), replace=False)
            neg_pairs = [neg_pairs[i] for i in neg_indices]
            neg_labels = [neg_labels[i] for i in neg_indices]
            num_neg_pairs = len(neg_pairs)
        
        # If we have too many positive pairs, limit them
        if num_pos_pairs > num_neg_pairs:
            # Limit positive pairs to match negative pairs
            pos_pairs = pos_pairs[:num_neg_pairs]
            pos_labels = pos_labels[:num_neg_pairs]
            num_pos_pairs = len(pos_pairs)
        
        # Log training data stats
        training_status["message"] = f"Created {num_pos_pairs} positive pairs and {num_neg_pairs} negative pairs"
        
        # Combine all pairs
        all_pairs = pos_pairs + neg_pairs
        all_labels = np.array(pos_labels + neg_labels)
        
        # Convert to format expected by Siamese network
        all_inputs = [
            np.array([pair[0] for pair in all_pairs]),
            np.array([pair[1] for pair in all_pairs])
        ]
        
        # Shuffle
        indices = np.random.permutation(len(all_labels))
        all_inputs[0] = all_inputs[0][indices]
        all_inputs[1] = all_inputs[1][indices]
        all_labels = all_labels[indices]
        
        # Split for validation
        split_idx = int(len(all_labels) * 0.8)
        train_inputs = [inp[:split_idx] for inp in all_inputs]
        train_labels = all_labels[:split_idx]
        val_inputs = [inp[split_idx:] for inp in all_inputs]
        val_labels = all_labels[split_idx:]
        
        # Training loop with progress updates
        for epoch in range(1, epochs + 1):
            # Check if stop was requested before starting this epoch
            if training_status.get("stop_requested", False):
                training_status["message"] = f"Training stopped by user at epoch {epoch-1}/{epochs}"
                training_status["is_training"] = False
                break
            
            training_status["epoch"] = epoch
            training_status["progress"] = int((epoch / epochs) * 100)
            training_status["message"] = f"Epoch {epoch}/{epochs}"
            
            # Train on batch
            history = siamese.model.fit(
                train_inputs,
                train_labels,
                batch_size=batch_size,
                epochs=1,
                validation_data=(val_inputs, val_labels),
                verbose=0
            )
            
            # Update status
            training_status["loss"] = float(history.history['loss'][0])
            training_status["accuracy"] = float(history.history['accuracy'][0])
            if 'val_loss' in history.history:
                training_status["val_loss"] = float(history.history['val_loss'][0])
            if 'val_accuracy' in history.history:
                training_status["val_accuracy"] = float(history.history['val_accuracy'][0])
            
            time.sleep(0.1)  # Small delay to allow status updates
            
            # Check again after epoch completes (in case stop was requested during training)
            if training_status.get("stop_requested", False):
                training_status["message"] = f"Training stopped by user after epoch {epoch}/{epochs}"
                training_status["is_training"] = False
                break
        
        # Save model - user-specific (only if training wasn't stopped)
        if not training_status.get("stop_requested", False):
            training_status["message"] = f"Saving model for {user_id}..."
            model_path = f"siamese_model_{user_id}.h5"
            siamese.save_model(model_path)
            training_status["message"] = f"✅ Training complete for {user_id}! Model saved to {model_path}"
            training_status["progress"] = 100
        else:
            # Training was stopped - don't save model
            training_status["message"] = f"⚠️ Training stopped by user. Model not saved."
            training_status["progress"] = int((training_status["epoch"] / epochs) * 100)
        
    except Exception as e:
        training_status["error"] = str(e)
        training_status["message"] = f"❌ Training failed: {str(e)}"
        import traceback
        print(f"Training error: {traceback.format_exc()}")
    finally:
        training_status["is_training"] = False


@router.post("/train-model", response_model=TrainModelResponse)
async def train_model(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """
    Start training the Siamese network model
    Training runs in the background - check /api/training-status for progress
    """
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(
            status_code=409,
            detail="Training is already in progress. Wait for it to complete or check status."
        )
    
    # Check if data directories exist
    data_dir = request.data_dir
    user_id = request.user_id
    
    # Check for user-specific structure first
    positive_dir = os.path.join(data_dir, user_id, 'positive')
    negative_dir = os.path.join(data_dir, user_id, 'negative')
    
    # Fallback to old structure
    if not os.path.exists(positive_dir):
        positive_dir = os.path.join(data_dir, 'positive')
        negative_dir = os.path.join(data_dir, 'negative')
    
    if not os.path.exists(positive_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Training data not found for {user_id}. Expected directory: {positive_dir}\n\n"
                   f"Please organize your data as:\n"
                   f"  data/{user_id}/positive/ (your images)\n"
                   f"  data/{user_id}/negative/ (other people's images)"
        )
    
    if not os.path.exists(negative_dir):
        raise HTTPException(
            status_code=400,
            detail=f"Negative images not found for {user_id}. Expected directory: {negative_dir}\n\n"
                   f"Please add negative examples (other people's images) to train the model."
        )
    
    # Start training in background
    background_tasks.add_task(
        train_model_background,
        user_id=user_id,
        epochs=request.epochs,
        batch_size=request.batch_size,
        data_dir=data_dir,
        embedding_dim=request.embedding_dim
    )
    
    return TrainModelResponse(
        status="started",
        message="Training started in background. Check /api/training-status for progress.",
        training_id="training_1"
    )


@router.get("/training-status")
def get_training_status():
    """
    Get current training status and progress
    """
    return training_status


@router.post("/stop-training")
def stop_training():
    """
    Stop training gracefully - will stop after current epoch completes
    """
    global training_status
    
    if not training_status["is_training"]:
        return {"status": "not_training", "message": "No training in progress"}
    
    # Set stop flag - training loop will check this at start of next epoch
    training_status["stop_requested"] = True
    training_status["message"] = "Stop requested - training will stop after current epoch completes"
    
    return {"status": "stop_requested", "message": "Stop requested. Training will complete current epoch and then stop."}
