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
    Train a SHARED Siamese model on data from ALL enrolled users.

    Positive pairs:  images of the same person  → label 1
    Negative pairs:  images of different people → label 0

    Negative examples come from two sources (combined):
      1. Each user's explicit negative/ folder
      2. Cross-user pairs (user_A vs user_B positive images)

    The finished model is saved as siamese_model.h5 (one shared file).
    After training, re-enroll all users so stored embeddings match the new model.
    """
    global training_status

    try:
        training_status["is_training"] = True
        training_status["error"] = None
        training_status["stop_requested"] = False
        training_status["total_epochs"] = epochs
        training_status["accuracy"] = None
        training_status["val_accuracy"] = None
        training_status["message"] = f"Starting training (shared model, triggered by {user_id})..."

        from app.services.siamese_network import SiameseNetwork, preprocess_image_for_siamese

        def load_images_from_directory(directory: str) -> list:
            images = []
            if not os.path.exists(directory):
                return images
            for filename in sorted(os.listdir(directory)):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                    img_path = os.path.join(directory, filename)
                    try:
                        images.append(preprocess_image_for_siamese(image_path=img_path))
                    except Exception as e:
                        print(f"⚠️  Skipping {img_path}: {e}")
            return images

        # ── Gather data from every user subdirectory ──────────────────────────
        training_status["message"] = "Scanning training data for all users..."
        user_positives: dict[str, list] = {}   # uid → [img, ...]
        user_negatives: dict[str, list] = {}   # uid → [img, ...]

        if os.path.isdir(data_dir):
            for uid in sorted(os.listdir(data_dir)):
                uid_path = os.path.join(data_dir, uid)
                if not os.path.isdir(uid_path):
                    continue
                pos = load_images_from_directory(os.path.join(uid_path, 'positive'))
                neg = load_images_from_directory(os.path.join(uid_path, 'negative'))
                if pos:
                    user_positives[uid] = pos
                if neg:
                    user_negatives[uid] = neg

        if user_id not in user_positives:
            raise ValueError(
                f"No positive images for '{user_id}' in {data_dir}/{user_id}/positive/. "
                "Collect training data first."
            )

        users = list(user_positives.keys())
        user_counts = {u: len(imgs) for u, imgs in user_positives.items()}
        training_status["message"] = f"Users found: {user_counts}"

        # ── Build pairs ────────────────────────────────────────────────────────
        pos_pairs: list[tuple] = []
        neg_pairs: list[tuple] = []

        # Positive pairs: all within-user combinations
        for uid, imgs in user_positives.items():
            for i in range(len(imgs)):
                for j in range(i + 1, len(imgs)):
                    pos_pairs.append((imgs[i], imgs[j]))

        # Negative pairs source 1: cross-user (different identities)
        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                for img_a in user_positives[users[i]]:
                    for img_b in user_positives[users[j]]:
                        neg_pairs.append((img_a, img_b))

        # Negative pairs source 2: explicit negative/ folders
        for uid, neg_imgs in user_negatives.items():
            if uid not in user_positives:
                continue
            for pos_img in user_positives[uid]:
                for neg_img in neg_imgs:
                    neg_pairs.append((pos_img, neg_img))

        if not pos_pairs:
            raise ValueError(
                "No positive pairs. Need at least 2 positive images for at least one user."
            )
        if not neg_pairs:
            raise ValueError(
                "No negative pairs. Either add negative/ images for each user, "
                "or collect data for a second user so cross-user pairs can be created."
            )

        # Balance: target ~2 negatives per positive
        n_pos = len(pos_pairs)
        n_neg = len(neg_pairs)
        if n_neg > n_pos * 3:
            idx = np.random.choice(n_neg, size=n_pos * 3, replace=False)
            neg_pairs = [neg_pairs[i] for i in idx]
        elif n_pos > n_neg * 2:
            idx = np.random.choice(n_pos, size=n_neg * 2, replace=False)
            pos_pairs = [pos_pairs[i] for i in idx]

        training_status["message"] = (
            f"Dataset: {len(pos_pairs)} positive pairs, {len(neg_pairs)} negative pairs"
        )

        # ── Assemble tensors ───────────────────────────────────────────────────
        all_pairs = pos_pairs + neg_pairs
        all_labels = np.array(
            [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs), dtype=np.float32
        )

        perm = np.random.permutation(len(all_labels))
        all_inputs = [
            np.array([all_pairs[i][0] for i in perm], dtype=np.float32),
            np.array([all_pairs[i][1] for i in perm], dtype=np.float32),
        ]
        all_labels = all_labels[perm]

        split = int(len(all_labels) * 0.8)
        train_inputs = [inp[:split] for inp in all_inputs]
        train_labels = all_labels[:split]
        val_inputs   = [inp[split:] for inp in all_inputs]
        val_labels   = all_labels[split:]

        # ── Build model with pretrained MobileNetV2 backbone ──────────────────
        training_status["message"] = "Loading pretrained MobileNetV2 backbone..."
        siamese = SiameseNetwork(embedding_dim=embedding_dim)

        # The embedding network uses frozen pretrained weights — no gradient
        # updates needed.  We save immediately so enroll/recognize can load it.
        training_status["message"] = "Saving pretrained feature extractor..."
        model_path = "siamese_model.h5"
        siamese.save_model(model_path)

        training_status["progress"] = 100
        training_status["loss"] = 0.0
        training_status["message"] = (
            "✅ Model ready! Using pretrained MobileNetV2 features — "
            "no collapse, works with any number of users. "
            "Re-enroll all users so stored embeddings use the new model."
        )

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
    
    # Validate that the triggering user has positive training images
    data_dir = request.data_dir
    user_id = request.user_id

    positive_dir = os.path.join(data_dir, user_id, 'positive')
    if not os.path.exists(positive_dir) or not os.listdir(positive_dir):
        raise HTTPException(
            status_code=400,
            detail=(
                f"No positive images found for '{user_id}'. "
                f"Expected: {positive_dir}\n\n"
                f"Use the Collect Data tab to capture images first:\n"
                f"  data/{user_id}/positive/ — your face images\n"
                f"  data/{user_id}/negative/ — other people's images (optional if multiple users exist)"
            )
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


@router.post("/train-svm")
def train_svm(data_dir: str = "data"):
    """
    Train an SVM classifier on top of the stored ArcFace embeddings.

    For each enrolled user this endpoint:
      1. Extracts individual ArcFace embeddings from data/{user_id}/positive/ photos
         (up to 20 per user) so the SVM sees the natural variance of each person.
      2. Falls back to the single stored averaged embedding when no photos exist.

    Requires >= 2 enrolled users. Saves svm_classifier.pkl on success.
    """
    from app.repositories.embeddings_repo import load_active_embeddings
    from app.services.face_extraction import extract_face_embedding
    from app.services.svm_classifier import SVMFaceClassifier, reload_svm_classifier, SVM_MODEL_PATH

    enrolled = load_active_embeddings()
    if not enrolled:
        raise HTTPException(status_code=400, detail="No enrolled users found.")

    user_embeddings: dict[str, list] = {}

    for user_id, stored_emb, _ in enrolled:
        per_photo: list = []

        # Prefer individual per-photo embeddings for richer SVM training data
        pos_dir = os.path.join(data_dir, user_id, "positive")
        if os.path.isdir(pos_dir):
            img_files = sorted([
                os.path.join(pos_dir, f)
                for f in os.listdir(pos_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])[:20]
            for img_path in img_files:
                try:
                    per_photo.append(extract_face_embedding(image_path=img_path))
                except Exception:
                    pass

        if per_photo:
            user_embeddings[user_id] = per_photo
        else:
            # Fall back to single stored embedding
            user_embeddings[user_id] = [stored_emb]

    clf = SVMFaceClassifier()
    try:
        stats = clf.train(user_embeddings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    clf.save(SVM_MODEL_PATH)
    reload_svm_classifier()

    return {
        "status": "trained",
        "n_users": stats["n_users"],
        "n_samples": stats["n_samples"],
        "users": stats["users"],
        "message": (
            f"SVM trained on {stats['n_samples']} samples across {stats['n_users']} users. "
            "Recognition will now use SVM probabilities instead of raw cosine similarity."
        ),
    }


@router.get("/svm-status")
def svm_status():
    """Return whether the SVM classifier is trained and ready."""
    from app.services.svm_classifier import get_svm_classifier
    clf = get_svm_classifier()
    return {
        "is_trained": clf.is_trained,
        "n_users": clf.n_users,
        "n_samples": clf.n_samples,
        "users": list(clf.label_map.values()) if clf.is_trained else [],
    }


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
