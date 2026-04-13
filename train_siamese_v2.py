"""
Training script for Siamese Network (V2 - matching notebook implementation)
Based on: https://github.com/nicknochnack/FaceRecognition
"""
import numpy as np
import os
import cv2
from app.services.siamese_network_v2 import (
    SiameseNetworkV2,
    preprocess_image,
    data_augmentation
)


def load_images_from_directory(directory: str, target_size=(100, 100)) -> list:
    """
    Load and preprocess images from a directory
    
    Args:
        directory: Path to directory containing images
        target_size: Target image size
        
    Returns:
        List of preprocessed image arrays
    """
    images = []
    
    if not os.path.exists(directory):
        print(f"⚠️  Directory not found: {directory}")
        return images
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(directory, filename)
            try:
                img = preprocess_image(img_path, target_size)
                images.append(img)
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
    
    return images


def prepare_training_data(anchor_dir: str, positive_dir: str, negative_dir: str):
    """
    Prepare training data in the format expected by the notebook
    
    Args:
        anchor_dir: Directory with anchor images
        positive_dir: Directory with positive images (same person as anchor)
        negative_dir: Directory with negative images (different people)
        
    Returns:
        Tuple of (anchor_images, positive_images, negative_images)
    """
    print("Loading anchor images...")
    anchor_images = load_images_from_directory(anchor_dir)
    print(f"  Loaded {len(anchor_images)} anchor images")
    
    print("Loading positive images...")
    positive_images = load_images_from_directory(positive_dir)
    print(f"  Loaded {len(positive_images)} positive images")
    
    print("Loading negative images...")
    negative_images = load_images_from_directory(negative_dir)
    print(f"  Loaded {len(negative_images)} negative images")
    
    # Ensure we have matching numbers
    min_count = min(len(anchor_images), len(positive_images), len(negative_images))
    if min_count == 0:
        raise ValueError("Need at least one image in each directory")
    
    anchor_images = anchor_images[:min_count]
    positive_images = positive_images[:min_count]
    negative_images = negative_images[:min_count]
    
    # Convert to numpy arrays
    anchor_array = np.array(anchor_images)
    positive_array = np.array(positive_images)
    negative_array = np.array(negative_images)
    
    print(f"\n✅ Prepared {min_count} training triplets")
    
    return anchor_array, positive_array, negative_array


def train_siamese_model(data_dir: str = "data",
                       epochs: int = 50,
                       batch_size: int = 16,
                       model_save_path: str = "siamese_model_v2.h5"):
    """
    Train Siamese network matching the notebook implementation
    
    Args:
        data_dir: Base directory containing anchor/, positive/, negative/ subdirectories
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Where to save the trained model
    """
    # Setup paths
    anchor_dir = os.path.join(data_dir, 'anchor')
    positive_dir = os.path.join(data_dir, 'positive')
    negative_dir = os.path.join(data_dir, 'negative')
    
    # Check if directories exist
    if not all(os.path.exists(d) for d in [anchor_dir, positive_dir, negative_dir]):
        print("❌ Error: Data directories not found!")
        print(f"   Expected structure:")
        print(f"   {data_dir}/")
        print(f"     anchor/")
        print(f"     positive/")
        print(f"     negative/")
        print("\n   Create these directories and add images to each.")
        return
    
    # Load training data
    try:
        anchor_images, positive_images, negative_images = prepare_training_data(
            anchor_dir, positive_dir, negative_dir
        )
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return
    
    # Create and compile model
    print("\nBuilding Siamese network...")
    siamese = SiameseNetworkV2()
    siamese.compile_model(learning_rate=0.0001)
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    history = siamese.train(
        anchor_images=anchor_images,
        positive_images=positive_images,
        negative_images=negative_images,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save model
    print(f"\nSaving model to {model_save_path}...")
    siamese.save_model(model_save_path)
    
    print("✅ Training complete!")
    print(f"   Final accuracy: {history.history['accuracy'][-1]:.2%}")
    
    return siamese, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Siamese Network V2")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Base directory with anchor/positive/negative subdirectories")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--model_path", type=str, default="siamese_model_v2.h5",
                       help="Path to save trained model")
    
    args = parser.parse_args()
    
    train_siamese_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_path
    )
