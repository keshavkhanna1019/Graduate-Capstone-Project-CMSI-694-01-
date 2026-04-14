"""
Training script for Siamese Network
This script trains a Siamese network on face pairs for better recognition accuracy.
"""
import numpy as np
import os
from app.services.siamese_network import (
    SiameseNetwork, 
    create_training_pairs,
    preprocess_face_for_siamese
)
from app.services.face_extraction import extract_face_embedding
import cv2


def load_training_images(data_dir: str):
    """
    Load training images from directory structure:
    data_dir/
      person1/
        img1.jpg
        img2.jpg
      person2/
        img1.jpg
        ...
    """
    images = []
    labels = []
    
    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if not os.path.isdir(person_path):
            continue
        
        for img_file in os.listdir(person_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    # Preprocess for Siamese network
                    img = preprocess_face_for_siamese(img)
                    images.append(img)
                    labels.append(person_dir)
    
    return images, labels


def train_siamese_model(data_dir: str = "training_data", 
                       epochs: int = 50,
                       batch_size: int = 32,
                       model_save_path: str = "siamese_model.h5"):
    """
    Train Siamese network on face pairs
    
    Args:
        data_dir: Directory containing training images organized by person
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_save_path: Where to save the trained model
    """
    print("Loading training images...")
    images, labels = load_training_images(data_dir)
    
    if len(images) < 10:
        raise ValueError(f"Need at least 10 images for training. Found {len(images)}")
    
    print(f"Loaded {len(images)} images from {len(set(labels))} people")
    
    # Create training pairs
    print("Creating training pairs...")
    pairs, pair_labels = create_training_pairs(images, labels, num_pairs=2000)
    print(f"Created {len(pairs)} training pairs")
    
    # Split into train/validation
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    train_labels = pair_labels[:split_idx]
    val_pairs = pairs[split_idx:]
    val_labels = pair_labels[split_idx:]
    
    # Prepare validation data
    val_images_a = np.array([pair[0] for pair in val_pairs])
    val_images_b = np.array([pair[1] for pair in val_pairs])
    val_data = ([val_images_a, val_images_b], np.array(val_labels))
    
    # Create and compile model
    print("Building Siamese network...")
    siamese = SiameseNetwork(input_shape=(112, 112, 3), embedding_dim=128)
    siamese.compile_model(learning_rate=0.00006)
    
    # Train
    print("Training Siamese network...")
    history = siamese.train(
        pairs=train_pairs,
        labels=train_labels,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save model
    print(f"Saving model to {model_save_path}...")
    siamese.save_model(model_save_path)
    
    print("Training complete!")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.2%}")
    
    return siamese, history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Siamese Network for Face Recognition")
    parser.add_argument("--data_dir", type=str, default="training_data",
                       help="Directory containing training images")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--model_path", type=str, default="siamese_model.h5",
                       help="Path to save trained model")
    
    args = parser.parse_args()
    
    train_siamese_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_path
    )
