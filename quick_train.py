"""
Quick training script - works with minimal data
Creates negative pairs from anchor/positive if negative folder is empty
"""
import numpy as np
import os

# Try to import dependencies
try:
    import cv2
    from app.services.siamese_network_v2 import (
        SiameseNetworkV2,
        preprocess_image,
    )
    DEPENDENCIES_OK = True
except ImportError as e:
    DEPENDENCIES_OK = False
    print(f"❌ Missing dependencies: {e}")
    print("\nPlease install:")
    print("  pip install tensorflow opencv-python")
    exit(1)

def load_images_from_directory(directory: str, target_size=(100, 100)) -> list:
    """Load and preprocess images from a directory"""
    images = []
    if not os.path.exists(directory):
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

def create_negative_pairs(anchor_images, positive_images):
    """
    Create negative pairs by mixing anchor and positive images
    This creates 'different person' pairs for training
    """
    negatives = []
    # Use anchor images as negatives (they're different from positives in the pair)
    # This is a simple approach when you don't have separate negative images
    for i in range(len(anchor_images)):
        # Use a different anchor as negative
        neg_idx = (i + 1) % len(anchor_images)
        negatives.append(anchor_images[neg_idx])
    return negatives

print("🚀 Quick Training Script")
print("=" * 50)

# Load data
anchor_dir = "data/anchor"
positive_dir = "data/positive"
negative_dir = "data/negative"

print("\n📁 Loading images...")
anchor_images = load_images_from_directory(anchor_dir)
positive_images = load_images_from_directory(positive_dir)
negative_images = load_images_from_directory(negative_dir)

print(f"   Anchor: {len(anchor_images)} images")
print(f"   Positive: {len(positive_images)} images")
print(f"   Negative: {len(negative_images)} images")

# If no negatives, create them from anchors
if len(negative_images) == 0:
    print("\n⚠️  No negative images found. Creating negative pairs from anchors...")
    negative_images = create_negative_pairs(anchor_images, positive_images)
    print(f"   Created {len(negative_images)} negative images")

# Check minimum requirements
min_count = min(len(anchor_images), len(positive_images), len(negative_images))
if min_count < 2:
    print("\n❌ Error: Need at least 2 images in each category")
    print("   Collect more data using: python collect_data.py")
    exit(1)

# Prepare arrays
anchor_array = np.array(anchor_images[:min_count])
positive_array = np.array(positive_images[:min_count])
negative_array = np.array(negative_images[:min_count])

print(f"\n✅ Ready to train with {min_count} triplets")

# Create and compile model
print("\n🔧 Building Siamese network...")
siamese = SiameseNetworkV2()
siamese.compile_model(learning_rate=0.0001)

# Train with fewer epochs for quick training
epochs = 10  # Reduced for quick training
print(f"\n🏋️  Training for {epochs} epochs (this may take a few minutes)...")

try:
    history = siamese.train(
        anchor_images=anchor_array,
        positive_images=positive_array,
        negative_images=negative_array,
        epochs=epochs,
        batch_size=min(4, min_count)  # Small batch size
    )
    
    # Save model
    model_path = "siamese_model_v2.h5"
    print(f"\n💾 Saving model to {model_path}...")
    siamese.save_model(model_path)
    
    print("\n✅ Training complete!")
    print(f"   Model saved: {model_path}")
    print(f"   Final accuracy: {history.history['accuracy'][-1]:.2%}")
    print("\n🎉 You can now use the system!")
    print("   Restart your server and try enrolling/recognizing faces.")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 Tips:")
    print("   - Make sure you have TensorFlow installed: pip install tensorflow")
    print("   - Try collecting more images: python collect_data.py")
    print("   - Add negative images to data/negative/ (different people)")
