"""
Check training status and diagnose issues
"""
import os
import sys

print("=" * 60)
print("TRAINING STATUS CHECK")
print("=" * 60)

# Check 1: Model files
print("\n1. Checking for trained models...")
models_found = []
if os.path.exists("siamese_model_v2.h5"):
    size = os.path.getsize("siamese_model_v2.h5") / (1024 * 1024)
    print(f"   ✅ siamese_model_v2.h5 found ({size:.2f} MB)")
    models_found.append("siamese_model_v2.h5")
else:
    print("   ❌ siamese_model_v2.h5 NOT found")

if os.path.exists("siamese_model.h5"):
    size = os.path.getsize("siamese_model.h5") / (1024 * 1024)
    print(f"   ✅ siamese_model.h5 found ({size:.2f} MB)")
    models_found.append("siamese_model.h5")
else:
    print("   ❌ siamese_model.h5 NOT found")

# Check 2: Training data
print("\n2. Checking training data...")
anchor_count = len([f for f in os.listdir("data/anchor") if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists("data/anchor") else 0
positive_count = len([f for f in os.listdir("data/positive") if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists("data/positive") else 0
negative_count = len([f for f in os.listdir("data/negative") if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists("data/negative") else 0

print(f"   Anchor images: {anchor_count}")
print(f"   Positive images: {positive_count}")
print(f"   Negative images: {negative_count}")

# Check 3: Dependencies
print("\n3. Checking dependencies...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow: {tf.__version__}")
except ImportError:
    print("   ❌ TensorFlow NOT installed")
    print("      Install: pip install tensorflow")

try:
    import cv2
    print(f"   ✅ OpenCV: {cv2.__version__}")
except ImportError:
    print("   ❌ OpenCV NOT installed")
    print("      Install: pip install opencv-python")

try:
    import numpy as np
    print(f"   ✅ NumPy: {np.__version__}")
except ImportError:
    print("   ❌ NumPy NOT installed")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if models_found:
    print(f"✅ Training appears complete! Found: {', '.join(models_found)}")
    print("   You can now use the system.")
else:
    print("❌ No trained model found. Training needed.")
    print("\nTo train:")
    if anchor_count >= 2 and positive_count >= 2:
        if negative_count == 0:
            print("   1. Run: python quick_train.py")
            print("      (This will auto-create negative pairs)")
        else:
            print("   1. Run: python train_siamese_v2.py --data_dir data --epochs 50")
        print("   2. Wait for training to complete")
        print("   3. Restart server: uvicorn app.main:app --reload")
    else:
        print("   1. Collect more data: python collect_data.py")
        print("      (Need at least 2 anchor and 2 positive images)")
        print("   2. Then train: python quick_train.py")

print("=" * 60)
