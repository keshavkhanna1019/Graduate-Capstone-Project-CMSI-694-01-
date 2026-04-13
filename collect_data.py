"""
Improved script to collect training data using webcam
Press 'a' for anchor, 'p' for positive, 'q' to quit
Uses face detection to automatically crop faces
"""
import cv2
import os
import uuid

# Create directories
os.makedirs("data/anchor", exist_ok=True)
os.makedirs("data/positive", exist_ok=True)

# Load face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    FACE_DETECTOR_AVAILABLE = True
except:
    FACE_DETECTOR_AVAILABLE = False
    print("⚠️  Face detector not available, using manual cropping")

def detect_and_crop_face(frame):
    """
    Detect face in frame and return cropped face region
    Falls back to center crop if detection fails
    """
    if FACE_DETECTOR_AVAILABLE:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Use the largest face detected
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            
            # Expand the bounding box a bit
            padding = 30
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Crop and resize to square
            face_region = frame[y:y+h, x:x+w]
            
            # Resize to 250x250 for consistency
            face_region = cv2.resize(face_region, (250, 250))
            
            return face_region, (x, y, w, h)
    
    # Fallback: center crop
    h, w = frame.shape[:2]
    size = min(h, w)
    start_y = (h - size) // 2
    start_x = (w - size) // 2
    face_region = frame[start_y:start_y+size, start_x:start_x+size]
    face_region = cv2.resize(face_region, (250, 250))
    
    return face_region, (start_x, start_y, size, size)

print("Starting webcam...")
print("Instructions:")
print("  - Press 'a' to save anchor image")
print("  - Press 'p' to save positive image")
print("  - Press 'q' to quit")
print("  - Make sure your face is clearly visible in the frame!")
print("\nPosition your face in the center and press keys to collect data!")

# Try different camera indices
cap = None
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"✅ Using camera {i}")
            break
    cap.release()

if cap is None or not cap.isOpened():
    print("❌ Error: Could not open camera")
    print("   Make sure your webcam is connected and not being used by another app")
    exit(1)

# Set camera resolution for better quality
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

anchor_count = 0
positive_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Detect and crop face
    face_region, bbox = detect_and_crop_face(frame)
    
    # Draw bounding box on original frame for feedback
    if FACE_DETECTOR_AVAILABLE:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show instructions on face region
    cv2.putText(face_region, "Press: a=anchor, p=positive, q=quit", 
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(face_region, f"Anchor: {anchor_count}, Positive: {positive_count}", 
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    # Show both views
    cv2.imshow('Full Frame (with detection)', frame)
    cv2.imshow('Face Region (will be saved)', face_region)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Save anchor
    if key == ord('a'):
        imgname = f"data/anchor/{uuid.uuid1()}.jpg"
        cv2.imwrite(imgname, face_region)
        anchor_count += 1
        print(f"✅ Saved anchor #{anchor_count}: {imgname}")
        # Flash green to confirm
        flash = face_region.copy()
        flash[:] = (0, 255, 0)
        cv2.imshow('Face Region (will be saved)', flash)
        cv2.waitKey(100)
    
    # Save positive
    if key == ord('p'):
        imgname = f"data/positive/{uuid.uuid1()}.jpg"
        cv2.imwrite(imgname, face_region)
        positive_count += 1
        print(f"✅ Saved positive #{positive_count}: {imgname}")
        # Flash blue to confirm
        flash = face_region.copy()
        flash[:] = (255, 255, 0)
        cv2.imshow('Face Region (will be saved)', flash)
        cv2.waitKey(100)
    
    # Quit
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Collection complete!")
print(f"   Collected {anchor_count} anchor images")
print(f"   Collected {positive_count} positive images")
print(f"\nNext steps:")
print(f"   1. Add negative images to data/negative/ (different people)")
print(f"   2. Train: python train_siamese_v2.py --data_dir data --epochs 50")
