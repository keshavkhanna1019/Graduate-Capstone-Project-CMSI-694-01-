"""
Clean Siamese Network Implementation for Face Recognition
Rebuilt from scratch with proper architecture to prevent embedding collapse
"""
import numpy as np
from typing import Tuple, Optional
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten,
        BatchNormalization, Dropout, GlobalAveragePooling2D
    )
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Model = None
    Layer = None


class L2Normalize(Layer):
    """Custom L2 normalization layer (serializable)"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        return tf.nn.l2_normalize(inputs, axis=1)
    
    def get_config(self):
        return super().get_config()


class L1Distance(Layer):
    """Kept for loading legacy .h5 models trained with BCE loss."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            anchor, positive = inputs[0], inputs[1]
        else:
            raise ValueError(f"L1Distance expects 2 inputs, got {type(inputs)}")
        return tf.reduce_sum(tf.abs(anchor - positive), axis=1, keepdims=True)

    def get_config(self):
        return super().get_config()


class EuclideanDistance(Layer):
    """Euclidean distance layer — output used directly with contrastive loss."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            a, b = inputs[0], inputs[1]
        else:
            raise ValueError(f"EuclideanDistance expects 2 inputs")
        # epsilon avoids NaN gradients at distance == 0
        return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=1, keepdims=True) + 1e-8)

    def get_config(self):
        return super().get_config()


def contrastive_loss(margin: float = 1.0):
    """
    Contrastive loss for metric learning.
    y=1 (same person) → minimise distance.
    y=0 (different person) → push distance above margin.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0.0))
        return tf.reduce_mean(y_true * square_pred + (1.0 - y_true) * margin_square)
    loss.__name__ = "contrastive_loss"
    return loss


def build_embedding_network(input_shape=(100, 100, 3), embedding_dim=128):
    """
    Face embedding network using frozen pretrained MobileNetV2.

    MobileNetV2 (trained on 1.4M ImageNet images) already produces
    highly discriminative visual features — no task-specific training needed.
    We just L2-normalise the 1280-dim backbone output so cosine similarity
    works correctly at inference.  Zero trainable parameters = zero collapse.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")

    inputs = Input(shape=input_shape, name='input_image')

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    backbone.trainable = False

    # MobileNetV2 expects pixels in [-1, 1]; our pipeline normalises to [0, 1]
    x = tf.keras.layers.Lambda(
        lambda img: img * 2.0 - 1.0, name='mobilenet_preprocess'
    )(inputs)
    x = backbone(x, training=False)
    x = GlobalAveragePooling2D(name='gap')(x)   # → 1280-dim
    outputs = L2Normalize(name='l2_normalize')(x)

    return Model(inputs=inputs, outputs=outputs, name='embedding_network')


def build_siamese_model(input_shape=(100, 100, 3), embedding_dim=128):
    """
    Build the Siamese network.  Output is Euclidean distance (used with contrastive loss).
    Embeddings are L2-normalised so cosine similarity == 1 - dist²/2 at inference time.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")

    embedding_network = build_embedding_network(input_shape, embedding_dim)

    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')

    anchor_embedding = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)

    distance = EuclideanDistance(name='euclidean_distance')([anchor_embedding, positive_embedding])

    return Model(
        inputs=[anchor_input, positive_input],
        outputs=distance,
        name='siamese_network'
    )


class SiameseNetwork:
    """
    Clean Siamese Network for face recognition
    """
    
    def __init__(self, model_path: Optional[str] = None, embedding_dim: int = 128):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        self.embedding_dim = embedding_dim
        self.model = None
        self.embedding_model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the Siamese model"""
        self.model = build_siamese_model(embedding_dim=self.embedding_dim)
        # Extract embedding network from the Siamese model
        self.embedding_model = self.model.get_layer('embedding_network')
    
    def compile_model(self, learning_rate: float = 0.0001):
        """Compile the model with contrastive loss for metric learning."""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=contrastive_loss(margin=1.0),
        )
    
    def train(
        self,
        anchor_images: np.ndarray,
        positive_images: np.ndarray,
        negative_images: np.ndarray,
        epochs: int = 50,
        batch_size: int = 16,
        validation_split: float = 0.2
    ):
        """
        Train the Siamese network
        
        Args:
            anchor_images: Anchor images (N, H, W, C)
            positive_images: Positive images (same person as anchor)
            negative_images: Negative images (different person)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
        """
        # Create positive pairs (anchor, positive) -> label 1
        pos_pairs = [anchor_images, positive_images]
        pos_labels = np.ones(len(anchor_images))
        
        # Create negative pairs (anchor, negative) -> label 0
        neg_pairs = [anchor_images, negative_images]
        neg_labels = np.zeros(len(anchor_images))
        
        # Combine
        all_inputs = [
            np.concatenate([pos_pairs[0], neg_pairs[0]], axis=0),
            np.concatenate([pos_pairs[1], neg_pairs[1]], axis=0)
        ]
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        # Shuffle
        indices = np.random.permutation(len(all_labels))
        all_inputs[0] = all_inputs[0][indices]
        all_inputs[1] = all_inputs[1][indices]
        all_labels = all_labels[indices]
        
        # Train
        history = self.model.fit(
            all_inputs,
            all_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single image
        
        Args:
            image: Preprocessed image array (H, W, C) or (1, H, W, C)
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        if self.embedding_model is None:
            raise ValueError("Model must be built or loaded before extracting embeddings")
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Extract embedding (already L2 normalized in the model)
        embedding = self.embedding_model.predict(image, verbose=0)
        return embedding[0]  # Return first (and only) embedding
    
    def verify(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[float, bool]:
        """
        Verify if two images are of the same person
        
        Args:
            image1: First image (H, W, C)
            image2: Second image (H, W, C)
            threshold: Similarity threshold
            
        Returns:
            Tuple of (similarity_score, is_match)
        """
        # Extract embeddings
        emb1 = self.extract_embedding(image1)
        emb2 = self.extract_embedding(image2)
        
        # Compute cosine similarity (embeddings are already L2 normalized)
        similarity = np.dot(emb1, emb2)
        is_match = similarity > threshold
        
        return float(similarity), bool(is_match)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("Model must be built before saving")
        
        # Save with custom objects
        self.model.save(
            filepath,
            save_format='h5',
            include_optimizer=False
        )
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            self.model = tf.keras.models.load_model(
                filepath,
                custom_objects={
                    'L1Distance': L1Distance,
                    'L2Normalize': L2Normalize,
                    'EuclideanDistance': EuclideanDistance,
                    'contrastive_loss': contrastive_loss(margin=1.0),
                },
                compile=False
            )
            
            # Extract embedding model
            self.embedding_model = self.model.get_layer('embedding_network')
            self.model_path = filepath
            print(f"✅ Model loaded from {filepath}")
            
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")


def preprocess_image_for_siamese(
    image_path: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_array: Optional[np.ndarray] = None,
    target_size: Tuple[int, int] = (100, 100)
) -> np.ndarray:
    """
    Preprocess image for Siamese network
    
    Args:
        image_path: Path to image file
        image_bytes: Image as bytes
        image_array: Image as numpy array
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image array (H, W, C) normalized to [0, 1]
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required. Install with: pip install opencv-python")
    
    img = None
    
    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path}")
    elif image_bytes:
        # Validate bytes
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Empty image bytes provided")
        
        # Try OpenCV first
        cv_error_msg = None
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as cv_error:
            img = None
            cv_error_msg = str(cv_error)
        
        # If OpenCV fails, try PIL as fallback (with HEIC support)
        if img is None:
            try:
                from PIL import Image
                import io
                
                # Try to register HEIF opener if available (for HEIC/HEIF support)
                try:
                    from pillow_heif import register_heif_opener
                    register_heif_opener()
                except ImportError:
                    pass  # pillow-heif not installed, continue without HEIC support
                
                pil_img = Image.open(io.BytesIO(image_bytes))
                
                # Convert to RGB if needed (handles RGBA, P, L, etc.)
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                # Convert PIL to numpy array
                img = np.array(pil_img)
                
                # Convert RGB to BGR for OpenCV compatibility
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                # Check if it's a HEIC file
                if image_bytes[:12].startswith(b'\x00\x00\x00') and b'ftyp' in image_bytes[:20]:
                    error_details = f"OpenCV error: {cv_error_msg or 'imdecode returned None'}, PIL error: {str(pil_error)}"
                    raise ValueError(
                        f"Could not decode HEIC/HEIF image. {error_details}\n\n"
                        f"💡 Solution: Install HEIC support with: pip install pillow-heif\n"
                        f"   Or convert your HEIC image to JPEG/PNG first.\n"
                        f"   Image size: {len(image_bytes)} bytes"
                    )
                else:
                    error_details = f"OpenCV error: {cv_error_msg or 'imdecode returned None'}, PIL error: {str(pil_error)}"
                    raise ValueError(f"Could not decode image from bytes. {error_details}. Image size: {len(image_bytes)} bytes, First 20 bytes: {image_bytes[:20]}")
        
        if img is None:
            raise ValueError("Could not decode image from bytes - both OpenCV and PIL failed")
    elif image_array is not None:
        img = image_array.copy()
    else:
        raise ValueError("Must provide image_path, image_bytes, or image_array")
    
    if img is None:
        raise ValueError("Failed to load image")
    
    # Convert BGR to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Face detection — crop to face region so the embedding is face-only, not background
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            # Pick the largest detected face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            pad = int(max(w, h) * 0.25)
            ih, iw = img.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(iw, x + w + pad)
            y2 = min(ih, y + h + pad)
            img = img[y1:y2, x1:x2]
    except Exception:
        pass  # if detection fails, use full image

    # Resize
    img = cv2.resize(img, target_size)

    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img
