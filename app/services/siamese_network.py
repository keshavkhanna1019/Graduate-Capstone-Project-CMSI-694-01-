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
        BatchNormalization, Dropout
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
    """Custom L1 distance layer for Siamese network"""
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


def build_embedding_network(input_shape=(100, 100, 3), embedding_dim=128):
    """
    Build the base embedding network (shared between twin networks)
    
    Architecture:
    - Conv layers for feature extraction
    - Dense layers for embedding
    - L2 normalization for proper cosine similarity
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    inputs = Input(shape=input_shape, name='input_image')
    
    # Convolutional feature extraction
    x = Conv2D(64, (10, 10), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = BatchNormalization(name='bn1')(x)
    
    x = Conv2D(128, (7, 7), activation='relu', name='conv2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    x = BatchNormalization(name='bn2')(x)
    
    x = Conv2D(128, (4, 4), activation='relu', name='conv3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)
    x = BatchNormalization(name='bn3')(x)
    
    x = Conv2D(256, (4, 4), activation='relu', name='conv4')(x)
    x = Flatten(name='flatten')(x)
    
    # Dense layers for embedding
    x = Dense(4096, activation='relu', name='dense1')(x)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(embedding_dim, activation='linear', name='dense2')(x)
    
    # L2 normalization - CRITICAL for cosine similarity
    outputs = L2Normalize(name='l2_normalize')(x)
    
    return Model(inputs=inputs, outputs=outputs, name='embedding_network')


def build_siamese_model(input_shape=(100, 100, 3), embedding_dim=128):
    """
    Build the complete Siamese network model
    
    Uses twin embedding networks with L1 distance comparison
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    # Build shared embedding network
    embedding_network = build_embedding_network(input_shape, embedding_dim)
    
    # Twin inputs
    anchor_input = Input(shape=input_shape, name='anchor_input')
    positive_input = Input(shape=input_shape, name='positive_input')
    
    # Generate embeddings
    anchor_embedding = embedding_network(anchor_input)
    positive_embedding = embedding_network(positive_input)
    
    # Compute L1 distance
    distance = L1Distance(name='l1_distance')([anchor_embedding, positive_embedding])
    
    # Classification (same person = 1, different = 0)
    output = Dense(1, activation='sigmoid', name='classification')(distance)
    
    return Model(
        inputs=[anchor_input, positive_input],
        outputs=output,
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
        """Compile the model for training"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
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
                    'L2Normalize': L2Normalize
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
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img
