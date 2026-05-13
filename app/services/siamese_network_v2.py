"""
Siamese Neural Network for Face Recognition
Based on: https://github.com/nicknochnack/FaceRecognition
Updated implementation matching the notebook architecture
"""
import numpy as np
from typing import Tuple, List
import os

# Optional imports for training
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten, Lambda
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None
    Model = None
    Layer = None
    Conv2D = None
    Dense = None
    MaxPooling2D = None
    Input = None
    Flatten = None
    Lambda = None


class L2Normalize(Layer):
    """
    Custom layer for L2 normalization
    This is serializable (unlike Lambda layers with lambda functions)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        import tensorflow as tf
        return tf.nn.l2_normalize(inputs, axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape


class L1Dist(Layer):
    """
    Custom layer to compute L1 distance between embeddings
    This is used in the Siamese network to compare two face embeddings
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        """
        Compute L1 distance between two embeddings
        
        Args:
            inputs: List containing [input_embedding, validation_embedding]
                   or nested list structure from Keras
            
        Returns:
            L1 distance tensor (sum of absolute differences)
        """
        # Handle nested list structure from Keras
        if isinstance(inputs, (list, tuple)):
            # Flatten nested lists if needed
            flat_inputs = []
            for item in inputs:
                if isinstance(item, (list, tuple)):
                    flat_inputs.extend(item)
                else:
                    flat_inputs.append(item)
            
            if len(flat_inputs) == 2:
                input_embedding, validation_embedding = flat_inputs[0], flat_inputs[1]
            else:
                raise ValueError(f"L1Dist expects 2 tensor inputs, got {len(flat_inputs)}")
        else:
            raise ValueError(f"L1Dist expects list/tuple input, got {type(inputs)}")
        
        # Compute L1 distance (sum of absolute differences)
        l1_dist = tf.reduce_sum(tf.math.abs(input_embedding - validation_embedding), axis=1, keepdims=True)
        
        return l1_dist
    
    def compute_output_shape(self, input_shape):
        """Compute output shape for the layer"""
        # Input shape is a list of two shapes, both should be (batch, 4096)
        # Output should be (batch, 1)
        if isinstance(input_shape, (list, tuple)):
            # Handle nested shapes
            first_shape = input_shape[0]
            if isinstance(first_shape, (list, tuple)) and len(first_shape) > 0:
                first_shape = first_shape[0]
            return (first_shape[0] if isinstance(first_shape, (list, tuple)) else None, 1)
        return (None, 1)


def make_embedding():
    """
    Create the base embedding network (shared between twin networks)
    Architecture matches the notebook implementation
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    inp = Input(shape=(100, 100, 3), name='input_image')
    
    # First block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D((2, 2), padding='same')(c1)
    
    # Second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D((2, 2), padding='same')(c2)
    
    # Third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D((2, 2), padding='same')(c3)
    
    # Fourth block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    # Changed from sigmoid to linear with L2 normalization
    # Sigmoid can cause embedding collapse (all embeddings become similar)
    # Linear + L2 normalization allows for better embedding diversity
    d1 = Dense(4096, activation='linear')(f1)
    # Add L2 normalization using custom layer (serializable, unlike Lambda)
    d1_normalized = L2Normalize()(d1)
    
    return Model(inputs=[inp], outputs=[d1_normalized], name='embedding')


def make_siamese_model():
    """
    Create the Siamese network model
    Uses twin embedding networks with L1 distance comparison
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    # Anchor and validation input images
    input_image = Input(name='input_img', shape=(100, 100, 3))
    validation_image = Input(name='validation_img', shape=(100, 100, 3))
    
    # Create the embedding network
    embedding = make_embedding()
    
    # Generate embeddings for both images
    in_emb = embedding(input_image)
    val_emb = embedding(validation_image)
    
    # Compute L1 distance between embeddings using the custom L1Dist layer
    # This matches the notebook implementation
    siamese_layer = L1Dist()
    distances = siamese_layer([in_emb, val_emb])
    
    # Classification layer
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')


class SiameseNetworkV2:
    """
    Siamese Network implementation matching the notebook
    https://github.com/nicknochnack/FaceRecognition
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize Siamese Network
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
        
        self.model = None
        self.embedding_model = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build the Siamese network model"""
        self.model = make_siamese_model()
        # Extract the embedding model from the Siamese model
        self.embedding_model = self.model.layers[2]  # The embedding network
    
    def compile_model(self, learning_rate=0.0001):
        """
        Compile the model with binary crossentropy loss
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, anchor_images: np.ndarray, positive_images: np.ndarray, 
              negative_images: np.ndarray, epochs=50, batch_size=16):
        """
        Train the Siamese network on triplets
        
        Args:
            anchor_images: Anchor images (N, 100, 100, 3)
            positive_images: Positive images (same person as anchor)
            negative_images: Negative images (different person)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # Create training pairs
        # Positive pairs: (anchor, positive) -> label 1
        # Negative pairs: (anchor, negative) -> label 0
        
        # Combine positive and negative
        pos_pairs = [(anchor_images[i], positive_images[i]) for i in range(len(anchor_images))]
        neg_pairs = [(anchor_images[i], negative_images[i]) for i in range(len(anchor_images))]
        
        # Create labels
        pos_labels = np.ones(len(pos_pairs))
        neg_labels = np.zeros(len(neg_pairs))
        
        # Combine
        all_pairs = pos_pairs + neg_pairs
        all_labels = np.concatenate([pos_labels, neg_labels])
        
        # Shuffle
        indices = np.random.permutation(len(all_pairs))
        shuffled_pairs = [all_pairs[i] for i in indices]
        shuffled_labels = all_labels[indices]
        
        # Prepare data
        anchor_batch = np.array([pair[0] for pair in shuffled_pairs])
        validation_batch = np.array([pair[1] for pair in shuffled_pairs])
        
        # Train
        history = self.model.fit(
            [anchor_batch, validation_batch],
            shuffled_labels,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from a single image
        
        Args:
            image: Preprocessed image array (100, 100, 3)
            
        Returns:
            Embedding vector
        """
        if self.embedding_model is None:
            raise ValueError("Model must be built or loaded before extracting embeddings")
        
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        embedding = self.embedding_model.predict(image, verbose=0)
        return embedding[0]  # Return first (and only) embedding
    
    def verify(self, input_image: np.ndarray, validation_image: np.ndarray, 
               threshold: float = 0.5) -> Tuple[float, bool]:
        """
        Verify if two images are of the same person
        
        Args:
            input_image: First image (100, 100, 3)
            validation_image: Second image (100, 100, 3)
            threshold: Similarity threshold (default 0.5)
            
        Returns:
            Tuple of (similarity_score, is_match)
        """
        # Add batch dimensions
        input_img = np.expand_dims(input_image, axis=0)
        val_img = np.expand_dims(validation_image, axis=0)
        
        # Predict
        prediction = self.model.predict([input_img, val_img], verbose=0)
        similarity = float(prediction[0][0])
        is_match = similarity > threshold
        
        return similarity, is_match
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model:
            # Save full model with custom objects for custom layers
            # This ensures L1Dist and L2Normalize layers can be loaded
            self.model.save(
                filepath, 
                save_format='h5',
                include_optimizer=False  # Don't save optimizer to avoid pickle issues
            )
            # Also save weights separately for easier loading if compatibility issues arise
            weights_path = filepath.replace('.h5', '_weights.h5')
            self.model.save_weights(weights_path)
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required")
        
        try:
            # Try loading with custom objects
            self.model = tf.keras.models.load_model(
                filepath,
                custom_objects={'L1Dist': L1Dist, 'L2Normalize': L2Normalize},
                compile=False
            )
        except Exception as e:
            # Handle compatibility issues with batch_shape parameter
            error_str = str(e)
            if 'batch_shape' in error_str or 'Unrecognized keyword arguments' in error_str:
                # Workaround: Rebuild model and extract weights from the incompatible file
                print(f"⚠️  Compatibility issue detected. Extracting weights from old model...")
                self._build_model()
                
                # Try to load weights directly from the incompatible model file
                # The weights are stored in 'model_weights' group in the HDF5 file
                try:
                    import h5py
                    # Open the old model file and extract weights
                    with h5py.File(filepath, 'r') as f:
                        if 'model_weights' in f.keys():
                            # Save weights to a temporary file that we can load
                            temp_weights_path = filepath.replace('.h5', '_extracted_weights.h5')
                            with h5py.File(temp_weights_path, 'w') as wf:
                                # Copy the model_weights group
                                f.copy('model_weights', wf, name='model_weights')
                            
                            # Now load the extracted weights into our new model
                            self.model.load_weights(temp_weights_path, by_name=True, skip_mismatch=True)
                            print("✅ Successfully extracted and loaded weights from incompatible model!")
                            
                            # Clean up temp file
                            if os.path.exists(temp_weights_path):
                                os.remove(temp_weights_path)
                        else:
                            raise ValueError("Could not find model_weights in the saved file")
                except Exception as weights_error:
                    print(f"⚠️  Could not extract weights: {weights_error}")
                    # Try loading from separate weights file if it exists
                    weights_path = filepath.replace('.h5', '_weights.h5')
                    if os.path.exists(weights_path):
                        try:
                            self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                            print("✅ Loaded weights from separate file")
                        except:
                            raise ValueError(
                                f"Model loading failed due to TensorFlow version compatibility.\n"
                                f"Please retrain the model with your current TensorFlow version.\n"
                                f"Original error: {error_str}"
                            ) from e
                    else:
                        raise ValueError(
                            f"Model file incompatible with current TensorFlow version.\n"
                            f"Error: {error_str}\n\n"
                            f"Solution: Please retrain the model:\n"
                            f"  1. python quick_train.py\n"
                            f"  2. Restart server"
                        ) from e
            else:
                raise
        
        # Extract embedding model
        # The model structure: Input -> Embedding -> L1Dist -> Dense
        # We need the embedding submodel (base network)
        try:
            # Find the embedding layer (it's a Model within the Siamese model)
            # The embedding is the shared base network
            for layer in self.model.layers:
                if isinstance(layer, Model) and 'embedding' in layer.name.lower():
                    self.embedding_model = layer
                    break
            
            # If not found, extract it from the model structure
            if self.embedding_model is None:
                # The embedding model is the shared base network
                # Get it from the first input's output before L1Dist
                input_img = self.model.input[0]
                # Find L1Dist layer
                l1_layer = None
                for layer in self.model.layers:
                    if isinstance(layer, L1Dist):
                        l1_layer = layer
                        break
                
                if l1_layer:
                    # Get the input to L1Dist (which is the embedding output)
                    embedding_output = l1_layer.input[0] if isinstance(l1_layer.input, list) else l1_layer.input
                    # Create embedding model
                    self.embedding_model = Model(inputs=input_img, outputs=embedding_output)
                else:
                    # Fallback: use make_embedding and try to load weights
                    self.embedding_model = make_embedding()
                    print("⚠️  Using default embedding architecture")
        except Exception as e:
            print(f"⚠️  Warning: Could not extract embedding model: {e}")
            # Fallback: create new embedding model
            self.embedding_model = make_embedding()
        
        self.model_path = filepath


def preprocess_image(image_path: str = None, image_bytes: bytes = None, target_size=(100, 100)) -> np.ndarray:
    """
    Preprocess image for Siamese network
    Matches the notebook preprocessing
    
    Args:
        image_path: Path to image file (optional)
        image_bytes: Image as bytes (optional)
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed image array (100, 100, 3) normalized to [0, 1]
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV required for preprocessing")
    
    # Read image - try path first, then bytes
    img = None
    if image_path:
        # Check if file exists
        if not os.path.exists(image_path):
            raise ValueError(f"Image file does not exist: {image_path}")
        
        # Try reading from file
        img = cv2.imread(image_path)
        if img is None:
            # If cv2.imread fails, try reading bytes and decoding
            try:
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as e:
                raise ValueError(f"Could not load image from path {image_path}: {str(e)}")
    
    if image_bytes and img is None:
        # Decode from bytes - try OpenCV first, then PIL as fallback
        cv_error = None
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                # Try with different flag
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            cv_error = e
            img = None
        
        # If OpenCV failed, try PIL
        if img is None:
            try:
                from PIL import Image
                import io
                pil_img = Image.open(io.BytesIO(image_bytes))
                # Convert PIL RGB to OpenCV BGR
                img_array = np.array(pil_img)
                if len(img_array.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                elif img_array.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:  # RGB
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                error_msg = f"Could not decode image from bytes. "
                if cv_error:
                    error_msg += f"OpenCV error: {str(cv_error)}. "
                error_msg += f"PIL error: {str(pil_error)}"
                raise ValueError(error_msg)
    
    if img is None:
        raise ValueError("Must provide either image_path or image_bytes")
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    return img


def data_augmentation(img: np.ndarray) -> List[np.ndarray]:
    """
    Apply data augmentation to an image
    Based on the notebook's augmentation function
    
    Args:
        img: Input image array
        
    Returns:
        List of augmented images
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow is required")
    
    augmented = []
    img_tensor = tf.convert_to_tensor(img)
    
    for i in range(9):
        # Random brightness
        aug = tf.image.stateless_random_brightness(img_tensor, max_delta=0.02, seed=(i, 2))
        # Random contrast
        aug = tf.image.stateless_random_contrast(aug, lower=0.6, upper=1, seed=(i, 3))
        # Random flip
        aug = tf.image.stateless_random_flip_left_right(aug, seed=(np.random.randint(100), np.random.randint(100)))
        # Random JPEG quality
        aug = tf.image.stateless_random_jpeg_quality(aug, min_jpeg_quality=90, max_jpeg_quality=100, 
                                                     seed=(np.random.randint(100), np.random.randint(100)))
        # Random saturation
        aug = tf.image.stateless_random_saturation(aug, lower=0.6, upper=1, seed=(np.random.randint(100), np.random.randint(100)))
        
        augmented.append(aug.numpy())
    
    return augmented
