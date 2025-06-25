# backend/security/sound_classifier.py
import os
import numpy as np
import pickle
import logging
import random
import librosa
import librosa.display # Often useful, though not strictly needed for just feature extraction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf # Import TensorFlow for session management/GPU checks

from config import (
    SAMPLE_RATE,
    SOUND_CLASSIFIER_MODEL_PATH,
    SOUND_CLASSIFIER_LABELS_PATH,
    DANGEROUS_SOUND_CATEGORIES,
    URBANSOUND8K_PATH,
    ESC50_PATH,
    AUDIO_SEGMENT_LENGTH_SECONDS
)
from audio_processing.audio_utils import preprocess_audio # Reuse audio utility

logger = logging.getLogger(__name__)

class SoundClassifier:
    """
    Implements a sound event classification model using TensorFlow/Keras.
    Supports real training with UrbanSound8K/ESC-50 and real-time inference.
    """
    def __init__(self):
        # Ensure the models directory exists
        os.makedirs(os.path.dirname(SOUND_CLASSIFIER_MODEL_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(SOUND_CLASSIFIER_LABELS_PATH), exist_ok=True)

        self.model = None # Initialize as None, will be loaded or simulated
        self.labels = [] # Initialize as empty, will be loaded or simulated

        # Try to load real model and labels, fallback to simulation if not found
        try:
            self.model = self._load_model()
            self.labels = self._load_labels()
            if self.model and self.labels:
                logger.info(f"SoundClassifier initialized with real model and {len(self.labels)} labels: {self.labels}")
            else:
                logger.warning("Could not load real model or labels. Falling back to simulated behavior.")
                self.model = self._simulate_model_loading()
                self.labels = self._load_labels(simulate_if_empty=True) # Ensure labels are populated for simulation
                logger.info(f"SoundClassifier initialized with simulated model and {len(self.labels)} labels.")

        except Exception as e:
            logger.error(f"Error during SoundClassifier initialization (might be missing trained model): {e}", exc_info=True)
            logger.warning("Falling back to simulated model due to initialization error.")
            self.model = self._simulate_model_loading()
            self.labels = self._load_labels(simulate_if_empty=True) # Ensure labels are populated for simulation
            logger.info(f"SoundClassifier initialized with simulated model and {len(self.labels)} labels.")


    def _load_model(self):
        """
        Loads a trained Keras deep learning model.
        Returns None if the model file does not exist.
        """
        logger.info(f"Attempting to load real sound classification model from {SOUND_CLASSIFIER_MODEL_PATH}")
        if os.path.exists(SOUND_CLASSIFIER_MODEL_PATH):
            try:
                # Disable eager execution if using older TensorFlow for graph mode benefits
                # tf.compat.v1.disable_eager_execution()
                model = load_model(SOUND_CLASSIFIER_MODEL_PATH)
                logger.info("Real sound classification model loaded successfully.")
                return model
            except Exception as e:
                logger.error(f"Error loading real sound classification model: {e}", exc_info=True)
                return None
        else:
            logger.warning(f"No real sound classification model found at {SOUND_CLASSIFIER_MODEL_PATH}. It might not have been trained yet.")
            return None

    def _simulate_model_loading(self):
        """
        Provides a dummy object that simulates prediction for development
        when a real model isn't trained yet.
        """
        class SimulatedSoundModel:
            def __init__(self, labels):
                self.labels = labels

            def predict(self, features):
                num_classes = len(self.labels) if self.labels else 10 # Fallback
                if num_classes == 0:
                    return [np.zeros(1)] # Return a dummy prediction if no labels

                # Simulate a "dangerous" sound detection with some probability
                if random.random() < 0.15: # 15% chance to simulate a dangerous sound
                    dangerous_indices = [self.labels.index(cat) for cat in DANGEROUS_SOUND_CATEGORIES if cat in self.labels]
                    if dangerous_indices:
                        predicted_class_index = random.choice(dangerous_indices)
                        probs = np.zeros(num_classes)
                        probs[predicted_class_index] = 0.95 # High probability for a dangerous sound
                        return [probs]

                # Otherwise, simulate a non-dangerous sound
                non_dangerous_indices = [i for i, label in enumerate(self.labels) if label not in DANGEROUS_SOUND_CATEGORIES]
                if non_dangerous_indices:
                    predicted_class_index = random.choice(non_dangerous_indices)
                else: # All labels are dangerous or no labels, just pick randomly
                    predicted_class_index = random.randint(0, num_classes - 1)

                probs = np.random.rand(num_classes)
                probs = probs / probs.sum() # Normalize to sum to 1
                probs[predicted_class_index] = probs[predicted_class_index] + 0.2 # Boost the predicted class
                probs = probs / probs.sum() # Re-normalize
                return [probs]

        # Pass the current instance's labels to the simulated model
        return SimulatedSoundModel(self.labels)

    def _load_labels(self, simulate_if_empty=False):
        """
        Loads classification labels from a pickle file.
        Provides default/simulated ones if the file doesn't exist or is empty
        and `simulate_if_empty` is True.
        """
        labels = []
        if os.path.exists(SOUND_CLASSIFIER_LABELS_PATH):
            try:
                with open(SOUND_CLASSIFIER_LABELS_PATH, 'rb') as f:
                    labels = pickle.load(f)
                logger.info("Sound classification labels loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading sound classification labels from file: {e}", exc_info=True)
        
        if not labels and simulate_if_empty:
            # If labels couldn't be loaded or were empty, provide some default/simulated ones
            # These should cover common classes from UrbanSound8K and ESC-50
            labels = ["air_conditioner", "car_horn", "children_playing",
                      "dog_bark", "drilling", "engine_idling", "gun_shot",
                      "jackhammer", "siren", "street_music",
                      "train", "rain", "clock_tick", "door_wood_knock", "scream",
                      "thunderstorm", "fire", "glass_breaking", "door_bell"] # Expanded for more realism
            logger.warning(f"Using default/simulated sound classification labels: {labels}")
            # Try to save these dummy labels for consistency
            try:
                os.makedirs(os.path.dirname(SOUND_CLASSIFIER_LABELS_PATH), exist_ok=True)
                with open(SOUND_CLASSIFIER_LABELS_PATH, 'wb') as f:
                    pickle.dump(labels, f)
                logger.info("Created dummy sound classification labels file for simulation.")
            except Exception as e:
                logger.error(f"Could not save dummy labels file: {e}")
        return labels

    def _extract_features_for_prediction(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Extracts features (Mel spectrogram) from preprocessed audio data
        that the sound classification model would expect. This should match
        the feature extraction used during training.
        """
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Pad or truncate audio to a fixed length for consistent feature extraction
        # This is crucial for models expecting fixed-size inputs.
        target_samples = SAMPLE_RATE * AUDIO_SEGMENT_LENGTH_SECONDS
        if len(audio_data) < target_samples:
            pad_width = target_samples - len(audio_data)
            audio_data = np.pad(audio_data, (0, pad_width), 'constant')
        elif len(audio_data) > target_samples:
            audio_data = audio_data[:target_samples]

        # Generate Mel Spectrogram
        # Parameters (n_fft, hop_length, n_mels) must match training
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio_data, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128
        )
        # Convert to dB scale (log scale is common for spectrograms)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Models often expect a specific input shape (e.g., (batch, height, width, channels))
        # For a 2D spectrogram, it might be (1, n_mels, num_frames, 1) for a CNN
        features = np.expand_dims(log_mel_spectrogram, axis=-1) # Add channel dimension
        features = np.expand_dims(features, axis=0) # Add batch dimension

        # logger.debug(f"Feature extraction for prediction: Output shape {features.shape}")
        return features

    def classify_sound(self, audio_data: np.ndarray, original_sr: int) -> tuple[str, float]:
        """
        Classifies the given audio data into a sound category and returns
        the predicted label and its confidence.
        """
        if self.model is None or not self.labels:
            logger.warning("Sound classifier model or labels not available. Cannot classify.")
            return "unknown_sound", 0.0

        logger.debug(f"Starting sound classification for audio of shape {audio_data.shape}, SR={original_sr}")
        try:
            # Preprocess the audio (resample, normalize)
            processed_audio = preprocess_audio(audio_data, original_sr)

            # Extract features suitable for the model
            features = self._extract_features_for_prediction(processed_audio)

            if features is None or features.size == 0:
                logger.error("Feature extraction failed or returned empty features.")
                return "classification_error", 0.0

            # Perform prediction
            predictions = self.model.predict(features)[0] # Get probabilities for the first (and only) sample

            # Get the predicted class index and confidence
            predicted_class_index = np.argmax(predictions)
            confidence = predictions[predicted_class_index]
            predicted_label = self.labels[predicted_class_index]

            logger.info(f"Sound classified as: '{predicted_label}' with confidence: {confidence:.2f}")
            return predicted_label, float(confidence)

        except Exception as e:
            logger.error(f"Error during sound classification: {e}", exc_info=True)
            return "classification_error", 0.0

    def is_dangerous_sound(self, sound_label: str) -> bool:
        """
        Checks if a given sound label is considered dangerous.
        """
        return sound_label in DANGEROUS_SOUND_CATEGORIES

    def train_model(self, dataset_path: str, epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2):
        """
        REAL IMPLEMENTATION: Trains a deep learning model for sound classification.

        Args:
            dataset_path (str): Path to the dataset (e.g., UrbanSound8K, ESC-50).
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training data to be used as validation data.
        """
        logger.info(f"Starting REAL AI model training with dataset from: {dataset_path}")
        logger.info(f"Training for {epochs} epochs with batch size {batch_size}, validation split {validation_split}")

        audio_paths = []
        labels = []
        dataset_name = os.path.basename(dataset_path)

        # --- 1. Data Loading and Label Collection ---
        if "UrbanSound8K" in dataset_name:
            metadata_path = os.path.join(dataset_path, 'metadata', 'UrbanSound8K.csv')
            if not os.path.exists(metadata_path):
                logger.error(f"UrbanSound8K metadata not found at {metadata_path}. Cannot train.")
                return
            df = pd.read_csv(metadata_path)
            for _, row in df.iterrows():
                fold = row['fold']
                filename = row['slice_file_name']
                class_name = row['class']
                audio_path = os.path.join(dataset_path, 'audio', f'fold{fold}', filename)
                if os.path.exists(audio_path):
                    audio_paths.append(audio_path)
                    labels.append(class_name)
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
            logger.info(f"Loaded {len(audio_paths)} audio paths from UrbanSound8K.")
        elif "ESC-50-master" in dataset_name: # Assumes your corrected path 'ESC-50-master'
            metadata_path = os.path.join(dataset_path, 'meta', 'esc50.csv')
            if not os.path.exists(metadata_path):
                logger.error(f"ESC-50 metadata not found at {metadata_path}. Cannot train.")
                return
            df = pd.read_csv(metadata_path)
            for _, row in df.iterrows():
                filename = row['filename']
                class_name = row['category'] # ESC-50 uses 'category' for label
                audio_path = os.path.join(dataset_path, 'audio', filename)
                if os.path.exists(audio_path):
                    audio_paths.append(audio_path)
                    labels.append(class_name)
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
            logger.info(f"Loaded {len(audio_paths)} audio paths from ESC-50.")
        else:
            logger.error(f"Unknown dataset name or structure for training: {dataset_name}. Skipping training.")
            return

        if not audio_paths:
            logger.error("No audio files found for training. Please check dataset path and structure.")
            return

        # --- 2. Label Encoding ---
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)
        class_labels = list(label_encoder.classes_) # Store the ordered list of labels

        logger.info(f"Detected {num_classes} classes: {class_labels}")

        # Save class labels immediately after encoding
        try:
            with open(SOUND_CLASSIFIER_LABELS_PATH, 'wb') as f:
                pickle.dump(class_labels, f)
            logger.info(f"Saved {num_classes} class labels to {SOUND_CLASSIFIER_LABELS_PATH}")
            self.labels = class_labels # Update instance labels
        except Exception as e:
            logger.error(f"Failed to save class labels: {e}", exc_info=True)


        # --- 3. Feature Extraction (Batch Processing for efficiency) ---
        logger.info("Extracting features from audio files...")
        all_features = []
        all_encoded_labels = []

        for i, path in enumerate(audio_paths):
            try:
                # Use librosa.load here to get raw audio data and its sample rate
                audio_data, sr_original = librosa.load(path, sr=None) # Load with original SR

                # Preprocess audio (resample to SAMPLE_RATE, normalize)
                processed_audio = preprocess_audio(audio_data, sr_original)

                # Extract Mel spectrogram features for training
                features = self._extract_features_for_prediction(processed_audio) # Re-use the prediction feature extractor

                if features is not None and features.size > 0:
                    all_features.append(features[0]) # Remove batch dimension for concatenation
                    all_encoded_labels.append(encoded_labels[i])
                else:
                    logger.warning(f"Skipping {path} due to feature extraction failure.")
            except Exception as e:
                logger.warning(f"Error processing {path}: {e}", exc_info=True)

        if not all_features:
            logger.error("No features extracted. Cannot proceed with training.")
            return

        X = np.array(all_features)
        y = tf.keras.utils.to_categorical(np.array(all_encoded_labels), num_classes=num_classes) # One-hot encode labels

        logger.info(f"Extracted features shape: {X.shape}, Labels shape: {y.shape}")

        # Determine input shape for the model
        input_shape = X.shape[1:] # (n_mels, num_frames, 1)

        # --- 4. Data Splitting ---
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42, stratify=y)
        logger.info(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # --- 5. Model Definition ---
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])

        # --- 6. Model Compilation ---
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary(print_fn=logger.info) # Print summary to logs

        # --- 7. Model Training ---
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        ]

        logger.info("Beginning model fitting...")
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks)
        logger.info("Model fitting complete.")

        # --- 8. Save Model ---
        try:
            model.save(SOUND_CLASSIFIER_MODEL_PATH)
            logger.info(f"Trained model saved to {SOUND_CLASSIFIER_MODEL_PATH}")
            self.model = model # Update the instance's model
        except Exception as e:
            logger.error(f"Failed to save trained model: {e}", exc_info=True)

        logger.info("Real AI model training complete.")


    def prepare_dataset(self, dataset_name: str):
        """
        Enhanced function to guide dataset preparation.
        """
        logger.info(f"Preparing dataset '{dataset_name}'.")
        target_path = ""
        if dataset_name.lower() == 'urbansound8k':
            target_path = URBANSOUND8K_PATH
            expected_audio_dir = os.path.join(target_path, 'audio')
            expected_metadata_file = os.path.join(target_path, 'metadata', 'UrbanSound8K.csv')
            logger.info(f"Checking UrbanSound8K structure at: {target_path}")
            if not os.path.exists(target_path) or not os.path.exists(expected_audio_dir) or not os.path.exists(expected_metadata_file):
                logger.warning(f"UrbanSound8K not found or incomplete. Please download and extract to: {target_path}")
                logger.warning("Expected: <target_path>/audio/fold[1-10]/*.wav, <target_path>/metadata/UrbanSound8K.csv")
                logger.info("Download UrbanSound8K: https://urbansound8k.readthedocs.io/en/latest/index.html")
            else:
                logger.info(f"UrbanSound8K dataset found and appears correctly structured at: {target_path}")

        elif dataset_name.lower() == 'esc50':
            target_path = ESC50_PATH
            expected_audio_dir = os.path.join(target_path, 'audio')
            expected_metadata_file = os.path.join(target_path, 'meta', 'esc50.csv')
            logger.info(f"Checking ESC-50 structure at: {target_path}")
            if not os.path.exists(target_path) or not os.path.exists(expected_audio_dir) or not os.path.exists(expected_metadata_file):
                logger.warning(f"ESC-50 not found or incomplete. Please download and extract to: {target_path}")
                logger.warning("Expected: <target_path>/audio/*.wav, <target_path>/meta/esc50.csv")
                logger.info("Download ESC-50: https://github.com/karolpiczak/ESC-50")
            else:
                logger.info(f"ESC-50 dataset found and appears correctly structured at: {target_path}")
        else:
            logger.warning(f"Unknown dataset name: {dataset_name}. Skipping preparation steps.")
