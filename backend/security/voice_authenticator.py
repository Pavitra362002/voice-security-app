# backend/security/voice_authenticator.py
import os
import numpy as np
import pickle
import logging
import random # For simulating embeddings
import time # Import time for random seed
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    AUTHENTICATION_THRESHOLD,
    VOICE_AUTHENTICATOR_MODEL_PATH,
    VOICE_FEATURE_LENGTH,
    REGISTERED_EMBEDDINGS_FILE, # Corrected: Import this variable
    DEFAULT_CHALLENGE_PHRASE,
    SAMPLE_RATE
)

logger = logging.getLogger(__name__)

# --- Simulated Voice Authenticator Model (Keras/TensorFlow placeholder) ---
class VoiceAuthenticatorModel:
    """
    A simulated Keras-like model for generating voice embeddings.
    In a real application, this would load a pre-trained deep learning model
    (e.g., from TensorFlow/Keras or a voice biometric library).
    """
    def __init__(self, model_path=None, feature_length=VOICE_FEATURE_LENGTH):
        self.feature_length = feature_length
        self.model = self._load_model(model_path)
        logger.info(f"VoiceAuthenticatorModel initialized. Feature length: {self.feature_length}")

    def _load_model(self, model_path):
        """
        Simulates loading a Keras model. In a real scenario, you'd use:
        `return tf.keras.models.load_model(model_path)`
        For this simulation, we just return a placeholder.
        """
        # Ensure the models directory exists
        os.makedirs(os.path.dirname(VOICE_AUTHENTICATOR_MODEL_PATH), exist_ok=True)
        if model_path and os.path.exists(model_path):
            logger.info(f"Simulating loading voice authenticator model from {model_path}")
            # Here you would load your actual Keras/TF model
            # For simulation, we don't need to load anything concrete.
            return True # Indicate 'model loaded'
        else:
            logger.warning(f"Voice authenticator model not found at {model_path}. Using simulated embedding generation.")
            return False # Indicate 'no real model'

    def generate_embedding(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Simulates generating a fixed-size embedding (feature vector) from audio data.
        In a real scenario, the loaded Keras model would process the audio.

        Args:
            audio_array (np.ndarray): The raw audio samples.
            sample_rate (int): The sample rate of the audio.

        Returns:
            np.ndarray: A 1D numpy array representing the voice embedding.
        """
        # In a real model, you'd preprocess audio and pass it to model.predict()
        # For simulation, generate a random embedding
        np.random.seed(int(time.time() * 1000) % (2**32)) # Seed for variety but also some determinism per call
        embedding = np.random.rand(self.feature_length).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding) # Normalize to unit vector
        logger.debug(f"Simulated embedding generated: {embedding[:5]}...") # Log first 5 values
        return embedding

class VoiceAuthenticator:
    def __init__(self):
        self.model = VoiceAuthenticatorModel(VOICE_AUTHENTICATOR_MODEL_PATH)
        self.enrolled_users = self._load_enrolled_users()
        logger.info(f"VoiceAuthenticator initialized. Enrolled users: {list(self.enrolled_users.keys())}")


    def _load_enrolled_users(self):
        """
        Loads enrolled user data (embeddings and challenge phrases) from a file.
        """
        if os.path.exists(REGISTERED_EMBEDDINGS_FILE): # Corrected: Use REGISTERED_EMBEDDINGS_FILE
            try:
                with open(REGISTERED_EMBEDDINGS_FILE, 'rb') as f:
                    data = pickle.load(f)
                    logger.info(f"Loaded {len(data)} enrolled users from {REGISTERED_EMBEDDINGS_FILE}")
                    return data
            except Exception as e:
                logger.error(f"Error loading enrolled users from {REGISTERED_EMBEDDINGS_FILE}: {e}", exc_info=True)
                return {} # Return empty if load fails
        logger.info("No enrolled users file found. Starting with empty enrollment.")
        return {}

    def _save_enrolled_users(self):
        """
        Saves current enrolled user data to a file.
        """
        try:
            # Ensure the directory exists before saving
            os.makedirs(os.path.dirname(REGISTERED_EMBEDDINGS_FILE), exist_ok=True)
            with open(REGISTERED_EMBEDDINGS_FILE, 'wb') as f: # Corrected: Use REGISTERED_EMBEDDINGS_FILE
                pickle.dump(self.enrolled_users, f)
            logger.info(f"Saved {len(self.enrolled_users)} enrolled users to {REGISTERED_EMBEDDINGS_FILE}")
        except Exception as e:
            logger.error(f"Error saving enrolled users to {REGISTERED_EMBEDDINGS_FILE}: {e}", exc_info=True)

    def enroll_user(self, username: str, audio_array: np.ndarray, samplerate: int, challenge_phrase: str) -> bool:
        """
        Enrolls a new user by generating and storing their voice embedding
        and associating it with a challenge phrase.
        """
        if username in self.enrolled_users:
            logger.warning(f"Enrollment failed: User '{username}' already exists.")
            return False

        # Simulate voice embedding generation
        embedding = self.model.generate_embedding(audio_array, samplerate)

        # Store embedding and challenge phrase
        self.enrolled_users[username] = {
            'embedding': embedding,
            'challenge_phrase': challenge_phrase.lower() # Store lowercased for case-insensitive comparison
        }
        self._save_enrolled_users()
        logger.info(f"User '{username}' enrolled with embedding and challenge phrase.")
        return True

    def authenticate_user(self, username: str, audio_array: np.ndarray, samplerate: int, spoken_phrase: str) -> bool:
        """
        Authenticates a user by comparing their live voice embedding to the stored one
        and verifying the spoken challenge phrase.
        """
        if username not in self.enrolled_users:
            logger.warning(f"Authentication failed for '{username}': User not found.")
            return False

        stored_data = self.enrolled_users[username]
        stored_embedding = stored_data['embedding']
        stored_challenge_phrase = stored_data['challenge_phrase']

        # Simulate live voice embedding generation
        live_embedding = self.model.generate_embedding(audio_array, samplerate)

        # Calculate cosine similarity between stored and live embeddings
        similarity = cosine_similarity(stored_embedding.reshape(1, -1), live_embedding.reshape(1, -1))[0][0]
        logger.info(f"Authentication for '{username}': Cosine similarity = {similarity:.4f}")

        # Check voice similarity
        voice_match = similarity >= AUTHENTICATION_THRESHOLD

        # Check challenge phrase (case-insensitive)
        phrase_match = spoken_phrase.lower() == stored_challenge_phrase

        logger.info(f"Authentication for '{username}': Voice match: {voice_match}, Phrase match: {phrase_match}")

        if voice_match and phrase_match:
            logger.info(f"Authentication successful for '{username}'.")
            return True
        else:
            if not voice_match:
                logger.warning(f"Authentication failed for '{username}': Voice not recognized (similarity: {similarity:.4f}).")
            if not phrase_match:
                logger.warning(f"Authentication failed for '{username}': Challenge phrase mismatch. Expected '{stored_challenge_phrase}', got '{spoken_phrase.lower()}'.")
            return False

    def get_enrolled_users(self) -> list[str]:
        """
        Returns a list of usernames that have been enrolled.
        """
        return list(self.enrolled_users.keys())

