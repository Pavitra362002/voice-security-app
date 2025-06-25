# backend/config.py
import os

# --- General Application Settings ---
LOGGING_LEVEL = 'INFO' # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

# --- Audio Configuration ---
SAMPLE_RATE = 16000     # Standard sample rate for many STT and ML models (Hz)
CHUNK_SIZE = 1024       # Number of frames per audio buffer for real-time processing
                        # Lower chunk size means lower latency but more CPU cycles.

# How often to process accumulated audio for STT and threat detection (in seconds)
AUDIO_SEGMENT_LENGTH_SECONDS = 3

# --- Paths ---
# Directory to store voice authentication models and data
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# This path is for storing registered user embeddings.
REGISTERED_EMBEDDINGS_FILE = os.path.join(MODELS_DIR, 'registered_embeddings.pkl')
VOICE_AUTHENTICATOR_MODEL_PATH = os.path.join(MODELS_DIR, 'voice_authenticator.h5') # Simulated Keras model

# NEW: AI Model Paths and Settings
SOUND_CLASSIFIER_MODEL_PATH = os.path.join(MODELS_DIR, 'sound_classifier_model.h5') # Simulated AI model
SOUND_CLASSIFIER_LABELS_PATH = os.path.join(MODELS_DIR, 'sound_classifier_labels.pkl') # For storing class labels

# NEW: Dataset Paths (Update these to your actual dataset locations)
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
URBANSOUND8K_PATH = os.path.join(DATASETS_DIR, 'UrbanSound8K')
ESC50_PATH = os.path.join(DATASETS_DIR, 'ESC-50-master')

# --- Voice Authentication Settings ---
# Cosine distance threshold: 0 (identical) to 2 (opposite). Lower means stricter.
# You will need to tune this with real data.
AUTHENTICATION_THRESHOLD = 0.4
VOICE_FEATURE_LENGTH = 128 # The size of the simulated embedding vector
# Default challenge phrase for enrollment/authentication
DEFAULT_CHALLENGE_PHRASE = "My voice is my password"


# --- Threat Detection Keywords ---
# Define keywords that, if detected in transcribed speech, will trigger an alert
THREAT_KEYWORDS = [
    "help", "fire", "intruder", "alarm", "break in", "danger",
    "emergency", "scream", "robbery", "weapon", "shoot", "gun",
    "police", "call police", "kill you", "kill u", "gonna kill", "murder", "stab", "attack",
    "die", "dead", "threat", "violence", "hostage", "kidnap",
    "bomb", "explosive", "detonate", "assault"
]

# NEW: Dangerous Sound Categories (from UrbanSound8K/ESC-50 examples, aligned with simulated labels)
DANGEROUS_SOUND_CATEGORIES = [
    "gun_shot", "siren", "scream", "fire", "glass_breaking", "explosion", # Aligned with simulated labels
    "car_horn", "drilling", "jackhammer" # Added more common noisy/potentially alarming sounds
]
# Ensure these categories are consistent with your model's output labels

# --- Sound Event Monitor Settings ---
# RMS amplitude threshold for "loud" noise. (0.0 to 1.0)
# You will need to tune this based on your microphone and environment.
VOLUME_THRESHOLD = 0.1 # Example: Increased slightly to reduce false positives
CONSECUTIVE_CHUNKS_THRESHOLD = 5 # Number of consecutive chunks exceeding threshold to trigger alert


# --- Application Launching ---
# A dictionary mapping 'application_name' (for dropdown) to 'path_to_executable' (for launching)
# IMPORTANT: Use double backslashes (\\) or forward slashes (/) for Windows paths.
# Update these paths to match the actual locations of executables on YOUR system.
PROTECTED_APPS = {
    "Notepad": "C:\\Windows\\System32\\notepad.exe",
    "Calculator": "C:\\Windows\\System32\\calc.exe",
    "Paint": "C:\\Windows\\System32\\mspaint.exe",
    "Command Prompt": "C:\\Windows\\System32\\cmd.exe",
    # Example for web browsers (paths may vary slightly on your system):
    # "Google Chrome": "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    # "Mozilla Firefox": "C:\\Program Files\\Mozilla Firefox\\firefox.exe",
}
