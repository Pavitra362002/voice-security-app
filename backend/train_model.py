# backend/train_model.py
import os
import sys
import logging

# --- Set up Logging for the training script ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Correction for Imports ---
# Add the backend directory to the system path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import necessary components from your backend
from security.sound_classifier import SoundClassifier
from config import URBANSOUND8K_PATH, ESC50_PATH, LOGGING_LEVEL

def run_training():
    """
    Initializes the SoundClassifier and runs its actual training method.
    """
    logger.info("Starting AI model training script...")

    # Set logging level for the classifier to see its internal logs
    classifier_logger = logging.getLogger('security.sound_classifier')
    classifier_logger.setLevel(getattr(logging, LOGGING_LEVEL.upper(), logging.INFO))


    try:
        classifier = SoundClassifier()
        logger.info("SoundClassifier instance created for training.")

        # --- Dataset Preparation Check ---
        # This will check if the datasets exist and log warnings if not.
        # You MUST have your datasets downloaded and extracted to the paths
        # specified in config.py for the training to find the audio files.
        classifier.prepare_dataset('urbansound8k')
        classifier.prepare_dataset('esc50')

        # --- Call the REAL train_model method ---
        logger.info("Calling real train_model method to begin training...")

        # You can choose to train on UrbanSound8K, ESC-50, or combine them
        # (if your data loading logic in train_model handles combining multiple datasets).
        # For simplicity, we'll start with UrbanSound8K.
        classifier.train_model(dataset_path=URBANSOUND8K_PATH, epochs=20, batch_size=16) # Increased epochs, reduced batch size for potentially better generalization

        # If you also want to train on ESC-50 (or train a separate model for it):
        # logger.info("Calling real train_model method for ESC-50...")
        # classifier.train_model(dataset_path=ESC50_PATH, epochs=20, batch_size=16)

        logger.info("AI model training script finished.")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)

if __name__ == "__main__":
    run_training()
