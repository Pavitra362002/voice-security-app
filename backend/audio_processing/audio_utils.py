# backend/audio_processing/audio_utils.py
import numpy as np
import librosa
import logging

from config import SAMPLE_RATE # Import TARGET_SAMPLE_RATE from config

logger = logging.getLogger(__name__)

def preprocess_audio(audio_data: np.ndarray, original_sr: int) -> np.ndarray:
    """
    Resamples, normalizes, and potentially trims/pads audio data.
    Assumes audio_data is a mono numpy array (float32).
    """
    logger.debug(f"Preprocessing audio: Original SR={original_sr}, Target SR={SAMPLE_RATE}")
    
    # Resample if necessary
    if original_sr != SAMPLE_RATE:
        logger.debug(f"Resampling from {original_sr} Hz to {SAMPLE_RATE} Hz.")
        # Ensure audio_data is float for librosa
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=SAMPLE_RATE)
        logger.debug(f"Resampling complete. New shape: {audio_data.shape}")

    # Normalize audio to prevent clipping and standardize volume
    # Find peak absolute value
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak # Normalize to -1.0 to 1.0 range
        logger.debug("Audio normalized.")
    else:
        logger.warning("Audio data is all zeros or silent, skipping normalization.")

    return audio_data

def pad_audio(audio_data: np.ndarray, target_length_samples: int) -> np.ndarray:
    """
    Pads audio with zeros to reach a target number of samples.
    If audio_data is longer than target_length_samples, it will be truncated.
    """
    if len(audio_data) >= target_length_samples:
        logger.debug(f"Truncating audio from {len(audio_data)} to {target_length_samples} samples.")
        return audio_data[:target_length_samples]
    else:
        pad_width = target_length_samples - len(audio_data)
        logger.debug(f"Padding audio from {len(audio_data)} to {target_length_samples} samples with {pad_width} zeros.")
        return np.pad(audio_data, (0, pad_width), 'constant')

