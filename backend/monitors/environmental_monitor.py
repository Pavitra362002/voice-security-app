# backend/monitors/environmental_monitor.py
import queue
import time
import numpy as np
import sounddevice as sd
import logging
import threading
import random

from config import (
    SAMPLE_RATE,
    CHUNK_SIZE,
    VOLUME_THRESHOLD,
    CONSECUTIVE_CHUNKS_THRESHOLD,
    DANGEROUS_SOUND_CATEGORIES,
    AUDIO_SEGMENT_LENGTH_SECONDS # Used for segmenting accumulated audio
)
from audio_processing.audio_utils import preprocess_audio
from security.sound_classifier import SoundClassifier

logger = logging.getLogger(__name__)

class EnvironmentalMonitor:
    def __init__(self, audio_chunk_queue: queue.Queue, log_queue: queue.Queue, sound_classifier: SoundClassifier):
        """
        Initializes the EnvironmentalMonitor.

        Args:
            audio_chunk_queue (queue.Queue): The central queue where raw audio chunks are put by the sounddevice stream.
            log_queue (queue.Queue): A queue to send detected environmental logs/warnings to.
            sound_classifier (SoundClassifier): An instance of the sound classification model.
        """
        self.audio_chunk_queue = audio_chunk_queue # This is where the sounddevice callback will put audio
        self.log_queue = log_queue
        self.sound_classifier = sound_classifier
        self.stream = None # sounddevice input stream
        self.is_monitoring_active = False # Flag to control if processing logic should run
        self.consecutive_loud_chunks = 0
        self.accumulated_audio = np.array([]) # Buffer for accumulating audio for sound classification
        self.last_classification_time = time.time()
        logger.info("EnvironmentalMonitor initialized.")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback function for the sounddevice stream.
        This function runs in a separate thread managed by sounddevice.
        It puts raw audio chunks (numpy array) into the central audio_chunk_queue.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
            # Consider logging to frontend if status indicates a serious issue
            self.log_queue.put({"type": "environmental_warning", "message": f"Audio callback warning: {status}", "timestamp": time.time()})

        # Put the raw audio chunk into the queue for the central processing thread
        # Convert to float32 if not already, as expected by numpy operations downstream
        if indata.dtype != np.float32:
            indata = indata.astype(np.float32)
        
        # Ensure it's mono if stereo, and then copy
        if indata.ndim > 1:
            audio_chunk = indata.mean(axis=1).copy()
        else:
            audio_chunk = indata.copy()

        try:
            self.audio_chunk_queue.put(audio_chunk)
            logger.debug(f"Audio chunk put into queue. Queue size: {self.audio_chunk_queue.qsize()}")
        except queue.Full:
            logger.warning("Audio chunk queue is full. Dropping audio data.")
            self.log_queue.put({"type": "environmental_warning", "message": "Audio buffer overloaded. Data dropped.", "timestamp": time.time()})
        except Exception as e:
            logger.error(f"Error putting audio into queue: {e}", exc_info=True)


    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Processes a single audio chunk for environmental monitoring (volume and sound classification).
        This method is called by the central `audio_processing_thread_func` in `app.py`.
        """
        if not self.is_monitoring_active:
            return # Do nothing if monitoring is not active

        # Volume monitoring
        rms = np.sqrt(np.mean(audio_chunk**2))
        if rms > VOLUME_THRESHOLD:
            self.consecutive_loud_chunks += 1
            logger.debug(f"Loud chunk detected. RMS: {rms:.4f}, Consecutive: {self.consecutive_loud_chunks}")
            if self.consecutive_loud_chunks >= CONSECUTIVE_CHUNKS_THRESHOLD:
                alert_id = f"LOUD_NOISE_{int(time.time())}_{random.randint(100, 999)}"
                self.log_queue.put({"type": "environmental_warning", "message": f"Sustained loud noise detected (RMS: {rms:.4f}).", "timestamp": time.time(), "id": alert_id})
                logger.warning(f"EnvironmentalMonitor: Sustained loud noise detected. RMS: {rms:.4f}")
                self.consecutive_loud_chunks = 0 # Reset after alert to prevent spamming
        else:
            self.consecutive_loud_chunks = 0 # Reset if silence or low volume detected

        # Sound classification (on accumulated segments)
        self.accumulated_audio = np.concatenate((self.accumulated_audio, audio_chunk))
        
        current_time = time.time()
        if (current_time - self.last_classification_time) >= AUDIO_SEGMENT_LENGTH_SECONDS:
            target_length_samples = int(SAMPLE_RATE * AUDIO_SEGMENT_LENGTH_SECONDS)
            if len(self.accumulated_audio) >= target_length_samples:
                segment_to_classify = self.accumulated_audio[:target_length_samples]
                
                predicted_label, confidence = self.sound_classifier.classify_sound(segment_to_classify, SAMPLE_RATE)
                
                if self.sound_classifier.is_dangerous_sound(predicted_label):
                    alert_id = f"DANGEROUS_SOUND_{int(time.time())}_{random.randint(100, 999)}"
                    message = f"Dangerous sound detected: '{predicted_label}' with confidence {confidence:.2f}."
                    self.log_queue.put({"type": "environmental_warning", "message": message, "timestamp": time.time(), "id": alert_id})
                    logger.warning(f"EnvironmentalMonitor: {message}")
                else:
                    logger.info(f"EnvironmentalMonitor: Classified as '{predicted_label}' (Confidence: {confidence:.2f}).")
                    # Optionally log non-dangerous sounds for debugging/full history
                    # self.log_queue.put({"type": "info", "message": f"Sound classified: '{predicted_2label}' ({confidence:.2f}).", "timestamp": time.time()})

                self.accumulated_audio = self.accumulated_audio[target_length_samples:] # Keep remaining audio
            else:
                logger.debug(f"EnvironmentalMonitor: Not enough accumulated audio ({len(self.accumulated_audio)} samples) for classification. Need {target_length_samples} samples.")

            self.last_classification_time = current_time # Reset timer


    def start_stream(self, device_id: str = None):
        """
        Starts the sounddevice audio input stream.
        """
        if self.stream and self.stream.active:
            logger.info("EnvironmentalMonitor: Audio stream is already active.")
            return

        selected_device_index = None
        if device_id and device_id != 'default':
            try:
                import sounddevice as sd
                all_devices = sd.query_devices()
                for i, dev in enumerate(all_devices):
                    if device_id == f"{dev.get('name')}-{i}": 
                        if dev['maxInputChannels'] > 0:
                            selected_device_index = i
                            logger.info(f"Mapped frontend device_id '{device_id}' to sounddevice index {selected_device_index}.")
                            break
            except Exception as e:
                logger.warning(f"Failed to map device_id '{device_id}' to index, attempting with default. Error: {e}")
        
        if selected_device_index is None: # Use sounddevice's default if no specific device or mapping failed
            try:
                import sounddevice as sd
                default_input_device_info = sd.query_devices(kind='input')
                if default_input_device_info:
                    selected_device_index = default_input_device_info['index']
                    logger.info(f"Using sounddevice default input device index: {selected_device_index}")
                else:
                    logger.error("No default input device found by sounddevice.")
                    self.log_queue.put({"type": "error", "message": "No default input device found for monitoring stream.", "timestamp": time.time()})
                    raise ValueError("No default input device found for monitoring stream.")
            except Exception as e:
                logger.error(f"Error querying default sound device for stream: {e}", exc_info=True)
                self.log_queue.put({"type": "error", "message": f"Failed to determine default audio device for stream: {e}", "timestamp": time.time()})
                raise

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                device=selected_device_index,
                channels=1, # Always use mono for processing
                dtype='float32', # Specify dtype
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_monitoring_active = True # Enable processing logic
            logger.info(f"EnvironmentalMonitor: Audio stream started on device index {selected_device_index}.")
            self.log_queue.put({"type": "info", "message": "Environmental monitoring stream started.", "timestamp": time.time()})
        except Exception as e:
            logger.error(f"EnvironmentalMonitor: Error starting audio stream: {e}", exc_info=True)
            self.log_queue.put({"type": "error", "message": f"Failed to start audio stream: {e}", "timestamp": time.time()})
            raise

    def stop_stream(self):
        """
        Stops the sounddevice audio input stream.
        """
        if self.stream and self.stream.active:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_monitoring_active = False # Disable processing logic
            self.consecutive_loud_chunks = 0 # Reset counters
            self.accumulated_audio = np.array([]) # Clear buffer
            logger.info("EnvironmentalMonitor: Audio stream stopped.")
            self.log_queue.put({"type": "info", "message": "Environmental monitoring stream stopped.", "timestamp": time.time()})
        else:
            logger.info("EnvironmentalMonitor: Audio stream is not active.")
