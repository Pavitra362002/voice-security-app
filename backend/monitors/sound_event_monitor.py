# backend/monitors/sound_event_monitor.py
import queue
import time
import numpy as np
import sounddevice as sd
import logging
import threading
import random # For generating unique alert IDs

from config import SAMPLE_RATE, CHUNK_SIZE, VOLUME_THRESHOLD, CONSECUTIVE_CHUNKS_THRESHOLD

logger = logging.getLogger(__name__)

class SoundEventMonitor:
    def __init__(self, audio_queue: queue.Queue, log_queue: queue.Queue):
        """
        Initializes the SoundEventMonitor.

        Args:
            audio_queue (queue.Queue): A queue to receive raw audio chunks from an audio stream.
            log_queue (queue.Queue): A queue to send detected sound event logs/warnings to.
        """
        self.audio_queue = audio_queue
        self.log_queue = log_queue
        self.running = threading.Event() # Event to signal the monitoring thread to stop
        self.loud_chunks_count = 0 # Counter for consecutive loud audio chunks
        self.stream = None # To hold the sounddevice stream
        logger.info("SoundEventMonitor initialized.")

    def _audio_callback(self, indata, frames, time_info, status):
        """
        This callback function is executed by sounddevice in a separate thread for each audio block.
        It puts raw audio (numpy array) into the audio_queue for processing.
        """
        if status:
            logger.warning(f"SoundDevice callback status: {status}")
        
        # Check if indata is valid and not empty
        if indata.shape[0] > 0:
            # Put the raw audio data and its original sample rate into the queue
            # The original_sr for sounddevice stream is the SAMPLE_RATE from config
            self.audio_queue.put((indata[:, 0], SAMPLE_RATE)) # Assuming mono audio (first channel)
        else:
            logger.debug("Received empty audio indata chunk.")


    def process_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Processes a single audio chunk for sound event detection (e.g., loud noises).

        Args:
            audio_chunk (np.ndarray): A numpy array representing the audio data for the current chunk.
        """
        # Calculate RMS (Root Mean Square) for volume detection
        # This gives a measure of the average amplitude of the audio signal.
        if audio_chunk.size > 0:
            rms = np.sqrt(np.mean(audio_chunk**2))
        else:
            rms = 0.0 # Handle empty chunks gracefully
        
        # logger.debug(f"Sound Event Monitor: RMS = {rms:.4f}") # Uncomment for detailed volume debugging

        # Check if the RMS exceeds the predefined volume threshold
        if rms > VOLUME_THRESHOLD:
            self.loud_chunks_count += 1
            # If a sufficient number of consecutive loud chunks are detected, trigger a warning
            if self.loud_chunks_count >= CONSECUTIVE_CHUNKS_THRESHOLD:
                alert_id = f"SOUND_ALERT_{int(time.time())}_{random.randint(100, 999)}"
                message = f"Loud Noise Detected! RMS: {rms:.4f}"
                self.log_queue.put({"type": "environmental_warning", "message": message, "timestamp": time.time(), "id": alert_id})
                logger.warning(f"Sound Event Monitor: {message}")
                self.loud_chunks_count = 0 # Reset counter after triggering an alert
        else:
            self.loud_chunks_count = 0 # Reset if silence is detected

    def start(self, device_index=None):
        """
        Starts monitoring audio from the specified input device.
        """
        if self.running.is_set():
            logger.info("SoundEventMonitor is already running.")
            return

        try:
            # Use a context manager for the stream to ensure it's closed properly
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                channels=1, # Assume mono input for simplicity
                dtype='float32',
                callback=self._audio_callback,
                device=device_index # Pass the selected device index
            )
            self.stream.start()
            self.running.set()
            logger.info(f"SoundEventMonitor started streaming from device {device_index}.")
        except Exception as e:
            self.running.clear()
            self.stream = None
            logger.error(f"Failed to start SoundEventMonitor audio stream on device {device_index}: {e}", exc_info=True)
            self.log_queue.put({"type": "error", "message": f"Sound Monitor failed to start: {e}. Check device/drivers.", "timestamp": time.time(), "id": f"MONITOR_SD_ERR_{int(time.time())}"})
            raise # Re-raise to propagate the error to Flask


    def stop(self):
        """
        Stops monitoring audio.
        """
        if not self.running.is_set():
            logger.info("SoundEventMonitor is not running.")
            return

        self.running.clear()
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("SoundEventMonitor audio stream stopped and closed.")
            except Exception as e:
                logger.error(f"Error stopping SoundEventMonitor stream: {e}", exc_info=True)
            finally:
                self.stream = None
        logger.info("SoundEventMonitor stopped.")

