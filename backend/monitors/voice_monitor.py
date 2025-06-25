# backend/monitors/voice_monitor.py
import queue
import time
import numpy as np
import speech_recognition as sr
import logging
import random # For alert IDs
import io
import soundfile as sf # For converting numpy array to WAV bytes for SpeechRecognition

from security.voice_authenticator import VoiceAuthenticator
from config import (
    SAMPLE_RATE,
    AUDIO_SEGMENT_LENGTH_SECONDS,
    THREAT_KEYWORDS
)

logger = logging.getLogger(__name__)

class VoiceSecurityMonitor:
    def __init__(self, log_queue: queue.Queue):
        """
        Initializes the VoiceSecurityMonitor.
        This monitor now receives audio chunks directly via process_audio_chunk.

        Args:
            log_queue (queue.Queue): A queue to send detected speech/threat logs/warnings to.
        """
        self.log_queue = log_queue
        self.recognizer = sr.Recognizer() # Initialize SpeechRecognition instance
        self.voice_authenticator = VoiceAuthenticator() # Instance of our voice authentication logic
        self.is_processing_active = False # Flag to control if processing logic should run
        self.accumulated_audio = np.array([]) # Buffer for accumulating audio for STT/Auth
        self.last_process_time = time.time() # To track when the last audio segment was processed
        logger.info("VoiceSecurityMonitor initialized.")

    def process_audio_chunk(self, audio_chunk: np.ndarray, samplerate: int):
        """
        Processes a single audio chunk for Speech-to-Text and threat keyword detection.
        This method is called by the central `audio_processing_thread_func` in `app.py`.
        It accumulates audio and processes it at fixed intervals.

        Args:
            audio_chunk (np.ndarray): The raw audio samples for the current chunk.
            samplerate (int): The sample rate of the audio_chunk.
        """
        if not self.is_processing_active:
            return # Do nothing if processing is not active

        # Append to accumulated audio buffer
        # Ensure audio_chunk is float32 to match accumulated_audio dtype
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        self.accumulated_audio = np.concatenate((self.accumulated_audio, audio_chunk))

        # Process accumulated audio every AUDIO_SEGMENT_LENGTH_SECONDS
        current_time = time.time()
        target_length_samples = int(SAMPLE_RATE * AUDIO_SEGMENT_LENGTH_SECONDS)

        # Process if enough audio accumulated OR if segment time has passed and some audio is available
        if (current_time - self.last_process_time) >= AUDIO_SEGMENT_LENGTH_SECONDS:
            if len(self.accumulated_audio) >= target_length_samples:
                segment_to_process = self.accumulated_audio[:target_length_samples]
                self._process_audio_segment_for_stt(segment_to_process, samplerate)
                # Keep the remaining audio for the next segment
                self.accumulated_audio = self.accumulated_audio[target_length_samples:]
            elif len(self.accumulated_audio) > 0: # Process if time passed and some audio but not full segment
                segment_to_process = self.accumulated_audio
                self._process_audio_segment_for_stt(segment_to_process, samplerate)
                self.accumulated_audio = np.array([]) # Clear buffer after processing a partial segment
            else:
                logger.debug(f"VoiceSecurityMonitor: No accumulated audio to process after timeout.")

            self.last_process_time = current_time # Reset timer for consistent processing


    def _process_audio_segment_for_stt(self, audio_segment: np.ndarray, samplerate: int):
        """
        Processes an accumulated audio segment for Speech-to-Text and threat keyword detection.
        """
        if audio_segment.size == 0:
            logger.debug("VoiceSecurityMonitor: Received empty audio segment for STT. Skipping.")
            if self.is_processing_active: # Only log to frontend if monitoring is active
                self.log_queue.put({"type": "info", "message": "STT: Empty audio segment. No speech detected.", "timestamp": time.time()})
            return

        # Ensure audio segment is float32 and within expected range for soundfile
        if audio_segment.dtype != np.float32:
            audio_segment = audio_segment.astype(np.float32)

        # Normalize to -1.0 to 1.0 range (though preprocess_audio should handle this)
        # This is an extra safeguard right before writing to WAV
        peak = np.max(np.abs(audio_segment))
        if peak > 0:
            audio_segment = audio_segment / peak
        else:
            logger.warning("VoiceSecurityMonitor: Audio segment is silent (all zeros) for STT. Skipping transcription.")
            if self.is_processing_active:
                self.log_queue.put({"type": "info", "message": "STT: Silent audio segment. No speech to transcribe.", "timestamp": time.time()})
            return

        try:
            wav_io = io.BytesIO()
            sf.write(wav_io, audio_segment, samplerate, format='wav') # Use provided samplerate
            wav_io.seek(0) # Rewind to the beginning

            logger.debug(f"VoiceSecurityMonitor: WAV bytes created for STT, size: {wav_io.getbuffer().nbytes} bytes")

            with sr.AudioFile(wav_io) as source:
                audio_listened = self.recognizer.record(source)

            logger.debug(f"VoiceSecurityMonitor: Audio data recognized by sr.Recognizer, size: {len(audio_listened.frame_data)} frames")

            transcribed_text = self.recognizer.recognize_google(audio_listened)
            logger.info(f"VoiceSecurityMonitor: Transcribed: '{transcribed_text}'")
            self.log_queue.put({"type": "speech_recognition", "message": f"Transcribed: '{transcribed_text}'", "timestamp": time.time()})

            # Check for threat keywords
            found_threats = [
                keyword for keyword in THREAT_KEYWORDS if keyword.lower() in transcribed_text.lower()
            ]
            if found_threats:
                alert_id = f"THREAT_DETECTED_{int(time.time())}_{random.randint(100, 999)}"
                message = f"Threat keyword(s) detected: {', '.join(found_threats)} in '{transcribed_text}'"
                self.log_queue.put({"type": "threat_warning", "message": message, "timestamp": time.time(), "id": alert_id})
                logger.warning(f"VoiceSecurityMonitor: {message}")

        except sr.UnknownValueError:
            logger.info("VoiceSecurityMonitor: Google Speech Recognition could not understand audio (or no clear speech detected).")
            if self.is_processing_active:
                self.log_queue.put({"type": "info", "message": "STT: No clear speech detected.", "timestamp": time.time()})
        except sr.RequestError as e:
            logger.error(f"VoiceSecurityMonitor: Could not request results from Google Speech Recognition service; {e}. Check internet connection or API limits.", exc_info=True)
            self.log_queue.put({"type": "error", "message": f"STT Service Error: {e}", "timestamp": time.time()})
        except Exception as e:
            logger.error(f"VoiceSecurityMonitor: General error during STT processing: {e}", exc_info=True)
            self.log_queue.put({"type": "error", "message": f"STT Processing Error: {e}", "timestamp": time.time()})

    def enroll_user(self, username: str, audio_array: np.ndarray, samplerate: int, challenge_phrase: str) -> bool:
        """
        Delegates to VoiceAuthenticator for user enrollment.
        """
        logger.info(f"Voice Monitor: Delegating enrollment for user '{username}' to VoiceAuthenticator.")
        success = self.voice_authenticator.enroll_user(username, audio_array, samplerate, challenge_phrase)
        if success:
            self.log_queue.put({"type": "info", "message": f"User '{username}' enrolled.", "timestamp": time.time()})
        else:
            self.log_queue.put({"type": "error", "message": f"User '{username}' enrollment failed.", "timestamp": time.time()})
        return success

    def authenticate_user(self, username: str, audio_array: np.ndarray, samplerate: int, challenge_phrase: str) -> bool:
        """
        Delegates to VoiceAuthenticator for authentication, including challenge phrase verification.
        """
        logger.info(f"Voice Monitor: Delegating authentication for user '{username}' to VoiceAuthenticator with challenge '{challenge_phrase}'.")
        is_authenticated = self.voice_authenticator.authenticate_user(username, audio_array, samplerate, challenge_phrase)
        # Authentication success/failure messages are handled by VoiceAuthenticator and put into log_queue there
        return is_authenticated

    def get_enrolled_users(self) -> list[str]:
        """Returns a list of enrolled usernames from the VoiceAuthenticator."""
        return self.voice_authenticator.get_enrolled_users()

    def enable_processing(self):
        """Enables the internal processing logic for audio chunks."""
        self.is_processing_active = True
        self.accumulated_audio = np.array([]) # Clear buffer on start
        self.last_process_time = time.time()
        logger.info("VoiceSecurityMonitor: Processing enabled.")

    def disable_processing(self):
        """Disables the internal processing logic for audio chunks."""
        self.is_processing_active = False
        self.accumulated_audio = np.array([]) # Clear buffer on stop
        logger.info("VoiceSecurityMonitor: Processing disabled.")
