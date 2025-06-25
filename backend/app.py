# backend/app.py
import os
import sys
import threading
import queue
import time
import logging
import base64
import numpy as np
import soundfile as sf
import io
import subprocess # For launching applications
from pydub import AudioSegment # For audio format conversion
from pydub.utils import get_prober_name # For checking pydub/ffmpeg integration

from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Set up Logging ---
from config import LOGGING_LEVEL, PROTECTED_APPS, SAMPLE_RATE
numeric_level = getattr(logging, LOGGING_LEVEL.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid logging level: {LOGGING_LEVEL}')
logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Correction for Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

logger.info(f"Backend current directory: {current_dir}")
logger.info(f"Python path (sys.path): {sys.path}")

# Import modules after path setup
from monitors.environmental_monitor import EnvironmentalMonitor
from monitors.voice_monitor import VoiceSecurityMonitor
from stt_processor import transcribe_audio
from security.sound_classifier import SoundClassifier

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Global Queues ---
audio_chunk_queue = queue.Queue() # For raw audio chunks from sounddevice to monitors
log_queue = queue.Queue()         # For logs/alerts from monitors to Flask app

# --- Global Monitor Instances ---
environmental_monitor = None
voice_security_monitor = None
sound_classifier = None # Global instance for SoundClassifier

# --- Initialization Flag ---
monitors_initialized = False

# Function to initialize monitors
def initialize_monitors():
    global environmental_monitor, voice_security_monitor, sound_classifier, monitors_initialized

    if not monitors_initialized:
        logger.info("Attempting to initialize security monitors...")
        try:
            # Explicitly set to None before trying to initialize to ensure a clean state
            environmental_monitor = None
            voice_security_monitor = None
            sound_classifier = None

            sound_classifier = SoundClassifier() # Initialize sound classifier
            # EnvironmentalMonitor is now responsible for putting data into audio_chunk_queue
            environmental_monitor = EnvironmentalMonitor(audio_chunk_queue, log_queue, sound_classifier)
            # VoiceSecurityMonitor now processes chunks from the central audio_processing_thread
            # Its __init__ no longer needs audio_chunk_queue as it receives chunks via process_audio_chunk
            voice_security_monitor = VoiceSecurityMonitor(log_queue) 

            monitors_initialized = True
            logger.info("Security monitors initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize security monitors: {e}", exc_info=True)
            monitors_initialized = False # Reset flag if initialization fails
            # Ensure global variables are set to None on failure
            environmental_monitor = None
            voice_security_monitor = None
            sound_classifier = None

# Thread to continuously process the audio_chunk_queue
# This ensures that audio processing doesn't block the sounddevice callback
def audio_processing_thread_func():
    global environmental_monitor, voice_security_monitor
    logger.info("Audio processing thread started.")
    while True:
        try:
            # Get a single audio chunk (numpy array) from the queue with a timeout
            audio_data_chunk = audio_chunk_queue.get(timeout=1)
            if audio_data_chunk is None: # Sentinel value to stop the thread
                logger.info("Audio processing thread received stop signal.")
                break

            # Process with EnvironmentalMonitor if it's active
            if environmental_monitor and environmental_monitor.is_monitoring_active:
                # EnvironmentalMonitor's process_audio_chunk expects only the audio_chunk
                environmental_monitor.process_audio_chunk(audio_data_chunk)
            
            # Process with VoiceSecurityMonitor if it's active
            if voice_security_monitor and voice_security_monitor.is_processing_active:
                # VoiceSecurityMonitor's process_audio_chunk expects the audio_chunk and SAMPLE_RATE
                voice_security_monitor.process_audio_chunk(audio_data_chunk, SAMPLE_RATE)

        except queue.Empty:
            # Queue was empty, continue looping to check for new chunks
            continue
        except Exception as e:
            logger.error(f"Error in audio processing thread: {e}", exc_info=True)
            log_queue.put({"type": "error", "message": f"Audio processing error: {e}", "timestamp": time.time(), "id": f"PROC_ERROR_{int(time.time())}"})

audio_processing_thread = threading.Thread(target=audio_processing_thread_func, daemon=True)
audio_processing_thread.start()


# --- Thread for continuously feeding logs to frontend ---
def log_feeder():
    """Continuously pulls logs from the log_queue and stores them."""
    global current_logs
    while True:
        try:
            log_entry = log_queue.get(timeout=0.1) # Short timeout
            current_logs.append(log_entry)
            if len(current_logs) > 500: # Keep last 500 logs
                current_logs = current_logs[-500:]
        except queue.Empty:
            time.sleep(0.05) # Small delay if queue is empty
        except Exception as e:
            logger.error(f"Error in log_feeder: {e}", exc_info=True)
            time.sleep(1) # Wait before retrying


current_logs = [] # In-memory storage for logs
log_feeder_thread = threading.Thread(target=log_feeder, daemon=True)
log_feeder_thread.start()
logger.info("Log feeder thread started.")


@app.route('/')
def health_check():
    """Basic health check endpoint."""
    return jsonify({"status": "healthy", "message": "Voice Security Backend is running."})

@app.route('/enroll', methods=['POST'])
def enroll_user_endpoint():
    """Endpoint to enroll a new user's voice."""
    data = request.json
    username = data.get('username')
    audio_data_b64 = data.get('audio_data_b64')
    challenge_phrase = data.get('challenge_phrase')

    if not all([username, audio_data_b64, challenge_phrase]):
        return jsonify({"status": "error", "message": "Missing username, audio data, or challenge phrase."}), 400
    if not voice_security_monitor:
        return jsonify({"status": "error", "message": "Voice security monitor not initialized."}), 500

    try:
        audio_bytes_webm = base64.b64decode(audio_data_b64)
        
        # Convert WebM to WAV using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes_webm), format="webm")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0) # Rewind to the beginning
        
        audio_array, samplerate = sf.read(wav_io)

        # Ensure audio is mono (if stereo, take one channel)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        success = voice_security_monitor.enroll_user(username, audio_array, samplerate, challenge_phrase)

        if success:
            return jsonify({"status": "success", "message": f"User '{username}' enrolled successfully."})
        else:
            return jsonify({"status": "error", "message": f"Enrollment for user '{username}' failed. User might already exist or voice not clear."}), 400
    except Exception as e:
        logger.error(f"Error during enrollment: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/authenticate', methods=['POST'])
def authenticate_user_endpoint():
    """Endpoint to authenticate a user's voice."""
    data = request.json
    username = data.get('username')
    audio_data_b64 = data.get('audio_data_b64')
    challenge_phrase = data.get('challenge_phrase')
    app_to_launch = data.get('app_to_launch') # Optional application to launch

    if not all([username, audio_data_b64, challenge_phrase]):
        return jsonify({"status": "error", "message": "Missing username, audio data, or challenge phrase."}), 400
    if not voice_security_monitor:
        return jsonify({"status": "error", "message": "Voice security monitor not initialized."}), 500

    try:
        audio_bytes_webm = base64.b64decode(audio_data_b64)
        
        # Convert WebM to WAV using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes_webm), format="webm")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0) # Rewind to the beginning

        audio_array, samplerate = sf.read(wav_io)

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        is_authenticated = voice_security_monitor.authenticate_user(username, audio_array, samplerate, challenge_phrase)

        if is_authenticated:
            response_message = f"Authentication successful for '{username}'."
            if app_to_launch and app_to_launch in PROTECTED_APPS:
                try:
                    app_path = PROTECTED_APPS[app_to_launch]
                    subprocess.Popen([app_path], shell=True) # Use shell=True for better compatibility on Windows
                    response_message += f" Launching '{app_to_launch}'."
                    logger.info(f"Launched application: {app_path}")
                except Exception as e:
                    response_message += f" Failed to launch '{app_to_launch}': {e}"
                    logger.error(f"Failed to launch app '{app_to_launch}': {e}", exc_info=True)
            return jsonify({"status": "success", "message": response_message})
        else:
            return jsonify({"status": "error", "message": f"Authentication failed for '{username}'. Voice not recognized or challenge phrase mismatch."}), 401
    except Exception as e:
        logger.error(f"Error during authentication: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/start_monitoring', methods=['POST'])
def start_monitoring_endpoint():
    """Endpoint to start real-time audio monitoring."""
    device_id = request.json.get('device_id') # Get device_id string from frontend
    if not environmental_monitor or not voice_security_monitor: # Check both monitors
        return jsonify({"status": "error", "message": "Monitors not initialized."}), 500
    try:
        # EnvironmentalMonitor starts the sounddevice stream and feeds audio_chunk_queue
        environmental_monitor.start_stream(device_id)
        # VoiceSecurityMonitor just needs to be enabled for processing chunks from the central thread
        voice_security_monitor.enable_processing() 
        return jsonify({"status": "success", "message": "Monitoring started."})
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring_endpoint():
    """Endpoint to stop real-time audio monitoring."""
    if not environmental_monitor or not voice_security_monitor:
        return jsonify({"status": "error", "message": "Monitors not initialized."}), 500
    try:
        environmental_monitor.stop_stream()
        voice_security_monitor.disable_processing()
        return jsonify({"status": "success", "message": "Monitoring stopped."})
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_logs', methods=['GET'])
def get_logs_endpoint():
    """Endpoint to retrieve accumulated logs."""
    global current_logs
    return jsonify({"status": "success", "logs": current_logs})


@app.route('/get_enrolled_users', methods=['GET'])
def get_enrolled_users_endpoint():
    """Endpoint to get the list of currently enrolled users."""
    if not voice_security_monitor:
        return jsonify({"status": "error", "message": "Voice security monitor not initialized."}), 500

    enrolled_users = voice_security_monitor.get_enrolled_users()
    logger.debug(f"Returning enrolled users: {enrolled_users}")
    return jsonify({"status": "success", "users": enrolled_users})

@app.route('/get_protected_apps', methods=['GET'])
def get_protected_apps_endpoint():
    """Endpoint to get the list of applications configured for protection/launch."""
    return jsonify({"status": "success", "apps": list(PROTECTED_APPS.keys())})

import sounddevice as sd
@app.route('/list_audio_devices', methods=['GET'])
def list_audio_devices_endpoint():
    """Endpoint to list available audio input devices on the system."""
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if d["maxInputChannels"] > 0:
                # Create a more robust deviceId by combining name and index
                # This helps in unique identification if names are duplicated
                input_devices.append({
                    "index": i,
                    "name": d["name"],
                    "deviceId": f"{d['name']}-{i}", # Unique ID for frontend dropdown
                    "max_input_channels": d["maxInputChannels"],
                    "hostapi_name": sd.query_hostapis(d['hostapi'])['name']
                })
        logger.info(f"Available input devices: {input_devices}")
        return jsonify({"status": "success", "devices": input_devices})
    except Exception as e:
        logger.error(f"Error listing audio devices: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/transcribe_other_audio', methods=['POST'])
def transcribe_other_audio_endpoint():
    """
    API endpoint for general purpose speech-to-text transcription
    without voice authentication or monitoring.
    """
    data = request.json
    audio_data_b64 = data.get('audio_data_b64')

    if not audio_data_b64:
        return jsonify({"status": "error", "message": "Missing audio data for transcription."}), 400

    try:
        audio_bytes_webm = base64.b64decode(audio_data_b64)
        
        # Convert WebM to WAV using pydub
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes_webm), format="webm")
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0) # Rewind to the beginning

        audio_array, samplerate = sf.read(wav_io)

        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        transcribed_text = transcribe_audio(audio_array, samplerate)

        return jsonify({
            "status": "success",
            "message": "Audio transcribed.",
            "transcribed_text": transcribed_text
        })
    except Exception as e:
        logger.error(f"Error during general audio transcription: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    # Initial check for pydub's backend (ffmpeg/libav)
    try:
        prober = get_prober_name()
        logger.info(f"pydub using prober: {prober}")
    except Exception as e:
        logger.warning(f"pydub backend (ffmpeg/libav) not found or configured. Some audio functionalities might fail. Please install ffmpeg and ensure it's in your PATH. Error: {e}")

    # Initialize monitors explicitly at startup
    initialize_monitors() # Moved here to ensure initialization before Flask runs

    # CRITICAL: Check if initialization was successful
    if not monitors_initialized:
        logger.critical("Security monitors failed to initialize. Exiting application to prevent further errors.")
        sys.exit(1) # Exit the application if monitors are not initialized

    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False) # use_reloader=False to prevent multiple monitor instances
