# backend/stt_processor.py
import speech_recognition as sr
import numpy as np
import io
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

def transcribe_audio(audio_array: np.ndarray, samplerate: int) -> str:
    """
    Transcribes a given audio numpy array to text using Google Speech Recognition.

    Args:
        audio_array (np.ndarray): The audio data as a NumPy array (float32).
        samplerate (int): The sample rate of the audio data.

    Returns:
        str: The transcribed text, or an empty string if transcription fails.
    """
    r = sr.Recognizer()

    if audio_array.size == 0:
        logger.warning("Attempted to transcribe an empty audio array.")
        return "No audio to transcribe."

    # Convert numpy array to WAV format in memory
    wav_io = io.BytesIO()
    try:
        sf.write(wav_io, audio_array, samplerate, format='WAV')
        wav_io.seek(0)  # Rewind to the beginning of the BytesIO object
    except Exception as e:
        logger.error(f"Error converting audio array to WAV for transcription: {e}", exc_info=True)
        return "Error: Audio format conversion failed."

    try:
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)  # Read the entire audio file
        
        text = r.recognize_google(audio_data)
        logger.info(f"Transcription successful: '{text}'")
        return text
    except sr.UnknownValueError:
        logger.info("Google Speech Recognition could not understand audio (likely no speech).")
        return "No speech detected or unclear audio."
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}", exc_info=True)
        return f"Error: Could not connect to speech recognition service; {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during transcription: {e}", exc_info=True)
        return f"Error: An unexpected transcription error occurred: {e}"

