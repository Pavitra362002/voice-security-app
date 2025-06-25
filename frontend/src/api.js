// frontend/src/api.js
// This file centralizes API calls to your Flask backend.

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://127.0.0.1:5000';

/**
 * Helper function for making POST requests to the backend.
 * @param {string} endpoint - The API endpoint (e.g., '/enroll', '/authenticate').
 * @param {object} data - The payload to send in the request body.
 * @returns {Promise<object>} - The JSON response from the backend.
 */
async function postRequest(endpoint, data) {
  try {
    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      // Attempt to parse error message from backend
      const errorData = await response.json().catch(() => ({ message: 'Unknown error occurred.' }));
      throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error in postRequest to ${endpoint}:`, error);
    throw error; // Re-throw to be handled by the calling component
  }
}

/**
 * Helper function for making GET requests to the backend.
 * @param {string} endpoint - The API endpoint (e.g., '/get_logs', '/get_enrolled_users').
 * @returns {Promise<object>} - The JSON response from the backend.
 */
async function getRequest(endpoint) {
  try {
    const response = await fetch(`${BACKEND_URL}${endpoint}`);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ message: 'Unknown error occurred.' }));
      throw new Error(errorData.message || `HTTP error! Status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error(`Error in getRequest to ${endpoint}:`, error);
    throw error; // Re-throw to be handled by the calling component
  }
}

// --- Voice Security System API Functions ---

export const enrollUser = async (username, audioDataBase64, challengePhrase) => {
  return postRequest('/enroll', { 
    username, 
    audio_data_b64: audioDataBase64,
    challenge_phrase: challengePhrase
  });
};

export const authenticateUser = async (username, audioDataBase64, challengePhrase, appToLaunch = null) => {
  return postRequest('/authenticate', {
    username,
    audio_data_b64: audioDataBase64,
    challenge_phrase: challengePhrase,
    app_to_launch: appToLaunch
  });
};

// Changed to accept deviceId string instead of index
export const startMonitoring = async (deviceId = null) => {
  return postRequest('/start_monitoring', {
    device_id: deviceId // Now sending device_id string
  });
};

export const stopMonitoring = async () => {
  return postRequest('/stop_monitoring', {});
};

export const getLogs = async () => {
  return getRequest('/get_logs');
};

export const getEnrolledUsers = async () => {
  return getRequest('/get_enrolled_users');
};

export const getProtectedApps = async () => {
  return getRequest('/get_protected_apps');
};

export const transcribeOtherAudio = async (audioDataBase64) => {
  return postRequest('/transcribe_other_audio', { 
    audio_data_b64: audioDataBase64 
  });
};

/**
 * Lists available audio input devices using browser API
 * @returns {Promise<Array>} - Array of audio input devices
 */
export const listAudioDevices = async () => {
  try {
    // First request microphone permission to get full device labels
    // This is crucial for getting meaningful device labels
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    // Stop tracks immediately after getting permission, as we only need the labels
    stream.getTracks().forEach(track => track.stop());
    
    // Then enumerate devices
    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter(device => device.kind === 'audioinput');
    
    // Add a 'Default' option manually if not already present
    const defaultOption = { deviceId: 'default', label: 'Default Microphone (Browser Selection)', groupId: 'default' };
    const processedDevices = [defaultOption, ...audioInputs.map(device => ({
      deviceId: device.deviceId,
      label: device.label || `Microphone ${device.deviceId.slice(0, 8)}...`, // Fallback label
      groupId: device.groupId
    }))];

    // Filter out duplicates if the default option is also returned by enumerateDevices with a specific ID
    const uniqueDevices = Array.from(new Map(processedDevices.map(item => [item['deviceId'], item])).values());
    
    return {
      status: 'success',
      devices: uniqueDevices
    };
  } catch (error) {
    console.error('Error listing audio devices:', error);
    return {
      status: 'error',
      message: error.message,
      devices: []
    };
  }
};
