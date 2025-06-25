// frontend/src/App.js
import React, { useState, useEffect, useRef, useCallback } from 'react'; // Added useCallback for memoization
import {
  enrollUser,
  authenticateUser,
  startMonitoring,
  stopMonitoring,
  getLogs,
  getEnrolledUsers,
  getProtectedApps,
  transcribeOtherAudio,
  listAudioDevices, // Added listAudioDevices import
} from './api.js'; // Corrected: Added .js extension to the import path

// --- Reusable Button Component ---
const Button = ({ onClick, children, className = '', disabled = false, variant = 'primary' }) => {
  const baseClasses = 'px-4 py-2 rounded-lg font-medium transition-all duration-150 focus:outline-none focus:ring-2';
  
  const variantClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500',
    secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-800 focus:ring-gray-400',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500',
    success: 'bg-green-600 hover:bg-green-700 text-white focus:ring-green-500',
    disabled: 'bg-gray-400 text-gray-700 cursor-not-allowed'
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${variantClasses[disabled ? 'disabled' : variant]} ${className}`}
    >
      {children}
    </button>
  );
};

// --- Card Component (for consistent UI styling) ---
const Card = ({ children, title, className = '' }) => (
  <div className={`bg-white rounded-lg shadow-md border border-gray-200 p-4 ${className}`}>
    {title && <h2 className="text-xl font-bold text-gray-800 mb-3">{title}</h2>}
    {children}
  </div>
);

// --- StatusMessage Component (for displaying feedback) ---
const StatusMessage = ({ message, type = 'info', onClose }) => {
  if (!message) return null;

  const typeClasses = {
    success: 'bg-green-100 border-green-400 text-green-700',
    error: 'bg-red-100 border-red-400 text-red-700',
    info: 'bg-blue-100 border-blue-400 text-blue-700'
  };

  return (
    <div className={`${typeClasses[type]} border px-4 py-3 rounded mb-4 relative`}>
      <span className="block sm:inline">{message}</span>
      <button onClick={onClose} className="absolute top-1 right-1 px-1">
        <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
        </svg>
      </button>
    </div>
  );
};


// --- AudioVisualizer Component (replaced SpectrogramDisplay) ---
const AudioVisualizer = ({ isActive, deviceId }) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceRef = useRef(null);
  const streamRef = useRef(null); // Ref to hold the MediaStream

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;

    // Only draw if analyser is valid
    if (!analyser) {
        if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
        }
        ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas if no analyser
        return;
    }
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    analyser.getByteFrequencyData(dataArray);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw frequency bars
    const barWidth = (canvas.width / bufferLength) * 2.5;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const barHeight = (dataArray[i] / 255) * canvas.height;
      const hue = i / bufferLength * 360;
      ctx.fillStyle = `hsl(${hue}, 80%, 50%)`;
      ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
      x += barWidth + 1;
    }

    animationRef.current = requestAnimationFrame(draw);
  }, []);

  useEffect(() => {
    const initAudio = async () => {
      try {
        // Cleanup existing audio resources before starting new ones
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current);
          animationRef.current = null;
        }
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        if (sourceRef.current) {
          sourceRef.current.disconnect();
          sourceRef.current = null;
        }
        if (analyserRef.current) {
          analyserRef.current.disconnect();
          analyserRef.current = null;
        }
        if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
          await audioContextRef.current.close();
          audioContextRef.current = null;
        }
       
        // Always explicitly request audio: true.
        // If deviceId is provided, nest it under the audio constraint.
        const constraints = { audio: deviceId ? { deviceId: { exact: deviceId } } : true };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        streamRef.current = stream; // Store the new stream
        
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
        analyserRef.current = audioContextRef.current.createAnalyser();
        analyserRef.current.fftSize = 256;
        
        sourceRef.current = audioContextRef.current.createMediaStreamSource(streamRef.current);
        sourceRef.current.connect(analyserRef.current);
        
        draw(); // Start the animation loop
      } catch (err) {
        console.error("Audio initialization error for AudioVisualizer:", err);
        // Ensure microphone is released even on error
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
        // Clear canvas on error
        const canvas = canvasRef.current;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        // Optionally, inform parent component or display error on UI
      }
    };

    if (isActive) {
      initAudio();
    } else {
      // Clean up when isActive becomes false
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
        sourceRef.current = null;
      }
      if (analyserRef.current) {
        analyserRef.current.disconnect();
        analyserRef.current = null;
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().then(() => {
          audioContextRef.current = null;
        });
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      // Clear canvas when inactive
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    // Cleanup on component unmount
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (sourceRef.current) {
        sourceRef.current.disconnect();
      }
      if (analyserRef.current) {
        analyserRef.current.disconnect();
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
    };
  }, [isActive, deviceId, draw]);

  return (
    <canvas 
      ref={canvasRef} 
      width="300" 
      height="120"
      className="w-full rounded bg-gray-900"
    />
  );
};


// --- AuthForm Component ---
const AuthForm = ({ 
  onSuccess, 
  onError, 
  enrolledUsers, 
  protectedApps, 
  mode = 'enroll',
  onModeChange,
  isInitialSetup = false
}) => {
  const [username, setUsername] = useState('');
  const [challengePhrase, setChallengePhrase] = useState('My voice is my password');
  const [selectedApp, setSelectedApp] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null); // To store the microphone stream

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream; // Store stream for cleanup
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        audioChunksRef.current = [];
        // Add null check before accessing streamRef.current.getTracks()
        if (streamRef.current) { 
          streamRef.current.getTracks().forEach(track => track.stop()); // Stop stream after recording
          streamRef.current = null;
        }
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAudioBlob(null); // Clear previous audio
    } catch (err) {
      onError(`Microphone access error: ${err.message}`);
      // Ensure stream is stopped on error
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
    // Ensure stream is stopped if recording stops manually
    if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
    }
  };

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const handleSubmit = async () => {
    if (!username || !audioBlob || !challengePhrase) {
      onError('Please fill all fields and record your voice');
      return;
    }

    try {
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      
      reader.onloadend = async () => {
        const base64Audio = reader.result.split(',')[1];
        let response;
        
        if (mode === 'enroll') {
          response = await enrollUser(username, base64Audio, challengePhrase);
        } else {
          response = await authenticateUser(username, base64Audio, challengePhrase, selectedApp);
        }
        
        onSuccess(response.message);
        setUsername('');
        setAudioBlob(null);
      };
    } catch (err) {
      onError(`Authentication error: ${err.message}`);
    }
  };

  return (
    <Card title={isInitialSetup ? "Initial Setup" : mode === 'enroll' ? "Enroll User" : "Authenticate"}>
      {!isInitialSetup && (
        <div className="flex mb-3 space-x-2">
          <Button 
            onClick={() => onModeChange('enroll')} 
            variant={mode === 'enroll' ? 'primary' : 'secondary'}
            className="flex-1"
          >
            Enroll
          </Button>
          <Button 
            onClick={() => onModeChange('authenticate')} 
            variant={mode === 'authenticate' ? 'primary' : 'secondary'}
            className="flex-1"
          >
            Authenticate
          </Button>
        </div>
      )}

      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            placeholder="Enter username"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Challenge Phrase</label>
          <input
            type="text"
            value={challengePhrase}
            onChange={(e) => setChallengePhrase(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            placeholder="Speak this phrase"
          />
        </div>

        {mode === 'authenticate' && protectedApps.length > 0 && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Application to Launch</label>
            <select
              value={selectedApp}
              onChange={(e) => setSelectedApp(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">None</option>
              {protectedApps.map(app => (
                <option key={app} value={app}>{app}</option>
              ))}
            </select>
          </div>
        )}

        <div className="pt-2">
          <AudioVisualizer isActive={isRecording} />
          <div className="flex space-x-2 mt-2">
            <Button
              onClick={isRecording ? stopRecording : startRecording}
              variant={isRecording ? 'danger' : 'primary'}
              className="flex-1"
            >
              {isRecording ? 'Stop Recording' : 'Start Recording'}
            </Button>
            {audioBlob && (
              <audio 
                src={URL.createObjectURL(audioBlob)} 
                controls 
                className="flex-1"
              />
            )}
          </div>
        </div>

        <Button
          onClick={handleSubmit}
          disabled={!username || !audioBlob || !challengePhrase}
          className="w-full mt-3"
        >
          {mode === 'enroll' ? 'Enroll User' : 'Authenticate'}
        </Button>
      </div>
    </Card>
  );
};

const MonitorControl = ({ isActive, onStart, onStop, selectedDevice, onDeviceChange, devices }) => {
  return (
    <Card title="Real-time Monitoring">
      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Audio Input</label>
          <select
            value={selectedDevice}
            onChange={(e) => onDeviceChange(e.target.value)}
            disabled={isActive}
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100"
          >
            <option value="">Default</option>
            {devices.map(device => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `Microphone ${device.deviceId.slice(0, 5)}`}
              </option>
            ))}
          </select>
        </div>

        <AudioVisualizer isActive={isActive} deviceId={selectedDevice} /> {/* Pass deviceId to AudioVisualizer */}

        <div className="flex space-x-2">
          <Button
            onClick={onStart}
            disabled={isActive}
            variant="success"
            className="flex-1"
          >
            Start Monitoring
          </Button>
          <Button
            onClick={onStop}
            disabled={!isActive}
            variant="danger"
            className="flex-1"
          >
            Stop Monitoring
          </Button>
        </div>

        {isActive && (
          <div className="flex items-center justify-center text-green-600 font-medium">
            <svg className="w-5 h-5 mr-2 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
            </svg>
            Monitoring Active
          </div>
        )}
      </div>
    </Card>
  );
};

const STTForm = ({ onSuccess, onError, selectedDevice }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [transcription, setTranscription] = useState('');
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null); // To store the microphone stream

  const startRecording = async () => {
    try {
      // Always explicitly request audio: true.
      // If selectedDevice is provided, nest it under the audio constraint.
      const constraints = { audio: selectedDevice ? { deviceId: { exact: selectedDevice } } : true };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream; // Store stream for cleanup
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };
      
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setAudioBlob(blob);
        audioChunksRef.current = [];
        // Add null check before accessing streamRef.current.getTracks()
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop()); // Stop stream after recording
          streamRef.current = null;
        }
      };
      
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAudioBlob(null); // Clear previous audio
      setTranscription(''); // Clear previous transcription
    } catch (err) {
      onError(`Microphone access error: ${err.message}`);
       // Ensure stream is stopped on error
       if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
     // Ensure stream is stopped if recording stops manually
     if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
    }
  };

   // Cleanup effect
   useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const handleTranscribe = async () => {
    if (!audioBlob) {
      onError('Please record audio first');
      return;
    }

    try {
      setTranscription('Processing...');
      const reader = new FileReader();
      reader.readAsDataURL(audioBlob);
      
      reader.onloadend = async () => {
        const base64Audio = reader.result.split(',')[1];
        const response = await transcribeOtherAudio(base64Audio);
        setTranscription(response.transcribed_text || 'No speech detected');
        onSuccess('Audio transcribed successfully');
      };
    } catch (err) {
      onError(`Transcription error: ${err.message}`);
      setTranscription('');
    }
  };

  return (
    <Card title="Speech-to-Text">
      <div className="space-y-3">
        <AudioVisualizer isActive={isRecording} deviceId={selectedDevice} /> {/* Pass selectedDevice */}
        
        <div className="flex space-x-2">
          <Button
            onClick={isRecording ? stopRecording : startRecording}
            variant={isRecording ? 'danger' : 'primary'}
            className="flex-1"
          >
            {isRecording ? 'Stop Recording' : 'Start Recording'}
          </Button>
          {audioBlob && (
            <audio 
              src={URL.createObjectURL(audioBlob)} 
              controls 
              className="flex-1"
            />
          )}
        </div>

        <Button
          onClick={handleTranscribe}
          disabled={!audioBlob}
          className="w-full"
        >
          Transcribe Audio
        </Button>

        {transcription && (
          <div className="mt-3 p-3 bg-gray-50 rounded border border-gray-200">
            <h3 className="font-medium text-gray-700 mb-1">Transcription:</h3>
            <p className="text-gray-800">{transcription}</p>
          </div>
        )}
      </div>
    </Card>
  );
};

const LogViewer = ({ logs }) => {
  const logContainerRef = useRef(null);

  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  const getLogColor = (type) => {
    switch (type) {
      case 'error': return 'bg-red-50 border-red-200 text-red-800';
      case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'success': return 'bg-green-50 border-green-200 text-green-800';
      case 'threat_warning': return 'bg-orange-100 border-orange-200 text-orange-800'; // Specific color for threat warnings
      case 'environmental_warning': return 'bg-purple-100 border-purple-200 text-purple-800'; // Specific color for env warnings
      case 'speech_recognition': return 'bg-blue-50 border-blue-200 text-blue-800'; // Specific color for STT
      default: return 'bg-gray-50 border-gray-200 text-gray-800'; // Default for info or unknown
    }
  };
  

  return (
    <Card title="Activity Logs" className="h-full">
      <div 
        ref={logContainerRef}
        className="h-64 overflow-y-auto space-y-2 pr-2"
      >
        {logs.length === 0 ? (
          <p className="text-gray-500 text-center py-8">No activity logs yet</p>
        ) : (
          logs.map((log, index) => (
            <div 
              key={`${log.timestamp}-${index}`}
              className={`p-2 rounded border ${getLogColor(log.type)}`}
            >
              <div className="flex justify-between items-start">
                <span className="text-sm">{log.message}</span>
                <span className="text-xs text-gray-500 whitespace-nowrap ml-2">
                  {new Date(log.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>
            </div>
          ))
        )}
      </div>
    </Card>
  );
};

// --- Main App Component ---
function App() {
  const [activeTab, setActiveTab] = useState('auth');
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('info');
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [logs, setLogs] = useState([]);
  const [enrolledUsers, setEnrolledUsers] = useState([]);
  const [protectedApps, setProtectedApps] = useState([]);
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedDevice, setSelectedDevice] = useState('');
  const [authMode, setAuthMode] = useState('enroll');

  const showMessage = (msg, type = 'info') => {
    setMessage(msg);
    setMessageType(type);
    setTimeout(() => setMessage(''), 5000);
  };

  const fetchData = async () => {
    try {
      const [usersRes, appsRes, logsRes, devicesRes] = await Promise.all([
        getEnrolledUsers(),
        getProtectedApps(),
        getLogs(),
        listAudioDevices()
      ]);

      if (usersRes.status === 'success') setEnrolledUsers(usersRes.users);
      if (appsRes.status === 'success') setProtectedApps(appsRes.apps);
      if (logsRes.status === 'success') setLogs(logsRes.logs);
      if (devicesRes.status === 'success') {
        setAudioDevices(devicesRes.devices);
        if (devicesRes.devices.length > 0) {
          // Find the default device if available, otherwise pick the first one
          const defaultDevice = devicesRes.devices.find(d => d.deviceId === 'default') || devicesRes.devices[0];
          setSelectedDevice(defaultDevice.deviceId);
        }
      }
    } catch (err) {
      showMessage(`Failed to fetch data: ${err.message}`, 'error');
    }
  };

  const handleStartMonitoring = async () => {
    try {
      const response = await startMonitoring(selectedDevice || null);
      if (response.status === 'success') {
        setIsMonitoring(true);
        showMessage('Monitoring started', 'success');
      }
    }
    catch (err) {
      showMessage(`Failed to start monitoring: ${err.message}`, 'error');
    }
  };

  const handleStopMonitoring = async () => {
    try {
      const response = await stopMonitoring();
      if (response.status === 'success') {
        setIsMonitoring(false);
        showMessage('Monitoring stopped', 'success');
      }
    } catch (err) {
      showMessage(`Failed to stop monitoring: ${err.message}`, 'error');
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const isInitialSetup = enrolledUsers.length === 0;

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto py-4 px-4 sm:px-6 lg:px-8">
          <h1 className="text-2xl font-bold text-gray-900">Voice Security System</h1>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <StatusMessage message={message} type={messageType} onClose={() => setMessage('')} />

        {isInitialSetup && (
          <div className="mb-4 bg-yellow-50 border-l-4 border-yellow-400 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-yellow-700">
                  Please complete the initial setup by enrolling your voice.
                </p>
              </div>
            </div>
          </div>
        )}

        {!isInitialSetup && (
          <div className="flex space-x-2 mb-4 overflow-x-auto pb-2">
            <Button
              onClick={() => setActiveTab('auth')}
              variant={activeTab === 'auth' ? 'primary' : 'secondary'}
            >
              Authentication
            </Button>
            <Button
              onClick={() => setActiveTab('monitor')}
              variant={activeTab === 'monitor' ? 'primary' : 'secondary'}
            >
              Monitoring
            </Button>
            <Button
              onClick={() => setActiveTab('stt')}
              variant={activeTab === 'stt' ? 'primary' : 'secondary'}
            >
              Speech-to-Text
            </Button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {isInitialSetup ? (
            <AuthForm
              onSuccess={(msg) => {
                showMessage(msg, 'success');
                fetchData();
              }}
              onError={(msg) => showMessage(msg, 'error')}
              enrolledUsers={enrolledUsers}
              protectedApps={protectedApps}
              mode="enroll"
              isInitialSetup={true}
            />
          ) : activeTab === 'auth' ? (
            <AuthForm
              onSuccess={(msg) => {
                showMessage(msg, 'success');
                fetchData();
              }}
              onError={(msg) => showMessage(msg, 'error')}
              enrolledUsers={enrolledUsers}
              protectedApps={protectedApps}
              mode={authMode}
              onModeChange={setAuthMode}
            />
          ) : activeTab === 'monitor' ? (
            <MonitorControl
              isActive={isMonitoring}
              onStart={handleStartMonitoring}
              onStop={handleStopMonitoring}
              selectedDevice={selectedDevice}
              onDeviceChange={setSelectedDevice}
              devices={audioDevices}
            />
          ) : (
            <STTForm
              onSuccess={(msg) => showMessage(msg, 'success')}
              onError={(msg) => showMessage(msg, 'error')}
              selectedDevice={selectedDevice}
            />
          )}

          <LogViewer logs={logs} />
        </div>
      </main>
    </div>
  );
}

export default App;
