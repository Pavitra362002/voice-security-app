Voice Security Application
Project Overview
This application provides a comprehensive voice-based security system. It integrates capabilities for voice authentication, environmental sound classification, and speech-to-text processing to enhance security and monitoring within an environment.

Features
Voice Authentication: Authenticate users based on their unique voice characteristics.
Environmental Sound Classification: Identify specific sound events in the environment (e.g., gunshots, dog barks, car horns) using machine learning.
Speech-to-Text Processing: Convert spoken audio into text for potential command recognition or logging.
Modular Design: Separate backend (Flask, Python) and frontend (React) components for scalability and ease of development.
Project Structure
voice_security_app/
├── backend/
│   ├── app.py                     # Main Flask application entry point
│   ├── config.py                  # Application configuration settings
│   ├── stt_processor.py           # Handles Speech-to-Text functionalities
│   ├── requirements.txt           # Python dependencies for the backend
│   ├── .env.example               # Example for environment variables (copy to .env)
│   ├── init.py                # Python package initializer
│   ├── monitors/
│   │   ├── init.py            # Python package initializer
│   │   ├── environmental_monitor.py # Logic for environmental sound monitoring
│   │   └── voice_monitor.py       # Logic for continuous voice activity monitoring
│   ├── security/
│   │   ├── init.py            # Python package initializer
│   │   ├── voice_authenticator.py # Handles voice registration and authentication
│   │   └── sound_classifier.py    # Classifies environmental sounds
│   ├── audio_processing/
│   │   ├── init.py            # Python package initializer
│   │   └── audio_utils.py         # Utilities for audio manipulation and feature extraction
│   ├── models/                    # Stores trained ML models (main models externally hosted)
│   │   ├── .gitkeep               # Placeholder for Git tracking
│   │   ├── registered_embeddings.pkl # Stores registered voice embeddings
│   │   └── sound_classifier_labels.pkl # Labels for sound classification
│   ├── datasets/                  # Datasets used for training (externally hosted)
│   │   ├── .gitkeep               # Placeholder for Git tracking
│   │   ├── UrbanSound8K/          # UrbanSound8K dataset structure
│   │   │   ├── audio/             # Contains audio files (fold1 to fold10)
│   │   │   └── metadata/
│   │   │       └── UrbanSound8K.csv # Metadata for UrbanSound8K
│   │   └── ESC-50-master/         # ESC-50 dataset structure
│   │       ├── audio/             # Contains audio files
│   │       └── esc50.csv          # Metadata for ESC-50
│   └── pycache/               # (Ignored) Compiled Python bytecode
│
├── frontend/
│   ├── public/                    # Public static assets for React app
│   │   ├── index.html             # Main HTML file
│   │   └── ... (other assets like favicon, logos)
│   ├── src/                       # React source code components and logic
│   │   ├── App.js                 # Main React application component
│   │   ├── api.js                 # API communication with backend
│   │   └── ... (other React files)
│   ├── node_modules/              # (Ignored) Node.js dependencies
│   ├── package.json               # Frontend dependencies and scripts
│   ├── package-lock.json          # Dependency lock file for frontend
│   ├── README.md                  # Frontend specific README (optional)
│   └── .env                       # (Ignored) Frontend environment variables
│
└── README.md                      # Overall project README (this file)

Setup and Installation
Follow these steps to get the Voice Security Application up and running on your local machine.

1. Clone the Repository
First, clone the project from GitHub:
git clone https://github.com/Pavitra362002/voice-security-app.git
cd voice-security-app

2. Download Large Datasets
This project utilizes large datasets (UrbanSound8K and ESC-50) which are too large to be hosted directly on GitHub. Please download them from the links below and place them in the correct directories as instructed.

UrbanSound8K Dataset:

Download Link: https://urbansounddataset.weebly.com/urbansound8k.html
Instructions: After downloading, extract the contents. You should have audio/ and metadata/UrbanSound8K.csv. Place these directly into the backend/datasets/UrbanSound8K/ folder. The final path should look like voice_security_app/backend/datasets/UrbanSound8K/audio/ and voice_security_app/backend/datasets/UrbanSound8K/metadata/.

ESC-50 Dataset:

Download Link: https://github.com/karolpiczak/ESC-50
Instructions: After downloading, extract the contents. You should have audio/ and esc50.csv. Place these directly into the backend/datasets/ESC-50-master/ folder. The final path should look like voice_security_app/backend/datasets/ESC-50-master/audio/ and voice_security_app/backend/datasets/ESC-50-master/esc50.csv.
3. Backend Setup (Python)
Navigate into the backend directory, create a Python virtual environment, install dependencies, and run the Flask application.

cd backend

Create a virtual environment (using the name 'venv_tf_gpu_310' as per original setup)
python -m venv venv_tf_gpu_310

Activate the virtual environment
On Windows PowerShell/CMD:
.\venv_tf_gpu_310\Scripts\activate

On Linux/macOS (Bash/Zsh):
source venv_tf_gpu_310/bin/activate
Install Python dependencies
pip install -r requirements.txt

Training the Sound Classifier Model
Important: This project includes the code necessary to train your own sound classifier model.
After downloading and placing the datasets (UrbanSound8K and ESC-50) as instructed above, you can run the following script to train the model:

python train_model.py

This script will generate the 'sound_classifier_model.h5' file
and 'sound_classifier_labels.pkl' in the 'backend/models/' directory
upon successful training.
Run the Flask backend application
python app.py

The backend server should start, typically listening on [suspicious link removed] or similar.

4. Frontend Setup (React)
Open a new terminal window, navigate into the frontend directory, and install its Node.js dependencies.

cd frontend

Install Node.js dependencies
npm install

Start the React development server
npm start

The frontend application should compile and open in your default web browser, usually at http://localhost:3000.

Usage
Once both the backend and frontend servers are running:

Access the application in your web browser at http://localhost:3000.
[Provide specific instructions on how to use your application, e.g., "Register your voice by navigating to the 'Register' tab and following the prompts. After registration, you can attempt voice authentication on the 'Login' tab."]
[e.g., "Explore the 'Environmental Monitor' section to see real-time sound classification."]
