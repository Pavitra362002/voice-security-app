// frontend/src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
// import reportWebVitals from './reportWebVitals'; // REMOVED: This file is not needed for core functionality

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// REMOVED: If you want to start measuring performance in your app, pass a function
// REMOVED: to log results (for example: reportWebVitals(console.log))
// REMOVED: or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
// reportWebVitals(); // REMOVED: No longer called as the file is not imported
