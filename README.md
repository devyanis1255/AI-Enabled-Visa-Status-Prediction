# AI-Enabled Visa Status Prediction 🌐

![Python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white) 
![React](https://img.shields.io/badge/-React-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white)
![Vite](https://img.shields.io/badge/-Vite-646CFF?logo=vite&logoColor=white)

## 📝 Modern Architecture Overview

**AI-Enabled Visa Status Prediction** is a high-performance machine learning application designed to structurally forecast the procedural processing days and likelihood outcomes for H-1B visa applications.

The ecosystem was recently overhauled from a static monolith into a decoupled, ultra-premium interface. It now leverages **React** and **Framer Motion** for a responsive, interactive UI capable of parsing complex statistical metrics natively, powered exclusively by a blazing-fast **FastAPI** backend routing real-time **XGBoost Regressor** pipeline inferences.

## 🛠️ Technology Ecosystem

### Interface Layer
- **Core Engine:** React 18, Vite JS
- **UX & Motion:** Framer Motion, Vanilla CSS Grid, Glassmorphism design patterns
- **Data Visualization:** Recharts, Lucide React

### Inference Layer
- **Microservice:** Python, FastAPI, Uvicorn
- **Prediction Architectures:** XGBoost Regression, Scikit-learn, Joblib
- **Data Modeling:** Pandas, Numpy, OpenPyXL

## 📁 Intelligent Repository Structure
This repository executes modularity utilizing standardized enterprise categorizations:
```
.
├── backend/            # FastAPI interface orchestrating API routing logic
├── frontend/           # The active React + Vite framework execution core
├── data/               # Robust storage for cleaned datasets & raw CSV/Excel dumps
├── models/             # Encrypted & serialized `.pkl` XGBoost Model files
├── scripts/            # Essential PyData logic maps for EDA and tuning processes
├── visualizations/     # Cached Matplotlib diagnostic rendering plots
├── logs/               # Consolidates historical runtime tracking csv sheets
└── Documents/          # Centralizes your legacy Excel metrics and reporting files
```

## 🚀 How to Run Locally

You will need to activate two simultaneous terminal tabs to bridge the framework instances together.

### 1. Boot up the Intelligence Layer (Backend API)
Navigate to the root directory and activate the core Python environment that mounts the XGBoost memory files.
```bash
# Ensure standard dependencies are installed primarily matching requirements.txt
pip install -r requirements.txt

# Shift scope into the backend cluster and initialize the FastAPI Microservice
cd backend
python api.py
```
*The backend API will establish local system routing on `http://localhost:8000`*

### 2. Boot up the User Interface (React Dashboard)
Leave the backend terminal running and independently open a **new terminal instance** in the root directory.
```bash
# Shift scope into the frontend UX engine
cd frontend

# Parse package binaries from npm
npm install

# Start the Vite deployment network locally
npm run dev
```
*Your Ultra-Premium predictive dashboard will now be fully active and operational at `http://localhost:5173`!*

---

## 📜 Legacy Configuration
For archival tracking purposes, the original flat-state Streamlit application architecture is preserved via earlier codebase instances at:
https://ai-enabled-visa-status-prediction.streamlit.app/

## 🛡️ License
This project complies under standard `LICENSE` protections automatically associated within this GitHub structure.
