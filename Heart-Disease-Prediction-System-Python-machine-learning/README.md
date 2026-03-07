# Heart Disease Risk Prediction System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Flask Version](https://img.shields.io/badge/Flask-2.2%2B-green.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit_Learn-orange.svg)

## Overview

This repository contains a full-stack Machine Learning application built with **Flask**. The application leverages a trained **Random Forest Classifier** to predict the 10-year risk of a patient developing heart disease based on their medical background and cardiovascular metrics.

This project demonstrates the end-to-end process of applied machine learning: from data preprocessing and model training to deploying the inference engine as an interactive web service.

## Features

- **Robust Backend ML Pipeline**: Implements pre-trained scaling and inference correctly.
- **Interactive UI**: A modern, mobile-responsive web interface powered by Bootstrap 5 with custom CSS targeting a professional aesthetic.
- **Input Validation & Error Handling**: Enhanced backend parsing with detailed debug logging.
- **Modular Codebase**: Architected cleanly with docstrings, separated static assets, and proper standard dependencies.

## Data Source & Algorithms
The model was trained on the Framingham Heart Study dataset, taking into account the following subset of health metrics:
* Demographics (Age, Gender)
* Behavioral factors (Smoking status, Cigarettes per day)
* Medical history (Stroke, Hypertension, Diabetes, BP Meds)
* Current medical metrics (Total Cholesterol, Systolic/Diastolic BP, BMI, Heart Rate, Glucose)

**Algorithm applied:** Random Forest Classifier (optimized through hyperparameter tuning) with a Standard Scaler for feature normalization.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-link>
   cd Heart-Disease-Prediction-System-Python-machine-learning
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```

5. **Access the Web Interface:**
   Open your browser and navigate to [http://localhost:5000](http://localhost:5000).

## Project Structure
```text
├── app.py                 # Main Flask application and API routes
├── rf_classifier.pkl      # Pre-trained Random Forest model
├── scaler.pkl             # Feature scaler model
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── static/
│   └── css/
│       └── style.css      # Custom UI styling
└── templates/
    └── index.html         # Frontend HTML layout
```

## Future Scope
- Incorporate a PostgreSQL database to securely save user inference history.
- Add user authentication via Flask-Login.
- Visualize input risk factors against aggregated statistics with dynamic charts.