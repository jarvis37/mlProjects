"""
Heart Disease Risk Prediction App
---------------------------------

A Flask-based Machine Learning web application that predicts a patient's 10-year risk 
of developing heart disease based on their health metrics.
"""

import os
from flask import Flask, request, render_template
import pickle
import numpy as np
import logging

# Configure basic logging for better debugging in production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for the ML models
classifier_model = None
feature_scaler = None

# Attempt to load the models at startup rather than per-request
try:
    with open('rf_classifier.pkl', 'rb') as f:
        classifier_model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        feature_scaler = pickle.load(f)
    logger.info("Successfully loaded ML model and scaler.")
except Exception as e:
    logger.error(f"Error loading models: {e}. Ensure 'rf_classifier.pkl' and 'scaler.pkl' are in the root directory.")

def evaluate_patient_risk(data):
    """
    Parses form data, applies transformations mapping text to numeric,
    scales features, and runs inference through the Random Forest model.
    """
    try:
        # Binary categorical conversions
        is_male = 1 if data.get('male', '').strip().lower() == "male" else 0
        is_smoker = 1 if data.get('currentSmoker', '').strip().lower() == "yes" else 0
        bp_meds = 1 if data.get('BPMeds', '').strip().lower() == "yes" else 0
        prev_stroke = 1 if data.get('prevalentStroke', '').strip().lower() == "yes" else 0
        prev_hyp = 1 if data.get('prevalentHyp', '').strip().lower() == "yes" else 0
        has_diabetes = 1 if data.get('diabetes', '').strip().lower() == "yes" else 0
        
        # Numeric parsing
        age = int(data.get('age', 0))
        cigs_per_day = float(data.get('cigsPerDay', 0.0))
        tot_chol = float(data.get('totChol', 0.0))
        sys_bp = float(data.get('sysBP', 0.0))
        dia_bp = float(data.get('diaBP', 0.0))
        bmi = float(data.get('BMI', 0.0))
        heart_rate = float(data.get('heartRate', 0.0))
        glucose = float(data.get('glucose', 0.0))

        # Create input array matching the training columns:
        # 1. male 2. age 3. currentSmoker 4. cigsPerDay 5. BPMeds 6. prevalentStroke 
        # 7. prevalentHyp 8. diabetes 9. totChol 10. sysBP 11. diaBP 12. BMI 13. heartRate 14. glucose
        features = np.array([[
            is_male, age, is_smoker, cigs_per_day, bp_meds,
            prev_stroke, prev_hyp, has_diabetes, tot_chol, sys_bp, dia_bp,
            bmi, heart_rate, glucose
        ]])

        scaled_features = feature_scaler.transform(features)
        prediction = classifier_model.predict(scaled_features)
        
        return int(prediction[0])
    except ValueError as ve:
        logger.warning(f"Validation Error: Could not parse some fields: {ve}")
        raise ValueError("Invalid numerical input in form fields.")
    except Exception as e:
        logger.error(f"Inference Error: {e}")
        raise

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if classifier_model is None or feature_scaler is None:
            return render_template('index.html', error="Model not loaded on the server.", form_data=request.form)

        try:
            pred_val = evaluate_patient_risk(request.form)
            if pred_val == 1:
                result_msg = "⚠️ High Risk: The patient has a high 10-year risk of developing heart disease."
                alert_type = "danger"
            else:
                result_msg = "✅ Low Risk: The patient has a low probability of developing heart disease."
                alert_type = "success"

            return render_template('index.html', result=result_msg, alert_type=alert_type, form_data=request.form)
        except ValueError as e:
            return render_template('index.html', error=str(e), form_data=request.form)
            
    return render_template('index.html', form_data={})

if __name__ == '__main__':
    # Use the PORT environment variable if it exists (Render requires this)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
