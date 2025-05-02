from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

app = Flask(__name__)

# Load models and scaler
rf_model = joblib.load('rf_model.joblib')
dt_model = joblib.load('dt_model.joblib')
lr_model = joblib.load('lr_model.joblib')
scaler = joblib.load('scaler.joblib')

# Load Label Encoders
label_encoder_sex = joblib.load('label_encoder_sex.joblib')
label_encoder_cp = joblib.load('label_encoder_cp.joblib')
label_encoder_ecg = joblib.load('label_encoder_ecg.joblib')
label_encoder_angina = joblib.load('label_encoder_angina.joblib')
label_encoder_slope = joblib.load('label_encoder_slope.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form
        
        # Transform the data
        features = {
            'Age': float(data['age']),
            'Sex': label_encoder_sex.transform([data['sex']])[0],
            'ChestPainType': label_encoder_cp.transform([data['chest_pain']])[0],
            'RestingBP': float(data['resting_bp']),
            'Cholesterol': float(data['cholesterol']),
            'FastingBS': float(data['fasting_bs']),
            'RestingECG': label_encoder_ecg.transform([data['resting_ecg']])[0],
            'MaxHR': float(data['max_hr']),
            'ExerciseAngina': label_encoder_angina.transform([data['exercise_angina']])[0],
            'Oldpeak': float(data['oldpeak']),
            'ST_Slope': label_encoder_slope.transform([data['st_slope']])[0]
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Scale the data
        input_scaled = scaler.transform(input_df)
        
        # Make predictions
        rf_pred = rf_model.predict_proba(input_scaled)[0][1]
        dt_pred = dt_model.predict_proba(input_scaled)[0][1]
        lr_pred = lr_model.predict_proba(input_scaled)[0][1]
        
        # Calculate average prediction
        avg_pred = (rf_pred + dt_pred + lr_pred) / 3
        
        # Prepare results
        results = {
            'rf_prediction': f'{rf_pred:.2%}',
            'dt_prediction': f'{dt_pred:.2%}',
            'lr_prediction': f'{lr_pred:.2%}',
            'average_prediction': f'{avg_pred:.2%}'
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001) 