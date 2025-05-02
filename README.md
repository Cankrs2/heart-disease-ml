# Heart Disease Prediction Application

This application is a web-based tool that predicts heart disease risk based on various health parameters. It uses three different machine learning models (Random Forest, Decision Tree, and Logistic Regression) to make predictions and provides an average of their results.

## Features

- User-friendly web interface
- Three different machine learning models for prediction
- Real-time results
- Responsive design

## Installation

1. Install required Python packages:
```bash
pip install -r requirements.txt
```

2. Train the models:
```bash
python models.py
```

3. Start the web application:
```bash
python app.py
```

4. Open your browser and go to `http://localhost:5001`

## Usage

1. Fill in the form fields:
   - Age
   - Sex
   - Chest pain type
   - Resting blood pressure
   - Cholesterol level
   - Fasting blood sugar
   - Resting ECG results
   - Maximum heart rate
   - Exercise-induced angina
   - ST depression
   - ST slope

2. Click "Make Prediction" button.

3. View the results:
   - Individual predictions from each model
   - Average prediction

## Technical Details

- Flask web framework
- Bootstrap 5 UI framework
- jQuery AJAX requests
- Scikit-learn machine learning models
- Joblib model saving/loading

## Model Performance

- Random Forest: 89.13% accuracy
- Decision Tree: 83.70% accuracy
- Logistic Regression: 87.68% accuracy

## Dataset

The application uses the Heart Disease dataset from UCI Machine Learning Repository. The dataset contains various health parameters that are used to predict the presence of heart disease.

## Contributing

Feel free to submit issues and enhancement requests! 