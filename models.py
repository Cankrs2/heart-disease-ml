import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('/Users/cengiz/Desktop/Veri Bilimi Kampi/Heart-Disease-Project/heart.csv')

# Create separate LabelEncoder for each categorical variable
label_encoder_sex = LabelEncoder()
label_encoder_cp = LabelEncoder()
label_encoder_ecg = LabelEncoder()
label_encoder_angina = LabelEncoder()
label_encoder_slope = LabelEncoder()

# Transform categorical variables
df['Sex'] = label_encoder_sex.fit_transform(df['Sex'])
df['ChestPainType'] = label_encoder_cp.fit_transform(df['ChestPainType'])
df['RestingECG'] = label_encoder_ecg.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = label_encoder_angina.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = label_encoder_slope.fit_transform(df['ST_Slope'])

X = df.drop('HeartDisease' , axis=1)
y = df['HeartDisease']

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X,y, test_size=0.3 , random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#1. Random Forest Model
print('Model 1: Random Forest Model')
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled,y_train)
rf_y_pred = rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(rf_y_pred,y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

#2. Decision Tree Model
print('\nModel 2: Decision Tree Model')
dt_model = DecisionTreeClassifier(
    max_depth=5,  # Limit maximum tree depth
    min_samples_split=10,  # Minimum samples required to split a node
    min_samples_leaf=5,  # Minimum samples required in a leaf node
    random_state=42
)
dt_model.fit(X_train_scaled,y_train)
dt_y_pred = dt_model.predict(X_test_scaled)
dt_accuracy = accuracy_score(dt_y_pred,y_test)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_y_pred))

#3. Logistic Regression Model
print("\nModel 3: Logistic Regression")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled,y_train)
lr_y_pred = lr_model.predict(X_test_scaled)
lr_accuracy = accuracy_score(lr_y_pred,y_test)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, lr_y_pred))

# Save models and scaler
joblib.dump(rf, 'rf_model.joblib')
joblib.dump(dt_model, 'dt_model.joblib')
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

# Save Label Encoders
joblib.dump(label_encoder_sex, 'label_encoder_sex.joblib')
joblib.dump(label_encoder_cp, 'label_encoder_cp.joblib')
joblib.dump(label_encoder_ecg, 'label_encoder_ecg.joblib')
joblib.dump(label_encoder_angina, 'label_encoder_angina.joblib')
joblib.dump(label_encoder_slope, 'label_encoder_slope.joblib')

print("\nModels and encoders successfully saved!")