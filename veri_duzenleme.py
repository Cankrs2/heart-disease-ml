import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


df= pd.read_csv('/Users/cengiz/Desktop/Veri Bilimi Kampi/Heart-Disease-Project/heart.csv')

string_columns = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
numeric_columns = ['Age','RestingBP','Cholesterol','FastingBS','MaxHR','Oldpeak']

le = LabelEncoder()
for i in string_columns:
    df[i] = le.fit_transform(df[i])
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
print(df.head())


df.to_csv('heart_duzenlenmis.csv', index=False)
