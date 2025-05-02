import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/cengiz/Desktop/Veri Bilimi Kampi/Heart-Disease-Project/heart.csv')

# df.hist(figsize=(18,12), bins = 20, edgecolor='black')
# plt.suptitle('Veri Dagilimi')
# plt.show()

# plt.figure(figsize=(10,6))
# sns.scatterplot(data=df,x='Age',y="Cholesterol", hue = 'HeartDisease')
# plt.show()

df1 = pd.read_csv('/Users/cengiz/Desktop/Veri Bilimi Kampi/Heart-Disease-Project/heart_duzenlenmis.csv')

plt.figure(figsize=(12,8))
sns.heatmap(df1.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()