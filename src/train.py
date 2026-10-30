import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load dataset

df = pd.read_csv('data/diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)   

# Train model

model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, predictions))

# Save model

joblib.dump(model, 'models/disease_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
