from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load('models/disease_model.pkl')
scaler = joblib.load('models/scaler.pkl')

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.get('/')
def home():
    return {'message': 'Disease Prediction API'}

@app.post('/predict')
def predict(data: PatientData):
    features = np.array([[
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]])

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

    return {'prediction': result}
