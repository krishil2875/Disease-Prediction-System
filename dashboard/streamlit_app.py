import streamlit as st
import numpy as np
import joblib

model = joblib.load('models/disease_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title('Disease Prediction System')

pregnancies = st.number_input('Pregnancies')
glucose = st.number_input('Glucose')
blood_pressure = st.number_input('Blood Pressure')
skin_thickness = st.number_input('Skin Thickness')
insulin = st.number_input('Insulin')
bmi = st.number_input('BMI')
dpf = st.number_input('Diabetes Pedigree Function')
age = st.number_input('Age')

if st.button('Predict'):
    data = np.array([[pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]])

    scaled = scaler.transform(data)
    prediction = model.predict(scaled)

    if prediction[0] == 1:
        st.error('Patient is likely diabetic')
    else:
        st.success('Patient is not diabetic')
