import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
import math

st.title('Diabetes Prediction')
# load model
model = load_model('dw_model.h5')
st.header('Prediction')
col1, col2 = st.columns(2)
with col1:
    pregnant = st.slider('Pregnancies', min_value=0, max_value=13, value=3)
with col2:
    glucose = st.slider('Glucose', min_value=44, max_value=199, value=120)
col3, col4 = st.columns(2)
with col3:
    bp = st.slider('Blood Pressure', min_value=40, max_value=104, value=70)
with col4:
    skin= st.slider('Skin Thickness', min_value=14.5, max_value=42.5, value=20.0, step=0.5)
col5, col6 = st.columns(2)
with col5:
    insulin = st.slider('Insulin', min_value=14, max_value=846, value=79)
with col6:
    bmi = st.slider('BMI', min_value=18.20, max_value=50.25, value=32.0, step=0.05)
col7, col8 = st.columns(2)
with col7:
    dpf = st.slider('Diabetes Pedigree Function', min_value=0.078, max_value=1.200, value=0.3725, step=0.001)
with col8:
    age = st.slider('Age', min_value=21, max_value=66, value=29)
# make prediction
data = np.array([[pregnant, glucose, bp, skin,insulin, bmi, dpf, age]])
prediction = model.predict(data)
if st.button('Predict'):
    if(round(prediction[0][0]) == 0):
        st.success('Prediction: You are not diabetic')
    else:
        st.success('Prediction: You are diabetic')
