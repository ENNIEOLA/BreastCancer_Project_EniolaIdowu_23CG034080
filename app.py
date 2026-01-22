# Breast Cancer Prediction System - Web GUI (Updated)
# =====================================================

import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load('model/breast_cancer_model.pkl')
scaler = joblib.load('model/scaler.pkl')

# Streamlit page configuration
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="🎗️", layout="centered")

# Title
st.title("Breast Cancer Prediction System 🎗️")
st.write("Enter tumor feature values below to predict whether the tumor is Benign or Malignant.")

# User inputs: start at 0 but allow negative and positive values
radius = st.number_input("Mean Radius", min_value=-100.0, max_value=1000.0, value=0.0, step=0.01)
texture = st.number_input("Mean Texture", min_value=-100.0, max_value=1000.0, value=0.0, step=0.01)
perimeter = st.number_input("Mean Perimeter", min_value=-100.0, max_value=1000.0, value=0.0, step=0.01)
area = st.number_input("Mean Area", min_value=-100.0, max_value=10000.0, value=0.0, step=0.01)
smoothness = st.number_input("Mean Smoothness", min_value=-10.0, max_value=10.0, value=0.0, step=0.001)

# Predict button
if st.button("Predict"):
    # Prepare user input for prediction
    user_data = np.array([[radius, texture, perimeter, area, smoothness]])
    user_data_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_data_scaled)
    prediction_proba = model.predict_proba(user_data_scaled)[0]

    # Display result
    result = "Benign ✅" if prediction[0] == 1 else "Malignant ❌"
    confidence = max(prediction_proba) * 100
    st.success(f"Prediction: **{result}** (Confidence: {confidence:.2f}%)")

# Footer
st.markdown("---")
st.markdown("⚠️ **Note:** This tool is strictly for educational purposes and is not a medical diagnostic system.")
