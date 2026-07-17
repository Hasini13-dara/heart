import streamlit as st
import joblib
import numpy as np

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# ---------------------------
# Load Model and Scaler
# ---------------------------
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------
# Title
# ---------------------------
st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient's health details below to predict the likelihood of heart disease.")

# ---------------------------
# User Inputs
# ---------------------------

age = st.number_input("Age", min_value=1, max_value=120, value=45)

sex = st.selectbox(
    "Sex",
    ["Female", "Male"]
)
sex = 1 if sex == "Male" else 0

cp = st.selectbox(
    "Chest Pain Type (cp)",
    [0, 1, 2, 3]
)

trestbps = st.number_input(
    "Resting Blood Pressure (trestbps)",
    min_value=80,
    max_value=250,
    value=120
)

chol = st.number_input(
    "Serum Cholesterol (chol)",
    min_value=100,
    max_value=600,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl (fbs)",
    [0, 1]
)

restecg = st.selectbox(
    "Resting ECG Results (restecg)",
    [0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina (exang)",
    [0, 1]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope",
    [0, 1, 2]
)

ca = st.selectbox(
    "Number of Major Vessels (ca)",
    [0, 1, 2, 3, 4]
)

thal = st.selectbox(
    "Thal",
    [0, 1, 2, 3]
)

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict"):

    input_data = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Probability (if supported)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)
        confidence = np.max(probability) * 100

    st.markdown("---")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    if hasattr(model, "predict_proba"):
        st.write(f"**Prediction Confidence:** {confidence:.2f}%")
