
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("/Users/macair/Movies/fraud_model.pkl")

st.title("Insurance Claim Fraud Detection")
st.write("Enter claim details to predict whether the claim is Genuine or Fraudulent.")

# Demo values
genuine_demo = {
    "months": 328,
    "age": 48,
    "premium": 1406.91,
    "umbrella": 0,
    "total": 71610,
    "injury": 6510,
    "property": 13020,
    "vehicle": 52080,
    "extra": [521585, 1000, 466132, 53300, 0, 5, 1, 1, 2, 2004]
}

fraud_demo = {
    "months": 165,
    "age": 37,
    "premium": 1137.03,
    "umbrella": 0,
    "total": 51590,
    "injury": 9380,
    "property": 9380,
    "vehicle": 32830,
    "extra": [429027, 1000, 603195, 0, 0, 23, 3, 2, 2, 2015]
}

# Session state for values
if "months" not in st.session_state:
    st.session_state.months = 100
    st.session_state.age = 30
    st.session_state.premium = 1000.0
    st.session_state.umbrella = 0.0
    st.session_state.total = 50000.0
    st.session_state.injury = 10000.0
    st.session_state.property = 10000.0
    st.session_state.vehicle = 30000.0
    st.session_state.extra = [0, 1000, 0, 0, 0, 12, 1, 0, 0, 2010]

# Demo buttons
col1, col2 = st.columns(2)

if col1.button("Load Genuine Demo"):
    for key in ["months", "age", "premium", "umbrella", "total", "injury", "property", "vehicle", "extra"]:
        st.session_state[key] = genuine_demo[key]
    st.rerun()

if col2.button("Load Fraud Demo"):
    for key in ["months", "age", "premium", "umbrella", "total", "injury", "property", "vehicle", "extra"]:
        st.session_state[key] = fraud_demo[key]
    st.rerun()

# 8 user inputs
st.number_input("Months as Customer", min_value=0, key="months")

st.number_input("Age", min_value=18, max_value=100, key="age")

st.number_input("Policy Annual Premium", min_value=0.0, key="premium")

st.number_input("Umbrella Limit", min_value=0.0, key="umbrella")

st.number_input("Total Claim Amount", min_value=0.0, key="total")

st.number_input("Injury Claim Amount", min_value=0.0, key="injury")

st.number_input("Property Claim Amount", min_value=0.0, key="property")

st.number_input("Vehicle Claim Amount", min_value=0.0, key="vehicle")

if st.button("Predict"):

    extra = st.session_state.extra

    features = np.array([[
        st.session_state.months,
        st.session_state.age,
        extra[0],
        extra[1],
        st.session_state.premium,
        st.session_state.umbrella,
        extra[2],
        extra[3],
        extra[4],
        extra[5],
        extra[6],
        extra[7],
        extra[8],
        st.session_state.total,
        st.session_state.injury,
        st.session_state.property,
        st.session_state.vehicle,
        extra[9]
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    if prediction == "Y":
        st.error("⚠️ Prediction: Fraudulent Claim")
        st.write(f"Fraud Probability: {probability[1] * 100:.2f}%")
    else:
        st.success("✅ Prediction: Genuine Claim")
        st.write(f"Genuine Probability: {probability[0] * 100:.2f}%")
