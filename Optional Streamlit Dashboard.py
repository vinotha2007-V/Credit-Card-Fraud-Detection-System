import streamlit as st
import joblib
import numpy as np

model = joblib.load("models/fraud_model.pkl")

st.title("💳 Fraud Detection App")

features = st.text_input("Enter transaction features (comma separated)")

if st.button("Predict"):
    data = list(map(float, features.split(",")))
    result = model.predict([data])
    
    if result[0] == 1:
        st.error("Fraud Transaction 🚨")
    else:
        st.success("Legitimate Transaction ✅")
