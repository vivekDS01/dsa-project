# app.py

import streamlit as st
import numpy as np
import joblib

# LOAD MODEL FILES
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.title("🏦 Loan Approval Prediction")

# INPUTS
dependents = st.number_input("Number of Dependents", min_value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")
cibil = st.number_input("CIBIL Score")

res_assets = st.number_input("Residential Assets Value")
com_assets = st.number_input("Commercial Assets Value")
lux_assets = st.number_input("Luxury Assets Value")
bank_assets = st.number_input("Bank Asset Value")

# ENCODING FUNCTION
def encode(col, val):
    return encoders[col].transform([val])[0]

# INPUT ARRAY
input_data = np.array([[ 
    dependents,
    encode("education", education),
    encode("self_employed", self_employed),
    income,
    loan_amount,
    loan_term,
    cibil,
    res_assets,
    com_assets,
    lux_assets,
    bank_assets
]])

# SCALE
input_scaled = scaler.transform(input_data)

# PREDICTION
if st.button("Predict Loan Status"):
    prediction = model.predict(input_scaled)
    result = encoders["loan_status"].inverse_transform(prediction)[0]
    
    if result == "Approved":
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")