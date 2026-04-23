# app.py

import streamlit as st
import numpy as np
import joblib

# LOAD FILES
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

st.set_page_config(page_title="Loan Prediction", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Fill in the details below:")

# ---------------------------
# INPUTS (USING ENCODER VALUES)
# ---------------------------

dependents = st.number_input("Number of Dependents", min_value=0, step=1)

education = st.selectbox(
    "Education",
    encoders["education"].classes_
)

self_employed = st.selectbox(
    "Self Employed",
    encoders["self_employed"].classes_
)

income = st.number_input("Annual Income", min_value=0.0)
loan_amount = st.number_input("Loan Amount", min_value=0.0)
loan_term = st.number_input("Loan Term", min_value=0.0)
cibil = st.number_input("CIBIL Score", min_value=0.0)

res_assets = st.number_input("Residential Assets Value", min_value=0.0)
com_assets = st.number_input("Commercial Assets Value", min_value=0.0)
lux_assets = st.number_input("Luxury Assets Value", min_value=0.0)
bank_assets = st.number_input("Bank Asset Value", min_value=0.0)

# ---------------------------
# ENCODING FUNCTION
# ---------------------------

def encode(col, val):
    return encoders[col].transform([val])[0]

# ---------------------------
# PREDICTION
# ---------------------------

if st.button("Predict Loan Status"):
    try:
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

        # SCALE INPUT
        input_scaled = scaler.transform(input_data)

        # PREDICT
        prediction = model.predict(input_scaled)
        result = encoders["loan_status"].inverse_transform(prediction)[0]

        # DISPLAY RESULT
        if result.lower() == "approved":
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

    except Exception as e:
        st.error(f"Error: {e}")