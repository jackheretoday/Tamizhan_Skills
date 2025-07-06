import streamlit as st
import pandas as pd
import pickle

# Load trained models
lr_model = pickle.load(open('lr_pipeline.pkl', 'rb'))
rf_model = pickle.load(open('rf_pipeline.pkl', 'rb'))

st.title("📊 Loan Eligibility Predictor")

# User Inputs
loan_id = st.text_input("Loan ID", "SAMPLE_ID")
name = st.text_input("Full Name", "SAMPLE_NAME")
phone_number = st.text_input("Phone Number", "1234567890")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, value=70000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=720)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
loan_amount = st.number_input("Loan Amount", min_value=10000, max_value=10000000, value=350000)

# Prediction Trigger
if st.button("🔮 Predict Loan Status"):

    sample_input = pd.DataFrame([{
        "loan_id": loan_id,
        "name": name,
        "phone_number": phone_number,
        "age": age,
        "income": income,
        "credit_score": credit_score,
        "education": education,
        "loan_amount": loan_amount
    }])

    # prediction
    pred_lr = lr_model.predict(sample_input)[0]
    pred_rf = rf_model.predict(sample_input)[0]

    st.subheader("🔍 Prediction Results")
    st.write(f"📌 Logistic Regression: **{pred_lr}**")
    st.write(f"📌 Random Forest: **{pred_rf}**")
