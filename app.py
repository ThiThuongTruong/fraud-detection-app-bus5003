# streamlit_app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import shap

# Load models
scaler = joblib.load('scaler.pkl')
rf_model = joblib.load('rf_model.pkl')
autoencoder = load_model('autoencoder_model.h5', custom_objects={'mse': MeanSquaredError()})

st.title("Fraud Detection Dashboard")

# User Input
st.header("Enter Provider Data")

features = {
    'ClaimID': st.number_input('Claim Count', min_value=0),
    'BeneID': st.number_input('Unique Patients'),
    'InscClaimAmtReimbursed': st.number_input('Reimbursed Amount'),
    'DeductibleAmtPaid': st.number_input('Deductible Paid'),
    'AdmitForDays': st.number_input('Admission Days'),
    'IPAnnualReimbursementAmt': st.number_input('IP Annual Reimbursed'),
    'IPAnnualDeductibleAmt': st.number_input('IP Annual Deductible'),
    'OperatingPhysician': st.number_input('Operating Physician Count'),
    'AttendingPhysician': st.number_input('Attending Physician Count'),
    'OtherPhysician': st.number_input('Other Physician Count'),
}

# Predict button
if st.button("Detect Fraud"):
    input_df = pd.DataFrame([features])
    X_scaled = scaler.transform(input_df)

    # Autoencoder
    recon = autoencoder.predict(X_scaled)
    recon_error = np.mean(np.square(X_scaled - recon))

    # RF Prediction
    rf_prob = rf_model.predict_proba(X_scaled)[:, 1][0]
    rf_pred = int(rf_prob > 0.5)

    st.subheader("Result:")
    st.write(f"🔍 Fraud Probability: **{rf_prob:.2f}**")
    st.write(f"🧠 Reconstruction Error (AE): **{recon_error:.4f}**")

    if rf_pred == 1 or recon_error > 0.95:  # example threshold
        st.error("🚨 Likely Fraudulent")
    else:
        st.success("✅ Provider looks clean")

    # SHAP explain
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled)
    shap_df = pd.DataFrame(shap_values[1], columns=input_df.columns)

    st.subheader("Top Contributing Factors")
    st.dataframe(shap_df.T.rename(columns={0: 'SHAP Value'}).sort_values('SHAP Value', key=abs, ascending=False))
