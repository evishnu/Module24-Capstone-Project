
import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.markdown(
    """
    <h2 style='text-align: center; margin-bottom: 0;'>🩺 Diabetic Readmission Prediction</h2>
    <p style='text-align: left; margin-top: 0; font-size: 16px;'>
        Using a Machine Learning Random Forest Model, predict whether a diabetic patient is likely to be readmitted within 30 days.
    </p>
     """,
    unsafe_allow_html=True
)
st.markdown(
    "<span style='color:red; font-weight:bold;'>⚠️ This prediction is part of a course project and is still under analysis. It should not be considered a final conclusion or used for medical decision-making. For any serious health concerns, please consult a licensed physician.</span>", unsafe_allow_html=True )


# Load the trained model and expected feature list
model, feature_names = joblib.load("DIRAPR.pkl")

# Define input form
with st.form("prediction_form"):
    st.subheader("Enter patient information:")

    time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=30, value=3)
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=150, value=40)
    num_medications = st.number_input("Number of Medications", min_value=0, max_value=100, value=10)
    number_inpatient = st.number_input("Number of Inpatient Visits", min_value=0, max_value=20, value=1)
    number_emergency = st.number_input("Number of Emergency Visits", min_value=0, max_value=20, value=0)
    number_outpatient = st.number_input("Number of Outpatient Visits", min_value=0, max_value=20, value=1)
    age = st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                                     "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Other"])

    submit = st.form_submit_button("Predict")

# Process and predict
if submit:
    age_mapping = {"[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
                   "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9}
    gender_mapping = {"Male": 1, "Female": 0}
    race_mapping = {"Caucasian": 0, "AfricanAmerican": 1, "Other": 2}

    input_data = pd.DataFrame([[
        time_in_hospital,
        num_lab_procedures,
        num_medications,
        number_inpatient,
        number_emergency,
        number_outpatient,
        age_mapping[age],
        gender_mapping[gender],
        race_mapping[race]
    ]], columns=[
        "time_in_hospital", "num_lab_procedures", "num_medications",
        "number_inpatient", "number_emergency", "number_outpatient",
        "age", "gender", "race"
    ])

    # Align to expected feature order with missing columns filled with 0
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.write("### Prediction Result:")
    if prediction == 1:
        st.success("✅ Patient likely to be readmitted within 30 days.")
    else:
        st.info("🟢 Patient unlikely to be readmitted within 30 days.")
    st.write(f"🔢 Probability of readmission: **{probability:.2f}**")
