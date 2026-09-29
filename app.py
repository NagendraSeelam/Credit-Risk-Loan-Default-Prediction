
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_risk_xgboost_pipeline.pkl")

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳"
)

st.title("💳 Credit Risk & Loan Default Prediction")

st.write(
    "Enter applicant information to estimate loan default risk."
)

person_age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

person_income = st.number_input(
    "Annual Income",
    min_value=0.0,
    value=50000.0
)

person_home_ownership = st.selectbox(
    "Home Ownership",
    ["RENT", "OWN", "MORTGAGE", "OTHER"]
)

person_emp_length = st.number_input(
    "Employment Length (Years)",
    min_value=0.0,
    max_value=100.0,
    value=5.0
)

loan_intent = st.selectbox(
    "Loan Intent",
    [
        "PERSONAL",
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
)

loan_grade = st.selectbox(
    "Loan Grade",
    ["A", "B", "C", "D", "E", "F", "G"]
)

loan_amnt = st.number_input(
    "Loan Amount",
    min_value=0.0,
    value=10000.0
)

loan_int_rate = st.number_input(
    "Loan Interest Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=10.0
)

loan_percent_income = st.number_input(
    "Loan Percent of Income",
    min_value=0.0,
    max_value=1.0,
    value=0.20
)

cb_person_default_on_file = st.selectbox(
    "Previous Default on File",
    ["Y", "N"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (Years)",
    min_value=0,
    max_value=100,
    value=5
)

input_data = pd.DataFrame([{
    "person_age": person_age,
    "person_income": person_income,
    "person_home_ownership": person_home_ownership,
    "person_emp_length": person_emp_length,
    "loan_intent": loan_intent,
    "loan_grade": loan_grade,
    "loan_amnt": loan_amnt,
    "loan_int_rate": loan_int_rate,
    "loan_percent_income": loan_percent_income,
    "cb_person_default_on_file": cb_person_default_on_file,
    "cb_person_cred_hist_length": cb_person_cred_hist_length
}])

if st.button("Predict Loan Risk"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Higher Default Risk")
    else:
        st.success("✅ Lower Default Risk")

    st.write(
        f"Estimated Default Probability: **{probability:.2%}**"
    )

    st.progress(float(probability))
