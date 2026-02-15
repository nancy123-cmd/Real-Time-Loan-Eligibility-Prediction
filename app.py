import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("loan_model.pkl", "rb"))

st.set_page_config(
    page_title="Loan Approval System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Loan Approval Analysis System")
st.caption("Enter applicant details to check loan eligibility in real time")

st.divider()

# ================= USER INPUT UI ================= #
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Monthly Income (₹)", min_value=0)
    coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0)
    loan_amount = st.number_input("Requested Loan Amount (₹ Thousands)", min_value=0)
    loan_term = st.selectbox("Loan Term (Months)", [120, 180, 240, 360])
    credit_history = st.selectbox("Credit History", ["Good", "Bad"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ================= DATA ENCODING ================= #
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
credit_history = 1 if credit_history == "Good" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# ================= ANALYSIS ================= #
input_data = np.array([[gender, married, dependents, education,
                         self_employed, applicant_income,
                         coapplicant_income, loan_amount,
                         loan_term, credit_history, property_area]])

prediction = model.predict(input_data)[0]
confidence = model.predict_proba(input_data)[0][prediction]

st.divider()

# ================= RESULT ================= #
if prediction == 1:
    st.success("✅ Loan Approved")
    st.metric("Approval Confidence", f"{confidence*100:.2f}%")
else:
    st.error("❌ Loan Not Approved")
    st.metric("Rejection Confidence", f"{confidence*100:.2f}%")

st.caption("Result is based on machine learning analysis of user inputs")
