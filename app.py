import streamlit as st
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prediction_pipeline import predict_churn
# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Project title
st.title("📊 Customer Churn Prediction System")

# Introduction
st.write("""
Welcome to the Customer Churn Prediction System!

This application collects customer information
to help identify customers who may leave the service.

Enter customer details below to get started.
""")

st.divider()

st.subheader("Customer Information")
st.write("Please enter the customer details below.")
st.subheader("Enter Customer Details")

# Customer Basic Information
customer_id = st.text_input("Customer ID")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.number_input(
    "Tenure Months",
    min_value=0,
    max_value=100,
    value=12
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=840.0
)
latitude = st.number_input("Latitude", value=0.0)

longitude = st.number_input("Longitude", value=0.0)

cltv = st.number_input("CLTV", min_value=0.0, value=0.0)

st.write("Customer details entered successfully!")
st.subheader("Internet and Phone Services")

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)
st.subheader("Contract and Payment Information")

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.write("All customer information fields are completed!")

st.divider()

st.subheader("🔮 Churn Prediction")

if st.button("Predict Customer Churn", type="primary"):

    customer = pd.DataFrame([{
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
        "Latitude": latitude,
        "Longitude": longitude,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
        "CLTV": cltv,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method
    }])

    try:
        result = predict_churn(customer)
        prediction = result["Prediction"].iloc[0]

        st.subheader("Prediction Result")

        if prediction == "Churn":
            st.error("Prediction: Churn")
            st.write("This customer may leave the service.")

        else:
            st.success("Prediction: No Churn")
            st.write("This customer may continue using the service.")

    except Exception as error:
        st.error(f"Prediction error: {error}")