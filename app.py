
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Import your existing prediction function
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from prediction_pipeline import predict_churn


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write(
    "Enter customer details to estimate whether the customer "
    "is likely to leave the service."
)

st.info(
    "Complete all fields. The prediction is an estimate from "
    "the trained machine-learning model, not a guarantee."
)

with st.form("customer_form"):

    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Female", "Male"])

    senior = st.selectbox(
        "Senior Citizen", ["No", "Yes"]
    )
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.number_input(
        "Tenure Months", min_value=0, max_value=100,
        value=12, step=1
    )

    phone = st.selectbox("Phone Service", ["Yes", "No"])

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
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

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing", ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0,
        step=1.0
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0,
        step=10.0
    )

    cltv = st.number_input(
        "CLTV",
        min_value=0,
        value=4000,
        step=100
    )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )


if submitted:

    # Basic input validation
    if tenure < 0 or monthly_charges < 0:
        st.error("Please enter valid non-negative values.")
    elif total_charges < 0 or cltv < 0:
        st.error("Charges and CLTV cannot be negative.")
    else:

        customer = pd.DataFrame([{
            "Gender": gender,
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": phone,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
            "CLTV": cltv
        }])

        try:
            result = predict_churn(customer)
            prediction = result["Prediction"].iloc[0]

            st.subheader("Prediction Result")

            if prediction == "Churn":
                st.error("Prediction: Churn")
                st.write(
                    "The model predicts that this customer may "
                    "leave the service. Consider reviewing their "
                    "service experience and retention options."
                )

            elif prediction == "No Churn":
                st.success("Prediction: No Churn")
                st.write(
                    "The model predicts that this customer may "
                    "remain with the service. Continue providing "
                    "good customer support."
                )

            else:
                st.warning(
                    f"The model returned an unexpected label: "
                    f"{prediction}"
                )

        except (ValueError, TypeError) as error:
            st.error(f"Please check the customer details: {error}")

        except Exception as error:
            st.error(
                "The prediction could not be completed. "
                "Please check the model and input data."
            )
            st.exception(error)
