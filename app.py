
import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------
# 1. Load trained ML model
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(
    BASE_DIR,
    "models",
    "best_logistic_regression.pkl"
)

model = joblib.load(model_path)

# --------------------------------
# 2. Page configuration
# --------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# --------------------------------
# 3. Project title
# --------------------------------

st.title("📊 Customer Churn Prediction System")

st.write("""
Welcome to the Customer Churn Prediction System!

This application uses Machine Learning to predict
whether a customer is likely to leave a service.
Enter customer details below to get a prediction.
""")

st.divider()

# --------------------------------
# 4. Customer information
# --------------------------------

st.header("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    customer_id = st.text_input("Customer ID")

    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen", ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner", ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents", ["Yes", "No"]
    )

    tenure = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=12
    )

    zip_code = st.number_input(
        "Zip Code",
        min_value=0,
        value=90001
    )

with col2:

    latitude = st.number_input(
        "Latitude",
        value=33.9731,
        format="%.4f"
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.2479,
        format="%.4f"
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

st.divider()

# --------------------------------
# 5. Internet and phone services
# --------------------------------

st.header("📱 Internet and Phone Services")

col3, col4 = st.columns(2)

with col3:

    phone_service = st.selectbox(
        "Phone Service", ["Yes", "No"]
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

with col4:

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

st.divider()

# --------------------------------
# 6. Contract and payment details
# --------------------------------

st.header("💳 Contract and Payment Information")

contract = st.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.selectbox(
    "Paperless Billing", ["Yes", "No"]
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

st.divider()

# --------------------------------
# 7. Customer churn prediction
# --------------------------------

st.header("🔮 Churn Prediction")

if st.button("Predict Customer Churn", type="primary"):

    customer = {
        "Zip Code": zip_code,
        "Lat Long": f"{latitude}, {longitude}",
        "Latitude": latitude,
        "Longitude": longitude,
        "Gender": gender,
        "Senior Citizen": (
            1 if senior_citizen == "Yes" else 0
        ),
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure,
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
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }

    try:

        customer_df = pd.DataFrame([customer])

        # Match exact model feature order
        customer_df = customer_df[
            model.feature_names_in_
        ]

        # Generate prediction
        prediction = model.predict(customer_df)[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ This customer is likely to churn!"
            )

        else:

            st.success(
                "✅ This customer is likely to stay!"
            )

        st.write("Customer ID:", customer_id)

        st.subheader("Customer Input Summary")
        st.dataframe(customer_df)

    except Exception as e:

        st.error(f"Prediction Error: {e}")
