

from pathlib import Path
import joblib
import pandas as pd

# Locate the project and saved model
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "final_selected_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

final_model = joblib.load(MODEL_PATH)

# Exact columns expected by the saved model
REQUIRED_COLUMNS = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "CLTV",
]


def predict_churn(customer_data):
    """Predict Churn or No Churn for customer records."""

    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError(
            "Input must be a DataFrame or customer dictionary."
        )

    if customer_data.empty:
        raise ValueError("Customer data cannot be empty.")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in customer_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep the correct columns in the correct order
    customer_data = customer_data[REQUIRED_COLUMNS].copy()

    # Treat blank strings as missing
    customer_data = customer_data.replace(
        r"^\s*$", pd.NA, regex=True
    )

    if customer_data.isnull().any().any():
        missing_values = customer_data.columns[
            customer_data.isnull().any()
        ].tolist()

        raise ValueError(
            f"Missing values in columns: {missing_values}"
        )

    # The saved pipeline handles its existing preprocessing
    predictions = final_model.predict(customer_data)

    labels = {1: "Churn", 0: "No Churn"}

    results = pd.DataFrame({
        "Prediction": [
            labels.get(prediction, str(prediction))
            for prediction in predictions
        ]
    })

    return results
