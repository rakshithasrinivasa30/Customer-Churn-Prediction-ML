
from pathlib import Path
import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "best_decision_tree_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

final_model = joblib.load(MODEL_PATH)


def predict_churn(customer_data):

    if isinstance(customer_data, dict):
        customer_data = pd.DataFrame([customer_data])

    if not isinstance(customer_data, pd.DataFrame):
        raise TypeError("Input must be a DataFrame.")

    if customer_data.empty:
        raise ValueError("Customer data cannot be empty.")

    required = [
        "Gender", "Senior Citizen", "Partner", "Dependents",
        "Tenure Months", "Latitude", "Longitude",
        "Monthly Charges", "Total Charges", "CLTV",
        "Phone Service", "Multiple Lines", "Internet Service",
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies",
        "Contract", "Paperless Billing", "Payment Method"
    ]

    missing = [col for col in required if col not in customer_data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    row = customer_data.iloc[0]

    features = {
        "Latitude": row["Latitude"],
        "Longitude": row["Longitude"],
        "Tenure Months": row["Tenure Months"],
        "Monthly Charges": row["Monthly Charges"],
        "Total Charges": row["Total Charges"],
        "CLTV": row["CLTV"],
        "Gender_Male": int(row["Gender"] == "Male"),
        "Senior Citizen_Yes": int(row["Senior Citizen"] == "Yes"),
        "Partner_Yes": int(row["Partner"] == "Yes"),
        "Dependents_Yes": int(row["Dependents"] == "Yes"),
        "Phone Service_Yes": int(row["Phone Service"] == "Yes"),
        "Multiple Lines_No phone service": int(row["Multiple Lines"] == "No phone service"),
        "Multiple Lines_Yes": int(row["Multiple Lines"] == "Yes"),
        "Internet Service_Fiber optic": int(row["Internet Service"] == "Fiber optic"),
        "Internet Service_No": int(row["Internet Service"] == "No"),
        "Online Security_No internet service": int(row["Online Security"] == "No internet service"),
        "Online Security_Yes": int(row["Online Security"] == "Yes"),
        "Online Backup_No internet service": int(row["Online Backup"] == "No internet service"),
        "Online Backup_Yes": int(row["Online Backup"] == "Yes"),
        "Device Protection_No internet service": int(row["Device Protection"] == "No internet service"),
        "Device Protection_Yes": int(row["Device Protection"] == "Yes"),
        "Tech Support_No internet service": int(row["Tech Support"] == "No internet service"),
        "Tech Support_Yes": int(row["Tech Support"] == "Yes"),
        "Streaming TV_No internet service": int(row["Streaming TV"] == "No internet service"),
        "Streaming TV_Yes": int(row["Streaming TV"] == "Yes"),
        "Streaming Movies_No internet service": int(row["Streaming Movies"] == "No internet service"),
        "Streaming Movies_Yes": int(row["Streaming Movies"] == "Yes"),
        "Contract_One year": int(row["Contract"] == "One year"),
        "Contract_Two year": int(row["Contract"] == "Two year"),
        "Paperless Billing_Yes": int(row["Paperless Billing"] == "Yes"),
        "Payment Method_Credit card (automatic)": int(row["Payment Method"] == "Credit card (automatic)"),
        "Payment Method_Electronic check": int(row["Payment Method"] == "Electronic check"),
        "Payment Method_Mailed check": int(row["Payment Method"] == "Mailed check")
    }

    input_data = pd.DataFrame([features])
    input_data = input_data[list(final_model.feature_names_in_)]

    prediction = final_model.predict(input_data)[0]

    label = "Churn" if prediction == 1 else "No Churn"

    return pd.DataFrame({"Prediction": [label]})
