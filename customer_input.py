
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_logistic_regression.pkl")

# Create sample customer data
customer = {
    'Zip Code': 90001,
    'Lat Long': '33.9731, -118.2479',
    'Latitude': 33.9731,
    'Longitude': -118.2479,
    'Gender': 'Female',
    'Senior Citizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'Tenure Months': 12,
    'Phone Service': 'Yes',
    'Multiple Lines': 'No',
    'Internet Service': 'Fiber optic',
    'Online Security': 'No',
    'Online Backup': 'Yes',
    'Device Protection': 'No',
    'Tech Support': 'No',
    'Streaming TV': 'Yes',
    'Streaming Movies': 'No',
    'Contract': 'Month-to-month',
    'Paperless Billing': 'Yes',
    'Payment Method': 'Electronic check',
    'Monthly Charges': 75.50,
    'Total Charges': 906.00
}

# Convert customer data into a DataFrame
customer_df = pd.DataFrame([customer])

# Match the exact training feature order
customer_df = customer_df[model.feature_names_in_]

# Display customer input data
print("\nCustomer Input Data:")
print(customer_df)

# Predict customer churn
prediction = model.predict(customer_df)

print("\nChurn Prediction:", prediction[0])
