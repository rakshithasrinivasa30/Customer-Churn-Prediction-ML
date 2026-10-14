from pathlib import Path
import joblib
import pandas as pd

# 1. Find the main project folder
project_folder = Path(__file__).resolve().parent.parent

# 2. Load the saved Logistic Regression model
model_path = (
    project_folder
    / "models"
    / "best_logistic_regression.pkl"
)

if not model_path.exists():
    raise FileNotFoundError(
        f"Model file not found: {model_path}"
    )

model = joblib.load(model_path)

print("Model loaded successfully!")
print("Model type:", type(model).__name__)

# 3. Find the customer dataset
possible_files = [
    project_folder / "data" / "cleaned_churn_data.csv",
    project_folder / "data" / "processed_churn_data.csv",
    project_folder / "data" / "Customer-Churn-Prediction.csv",
]

data_path = next(
    (path for path in possible_files if path.exists()),
    None
)

if data_path is None:
    raise FileNotFoundError(
        "Customer CSV not found. Check the filenames "
        "inside your data folder."
    )

# 4. Read the dataset
df = pd.read_csv(data_path)

print("\nDataset loaded:", data_path.name)
print("Number of rows:", len(df))

# 5. Find the exact input columns expected by the model
if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "The model does not contain saved feature names. "
        "Check the original training notebook."
    )

required_columns = list(model.feature_names_in_)

# Remove the target column if it exists in the dataset
target_columns = [
    col for col in df.columns
    if col.strip().lower() in ["churn", "exited"]
]

features_df = df.drop(columns=target_columns)

# Check that all required columns exist
missing_columns = [
    col for col in required_columns
    if col not in features_df.columns
]

if missing_columns:
    print("\nMissing columns:")
    for col in missing_columns:
        print("-", col)

    print("\nAvailable dataset columns:")
    print(features_df.columns.tolist())

    raise ValueError(
        "The dataset does not contain all columns "
        "required by the saved model. Use the same "
        "input dataset and column names used during training."
    )

# 6. Select the correct columns in the correct order
X = features_df[required_columns]

# 7. Predict churn for one customer
customer = X.iloc[[0]]
prediction = model.predict(customer)

print("\nCustomer prediction:", prediction[0])

# 8. Display a readable result
if prediction[0] == 1 or str(prediction[0]).lower() in [
    "yes", "true"
]:
    print("Result: Customer is predicted to churn.")
else:
    print("Result: Customer is predicted not to churn.")

# 9. Display probability, if supported by the model
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(customer)[0]

    print("\nPrediction probabilities:")
    for label, probability in zip(
        model.classes_, probabilities
    ):
        print(f"{label}: {probability:.2%}")