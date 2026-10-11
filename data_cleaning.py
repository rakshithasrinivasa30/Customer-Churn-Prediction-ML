import pandas as pd

data = pd.read_csv("data/Customer-Churn-Prediction.csv")

print(data.head())

print("Shape of dataset:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nData types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isnull().sum())
print("\nDuplicate rows:")
print(data.duplicated().sum())
print("\nChurn Label distribution:")
print(data["Churn Label"].value_counts())
columns_to_drop = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Churn Reason",
    "Churn Score",
    "Churn Value"
]

data = data.drop(columns=columns_to_drop)

print("\nColumns after cleaning:")
print(data.columns)
print("\nMissing values after removing columns:")
print(data.isnull().sum())
data.to_csv("data/cleaned_churn_data.csv", index=False)

print("\nCleaned dataset saved successfully!")