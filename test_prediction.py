
from pathlib import Path
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from prediction_pipeline import predict_churn

# Find X_test.csv
X_TEST_PATH = next(
    PROJECT_DIR.rglob("X_test.csv"),
    None
)

if X_TEST_PATH is None:
    raise FileNotFoundError(
        "X_test.csv was not found in the project folder."
    )

X_test = pd.read_csv(X_TEST_PATH)

print("Test dataset:", X_TEST_PATH)
print("Dataset shape:", X_test.shape)
print("Dataset columns:", X_test.columns.tolist())

# Test with three customer records
sample_customers = X_test.head(3).copy()

results = predict_churn(sample_customers)

print("\nSample Customer Predictions:")
print(results)

# Save predictions
OUTPUT_DIR = PROJECT_DIR / "models" / "day10_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "sample_predictions.csv"
results.to_csv(OUTPUT_PATH, index=False)

print("\nPredictions saved to:", OUTPUT_PATH)
