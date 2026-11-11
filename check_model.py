import joblib

model = joblib.load("best_logistic_regression.pkl")

print("Model loaded successfully!")

print("\nRequired Features:")
print(model.feature_names_in_)